from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

from cl_lora.slice.cache import SliceCacheEntry, load_slice_cache, save_slice_cache
from cl_lora.slice.apply import apply_slice_inits
import cl_lora.slice.compute as slice_compute
from cl_lora.slice.compute import _model_cache_identity, summarize_projection_stats
from cl_lora.slice.config import SliceInitConfig
from cl_lora.slice.decompose import build_ab_from_gradient
from cl_lora.slice.gradients import accumulate_gradients
from cl_lora.slice.projections import project_gradients_advanced


def test_lora_ga_rejects_rank_without_two_disjoint_subspaces():
    with pytest.raises(ValueError, match=r"2 \* rank <= min\(G.shape\).+rank=3.+shape=\(4, 8\)"):
        build_ab_from_gradient(torch.randn(4, 8), r=3, weight_var=1.0)


def test_lora_ga_documents_and_returns_approximate_not_exact_zero_product():
    torch.manual_seed(0)
    ab = build_ab_from_gradient(torch.randn(8, 8), r=3, weight_var=1.0)
    assert torch.isfinite(ab["B"] @ ab["A"]).all()
    assert "approximately" in build_ab_from_gradient.__doc__.lower()
    assert "exact" not in build_ab_from_gradient.__doc__.lower()


def test_cache_round_trip_persists_tensors_and_metadata(tmp_path: Path):
    entry = SliceCacheEntry({"layer.weight": {"A": torch.randn(2, 3), "B": torch.randn(4, 2)}})
    save_slice_cache(str(tmp_path), "key", entry, meta={"payload": {"lora": {"lora_alpha": 7.0}}})
    loaded = load_slice_cache(str(tmp_path), "key")
    assert loaded is not None
    assert torch.equal(loaded.inits["layer.weight"]["A"], entry.inits["layer.weight"]["A"])
    assert json.loads((tmp_path / "key" / "meta.json").read_text())["payload"]["lora"]["lora_alpha"] == 7.0


def test_load_or_compute_produces_real_second_call_cache_hit(tmp_path: Path, monkeypatch):
    calls = 0

    def fake_compute(*, model, config):
        nonlocal calls
        calls += 1
        return {"layer.weight": {"A": torch.ones(1, 2), "B": torch.ones(2, 1)}}

    monkeypatch.setattr(slice_compute, "compute_loram_inits", fake_compute)
    model = torch.nn.Linear(2, 2)
    config = SliceInitConfig(cache_dir=str(tmp_path), init_method="loram", rank=1, lora_alpha=7.0)
    first, root, _ = slice_compute.load_or_compute_slice_inits(model, None, "task", [], config=config)
    second, second_root, _ = slice_compute.load_or_compute_slice_inits(model, None, "task", [], config=config)
    assert calls == 1
    assert root == second_root
    assert torch.equal(first["layer.weight"]["A"], second["layer.weight"]["A"])
    meta = json.loads((Path(root) / "meta.json").read_text())
    assert meta["payload"]["lora"]["lora_alpha"] == 7.0


def test_model_cache_identity_uses_revision_and_live_weights():
    model = torch.nn.Linear(2, 2)
    model.config = types.SimpleNamespace(_commit_hash="abc123", _name_or_path="mutable/model")
    identity = _model_cache_identity(model)
    assert identity["revision"] == "abc123"
    assert "weight_fingerprint" in identity


def test_model_cache_identity_fingerprints_weights_without_revision():
    model = torch.nn.Linear(2, 2)
    model.config = types.SimpleNamespace(_name_or_path="local/model")
    first = _model_cache_identity(model)
    with torch.no_grad():
        model.weight[0, 0].add_(1)
    second = _model_cache_identity(model)
    assert first["weight_fingerprint"] != second["weight_fingerprint"]


def _advanced(**overrides):
    kwargs = dict(
        method="gradvac", cosine_threshold=None, per_layer_threshold=False,
        per_layer_threshold_delta=0.0, pcgrad_c=0.5, gradvac_phi=0.5,
        gradvac_beta=0.5, magnitude_preserve=False, nullspace_rank=1,
        nullspace_sv_threshold=0.0, always_project=False,
        add_retain_grad=False, global_projection=False,
    )
    kwargs.update(overrides)
    return kwargs


def test_gradvac_uses_its_phi_gate_for_positive_cosine():
    current = {"x": torch.tensor([1.0, 0.0])}
    retain = {"x": torch.tensor([0.2, 0.98])}
    projected, stats = project_gradients_advanced(current, retain, **_advanced())
    assert stats["modules"]["x"]["action"] == "gradvac"
    assert not torch.equal(projected["x"], current["x"])


def test_gradvac_phi_state_persists_across_calls():
    state = {}
    current = {"x": torch.tensor([1.0, 0.0])}
    retain = {"x": torch.tensor([0.0, 1.0])}
    project_gradients_advanced(current, retain, gradvac_state=state, **_advanced(gradvac_phi=0.6, gradvac_beta=0.25))
    assert state["x"] == pytest.approx(0.45)
    project_gradients_advanced(current, retain, gradvac_state=state, **_advanced(gradvac_phi=0.9, gradvac_beta=0.25))
    assert state["x"] == pytest.approx(0.3375)


def test_advanced_summary_reports_per_module_changes():
    stats = {
        "applied": True, "method": "gradvac", "mode": "per_module",
        "modules": {"x": {"action": "gradvac", "cos": 0.2, "current_norm": 2.0,
                           "retain_norm": 1.0, "projected_norm": 2.5}},
    }
    summary = summarize_projection_stats(stats)
    assert summary["fired"] is True
    assert summary["n_modules_changed"] == 1
    assert summary["rel_change"] is None  # norms alone cannot recover vector difference


class _FailingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = types.SimpleNamespace(use_cache=True)
        self.is_gradient_checkpointing = False
        self.gc_disabled = False

    def gradient_checkpointing_enable(self, **kwargs):
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False
        self.gc_disabled = True

    def forward(self, **batch):
        self.config.use_cache = False
        raise RuntimeError("boom")


def test_gradient_capture_restores_state_after_forward_exception():
    model = _FailingModel()
    with pytest.raises(RuntimeError, match="boom"):
        accumulate_gradients(model, [{"x": torch.ones(1)}], {"weight": model.weight}, torch.device("cpu"), 1)
    assert model.weight.requires_grad is False
    assert model.is_gradient_checkpointing is False
    assert model.gc_disabled is True
    assert model.config.use_cache is True


def test_gradient_capture_restores_requires_grad_when_checkpoint_enable_fails():
    model = _FailingModel()

    def fail_enable(**kwargs):
        raise RuntimeError("checkpoint failure")

    model.gradient_checkpointing_enable = fail_enable
    with pytest.raises(RuntimeError, match="checkpoint failure"):
        accumulate_gradients(model, [], {"weight": model.weight}, torch.device("cpu"), 1)
    assert model.weight.requires_grad is False
    assert model.config.use_cache is True


def test_slice_config_has_explicit_alpha_and_persistent_gradvac_state():
    config = SliceInitConfig(lora_alpha=9.0)
    assert config.lora_alpha == 9.0
    assert config.gradvac_state == {}


def test_apply_requires_every_computed_init_to_match(monkeypatch):
    class FakeLoraLinear(torch.nn.Module):
        pass

    peft = types.ModuleType("peft")
    tuners = types.ModuleType("peft.tuners")
    lora = types.ModuleType("peft.tuners.lora")
    lora.Linear = FakeLoraLinear
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "peft.tuners", tuners)
    monkeypatch.setitem(sys.modules, "peft.tuners.lora", lora)

    with pytest.raises(RuntimeError, match="complete application required"):
        apply_slice_inits(
            torch.nn.Sequential(),
            {"missing.weight": {"A": torch.ones(1, 1), "B": torch.ones(1, 1)}},
            r=1,
            skip_absorption=True,
        )


def test_documented_scripts_only_advertise_implemented_methods():
    root = Path(__file__).parents[1]
    runnable = "\n".join(
        (root / path).read_text()
        for path in ("scripts/test_init_x_cl_methods.sh", "scripts/test_init_x_cl_methods_lean.sh")
    )
    assert "inflora" not in runnable.lower()
    assert "sapt" not in runnable.lower()
    readme = (root / "README.md").read_text()
    assert "cagrad" not in readme.lower()
    assert "--slice-pcgrad-c" in readme
    assert "quarantined" in readme.lower()


def test_portable_scripts_do_not_contain_personal_mount_paths():
    root = Path(__file__).parents[1]
    for name in ("eval.sh", "check_evals.sh"):
        text = (root / "scripts" / name).read_text()
        assert "/mnt/" not in text
        assert 'REPO_ROOT="${SCRIPT_DIR}/.."' in text


def test_dependencies_are_bounded_and_environment_defers_to_requirements():
    root = Path(__file__).parents[1]
    requirements = (root / "requirements.txt").read_text()
    for package in ("torch", "datasets", "peft", "requests", "typer", "lm-eval"):
        line = next(line for line in requirements.splitlines() if line.startswith(package))
        assert any(op in line for op in ("==", ">=", "<="))
    assert "git+" not in requirements
    environment = (root / "environment.yml").read_text()
    assert "- -r requirements.txt" in environment
    assert "cuda-version=12.9" not in environment
