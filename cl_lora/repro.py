from __future__ import annotations

import os
import random
import warnings
from typing import Any, Dict


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
    set_env: bool = True,
) -> Dict[str, Any]:
    """Set RNG seeds and return a report containing any reproducibility failures."""
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    report: Dict[str, Any] = {"seed": seed, "deterministic": deterministic, "failures": []}

    def failed(component: str, exc: BaseException) -> None:
        message = f"{component} seeding failed: {type(exc).__name__}: {exc}"
        report["failures"].append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    if set_env:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    random.seed(seed)

    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception as exc:
        failed("NumPy", exc)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
            except TypeError:
                torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as exc:
        failed("PyTorch", exc)

    try:
        from transformers import set_seed as hf_set_seed
        hf_set_seed(seed)
    except Exception as exc:
        failed("Transformers", exc)

    return report
