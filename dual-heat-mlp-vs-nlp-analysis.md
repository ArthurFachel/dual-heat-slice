# Dual-Heat: Análise Técnica Comparativa entre MLP e NLP/LoRA

**Arthur Fachel — MALTA Lab, PUCRS**

---

## 1. Visão Geral

### 1.1 O Problema

Catastrophic forgetting (McCloskey & Cohen, 1989) ocorre quando o treino em uma nova tarefa **T₂** sobrescreve representações aprendidas na tarefa anterior **T₁**. A acurácia em **T₁** cai abruptamente após o treino em **T₂**.

### 1.2 Dual-Heat: Intuição Geral

Dual-Heat propõe um mecanismo de regularização neural de **duas escalas temporais** que opera simultaneamente no forward pass (inibição lateral divisiva) e no backward pass (modulação do gradiente por neurônio). A ideia central é:

1. **A curto prazo (fast heat)**: neurônios competem entre si via inibição lateral. Um neurônio que dispara forte suprime os demais, forçando especialização por tarefa.
2. **A longo prazo (slow heat)**: neurônios com alta atividade média acumulam proteção EWC — seus gradientes são reduzidos, prevenindo que conhecimento consolidado seja sobrescrito.

A medida de importância usada é a **magnitude de ativação pós-inibição** (`|output|`), um proxy barato (O(n_neurons)) para a sensibilidade da loss ao neurônio.

---

## 2. Dual-Heat Original para MLP

**Fonte:** `dual_heat_module.py` (v3), linhas 1–235.

### 2.1 Arquitetura

O `DualHeatLinear` substitui `nn.Linear` camada a camada. Mantém todos os parâmetros de uma linear padrão mais os buffers de calor:

```python
self.weight = nn.Parameter(torch.randn(out_features, in_features))
self.bias = nn.Parameter(torch.zeros(out_features))

self.register_buffer("fast_heat", torch.zeros(out_features))   # [N]
self.register_buffer("slow_heat", torch.zeros(out_features))   # [N]
self.register_buffer("slow_n", torch.ones(1))                   # contador
```

Um `DualHeatMLP` é um `nn.Sequential` de `DualHeatLinear + ativação` para as camadas ocultas, terminando com `nn.Linear` padrão (sem heat) na última camada.

### 2.2 Parâmetros Envolvidos

| Símbolo | Código | Descrição |
|---------|--------|-----------|
| α | `fast_decay` | Decaimento EMA do fast heat (0.85–0.97) |
| γ | `fast_strength` | Força da inibição lateral (1.0–5.0) |
| δ | `fast_decay_rate` | Decaimento ativo por passo (0.02–0.08) |
| β | `slow_strength` | Força EWC no gradiente (0.0–5.0) |
| W | `slow_window` | Janela de memória do slow heat (None = ∞) |

**Treinados:** `self.weight` (W ∈ ℝ^{N×D}), `self.bias` (b ∈ ℝ^N).

**Congelados:** `fast_heat`, `slow_heat`, `slow_n` (buffers, sem gradiente).

### 2.3 Fluxo de Treinamento — Passo a Passo

Para uma camada com `N` neurônios de saída, entrada `x ∈ ℝ^B×D`:

**Passo 1 — Pré-ativação** (linha 126):
$$
z = W x + b, \quad z \in \mathbb{R}^{B \times N}
$$

**Passo 2 — Inibição Lateral Divisiva** (linhas 129–135):
$$
\text{output}_i = \frac{z_i}{1 + \gamma \cdot \frac{1}{N-1} \sum_{j \neq i} \text{fast\_heat}_j^{(t-1)}}, \quad i = 1, \ldots, N
$$

A inibição é **lateral** (cada neurônio é inibido pela média dos outros) e **divisiva** (escala o sinal, sem inverter). O fast heat usado é do passo anterior, evitando dependência circular.

Se `fast_strength <= 0` ou `N == 1`, `output = z` (sem inibição).

**Passo 3 — Atualização do Fast Heat** (linhas 143–145):

Primeiro calcula-se a magnitude pós-inibição, reduzindo sobre o batch:
$$
\text{post\_mag}_i = \frac{1}{B} \sum_{k=1}^{B} |\text{output}_{k,i}|
$$

Depois:
$$
\text{fast\_heat}_i^{(t)} = \max\!\big(0,\; \alpha \cdot \text{fast\_heat}_i^{(t-1)} + (1-\alpha) \cdot \text{post\_mag}_i - \delta\big)
$$

**Interpretação:** α controla a inércia do EMA. O termo −δ funciona como limiar de atividade: se `post_mag < δ/(1-α)`, o fast heat zera.

**Passo 4 — Atualização do Slow Heat** (linhas 147–151):

$$
n_{\text{eff}} = \begin{cases}
\min(n, W) & \text{se } W \text{ fornecida} \\
n & \text{caso contrário}
\end{cases}
$$

$$
\text{slow\_heat}_i^{(t)} = \text{slow\_heat}_i^{(t-1)} + \frac{\text{post\_mag}_i - \text{slow\_heat}_i^{(t-1)}}{n_{\text{eff}}}
$$

Para `n <= W` (ou `W = None`), isso é **exatamente a média amostral** (unbiased, consistent estimator). Para `n > W`, comporta-se como EMA com taxa efetiva `η = 1/W`.

O contador `slow_n` é incrementado a cada passo (linha 151).

**Passo 5 — EWC Gradient Hook** (linhas 155–162):

O hook é registrado **no próprio `self.weight`** e **no `self.bias`** no `__init__` (linhas 120–122):

```python
self.weight.register_hook(self._ewc_hook())
self.bias.register_hook(self._ewc_hook())
```

O hook (linhas 157–162):
$$
\nabla_{w_{ij}} \leftarrow \frac{\nabla_{w_{ij}}}{1 + \beta \cdot \text{slow\_heat}_i}, \quad \nabla_{b_i} \leftarrow \frac{\nabla_{b_i}}{1 + \beta \cdot \text{slow\_heat}_i}
$$

Em código:
```python
scale = 1.0 / (1.0 + self.slow_strength * self.slow_heat)  # (N,)
return grad * scale.view(-1, *([1] * (grad.dim() - 1)))
```

Para `self.weight` (grad shape [N, D]), `scale.view(-1, 1)` faz broadcasting correto: cada linha `i` da matriz de pesos (todas as conexões de entrada do neurônio `i`) é escalada por `slow_heat[i]`.

### 2.4 Mecanismo de Continual Learning

O ciclo completo:

```
1. forward(z = Wx + b)                     — pré-ativação padrão
2. output = z / (1 + γ · mean_others)      — inibição lateral (competição)
3. fast_heat atualizado com |output|       — memória curta (quem está ativo agora)
4. slow_heat atualizado com |output|       — memória longa (quem foi importante)
5. backward: grad /= (1 + β · slow_heat)   — proteção EWC
6. optimizer.step()                        — pesos atualizados com gradiente já escalado
```

**Como o conhecimento anterior é representado:** `slow_heat` — um vetor per-neuron contendo a média amostral de `|output|` desde o início do treino. É uma _proxy_ de importância: neurônios com alta ativação média são presumidos importantes.

**Como o novo conhecimento é incorporado:** através da atualização normal dos pesos (`W`, `b`) via SGD/Adam, mas com o gradiente modulado. O LR efetivo de cada neurônio é:
$$
\eta_{\text{eff}, i} = \frac{\eta}{1 + \beta \cdot \text{slow\_heat}_i}
$$

**Onde ocorre a competição:** no forward pass (Passo 2), usando `fast_heat`. Neurônios quentes inibem os outros, forçando que diferentes tarefas recrutem diferentes subconjuntos de neurônios.

**Onde ocorre a proteção contra forgetting:** no backward pass (Passo 5), usando `slow_heat`. Neurônios com alta importância histórica têm gradiente reduzido.

### 2.5 Pseudocódigo

```
Para cada passo de treino t:
    Para cada camada DualHeatLinear:
        z = W @ x + b
        mean_others = (sum(fast_heat) - fast_heat) / (N - 1)
        output = z / (1 + γ * mean_others)           # inibição lateral
        post_mag = |output|.mean(dim=batch)          # [N]

        fast_heat = max(0, α·fast_heat + (1-α)·post_mag - δ)
        slow_heat += (post_mag - slow_heat) / min(slow_n, W)

    loss = CrossEntropy(output_final, target)
    loss.backward()    # → hooks disparam: grad_W /= (1 + β·slow_heat)
    optimizer.step()
```

---

## 3. Dual-Heat para NLP com LoRA

**Fonte:** `dual-heat-slice/cl_lora/cl_methods/dual_heat.py`, linhas 1–458.

### 3.1 Arquitetura

Diferentemente do MLP, a versão NLP **não substitui as camadas lineares do Transformer**. Em vez disso, utiliza-se o ecossistema PEFT (HuggingFace), onde LoRA adiciona dois parâmetros treináveis (`lora_A`, `lora_B`) a cada módulo alvo (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — ver `lora_config.py` linha 9–16), mantendo os pesos pré-treinados congelados.

A arquitetura segue o padrão `CLMethod` (interface em `base.py`):

```
CLMethod
  ├── pre_train(lora_model)     → registra hooks
  ├── aux_loss(lora_model)      → None (DualHeat não usa loss adicional)
  ├── post_train(lora_model)    → captura estado
  ├── save(state_dir)           → persiste heat state entre tasks
  └── load(state_dir)           → restaura heat state entre tasks
```

O `DualHeatCLMethod` gerencia uma coleção de `_DualHeatModule`, um por módulo LoRA ativo.

### 3.2 Parâmetros Envolvidos

**Treinados (PEFT LoRA padrão):** `lora_A.weight` (A ∈ ℝ^{r×D}) e `lora_B.weight` (B ∈ ℝ^{N×r}) para cada módulo alvo. `r` é o rank (default 64).

**Congelados:** Todos os pesos do Transformer base (`base_weight`, `base_bias`), gerenciados pelo PEFT.

**Buffers (não treináveis):** `fast_heat`, `slow_heat`, `slow_n`, `step`, armazenados em `_DualHeatModule`, que por sua vez mantém dicionários por-device (`_per_device`) para compatibilidade com DataParallel.

### 3.3 Integração no Pipeline de Treinamento

O ciclo de vida completo, conforme `orchestrator.py`:

**Entre tarefas:**
```
1. cl_method.load(cl_state_dir)       — carrega heat state da tarefa anterior
2. train_on_task(...)                 — treina nova tarefa
   a. lora_model = get_peft_model(...)  — cria adaptadores LoRA
   b. cl_method.pre_train(lora_model)   → registra hooks
   c. Trainer.train()                   → treino com hooks ativos
   d. cl_method.post_train(...)         → salva snapshot do heat
3. cl_method.save(cl_state_dir)       — persiste heat state
```

O heat state é preservado **entre tarefas**: `slow_heat` carregado da tarefa anterior é transferido para os novos adaptadores LoRA (criados frescos a cada task) via `load_state_snapshot` → `_get_or_restore`.

### 3.4 Registro de Hooks

Em `pre_train` (linhas 331–354), `_register_hooks` (linhas 291–319):

Para cada módulo LoRA ativo:

1. **Cria um `_DualHeatModule`** com os mesmos hiperparâmetros (`α`, `γ`, `δ`, `β`, `W`) do original.

2. **Registra forward hook** na saída do módulo LoRA (linha 314):
```python
fwd_hook = self._make_forward_hook(name, out_features)
self._fwd_handles.append(mod.register_forward_hook(fwd_hook))
```

3. **Registra backward hook** em `lora_B.weight` (linhas 316–317):
```python
bwd_hook = self._make_ewc_hook(name, B_w)
self._bwd_handles.append(B_w.register_hook(bwd_hook))
```

### 3.5 Fluxo Durante o Treino

**Forward hook** (linhas 257–289):

Para um módulo LoRA com saída `output` (já calculada pelo PEFT: `base(x) + B·A·x·scaling`):

```
1. Se lateral_inhibition e training:
     state = dh_mod._get_or_restore(output.device)
     mean_others = (sum(fast_heat) - fast_heat) / (N - 1)
     output = output / (1 + γ · mean_others)

2. Se training:
     dh_mod.update_heat(output_para_rastrear)
```

**`update_heat`** (linhas 151–169 do `_DualHeatModule`):

```python
reduce_dims = tuple(range(output.dim() - 1))  # todas exceto última
post_mag = output.detach().abs().mean(dim=reduce_dims)

# Fast heat: EMA + decay (idêntico ao original)
state["fast_heat"].mul_(self.fast_decay).add_(
    (1.0 - self.fast_decay) * post_mag, alpha=1.0
).sub_(self.fast_decay_rate).clamp_(min=0.0)

# Slow heat: capped incremental mean (idêntico ao original)
n_eff = min(state["slow_n"], slow_window) if slow_window else state["slow_n"]
state["slow_heat"].add_((post_mag - state["slow_heat"]) / float(n_eff))
state["slow_n"] += 1
```

Note que, diferentemente do MLP, o `reduce_dims` aqui é `tuple(range(output.dim() - 1))` em vez de `0`. Isso permite que o output seja 2D (batch, features), 3D (batch, seq_len, hidden), ou qualquer dimensionalidade — essencial para Transformers que operam com sequências.

**Backward hook** (linhas 244–255):

```python
def hook(grad: torch.Tensor) -> torch.Tensor:  # grad shape: (N, r)
    dh_mod = self._dual_modules.get(name)
    if dh_mod is None:
        return grad
    scale = dh_mod.get_ewc_scale(grad.device, dtype=grad.dtype)  # (N,)
    return grad * scale.view(-1, 1).to(dtype=grad.dtype, device=grad.device)
```

O hook atua **diretamente no gradiente de `lora_B.weight`** (shape [N, r]). O fator `scale.view(-1, 1)` faz broadcasting: cada linha `i` de B (as `r` conexões do neurônio de saída `i`) tem seu gradiente escalado por `slow_heat[i]`.

### 3.6 Persistência do Estado

**Entre tarefas**, o heat state é salvo via `save()` (linhas 386–412) e carregado via `load()` (linhas 414–431) em um arquivo `dual_heat_state.pt` no diretório `cl_state_dir/`.

O estado inclui, para cada módulo LoRA:
- `fast_heat`, `slow_heat`, `slow_n`, `step`
- Hiperparâmetros (`α`, `γ`, `δ`, `β`, `W`)

No início da próxima tarefa, `pre_train` → `_register_hooks` → `restore_heat_state` (linhas 447–458) transfere o estado carregado (CPU) para os novos `_DualHeatModule` (no device correto via `_get_or_restore`).

### 3.7 Equações Matemáticas (NLP)

**Forward (por módulo LoRA):**

$$
z = \underbrace{W_{\text{base}} x + b_{\text{base}}}_{\text{congelado}} + \underbrace{\frac{\alpha_{\text{lora}}}{r} \cdot B \cdot A \cdot x}_{\text{treinável}}, \quad B \in \mathbb{R}^{N \times r}, \; A \in \mathbb{R}^{r \times D}
$$

**Inibição lateral (forward hook):**
$$
\text{output}_i = \frac{z_i}{1 + \gamma \cdot \frac{1}{N-1} \sum_{j \neq i} \text{fast\_heat}_j^{(t-1)}}
$$

**Atualização dos heats** (idêntico ao MLP):
$$
\text{fast\_heat}_i = \max(0, \alpha \cdot \text{fast\_heat}_i + (1-\alpha) \cdot |\text{output}|_i - \delta)
$$
$$
\text{slow\_heat}_i = \text{slow\_heat}_i + \frac{|\text{output}|_i - \text{slow\_heat}_i}{\min(n, W)}
$$

onde:
$$
|\text{output}|_i = \frac{1}{B \cdot S} \sum_{k=1}^{B} \sum_{l=1}^{S} |\text{output}_{k,l,i}|
$$
(para sequências de comprimento S — média sobre batch e tempo)

**Backward hook (diretamente em B):**
$$
\nabla_{B_{ij}} \leftarrow \frac{\nabla_{B_{ij}}}{1 + \beta \cdot \text{slow\_heat}_i}
$$

**Efeito no gradiente de A (via chain rule):**
$$
\nabla_{A} = \frac{\alpha_{\text{lora}}}{r} \cdot B^{\top} \cdot \nabla_{z}
$$

onde `∇_z` já foi escalado pelo hook... **exceto que** o hook está em B, não em z ou delta.

Quem precisa de atenção aqui: o hook está registrado em `lora_B.weight`. Quando o backward passa por B, o gradiente de B é escalado. Mas o gradiente que flui de volta para A passa **através** de B — e ele é escalado já no nível da saída? Não exatamente. O gradiente chega em `z` (a saída combinada base+LoRA), vem da loss. A regra da cadeia para o delta:

∂L/∂A = (∂L/∂z) · (∂z/∂A) = (∂L/∂z) · (scaling · B^T · x)

Mas o hook está em B, não em z. Então **∂L/∂z não é escalado** (ele vem diretamente da loss). A escala acontece quando o gradiente passa por B:

∂L_grad_B_scaled / ∂B = ∂L/∂z * (scaling · A · x)^T  ... escalado por slow_heat

E o gradiente que volta para A:
∂L/∂A = B^T · ∂L/∂z · scaling · x ... este gradiente NÃO É DIRETAMENTE escalado.

**MAS** na prática, como B é atualizado menos (gradiente escalado), as mudanças em B são menores, e isso se propaga para futuras atualizações de A. Mas não há proteção direta no gradiente de A.

---

## 4. Comparação Matemática

### 4.1 Tabela Comparativa

| Aspecto | MLP (DualHeatLinear) | NLP + LoRA (DualHeatCLMethod) |
|---|---|---|
| **Parâmetros atualizados** | `W` (N×D), `b` (N) — pesos completos | `B` (N×r), `A` (r×D) — fatores LoRA |
| **Parâmetros congelados** | `fast_heat`, `slow_heat` (buffers) | Pesos base do Transformer + `fast_heat`, `slow_heat` |
| **Representação do conhecimento anterior** | `slow_heat` ∈ ℝ^N por camada DualHeatLinear | `slow_heat` ∈ ℝ^N por módulo LoRA |
| **Estado antigo** | `fast_heat` + `slow_heat` no módulo | `_DualHeatModule._per_device[device]` (dict) |
| **Novo conhecimento** | ΔW via SGD (gradiente escalado) | ΔB, ΔA via SGD (apenas gradiente de B escalado) |
| **Atualização** | `W ← W - η · ∇_W / (1 + β·slow_heat)` | `B ← B - η · ∇_B / (1 + β·slow_heat)`; `A ← A - η · ∇_A` (sem escala direta) |
| **Gradientes** | Hook em `self.weight` e `self.bias` — todos os parâmetros protegidos | Hook apenas em `lora_B.weight` — B protegido, A protegido indiretamente |
| **Espaço dos parâmetros** | Completo: ℝ^{N×D} | Baixo posto: ℝ^{N×r} + ℝ^{r×D} |
| **Mecanismo de proteção contra forgetting** | Escala do gradiente por slow_heat, por neurônio de saída | Escala do gradiente de B por slow_heat, por neurônio de saída (A sem escala direta) |
| **Inibição lateral** | Divisiva, dentro do `forward` da camada | Divisiva, via forward hook na saída do módulo LoRA |
| **Dimensionalidade do output** | 2D (batch, features) | N-D (batch, [seq_len, ...], features) |

### 4.2 Diferenças Matemáticas Críticas

**Proteção assimétrica no LoRA.** No MLP original, o hook em `self.weight` protege **todos os parâmetros** que convergem para o neurônio `i`:
$$
\nabla_{W_{i,:}} \leftarrow \frac{\nabla_{W_{i,:}}}{1 + \beta \cdot \text{slow\_heat}_i}
$$

No LoRA, apenas `B` é diretamente protegido:
$$
\nabla_{B_{i,:}} \leftarrow \frac{\nabla_{B_{i,:}}}{1 + \beta \cdot \text{slow\_heat}_i}
$$

O gradiente de `A` não recebe escala direta. A proteção em A é **indireta**: como B muda menos, o produto B·A muda menos, mas isso não impede que A mude arbitrariamente nas direções que não afetam B (e.g., mudanças no espaço nulo de B).

**Equivalência matemática incompleta.** Considere a atualização do delta LoRA:
$$
\Delta_t = \frac{\alpha}{r} B_t A_t, \quad \Delta_{t+1} = \frac{\alpha}{r} B_{t+1} A_{t+1}
$$

A mudança é:
$$
\Delta_{t+1} - \Delta_t = \frac{\alpha}{r} \big[ (B_t + \Delta B)(A_t + \Delta A) - B_t A_t \big]
= \frac{\alpha}{r} \big[ \Delta B \cdot A_t + B_t \cdot \Delta A + \Delta B \cdot \Delta A \big]
$$

No MLP original, a proteção age sobre **todo** ΔW (protegendo todas as direções no espaço de pesos completo). No LoRA, ΔB é diretamente reduzido (pelo hook), mas ΔA não. Como resultado, o termo `B_t · ΔA` pode mover livremente a saída mesmo com `ΔB ≈ 0`, desde que `B_t ≠ 0`. Este é um **vazamento de atualização** que não existe no MLP original.

### 4.3 A Visão do DualHeatLoRALinear (Standalone)

Nota-se que existe uma segunda implementação LoRA, `~/Desktop/dual_heat/dual_heat_LoRA_module.py`, que adota uma estratégia **diferente** e potencialmente **mais correta**: ela hooka o tensor `delta` (a contribuição LoRA), não `lora_B.weight`:

```python
delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
if self.slow_strength > 0.0 and delta.requires_grad:
    delta.register_hook(self._ewc_scale_output)
z = base + delta
```

Aqui, `_ewc_scale_output` escala o gradiente que chega em `delta` com shape `(..., N)`. Pela regra da cadeia:

$$
\nabla_{B} = \nabla_{\text{delta}} \cdot (A \cdot x)^{\top}, \quad \nabla_{A} = B^{\top} \cdot \nabla_{\text{delta}} \cdot x^{\top}
$$

Como `∇_delta` já está escalado, **ambos** `∇_B` e `∇_A` herdam a escala. Isto é matematicamente equivalente ao comportamento do MLP original, onde o hook em `self.weight` escala o gradiente **antes** dele se propagar para qualquer parâmetro.

**A implementação DualHeatCLMethod (usada no pipeline) não faz isso** — ela hooka `lora_B.weight` diretamente, perdendo a proteção simétrica sobre A.

---

## 5. O que Permanece Igual

### 5.1 O Invariante Conceitual do Dual-Heat

A despeito das diferenças arquiteturais, o **invariante** de ambas as implementações é:

> **A intensidade de atualização de cada parâmetro que contribui para a saída do neurônio `i` é modulada por um fator `1/(1 + β · s_i)`, onde `s_i` é a média histórica de `|output_i|`.**

Formalmente, ambas as implementações seguem o esquema:

$$
\theta_{t+1} = \theta_t - \eta \cdot \underbrace{\frac{1}{1 + \beta \cdot s_i}}_{\text{escala de proteção}} \cdot \nabla_{\theta} \mathcal{L}
$$

Onde:
- `θ` são todos os parâmetros que **diretamente** produzem a saída do neurônio `i`
- `s_i = slow_heat[i]` — média amostral de `|output_i|`
- O fator de escala é o mesmo em ambas (linha 160 do MLP vs linha 176 do NLP)
- A inibição lateral (Passo 2) é **idêntica** em ambas: `output = z / (1 + γ · mean(fast_heat_others))`
- A atualização dos heats (Passos 3–4) é **idêntica** em ambas, com os mesmos hiperparâmetros `α, γ, δ, β`

### 5.2 O Mesmo Algoritmo em Pseudocódigo Abstrato

```
Para cada passo de treino:
    Para cada neurônio i (em cada camada/módulo):
        z_i = f(x; θ_i)                          # pré-ativação
        output_i = z_i / (1 + γ · M_i)           # inibição lateral (M_i = média dos fast_heat dos outros)
        m_i = |output_i|                         # magnitude pós-inibição

        fast_heat[i] = max(0, α·fast_heat[i] + (1-α)·m_i - δ)
        slow_heat[i] += (m_i - slow_heat[i]) / n_eff

    loss.backward()

    Para cada parâmetro θ:
        se θ contribui para o output do neurônio i:
            grad[θ] /= (1 + β · slow_heat[i])
```

### 5.3 Componentes Idênticos

| Componente | MLP | NLP | Status |
|---|---|---|---|
| Fórmula da inibição lateral | `z / (1+γ·mean_others)` | `output / (1+γ·mean_others)` | Idêntica |
| Fast heat update | `α·h + (1-α)·m - δ`, clamp(0) | `α·h + (1-α)·m - δ`, clamp(0) | Idêntica |
| Slow heat update (capped incremental mean) | `h += (m-h) / min(n,W)` | `h += (m-h) / min(n,W)` | Idêntica |
| EWC scale formula | `1/(1+β·slow_heat)` | `1/(1+β·slow_heat)` | Idêntica |
| Hiperparâmetros | α, γ, δ, β, W | α, γ, δ, β, W | Idênticos |
| Escopo da proteção | Por neurônio de saída | Por neurônio de saída | Mesmo princípio |
| Medida de importância | `mean(|output|)` | `mean(|output|)` | Idêntica |

---

## 6. O que Muda

### 6.1 Mudanças Arquiteturais

| Aspecto | MLP | NLP | Causa |
|---|---|---|---|
| Estrutura da camada | `DualHeatLinear` (subclasse de `nn.Linear`) | LoRA via PEFT + hooks | LoRA não substitui camadas; adiciona adaptadores |
| Onde os parâmetros vivem | `self.weight`, `self.bias` no módulo | `lora_A`, `lora_B` no módulo PEFT | Diferença fundamental entre peso cheio e adaptador LoRA |
| Onde o heat vive | Buffers no próprio módulo | `_DualHeatModule` separado, gerenciado pelo `CLMethod` | Separação de concerns: CLMethod gerencia estado CL |
| Quantidade de módulos heat | Quantidade de camadas ocultas | Quantidade de módulos LoRA × (q,k,v,o,gate,up,down) | Transformers têm muitas projeções lineares |
| Dimensionalidade do output | 2D (batch, features) | N-D (batch, [seq, ...], features) | Transformers operam em sequências |

### 6.2 Mudanças na Parametrização LoRA

**O delta LoRA** é a mudança arquitetural mais significativa:
$$
\Delta W_{\text{efetivo}} = \frac{\alpha}{r} B A
$$

Isto significa que o **número de parâmetros treináveis** por neurônio de saída cai de `D` (full rank) para `r + D·r/N` (aproximadamente). Por exemplo, para um módulo com N=4096, D=4096, r=64:
- MLP: 4096 parâmetros por neurônio de saída
- LoRA: 64 parâmetros por neurônio de saída (em B) + contribuição compartilhada para A

### 6.3 Mudanças na Proteção — O Ponto Mais Importante

**No MLP original:** O hook protege `self.weight` (N×D). Cada neurônio de saída tem sua **linha inteira** de D pesos protegida. Isso é proteção **completa e simétrica**.

**No DualHeatCLMethod:** O hook protege apenas `lora_B.weight` (N×r). As `r` conexões por neurônio de saída são protegidas, mas `A` não tem proteção no gradiente.

**No DualHeatLoRALinear (standalone):** O hook protege o tensor `delta` (...,N), que por chain rule protege **tanto B quanto A**. Isso é **equivalente** ao MLP.

**Conclusão:** O `DualHeatCLMethod` (usado no pipeline NLP) tem **proteção incompleta** em relação ao MLP original. O parâmetro A pode mudar sem ser escalado, o que cria um caminho para forgetting mesmo com B protegido.

### 6.4 Mudanças na Persistência

No MLP original, os heats são **automáticos**: como vivem em buffers do módulo, persistem enquanto o modelo existe. Entre tarefas no mesmo modelo, o heat simplesmente continua acumulando.

No NLP, como cada tarefa cria **novos adaptadores LoRA** (ver `orchestrator.py`), o heat state precisa ser **explicitamente salvo e carregado**. O `pre_train` restaura o heat via `restore_heat_state`. Isso é uma mudança de implementação (não conceitual) necessária para o pipeline.

### 6.5 Mudanças no Forward Hook

No MLP, a inibição lateral e a atualização dos heats ocorrem **dentro do `forward()`** da camada. É parte da definição da camada.

No NLP, esses mecanismos são implementados via **forward hooks** registrados pelo CLMethod. Isto significa que:
- O hook recebe o output já calculado pelo PEFT LoRA
- Se `lateral_inhibition = False`, o output não é modificado (apenas o tracking de heat ocorre)
- O hook retorna o output (possivelmente modificado pela inibição) para continuar o forward

Essa é uma mudança de implementação: o efeito é o mesmo, mas o mecanismo é diferente (hook vs override de forward).

### 6.6 Mudança de Implementação vs Mudança Conceitual

| Mudança | Tipo | Impacto |
|---|---|---|
| Hook em B vs hook em delta | **Conceitual** | Proteção de A é indireta |
| Forward hook vs override | **Implementação** | Efeito idêntico |
| Per-device heat dicts | **Implementação** | DataParallel compat |
| Persistência explícita | **Implementação** | Pipeline exige |
| `reduce_dims` flexível | **Implementação** | 3D vs 2D |
| Última camada sem heat | **Ambas** | MLP não protege última; NLP protege todos os módulos LoRA (incluindo última) |

---

## 7. Dual-Heat como Método Independente da Arquitetura

### 7.1 Formulação Abstrata

Seja:
- `θ` o conjunto de parâmetros treináveis do modelo
- `g = ∇_θ L` o gradiente da loss
- `s_i` o estado de "calor lento" do neurônio de saída `i` (proxy de importância)
- `h_i` o estado de "calor rápido" do neurônio de saída `i` (competição)
- `f_i(x; θ)` a função que produz a saída do neurônio `i`

A regra geral de atualização do Dual-Heat é:

**Forward:**
$$
\tilde{f}_i(x; θ) = \frac{f_i(x; θ)}{1 + \gamma \cdot \frac{1}{N-1} \sum_{j \neq i} h_j}
$$
$$
h_i \leftarrow \max\big(0,\; \alpha h_i + (1-\alpha)|\tilde{f}_i| - \delta\big)
$$
$$
s_i \leftarrow s_i + \frac{|\tilde{f}_i| - s_i}{n_{\text{eff}}}
$$

**Backward:**
$$
g_i \leftarrow \frac{g_i}{1 + \beta s_i}, \quad \forall \theta \in \text{params}(f_i)
$$

### 7.2 Instanciações

**Instanciação MLP (DualHeatLinear):**
- `f_i(x; θ) = W_i x + b_i` — produto escalar + bias
- `params(f_i) = {W_{i,:}, b_i}` — linha da matriz de pesos + bias
- A proteção atua sobre o gradiente de `W_{i,:}` (D parâmetros) e `b_i` (1 parâmetro)

**Instanciação LoRA (DualHeatLoRALinear — hook no delta):**
- `f_i(x; θ) = base_i(x) + (α/r) B_i A x`
- `params(f_i) = {B_{i,:}, A_{:, :}}` — linha de B + matriz A inteira
- A proteção atua sobre o gradiente de `delta`, que se propaga para B e A

**Instanciação LoRA (DualHeatCLMethod — hook em B):**
- `f_i(x; θ) = base_i(x) + (α/r) B_i A x`
- `params(f_i)` como acima, mas apenas `B_{i,:}` recebe escala direta
- `A` não tem seus gradientes diretamente escalados

### 7.3 Dual-Heat é Independente da Arquitetura?

**Sim, com ressalvas.** O núcleo do algoritmo — inibição lateral divisiva + EMA rápido + média incremental lenta + escala EWC por neurônio — é totalmente independente da arquitetura. Ele opera no **espaço de ativação** (magnitude de saída dos neurônios), não no espaço dos parâmetros.

As adaptações necessárias são:

1. **Identificar "neurônios"**: no MLP são os nós das camadas ocultas; no Transformer são as dimensões ocultas de cada projeção LoRA (q, k, v, o, gate, up, down). Ambos são canais de saída de uma transformação linear.

2. **Identificar parâmetros protegidos**: no MLP são os pesos completos; no LoRA são os fatores do adaptador. O princípio — proteger os parâmetros que produzem a saída do neurônio `i` — é o mesmo, mas a implementação muda.

3. **Gerenciar estado entre tarefas**: no MLP os heats vivem nos módulos e persistem naturalmente; no pipeline NLP precisam ser serializados.

O **invariante abstrato** que define Dual-Heat como método:

> Um sistema dinâmico acoplado de dois tempos que (a) regula a competição entre neurônios no forward via inibição lateral com memória curta, e (b) protege neurônios importantes no backward via modulação do gradiente com memória longa, usando a magnitude de ativação pós-inibição como única medida de importância.

---

## 8. Análise Crítica da Implementação NLP

### 8.1 Diferença Crítica: Hook em B vs Hook em Delta

Esta é a diferença mais significativa entre a implementação MLP original e a implementação NLP via `DualHeatCLMethod`.

**No MLP original (dual_heat_module.py, linha 120):**
```python
self.weight.register_hook(self._ewc_hook())
```
O hook age sobre `∇_W` (shape [N, D]). Cada neurônio `i` tem sua linha inteira protegida.

**No DualHeatCLMethod (dual_heat.py, linha 316–317):**
```python
bwd_hook = self._make_ewc_hook(name, B_w)
self._bwd_handles.append(B_w.register_hook(bwd_hook))
```
O hook age sobre `∇_B` (shape [N, r]). Apenas B é diretamente protegido.

**Consequência matemática:**

Seja a atualização SGD:
$$
B_{t+1} = B_t - \eta \cdot \frac{\nabla_B}{1 + \beta s}
$$
$$
A_{t+1} = A_t - \eta \cdot \nabla_A \quad \text{(sem escala)}
$$

A mudança efetiva no delta LoRA:
$$
\Delta_{t+1} - \Delta_t \propto (\Delta B) \cdot A_t + B_t \cdot (\Delta A) + (\Delta B) \cdot (\Delta A)
$$

O termo `B_t · (ΔA)` não é controlado pela proteção. Se `B_t` tem posto completo (o que é esperado após treino), mudanças em A produzem mudanças em `Δ` mesmo com `ΔB ≈ 0`. Isto é um **vazamento de atualização**.

**No DualHeatLoRALinear (standalone, delta hook):**
```python
delta.register_hook(self._ewc_scale_output)
```
O hook age sobre `∇_delta` (shape [..., N]). Pela regra da cadeia:
$$
\nabla_B \propto \nabla_{\text{delta}} \cdot (A x)^{\top}, \quad \nabla_A \propto B^{\top} \cdot \nabla_{\text{delta}} \cdot x^{\top}
$$

Ambos herdam a escala de `∇_delta`, e ambos são protegidos. Isto é **conceitualmente equivalente** ao MLP original.

### 8.2 Por que o Hook em Delta é Preferível

O hook em `delta` é a generalização correta porque:

1. **Simetria**: protege igualmente B e A (MLP original protege igualmente todos os pesos da camada)
2. **Fidelidade ao princípio**: "protege a contribuição do neurônio de saída" — a contribuição é `delta_i`, que deve ser protegida antes de se decompor em B e A
3. **Ausência de vazamento**: não há caminho de gradiente não-escalado para modificar a saída

### 8.3 Avaliação do Impacto Prático

**O vazamento via A pode ser relevante ou não, dependendo de:**

1. **Magnitude de B**: se `B` já foi protegido (slow_heat alto), `B` mudou pouco. Mas se `B` já tem valores significativos (não-zero), mesmo pequenas mudanças em A podem causar mudanças consideráveis em `B·A`.

2. **Razão r/N**: quanto menor o rank, mais os parâmetros estão em A (r·D) vs B (N·r). Para r=64, N=4096, D=4096: A tem 262K parâmetros, B tem 262K parâmetros. A metade dos parâmetros não está diretamente protegida.

3. **Inicialização de B**: LoRA tipicamente inicializa B=0. No começo do treino, `B·A = 0`, então o vazamento é zero. Mas conforme o treino progride e B se torna não-zero, o vazamento cresce.

### 8.4 Diferenças Adicionais

**Proteção da última camada.** No MLP original, a última camada é explicitamente um `nn.Linear` padrão **sem** DualHeat (linha 229). No NLP, o LoRA é aplicado a **todos** os módulos alvo, incluindo `down_proj` que são projeções de saída. Isto significa que a versão NLP protege mais camadas que a MLP.

**Reset de heat entre tarefas.** No MLP, `slow_heat` acumula por todo o treino (a menos que `slow_window` force esquecimento). No NLP, o heat é preservado entre tarefas via save/load, mas **não é explícito se o heat é resetado ao mudar de tarefa ou continua acumulando**. Pelo código, `load_state_snapshot` carrega o heat do checkpoint — ele continua onde parou. Isto é conceitualmente consistente com o MLP.

**Múltiplos módulos LoRA.** Um Transformer contém muitos módulos LoRA (q, k, v, o, gate, up, down para cada camada). Cada um tem seu próprio `_DualHeatModule` independente. Isto significa que o Dual-Heat opera em **72 módulos** (para 12 camadas × 6 módulos LoRA) em vez de 2–3 camadas MLP. O comportamento global depende da interação entre todos esses módulos — algo que não existe na versão MLP.

### 8.5 Inconsistências Potenciais

1. **A proteção de A é conceitualmente diferente do MLP** (seção 8.1). Isto **pode** fazer com que a implementação NLP tenha menor efetividade de CL que a MLP, especialmente em regimes onde A domina as atualizações.

2. **O DualHeatLoRALinear (standalone) não tem esse problema.** Ele está em `~/Desktop/dual_heat/dual_heat_LoRA_module.py` e implementa a estratégia correta. Se o objetivo é fidelidade ao MLP original, esta implementação é superior.

3. **Não há testes comparativos** entre as duas estratégias de hook (B vs delta). O sanity check no `DualHeatLoRALinear` (linhas 172–197) confirma que o hook no delta funciona, mas não há equivalente no `DualHeatCLMethod`.

---

## 9. Conclusão

### 9.1 O Dual-Heat Original e a Versão NLP São o Mesmo Método em Essência?

**Sim, na intenção e no algoritmo central.** Ambos implementam:
- Inibição lateral divisiva com fast heat (EMA + decay)
- Proteção EWC com slow heat (capped incremental mean)
- A mesma fórmula de escala `1/(1 + β·s)`
- Os mesmos hiperparâmetros (α, γ, δ, β, W)

### 9.2 Partes Invariantes

- As **equações de atualização dos heats** (fast e slow) são idênticas
- A **fórmula da inibição lateral** é idêntica
- A **fórmula do EWC** (`1/(1+β·slow_heat)`) é idêntica
- A **medida de importância** (magnitude de ativação pós-inibição) é idêntica
- O **ciclo de realimentação negativa** é idêntico
- A **filosofia de duas escalas temporais** é idêntica

### 9.3 Partes Específicas de MLP

- Pesos completos como parâmetros treináveis (`W ∈ ℝ^{N×D}`)
- Hook no gradiente do `self.weight` — proteção completa e simétrica
- Heats como buffers no próprio módulo (persistência automática)
- Dimensionalidade 2D (batch, features)
- Última camada explicitamente excluída do mecanismo

### 9.4 Partes Específicas de LoRA/Transformer

- Fatoração de baixo posto (`B ∈ ℝ^{N×r}, A ∈ ℝ^{r×D}`)
- Pesos base congelados (via PEFT)
- Hook apenas em `B` (DualHeatCLMethod) ou no tensor `delta` (DualHeatLoRALinear)
- Heats em `_DualHeatModule` separado (serialização explícita)
- Dimensionalidade N-D (com suporte a sequências)
- Múltiplos módulos por camada (q, k, v, o, gate, up, down)

### 9.5 A Versão NLP é uma Generalização?

**Sim, conceitualmente.** O algoritmo se aplica sem modificações ao cenário Transformer+LoRA, pois opera no espaço de ativação (invariante arquitetural). A formulação abstrata (Seção 7.1) descreve exatamente o mesmo método para ambas as arquiteturas.

**Mas há um problema na implementação `DualHeatCLMethod`:** o hook em `B` em vez de `delta` quebra a simetria da proteção, deixando `A` sem escala direta de gradiente. Isto **não é uma limitação do método**, é uma escolha de implementação que pode ser corrigida. A implementação `DualHeatLoRALinear` (standalone) mostra como fazer corretamente.

### 9.6 Formulação Matemática Mais Limpa

Dual-Heat, independente da arquitetura, é definido por:

**Equações de estado:**
$$
h_i^{(t)} = \max\left(0, \alpha h_i^{(t-1)} + (1-\alpha) |\tilde{f}_i(x; \theta^{(t)})| - \delta\right)
$$
$$
s_i^{(t)} = s_i^{(t-1)} + \frac{|\tilde{f}_i(x; \theta^{(t)})| - s_i^{(t-1)}}{\min(t, W)}
$$

**Forward modificado:**
$$
\tilde{f}_i(x; \theta) = \frac{f_i(x; \theta)}{1 + \gamma \cdot \frac{1}{N-1} \sum_{j \neq i} h_j}
$$

**Backward modificado:**
$$
\nabla_{\theta} \mathcal{L} \to \frac{\nabla_{\theta} \mathcal{L}}{1 + \beta \cdot s_i}, \quad \forall \theta \in \text{Params}(f_i)
$$

**Onde `Params(f_i)` são todos os parâmetros que contribuem para a saída do neurônio `i`.** No MLP, são `W_i` e `b_i`. No LoRA, são `B_i` e **todos os elementos de `A`** (já que A contribui para todas as saídas). Portanto, a implementação correta para LoRA deve proteger tanto B quanto A, seja via hook no delta ou via hooks separados em ambos.

### 9.7 Verificação Final

| Pergunta | Resposta |
|---|---|
| Mesmo método em essência? | **Sim** — algoritmo, equações e intenção são os mesmos |
| Partes invariantes? | Heats, inibição lateral, EWC scale, medida de importância |
| Partes específicas de MLP? | Pesos completos, persistência automática, proteção simétrica |
| Partes específicas de LoRA/Transformer? | Fatoração baixo posto, serialização, hook parcial (B apenas) |
| Generalização? | **Conceitualmente sim, mas implementação `DualHeatCLMethod` tem proteção assimétrica** |
| Formulação mais limpa? | Seção 9.6 — opera no espaço de ativação, invariante arquitetural |

### 9.8 Recomendação

Se o objetivo é equivalência matemática com o MLP original, a implementação `DualHeatCLMethod` deveria usar hook no tensor `delta` (como `DualHeatLoRALinear` faz) em vez de hook em `lora_B.weight`. A mudança é simples (substituir o backward hook de `B` para uma closure que captura `delta` no forward hook) e eliminaria a proteção assimétrica.
