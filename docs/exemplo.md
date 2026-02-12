# FedRankLite-SLM — Arquitetura de Implementacao

Harness de experimento config-driven para fine-tuning federado de SLMs com rank LoRA adaptativo.

## Objetivos

1. Rodar **FAH-QLoRA-like** como baseline (scheduler CVX ou greedy)
2. Rodar o **nosso** (Scheduler Lite) trocando apenas o scheduler via config
3. Comparar via **ablacoes** (fixed rank, IID vs non-IID, com/sem deadline)
4. Tudo controlado por **YAML** — nenhum parametro hardcoded

---

## Stack

| Componente | Lib |
|---|---|
| Treino | PyTorch + HuggingFace Transformers |
| PEFT | peft (LoRA) |
| Quantizacao | bitsandbytes (4-bit NF4 / 8-bit) |
| Dados | datasets (HuggingFace) |
| Solver (baseline) | cvxpy (opcional) |
| Config | PyYAML |

---

## Estrutura do repositorio

```
fedranklite-slm/
  configs/
    base.yaml                        # defaults globais (modelo, treino, tiers, metricas)
    experiments/
      baseline_cvx.yaml              # override: scheduler CVX
      baseline_fixed.yaml            # override: rank fixo (ablacao)
      lite_iid.yaml                  # override: Lite + IID
      lite_noniid_deadline.yaml      # override: Lite + non-IID severo + deadline
  src/
    __init__.py
    config.py                        # carrega/merge YAML (base + overrides) -> dict
    models.py                        # backbone + quantizacao + LoRA + mask de rank
    data.py                          # dataset HF + particionamento non-IID (Dirichlet)
    client.py                        # treino local (1 funcao = 1 cliente, 1 rodada)
    aggregation.py                   # agregacao LoRA com ranks heterogeneos
    engine.py                        # loop federado principal
    metrics.py                       # telemetria, fairness, time-to-target, checkpoints
    schedulers/
      __init__.py                    # factory: build_scheduler(cfg)
      base.py                        # ABC: allocate(round, telemetry) -> assignments
      cvx.py                         # baseline CVX / greedy fallback
      lite.py                        # nosso scheduler leve O(N)
      fixed.py                       # rank fixo (ablacao / sanity check)
  scripts/
    run.py                           # entry point: python run.py --config <yaml>
    run_slurm.sh                     # wrapper Slurm
  tests/
    test_aggregation.py              # agregacao com ranks diferentes
    test_scheduler.py                # scheduler respeita bounds e budget
    test_masking.py                  # mask de rank congela dimensoes certas
    test_metrics.py                  # fairness, time-to-target
    test_data_partition.py           # distribuicao Dirichlet, min_samples
  results/                           # output dos experimentos (gitignored)
  pyproject.toml
  .gitignore
```

---

## Config YAML — ponto de entrada unico

Tudo que muda entre experimentos esta no YAML. Os experiment YAMLs herdam de `base.yaml` via `_base_` e fazem override apenas dos campos necessarios.

### Campos do `base.yaml`

```yaml
seed: 42

model:
  name: "Qwen/Qwen2.5-1.5B"
  quantization: "4bit"            # 4bit | 8bit | none
  lora:
    r_max: 16
    r_min: 2
    alpha: 32
    target_modules: ["q_proj", "v_proj"]
    dropout: 0.05

federation:
  num_rounds: 100
  clients_per_round: 6
  total_clients: 10
  aggregation:
    method: "fedavg"              # fedavg | fedavg_blockwise
    weighting: "num_samples"      # num_samples | uniform
  scheduler:
    type: "lite"                  # cvx | lite | fixed
    lite:
      r_bar: 8
      ema_alpha: 0.3
      gain_weight: 0.6
      throughput_weight: 0.4
      deadline_penalty: 0.5
    cvx:
      solver: "ECOS"
      objective: "min_round_time"
    fixed:
      rank: 8

training:
  local_epochs: 3
  batch_size: 8
  lr: 2.0e-4
  max_grad_norm: 1.0
  optimizer: "adamw"
  warmup_ratio: 0.06

data:
  dataset: "glue/sst2"
  partition:
    method: "dirichlet"           # dirichlet | label_skew | iid
    alpha: 0.3
    min_samples: 50
  eval_split: 0.1

tiers:
  - name: "gpu_fast"
    count: 3
    compute_factor: 1.0
    memory_mb: 24000
  - name: "gpu_medium"
    count: 3
    compute_factor: 0.7
    memory_mb: 32000
  - name: "cpu_slow"
    count: 4
    compute_factor: 0.15
    memory_mb: 64000

deadline:
  enabled: true
  seconds: 120
  straggler_policy: "drop"        # drop | wait | use_partial

evaluation:
  target_score: 0.88
  metric: "accuracy"
  eval_every: 5

metrics:
  track_fairness: true
  percentiles: [10, 50, 90]
  track_comm_bytes: true
  track_peak_memory: true

logging:
  output_dir: "results/${experiment_name}"
  save_every: 10
  log_telemetry: true
  wandb:
    enabled: false
    project: "fedranklite"

experiment_name: "default"
```

### Exemplo de override (experimento)

```yaml
# configs/experiments/lite_noniid_deadline.yaml
_base_: "../base.yaml"
experiment_name: "lite_noniid_hard"
data:
  partition:
    method: "dirichlet"
    alpha: 0.1
federation:
  scheduler:
    type: "lite"
deadline:
  enabled: true
  seconds: 90
  straggler_policy: "use_partial"
```

---

## Decisoes de design

### 1) Simulacao local (nao distribuido)

O loop federado roda **sequencialmente em uma unica maquina**. Cada cliente treina em sequencia, e o tempo e ajustado pelo `compute_factor` do tier. Isso e suficiente para paper e muito mais reprodutivel que comunicacao real entre maquinas.

Para rodar distribuido de verdade (Slurm), basta usar o `run_slurm.sh` como wrapper — a logica nao muda.

### 2) LoRA global como state_dict parcial

Apenas parametros `lora_A` e `lora_B` sao extraidos, transmitidos e agregados. O backbone fica congelado e quantizado localmente.

### 3) Rank heterogeneo via mascaramento

Inicializa LoRA com `r_max`. Quando o cliente recebe `r_i < r_max`:
- Forward usa `A[:r_i, :]` e `B[:, :r_i]`
- Gradientes alem de `r_i` sao zerados via hook
- Upload envia apenas as dimensoes ativas (truncation)

Implementado em `models.py`: `mask_lora_rank()`, `truncate_lora_state()`, `pad_lora_state()`.

### 4) Agregacao com ranks diferentes

Dois metodos disponiveis via config (`federation.aggregation.method`):

- **fedavg**: zero-pad todos os deltas para `r_max`, media ponderada por `num_samples` ou `uniform`.
- **fedavg_blockwise**: agrega por bloco — dimensao `r` so recebe contribuicao de clientes com `rank_used > r`. Evita diluicao nas dimensoes superiores.

### 5) Scheduler plugavel

Interface unica: `allocate(round_id, client_ids, telemetry) -> {cid: {"rank": int}}`.

| Scheduler | Descricao |
|---|---|
| `fixed` | Rank fixo para todos. Ablacao / sanity check. |
| `cvx` | Minimiza max round time (cvxpy ou greedy fallback). Baseline do paper. |
| `lite` | **Nosso**. Score = gain_weight * EMA(loss_drop) + throughput_weight * EMA(throughput). Distribui budget `r_bar * N` proporcional a score. Penaliza quem estourou deadline. O(N) por rodada, sem solver. |

#### Scheduler Lite — formula

```
score_i = gain_weight * EMA(gain_i) + throughput_weight * EMA(throughput_i)
if exceeded_deadline: score_i *= deadline_penalty

raw_rank_i = (score_i / sum(scores)) * budget
rank_i = clamp(round(raw_rank_i), r_min, r_max)

# correcao greedy para sum(ranks) == budget
```

Inicializacao (round 0): todos recebem `r_bar`.

---

## Fluxo de execucao por rodada

```
engine.py (round t):
  1. Seleciona subset de clientes (amostragem aleatoria)
  2. Scheduler decide rank por cliente
  3. Para cada cliente selecionado:
     a. Carrega LoRA global no modelo
     b. Aplica mask de rank
     c. Treina local por E epochs
     d. Retorna: delta LoRA truncado + telemetria
     e. Aplica compute_factor do tier ao tempo medido
     f. Verifica deadline (drop / wait / use_partial)
  4. Agrega deltas no LoRA global (fedavg ou blockwise)
  5. Avalia globalmente a cada eval_every rounds
  6. Loga telemetria (fairness, bytes, deadline compliance)
  7. Checkpoint a cada save_every rounds
```

---

## Metricas coletadas

| Metrica | Onde | Descricao |
|---|---|---|
| **time-to-target** | `metrics.py` | Round em que `eval_score >= target_score` |
| **deadline compliance** | `metrics.py` | % de rounds sem straggler |
| **fairness (p10/p50/p90)** | `metrics.py` | Percentis da loss por cliente por rodada |
| **bytes/rodada** | `client.py` | `num_params_ativos * dtype_size` por cliente |
| **memoria pico** | `client.py` | `torch.cuda.max_memory_allocated()` ou `psutil` |
| **accuracy / score** | `metrics.py` | Avaliacao no eval set global |

---

## Tiers de hardware (simulados)

Definidos no YAML (`tiers`). Cada tier tem `count` (clientes), `compute_factor` (1.0 = mais rapido) e `memory_mb`.

| Tier | Hardware real | compute_factor | Clientes |
|---|---|---|---|
| gpu_fast | GPU 24GB (servers) | 1.0 | 3 |
| gpu_medium | GPU 32GB (Apolo node) | 0.7 | 3 |
| cpu_slow | CPU 256 cores (Apolo) | 0.15 | 4 |

O `engine.py` divide `train_time` pelo `compute_factor` para simular a diferenca de velocidade.

---

## Testes

| Teste | O que valida |
|---|---|
| `test_aggregation.py` | Agregacao com ranks iguais e diferentes, weighting, blockwise |
| `test_scheduler.py` | Bounds respeitados, budget exato, penalty funciona |
| `test_masking.py` | Mask zera dimensoes certas, gradientes zerados, truncate/pad roundtrip |
| `test_metrics.py` | Time-to-target, deadline compliance, fairness percentis |
| `test_data_partition.py` | IID cobre todos os samples, Dirichlet cria skew, min_samples respeitado |

---

## Como rodar

```bash
# instalar
pip install -e ".[dev]"

# rodar experimento
python scripts/run.py --config configs/experiments/lite_noniid_deadline.yaml

# rodar testes
pytest tests/ -v

# via slurm
sbatch scripts/run_slurm.sh configs/experiments/lite_noniid_deadline.yaml
```
