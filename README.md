# DASE7506 MP1 — Small Language Model Challenge

A causal Transformer trained from scratch on WikiText-2, with knowledge distillation during training and, at inference, a training-derived Kneser–Ney count model and a similarity-gated cache that copies only from earlier positions of the current 256-token window. The supplied tokenizer, data, baseline and scorer are unchanged.

| Result (CPU, FP32, 4 threads) | Value |
|---|---|
| **Full-test BPB** | **1.4310428473355665** (baseline 2.101256834929385) |
| Validation BPB | 1.415455287338549 (baseline 2.071081) |
| CPU scoring time vs. baseline | 4.255× (validation, worst of two sessions); 4.235× (test) — limit 5× |
| Peak process-tree RAM | 1,796,526,080 bytes (1.67 GiB) — limit 4 GiB |
| Inference assets | 46,549,381 bytes (44.4 MiB) — limit 64 MiB |
| Final checkpoint SHA-256 | `0ccd91e76520f7b83a1aee05584d88007708ca6b33e15f9f9e1586335affd7ee` |

The method, experiments, ablations and costs are described in the [report](report.pdf) ([Markdown source](REPORT.md)).

## 1. Install

Python 3.12 and PyTorch 2.7.1. A CPU-only environment is enough to reproduce the score.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt -r requirements-extra.txt
```

`requirements.txt` is the course file (NumPy 2.5.3, tokenizers 0.21.4). `requirements-extra.txt` adds psutil 7.2.2, which is only used to measure peak RAM.

## 2. Download the checkpoint

Checkpoints are not stored in git. Download them from the [v1.0 release](https://github.com/XFY666/dase7506-mp1/releases/tag/v1.0) and place them at these exact paths:

| Release asset | Path in the repository | SHA-256 |
|---|---|---|
| `checkpoint.pt` | `checkpoints/final/checkpoint.pt` | `0ccd91e76520f7b83a1aee05584d88007708ca6b33e15f9f9e1586335affd7ee` |
| `baseline-checkpoint.pt` (only for timing) | `checkpoints/baseline/checkpoint.pt` | `1ed624aaac94411664e8168c4bf8d852d167ab19ccba62155e3a08dd3ea36def` |

The baseline checkpoint was trained locally with the unchanged course `train.py` and `model.py` (seed 17, 1,200 updates, `configs/baseline.json`). The baseline BPB values above come from it.

The count table `assets/ngram_mkn5.npz` is already in the repository. The final predictor needs only `checkpoints/final/checkpoint.pt`, `assets/ngram_mkn5.npz` and the Python sources. It needs no teacher model, network access or retraining.

## 3. Reproduce the test score

From the repository root:

```bash
python verify_integrity.py      # course data, baseline and scorer are unchanged
python verify_frozen.py         # checkpoint, sources, data and count table match the freeze
python -m unittest discover -s tests -v
python evaluate.py --checkpoint checkpoints/final/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test --output reproduced-test.json
```

`reproduced-test.json` should report `"bpb": 1.4310428473355665`. The recorded score files are in `results/`. Different CPUs or library builds can change the last digits slightly.

## 4. Measure time and memory

Resource use was measured as the course README describes, on an otherwise idle Ryzen 5 7600 with 4 threads. Each session runs one warm-up pair, then three alternating baseline/candidate pairs (B C, C B, B C). It reports the ratio of the median scoring times. `measure_evaluation.py` runs the unchanged scorer in a subprocess and records its scoring time and peak process-tree RAM. `summarize_resources.py` checks and summarizes the records.

```bash
python measure_evaluation.py --checkpoint checkpoints/baseline/checkpoint.pt --split validation --output m/w_base.json
python measure_evaluation.py --checkpoint checkpoints/final/checkpoint.pt    --split validation --output m/w_final.json
python measure_evaluation.py --checkpoint checkpoints/baseline/checkpoint.pt --split validation --output m/b1.json
python measure_evaluation.py --checkpoint checkpoints/final/checkpoint.pt    --split validation --output m/c1.json
python measure_evaluation.py --checkpoint checkpoints/final/checkpoint.pt    --split validation --output m/c2.json
python measure_evaluation.py --checkpoint checkpoints/baseline/checkpoint.pt --split validation --output m/b2.json
python measure_evaluation.py --checkpoint checkpoints/baseline/checkpoint.pt --split validation --output m/b3.json
python measure_evaluation.py --checkpoint checkpoints/final/checkpoint.pt    --split validation --output m/c3.json
python summarize_resources.py --baseline-records m/b1.json m/b2.json m/b3.json --candidate-records m/c1.json m/c2.json m/c3.json --output m/summary.json
```

Recorded results:

| Session | Baseline median | Model median | Ratio | Peak RSS |
|---|---:|---:|---:|---:|
| Validation 1 | 5.43 s | 23.09 s | 4.255 | 1,796,526,080 B |
| Validation 2 | 5.42 s | 22.95 s | 4.232 | 1,796,218,880 B |
| Test | 6.18 s | 26.17 s | 4.235 | 1,796,079,616 B |

The ratio varies between sessions. Before the re-save described in Section 8, the same predictor measured 4.250× and 4.284× on validation and 4.283× on test. Earlier checkpoints with the same architecture and inference code measured 4.49–4.58×. All sessions stayed below 5×. The first validation session is stored with the freeze in `checkpoints/final/resource_evidence.json`, and the test session is in `results/test_resource_summary.json`. These evidence files record the local paths of the machine they were measured on.

Inference assets: the checkpoint (27,821,877 B), the count table (18,689,724 B) and the six inference source files (37,780 B), 46,549,381 B in total.

## 5. How the predictor works

- **Network** (`student.py`). 8 blocks, width 256, 4 heads, SwiGLU MLP of width 704, RoPE, RMSNorm, tied embeddings; 6,951,168 parameters.
- **Training** (`train_experiment.py`). 32,000 updates of 32×256 tokens, AdamW, cosine schedule and EMA. The loss is 0.5 × cross-entropy + 0.5 × KL to a frozen teacher ensemble.
- **Count model** (`fit_ngram.py`, `ngram_expert.py`). Order-5 modified Kneser–Ney fitted on the training text only.
- **Cache** (`student.py`, `hybrid.py`). Final-layer hidden states of earlier positions in the same window vote for their next tokens. The cache is used only after 32 positions, and only if some earlier state has cosine similarity ≥ 0.7.
- **Mixture** (`hybrid.py`). p = (1 − β)[(1 − α) p_net + α p_cache] + β p_counts, with network temperature 1.05, α = 0.15 and β = 0.10 for the first 64 positions of a window, 0.05 afterwards.
- **CPU scheduling** (`hybrid.py`). Windows are batched through the network 8 at a time and mixed 4 at a time. This only changes speed. Log-probabilities changed by at most 1.9e-6 on an earlier checkpoint, and validation BPB of the final checkpoint by 8e-11.

Every window is scored independently. The cache is rebuilt from scratch for each window, and the count table is never updated during evaluation.

## 6. Reproduce training

Retraining reproduces the recipe, not bit-identical weights: training uses CUDA BF16, which is not deterministic. Expected GPU time on an RTX 4070 is about 65 minutes in total. The commands use a CUDA build of PyTorch 2.7.1 for training and the CPU environment above for scoring. Run them from the repository root in a POSIX shell (bash, or Git Bash on Windows). Keep `RUNS` outside the repository as an **absolute** path, because checkpoints record the paths of their parents.

```bash
RUNS=/absolute/path/to/runs
COMMON="--implementation student --device cuda --precision bf16 --threads 4 --seed 17 --batch-size 32 \
  --lr 0.001 --min-lr-ratio 0.05 --warmup 200 --weight-decay 0.1 --beta2 0.95 --eval-every 1000 \
  --ema-decay 0.999 --ema-start 1000 --patience 0 --save-every 1000 --log-every 1000"

# 1. Count table (CPU, ~10 s). The result should equal assets/ngram_mkn5.npz.
python fit_ngram.py --output $RUNS/counts/ngram_mkn5.npz --max-order 5 --min-count 2

# 2. Two directly trained models (20k updates each, ~11 min each).
python train_experiment.py $COMMON --steps 20000 --config configs/direct-6x288.json --run-dir $RUNS/direct-6x288
python train_experiment.py $COMMON --steps 20000 --config configs/direct-8x256.json --run-dir $RUNS/direct-8x256

# 3. Teacher A: equal-weight mixture at temperature 1.15.
python build_teacher_ensemble.py --first-checkpoint $RUNS/direct-6x288/checkpoint.pt \
  --second-checkpoint $RUNS/direct-8x256/checkpoint.pt --output-dir $RUNS/teacher-a \
  --device cuda --temperatures 1.15 --alphas .5

# 4. Student distilled from teacher A, dropout 0.15 (~18 min).
python train_experiment.py $COMMON --steps 20000 --config configs/kd-8x256-dropout15.json --run-dir $RUNS/kd-8x256 \
  --teacher-checkpoint $RUNS/teacher-a/checkpoint.pt --distill-weight 0.5 --distill-temperature 1.0

# 5. Teacher B: 0.25 x direct 6x288 + 0.75 x distilled student, temperature 1.05 (CPU, ~1 min).
python reproduction/build_teacher_b.py --first-checkpoint $RUNS/direct-6x288/checkpoint.pt \
  --second-checkpoint $RUNS/kd-8x256/checkpoint.pt --output-dir $RUNS/teacher-b

# 6. Final student: 32k updates distilled from teacher B, dropout 0.10 (~25 min).
python train_experiment.py $COMMON --steps 32000 --config configs/final-8x256-dropout10.json --run-dir $RUNS/final-32k \
  --teacher-checkpoint $RUNS/teacher-b/checkpoint.pt --distill-weight 0.5 --distill-temperature 1.0

# 7. Take the EMA weights at step 32,000 from resume.pt. The run's own checkpoint.pt keeps the
#    network-only best step, which is not the selected model.
python reproduction/finalize_checkpoint.py ema --input $RUNS/final-32k/resume.pt --output $RUNS/final-ema/checkpoint.pt

# 8. Attach the selected mixture settings, then the CPU scheduling keys.
python tune_hybrid.py --checkpoint $RUNS/final-ema/checkpoint.pt --output-dir $RUNS/final-hybrid --device cuda \
  --logit-temperatures 1.05 --cache-layer -1 --cache-weights .15 --cache-temperatures 12 --decays 0 \
  --min-histories 32 --min-similarities .7 --ngram-weights .05 --ngram-confidence-powers 0 \
  --early-weights .1 --early-cutoffs 64
python reproduction/finalize_checkpoint.py schedule --input $RUNS/final-hybrid/checkpoint.pt --output $RUNS/final/checkpoint.pt

# 9. Validation (scores only the validation text), then test.
python ablate_hybrid.py --checkpoint $RUNS/final/checkpoint.pt --output $RUNS/final/validation_ablation.json --device cpu --threads 4
python evaluate.py --checkpoint $RUNS/final/checkpoint.pt --device cpu --precision fp32 --threads 4 --split test
```

Reference values from the original runs (validation BPB): direct 6×288 1.50074, direct 8×256 1.50233, teacher A 1.44394, distilled student 1.45259, teacher B 1.43448, final network (EMA at 32k) 1.44872, final predictor 1.415455.

The non-training steps 3, 5, 7 and 8 were checked against the original artifacts; the training steps 2, 4 and 6 were not rerun. Given the original training outputs, `build_teacher_ensemble.py` and `reproduction/build_teacher_b.py` rebuild teachers A and B with identical tensors and configuration. Steps 7–8 rebuild the frozen checkpoint with identical tensors and configuration.

The original validation selections are applied here as fixed choices:
- **Endpoint**: EMA at 32k, the best of eight raw/EMA endpoints at 20k, 24k, 28k and 32k. `train_experiment.py` keeps only the last EMA state, so comparing endpoints requires saving `resume.pt` copies at those steps.
- **Mixture settings**: the 11 settings, from a 44,980-setting search. `tune_hybrid.py` can repeat it with comma-separated grids, run once per cache layer.
- **Teacher A**: temperature and weight from a 25-setting grid of `build_teacher_ensemble.py` plus a 20-setting refinement.
- **Teacher B**: temperature and weight from a 24-setting search.

## 7. Repository layout

| Path | Contents |
|---|---|
| `student.py`, `hybrid.py`, `ngram_expert.py` | Final predictor: network, cache and mixture, count model |
| `common.py`, `evaluate.py`, `model.py`, `train.py` | Course scorer, data loading and baseline (unchanged) |
| `train_experiment.py`, `teacher_ensemble.py`, `build_teacher_ensemble.py` | Training with EMA and distillation; teacher ensembles |
| `fit_ngram.py`, `tune_cache.py`, `tune_hybrid.py`, `ablate_hybrid.py` | Count fitting, mixture selection and ablation on validation |
| `average_checkpoints.py`, `benchmark_architectures.py` | Tools for experiments reported as negative or exploratory |
| `measure_evaluation.py`, `summarize_resources.py`, `evidence.py` | Time and memory measurement |
| `freeze_candidate.py`, `verify_frozen.py`, `verify_integrity.py` | Pre-test freeze and hash checks |
| `reproduction/` | Teacher B builder and checkpoint finalization for retraining |
| `configs/` | Baseline and training configurations |
| `data/`, `assets/` | Course data and tokenizer; fitted count table |
| `checkpoints/final/` | Freeze manifest and the evidence it binds (checkpoint from the release) |
| `results/` | Official test and baseline score files and the test resource summary |
| `tests/` | Course contract tests plus causality, normalization, chunking and estimator tests |
| `figures/` | Validation curves of the final training run |

## 8. Selection, freeze and test

All training used only the training text. All model, checkpoint and setting choices used the validation split. Among candidates whose CPU time ratio stayed at or below 4.8× in two independent measurement sessions, the final model had the lowest complete validation BPB. Before the test split was scored, `freeze_candidate.py` wrote `checkpoints/final/freeze.json`. It records SHA-256 hashes of 20 files: the checkpoint, the six inference sources, the data, the count table, the five freeze and measurement tools and the validation/resource evidence. The test split was then scored in a single paired measurement campaign. All four model passes gave the same BPB, and a later reproduction run from this repository matched it. No predictor weight or setting was changed after the test.

**Re-saved release files.** The checkpoint frozen before the test (SHA-256 `91710c26…`) stored local build metadata: file paths of the training machine and internal notes. After the test, the checkpoint was re-saved with only the fields the scorer needs plus a short description, and one comment line in `hybrid.py` was reworded. The weights, configuration and computation are unchanged. All eight validation ablation rows are bit-identical, and the per-window test losses of all 1,674 windows are identical. The file hashes changed, so the two resource sessions, the freeze (13:26 UTC+8) and the test campaign (13:26–13:30) were repeated for the released files, again giving test BPB 1.4310428473355665. The documentation, tests and reproduction helpers were also edited or added after the test; none of them is used for inference.

One deviation from our own plan: it set an internal deadline of 08:30 (UTC+8, 28 September) for the final model's two resource sessions. The AI assistant running the measurements ran out of usage quota, so the sessions ran at 09:56–10:04 instead. Before any test result existed, I decided to accept them under the unchanged 4.8× rule. Otherwise the previously qualified model (the 24k-update continuation, validation BPB 1.419230) would have been submitted. The freeze (10:20) and the test campaign (10:22–10:26) followed.

Checkpoints store build metadata, including the author's local file paths, for provenance. This metadata is not used at inference.

## 9. My role, learning goals and AI assistance

**My role and learning goals.** I set the project's goal of improving BPB within the course constraints, allocated local computing resources, and requested independent review of the implementation and experimental choices. I directed the collaboration between the two assistants and approved key decisions, including accepting the delayed resource sessions before testing and proceeding to the final freeze. My interest in the project extends beyond the score: I want to understand how distillation, causal caching and training-derived counts interact, and why a lower validation loss must be considered alongside computational cost and reproducibility.

**Scope of assistance.** As permitted by the course, I used OpenAI Codex and Anthropic Claude substantially for technical explanations, implementation, experiments and documentation:

- Codex implemented much of the model and experimental tooling, executed training and validation experiments, and prepared initial documentation.
- Claude provided a second technical review, suggested experiments, contributed CPU scheduling prototypes, and completed the final measurement, freezing, testing and repository preparation.
- Their assistance also covered debugging, correctness tests, provenance and hash checks, code cleanup, and editing the README and report. These checks explain the detailed verification tooling in the repository; they were developed with AI assistance.

The responsibilities above distinguish my project direction and decisions from the assistants' implementation and execution. I am responsible for the submitted work, for checking its claims and for being able to explain the method and its limitations.

## 10. Credits and data

The course starter supplied the baseline model, trainer, scorer, tokenizer, data and contract tests. RoPE, RMSNorm, SwiGLU, modified Kneser–Ney smoothing, the continuous cache, weight averaging and knowledge distillation are published methods, cited in the report. No external text or pretrained weights were used.

WikiText-2 was introduced by Merity et al., [*Pointer Sentinel Mixture Models*](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors and is available under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GFDL](https://www.gnu.org/licenses/fdl-1.3.html). The split revision and byte hashes are in `data/manifest.json`.
