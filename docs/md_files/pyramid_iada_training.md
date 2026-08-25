# PyramidFusion OPV2V-to-DAIR training commands

All runs inherit the same PyramidFusion detector configuration. The optional
single-agent occupancy loss remains disabled in the baseline, comparison
methods, IADA, and every ablation.

## Source-only baseline

```bash
CONFIG=opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_pyramid_iada.yaml

CUDA_VISIBLE_DEVICES=0 uv run python opencood/tools/train_iada.py \
  --hypes_yaml "$CONFIG" \
  --stage baseline \
  --half
```

After training, resolve the newly produced baseline once and reuse it for all
adaptation runs:

```bash
BASELINE="$(ls -dt \
  opencood/logs/opv2v_to_dairv2x_pointpillar_pyramid_da_baseline_* \
  | head -n 1)"
```

## Comparison methods

Each method starts independently from the same source-only baseline.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python opencood/tools/train_iada.py \
  --hypes_yaml "$CONFIG" \
  --stage grl \
  --pretrained_model_dir "$BASELINE" \
  --half

CUDA_VISIBLE_DEVICES=0 uv run python opencood/tools/train_iada.py \
  --hypes_yaml "$CONFIG" \
  --stage dusa \
  --pretrained_model_dir "$BASELINE" \
  --half

CUDA_VISIBLE_DEVICES=0 uv run python opencood/tools/train_iada.py \
  --hypes_yaml "$CONFIG" \
  --stage cudax \
  --pretrained_model_dir "$BASELINE" \
  --half

CUDA_VISIBLE_DEVICES=0 uv run python opencood/tools/train_iada.py \
  --hypes_yaml "$CONFIG" \
  --stage iada \
  --pretrained_model_dir "$BASELINE" \
  --half
```

The PyramidFusion YAML already contains the frozen OPV2V source-only CUDA-X
residual bounds. Do not recompute them from DAIR labels.

## Cumulative IADA ablations

The four configurations progressively enable the identity-initialized gate,
source safety supervision, global conditional alignment, and local conditional
alignment. Every run starts from the same baseline rather than from the
previous ablation.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python opencood/tools/train_iada.py \
  --hypes_yaml opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_pyramid_iada_gate.yaml \
  --pretrained_model_dir "$BASELINE" \
  --half

CUDA_VISIBLE_DEVICES=0 uv run python opencood/tools/train_iada.py \
  --hypes_yaml opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_pyramid_iada_source.yaml \
  --pretrained_model_dir "$BASELINE" \
  --half

CUDA_VISIBLE_DEVICES=0 uv run python opencood/tools/train_iada.py \
  --hypes_yaml opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_pyramid_iada_consistency.yaml \
  --pretrained_model_dir "$BASELINE" \
  --half

CUDA_VISIBLE_DEVICES=0 uv run python opencood/tools/train_iada.py \
  --hypes_yaml opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_pyramid_iada_full.yaml \
  --pretrained_model_dir "$BASELINE" \
  --half
```

Use separate GPUs or add log redirection when launching methods in parallel.
