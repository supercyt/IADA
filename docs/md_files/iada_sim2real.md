# OPV2V to DAIR-V2X domain adaptation

This experiment suite compares source-only training, a conventional GRL
discriminator, DUSA, CUDA-X, and IADA under one fusion-agnostic PointPillar
interface. The target domain is DAIR-V2X and the source domain is OPV2V.

## Configurations

All three fusion backbones use `point_pillar_baseline`. The adapter receives
the per-agent feature before fusion and the collaborative ego-centric feature
after the native fusion module, so the detector and fusion parameter names stay
compatible with their source-only checkpoints.

| Fusion backbone | Configuration | Adapter feature map |
| --- | --- | --- |
| single-scale AttFuse | [AttFuse YAML](../../opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_attfuse_iada.yaml) | `384 x 100 x 252` |
| DiscoNet, student-only | [DiscoNet YAML](../../opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_disconet_da.yaml) | `256 x 100 x 252` |
| V2X-ViT | [V2X-ViT YAML](../../opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_v2xvit_da.yaml) | `256 x 100 x 252` |

The AttFuse filename is retained for compatibility with existing commands, but
the file now supports every stage below. DiscoNet deliberately uses the common
student-only fusion module and `point_pillar_loss`; it does not use the
teacher/KD-only `point_pillar_disconet` wrapper.

V2X-ViT keeps a stride-1 channel shrink so it uses the same `100 x 252`
detection grid as the other two backbones. Its pyramid windows are `[1, 2, 4]`:
the implementation does not pad window inputs, and these sizes divide both
spatial dimensions. The standard `[4, 8, 16]` windows would be invalid on this
shared OPV2V-to-DAIR geometry. Their relative-position parameter shapes also
differ, so a standard-window V2X-ViT checkpoint is not a compatible warm start;
train the matching source-only baseline from this comparison YAML.

V2X-ViT additionally consumes an explicit per-scene heterogeneity prior named
`prior_encoding`, whose three channels are `[velocity, time_delay,
infrastructure_type]`. The accepted shapes are `[B, L, 3]` and
`[B, L, 3, 1, 1]`; padded agents must be zero and `infrastructure_type` must be
exactly `0` or `1`. Source OPV2V batches pass an all-zero prior. Target and
mixed training batches mark only local agent index 1 as infrastructure when
that scene has at least two valid agents. An explicit input always overrides
the model fallback, so source training never infers a role from agent order.

For target inference paths that do not expose this field, the V2X-ViT YAML
sets `prior_encoding_fallback: local_index_1_infra`. This ordering assumption
is specific to the current DAIR-V2X loader: it marks index 1 only when
`record_len > 1`, and leaves single-agent scenes and padding at zero. A dataset
with different agent ordering should supply `prior_encoding` explicitly, or
change the policy to `zeros` or `error`. The fusion boundary rejects mismatched
shape, device, dtype, padding mask, non-finite values, and invalid role values
instead of silently broadcasting an ambiguous prior.

The configs share the target-centric range
`[-100.8, -40, -3.5, 100.8, 40, 1.5]`, voxel size `[0.4, 0.4, 5]`,
`max_cav: 2`, and `comm_range: 100`. Update the OPV2V and DAIR-V2X paths in a
config if the datasets are stored elsewhere. The top-level dataset remains
DAIR-V2X so a saved run can be evaluated without rewriting its config;
`domain_adaptation.source` supplies only the OPV2V overrides.

## Training stages

Every config defaults to the safe source-only stage. First train one baseline
for the selected fusion backbone:

```bash
CONFIG=opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_attfuse_iada.yaml

python opencood/tools/train_iada.py \
  --hypes_yaml "$CONFIG" \
  --stage baseline
```

Then start each adaptation method independently from that exact baseline:

```bash
BASELINE=opencood/logs/<matching_baseline_run>

python opencood/tools/train_iada.py -y "$CONFIG" \
  --stage grl --pretrained_model_dir "$BASELINE"

python opencood/tools/train_iada.py -y "$CONFIG" \
  --stage dusa --pretrained_model_dir "$BASELINE"

python opencood/tools/train_iada.py -y "$CONFIG" \
  --stage cudax --pretrained_model_dir "$BASELINE"

python opencood/tools/train_iada.py -y "$CONFIG" \
  --stage iada --pretrained_model_dir "$BASELINE"
```

The five accepted stages are:

- `baseline`: OPV2V detection only; the adapter is disabled.
- `grl`: conventional global agent-feature discriminator with gradient
  reversal.
- `dusa`: location-selective source/target alignment plus confidence-aware
  vehicle/infrastructure alignment.
- `cudax`: the repository's paper-based reproduction of CUDA-X CKT, BLC, and
  CPA.
- `iada`: interventional collaboration-advantage adaptation. It compares the
  native collaborative prediction with an ego-only counterfactual and does
  not use a graph/domain discriminator.

DUSA uses its paper coefficients `0.05` for LSA and `0.1` for CIA directly;
they are fixed adversarial scales and are not multiplied by the global GRL
schedule used by the other adapters.

Use a different matching baseline for AttFuse, DiscoNet, and V2X-ViT. Never
initialize one adaptation method from another adaptation checkpoint. The
loader verifies the data protocol and shared detector/fusion keys; only fresh
`domain_adapter.*` parameters may be absent from a baseline checkpoint.

Add `--half` for CUDA automatic mixed precision. Use `--model_dir` only to
resume the same stage with its full training state. `training_state_latest.pth`
contains the model, optimizer, scheduler, scaler, random state, and checkpoint
selection state; the `net_epoch*.pth` files remain model-only inference
checkpoints.

## Incremental IADA configurations

The four AttFuse IADA files use `base_config` inheritance, so the dataset,
detector, optimizer, and protocol remain defined in the main YAML, while
`pointpillar_attfuse_iada_common.yaml` stores the shared stage and baseline
checkpoint. Each experiment file progressively enables another part of the
same method:

| Configuration | Utility gate | Source safe/correction/utility | Target intervention consistency | Effect memory |
| --- | --- | --- | --- | --- |
| `pointpillar_attfuse_iada_gate.yaml` | yes | no | no | no |
| `pointpillar_attfuse_iada_source.yaml` | yes | yes | no | no |
| `pointpillar_attfuse_iada_consistency.yaml` | yes | yes | yes | no |
| `pointpillar_attfuse_iada_full.yaml` | yes | yes | yes | yes |

Every incremental config sets `stage: iada` and contains the current matching
baseline path. Run it without changing command-line stages:

```bash
python opencood/tools/train_iada.py -y \
  opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_attfuse_iada_gate.yaml

python opencood/tools/train_iada.py -y \
  opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_attfuse_iada_source.yaml

python opencood/tools/train_iada.py -y \
  opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_attfuse_iada_consistency.yaml

python opencood/tools/train_iada.py -y \
  opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/pointpillar_attfuse_iada_full.yaml
```

These runs are cumulative *module ablations* but independently warm-start from
the same source-only AttFuse baseline. Do not warm-start one ablation from a
different IADA ablation, because that would confound module gains with extra
optimization epochs. When a new baseline is trained, only update
`domain_adaptation.pretrained_model_dir` in
`pointpillar_attfuse_iada_common.yaml`.

The source stage uses OPV2V labels to supervise non-degradation, continuous
box correction, and signed collaboration utility. The target stage never
receives DAIR labels: it compares a channel-dropout intervention view against
an EMA copy of the utility gate, and aligns reliable effect tokens to
source-only discovery, suppression, and refinement prototypes grouped into
near/middle/far relative-geometry contexts.

## Unified adapter and removed interaction logits

Domain adaptation is attached after the existing encoder and native fusion
module. GRL, DUSA, and CUDA-X retain the comparison implementation described
above. IADA additionally extracts the first agent as an ego-only
counterfactual and applies a bounded residual gate:

`F_calibrated = F_ego + gate * (F_fused - F_ego)`.

The gate head is zero-initialized, so `gate == 1` and the first forward exactly
matches native fusion. At inference the calibrated feature is passed through
the unchanged PointPillar heads; teacher/intervention views and effect-memory
losses are training-only.

`interaction_logits`, the old graph-domain discriminator, graph variance
floor, and the old `iada_v1` / `iada_v2` stage split have been removed.
Checkpoints from graph-based IADA are not compatible exact-resume checkpoints;
start the new method from the matching source-only baseline.

## CUDA-X residual bounds

CUDA-X in this repository is a reproduction from the published CKT/BLC/CPA
description, not an import of an official released implementation. The CKT
path shuffles channels within each slice before concatenation. Because the
paper does not disclose the within-slice shuffle grouping, the configs expose
the explicit reproduction assumption `ckt_shuffle_groups: 2`.

The BLC branch divides six non-angular encoded box residuals into bins and
discards yaw. The CUDA-X paper text also mentions a `K x 7` prediction in one
place, while its residual analysis is described over six spatial coordinates;
with no released reference code, this repository records the six-coordinate
choice explicitly rather than silently guessing a periodic yaw bound. Each
YAML intentionally contains:

```yaml
cudax_residual_bounds: []
```

The empty value makes `--stage cudax` fail until the experimenter supplies six
positive bounds. Compute the paper's max-absolute statistic from the OPV2V
source training split with the included source-only utility:

```bash
python opencood/tools/compute_cudax_bounds.py -y "$CONFIG"
```

Copy the printed `cudax_residual_bounds` list into the YAML, then keep it frozen
for the whole comparison. The utility performs the following operations:

1. Run the selected fusion config's source loader without constructing or
   inspecting a DAIR target batch.
2. From `label_dict["targets"]`, retain only anchors selected by
   `label_dict["pos_equal_one"]`.
3. Reshape the seven encoded residuals per anchor, discard yaw, and calculate
   the maximum absolute value for each remaining coordinate.
4. Store the six frozen values in encoded-target order
   `[dx, dy, dz, dlog_h, dlog_w, dlog_l]`, corresponding to OpenCOOD's
   `[x, y, z, h, w, l]` box order, and record the statistic used.

Do not estimate, tune, or recompute these bounds from DAIR labels, a mixed
source/target loader, DAIR validation AP, or adaptation results. The
`model.args.domain_adapter.cudax_bin_count` and
`domain_adaptation.cudax_bin_count` values must also remain equal.

## Target-label policy

Adaptation is unsupervised with respect to DAIR detection labels. Source and
target scenes are collated separately, then the merge helper copies only model
inputs such as LiDAR tensors, poses, transforms, and `record_len`. Detection
loss is evaluated only on the source slice with the OPV2V label dictionary.
CUDA-X BLC bin supervision also uses only those OPV2V labels.

DAIR training labels must not enter the merged model batch or any loss. DAIR
validation labels are reserved for final reporting, while checkpoint selection
uses OPV2V source validation loss. DUSA's target-side vehicle/infrastructure
loss additionally accepts only DAIR scenes with exactly the expected two-agent
ordering; it must not infer agent roles from local indices on other datasets.

## Evaluation

Evaluate a saved target-centric run with the standard intermediate-fusion
entry point:

```bash
python opencood/tools/inference.py \
  --model_dir opencood/logs/<run> \
  --fusion_method intermediate
```

To evaluate a specific periodic checkpoint instead of the best-validation
checkpoint, pass its epoch number explicitly:

```bash
python opencood/tools/inference.py \
  --model_dir opencood/logs/<run> \
  --fusion_method intermediate \
  --checkpoint_epoch 10
```

This loads `net_epoch10.pth` from the run directory. Omitting
`--checkpoint_epoch` preserves the default selection rule: prefer the single
`net_epoch_bestval_at*.pth` checkpoint, then fall back to the latest periodic
checkpoint.

For a controlled table, report the baseline and all requested adaptation
methods for the same fusion config, seed, source baseline, residual-bound rule,
and DAIR split.

## Tests

```bash
MPLCONFIGDIR=/tmp/coalign-mpl \
  .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```
