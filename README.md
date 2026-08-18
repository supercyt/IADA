# IADA

Research code under active development.

## Interventional Advantage Domain Adaptation

The current IADA implementation adapts the prediction change caused by
collaboration relative to an ego-only counterfactual. It uses an
identity-initialized utility gate, source-only safe/correction supervision,
target intervention consistency with an EMA gate teacher, and a source-only
context-matched effect memory. It does not use the former graph discriminator.

Four incremental AttFuse configurations can be executed directly. They all
inherit the common experiment protocol from `pointpillar_attfuse_iada.yaml`
and warm-start from the baseline directory recorded in each small config:

```bash
for CONFIG in \
  pointpillar_attfuse_iada_gate.yaml \
  pointpillar_attfuse_iada_source.yaml \
  pointpillar_attfuse_iada_consistency.yaml \
  pointpillar_attfuse_iada_full.yaml
do
  python opencood/tools/train_iada.py \
    -y "opencood/hypes_yaml/domain_adaptation/opv2v_to_dair/${CONFIG}"
done
```

These are independent cumulative ablations from the same source baseline, not
checkpoint chaining between ablations. If the baseline is retrained, update
only `domain_adaptation.pretrained_model_dir` in
`pointpillar_attfuse_iada_common.yaml`.
See [the full experiment guide](docs/md_files/iada_sim2real.md) for details.

## Selective Shift (SSDA)

The domain-adaptation training path supports the FSA and SAA modules from
*Selective Shift: Towards Personalized Domain Adaptation in Multi-Agent
Collaborative Perception*:

- FSA applies a fixed four-band Haar transform, adaptive frequency refinement,
  scene-local statistical attribute obfuscation, and statistical feature
  weighting to every agent feature before intermediate fusion.
- SAA applies entropy-guided global ego alignment and local all-agent alignment
  through gradient-reversal domain classifiers.
- Only source-domain labels contribute to the detector loss. Target training
  labels are not passed to the model or any loss.

The paper-aligned V2XSet-to-DAIR-V2X configuration is
`opencood/hypes_yaml/domain_adaptation/v2xset_to_dair/pointpillar_attfuse_ssda.yaml`.
Its global/local loss weights are `0.5` and `1.0`, respectively. Update the
dataset paths in that file when the datasets are stored elsewhere.

First train the matching common-geometry source baseline:

```bash
python opencood/tools/train_iada.py \
  --hypes_yaml opencood/hypes_yaml/domain_adaptation/v2xset_to_dair/pointpillar_attfuse_ssda.yaml \
  --stage baseline
```

Then warm-start SSDA from the baseline output directory:

```bash
python opencood/tools/train_iada.py \
  --hypes_yaml opencood/hypes_yaml/domain_adaptation/v2xset_to_dair/pointpillar_attfuse_ssda.yaml \
  --stage ssda \
  --pretrained_model_dir opencood/logs/<baseline-run-directory>
```

For a from-scratch comparison, omit `--pretrained_model_dir`. The detector,
fusion module, and domain adapter will all be randomly initialized and trained
at the optimizer base learning rate:

```bash
python opencood/tools/train_iada.py \
  --hypes_yaml opencood/hypes_yaml/domain_adaptation/v2xset_to_dair/pointpillar_attfuse_ssda.yaml \
  --stage ssda
```

Evaluate the resulting checkpoint on the target split with:

```bash
python opencood/tools/inference.py \
  --model_dir opencood/logs/<ssda-run-directory> \
  --fusion_method intermediate
```

Use `--model_dir` with the same `--stage ssda` to resume a saved SSDA run,
including optimizer, scheduler, AMP scaler, and random state.
