# IADA

Research code under active development.

Documentation and reproducible experiment instructions will be added when the
project is ready for release.

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

Evaluate the resulting checkpoint on the target split with:

```bash
python opencood/tools/inference.py \
  --model_dir opencood/logs/<ssda-run-directory> \
  --fusion_method intermediate
```

Use `--model_dir` with the same `--stage ssda` to resume a saved SSDA run,
including optimizer, scheduler, AMP scaler, and random state.
