import unittest
from pathlib import Path

import yaml

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.train_iada import _configure_stage


class TargetModelConfigTest(unittest.TestCase):
    ROOT = Path("opencood/hypes_yaml/domain_adaptation")
    TARGETS = {
        "opv2v_to_zut": ("opv2v", [80, 160]),
        "opv2v_to_v2v4real": ("v2v4real", [48, 176]),
    }
    MODELS = {
        "attfuse": ("attfuse", "att"),
        "cobevt": ("cobevt", "cobevt"),
        "v2xvit": ("v2xvit", "v2xvit"),
        "pyramidfusion": ("pyramid", "pyramid"),
    }

    def test_model_directories_and_training_stages(self):
        for target, (dataset_name, feature_size) in self.TARGETS.items():
            for directory, (token, fusion_method) in self.MODELS.items():
                with self.subTest(target=target, model=directory):
                    folder = self.ROOT / target / directory
                    config_files = list(folder.glob("*.yaml"))
                    self.assertEqual(len(config_files), 8)

                    for path in config_files:
                        raw = yaml.safe_load(path.read_text())
                        base_config = raw.get("base_config")
                        if base_config is not None:
                            self.assertEqual(Path(base_config).parent, Path("."))
                            self.assertTrue((folder / base_config).is_file())
                        self.assertNotIn("opv2v_to_dair", path.read_text())

                    main = folder / f"pointpillar_{token}_iada.yaml"
                    self.assertNotIn(
                        "base_config",
                        yaml.safe_load(main.read_text()),
                    )
                    hypes = load_yaml(str(main), None)
                    self.assertEqual(hypes["fusion"]["dataset"], dataset_name)
                    self.assertEqual(
                        hypes["model"]["args"]["fusion_method"],
                        fusion_method,
                    )
                    self.assertEqual(
                        hypes["model"]["args"]["domain_adapter"][
                            "feature_size"
                        ],
                        feature_size,
                    )
                    bounds = hypes["domain_adaptation"][
                        "cudax_residual_bounds"
                    ]
                    self.assertEqual(len(bounds), 6)
                    self.assertTrue(all(value > 0 for value in bounds))

                    comparison = (
                        folder / f"pointpillar_{token}_comparison.yaml"
                    )
                    for stage in ("grl", "dusa", "cudax", "ssda"):
                        stage_hypes = load_yaml(str(comparison), None)
                        self.assertEqual(
                            _configure_stage(stage_hypes, stage), stage
                        )
                        self.assertEqual(
                            stage_hypes["train_params"]["epoches"], 15
                        )

                    full = folder / f"pointpillar_{token}_iada_full.yaml"
                    full_hypes = load_yaml(str(full), None)
                    self.assertEqual(
                        _configure_stage(full_hypes, "iada"), "iada"
                    )


if __name__ == "__main__":
    unittest.main()
