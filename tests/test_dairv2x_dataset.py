import unittest

from opencood.data_utils.datasets.basedataset.dairv2x_basedataset import (
    _filter_dair_labels,
)


class DAIRLabelFilterTest(unittest.TestCase):
    def test_car_only_filter_is_case_insensitive_and_strict(self):
        labels = [
            {"type": "Car", "id": 1},
            {"type": "car", "id": 2},
            {"type": "Van", "id": 3},
            {"type": "Truck", "id": 4},
            {"type": "Bus", "id": 5},
            {"id": 6},
        ]

        filtered = _filter_dair_labels(labels, car_only=True)

        self.assertEqual([label["id"] for label in filtered], [1, 2])

    def test_explicit_opt_out_preserves_all_labels(self):
        labels = [{"type": "Car"}, {"type": "Van"}]

        filtered = _filter_dair_labels(labels, car_only=False)

        self.assertIs(filtered, labels)


if __name__ == "__main__":
    unittest.main()
