import os
import tempfile
import unittest

import torch

from opencood.tools import train_utils


class LoadSavedModelTest(unittest.TestCase):
    def test_requested_epoch_overrides_best_validation_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            requested = torch.nn.Linear(1, 1, bias=False)
            best = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                requested.weight.fill_(10.0)
                best.weight.fill_(3.0)
            torch.save(
                requested.state_dict(),
                os.path.join(directory, "net_epoch10.pth"),
            )
            torch.save(
                best.state_dict(),
                os.path.join(directory, "net_epoch_bestval_at3.pth"),
            )

            loaded = torch.nn.Linear(1, 1, bias=False)
            epoch, loaded = train_utils.load_saved_model(
                directory, loaded, checkpoint_epoch=10
            )

            self.assertEqual(epoch, 10)
            torch.testing.assert_close(loaded.weight, requested.weight)

    def test_requested_missing_epoch_fails_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(1, 1, bias=False)
            torch.save(
                model.state_dict(),
                os.path.join(directory, "net_epoch_bestval_at3.pth"),
            )

            with self.assertRaisesRegex(
                FileNotFoundError, "net_epoch10[.]pth"
            ):
                train_utils.load_saved_model(
                    directory, model, checkpoint_epoch=10
                )


if __name__ == "__main__":
    unittest.main()
