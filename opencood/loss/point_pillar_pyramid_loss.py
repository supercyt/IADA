"""Optional single-agent occupancy supervision for PyramidFusion."""

import torch
import torch.nn.functional as F

from opencood.loss.point_pillar_loss import (
    PointPillarLoss,
    sigmoid_focal_loss,
)


class PointPillarPyramidLoss(PointPillarLoss):
    """Standard PointPillar loss with an opt-in Pyramid auxiliary term.

    With ``pyramid_aux_loss.enabled: false`` the default forward is exactly
    :class:`PointPillarLoss`; no single-agent labels are consumed.
    """

    def __init__(self, args):
        super().__init__(args)
        config = args.get("pyramid_aux_loss", {})
        self.pyramid_aux_enabled = bool(config.get("enabled", False))
        self.relative_downsample = config.get(
            "relative_downsample", [1, 2, 4]
        )
        self.pyramid_weights = config.get("weight", [0.4, 0.2, 0.1])

    def forward(self, output_dict, target_dict, suffix=""):
        if suffix != "_single":
            return super().forward(output_dict, target_dict, suffix)
        if not self.pyramid_aux_enabled:
            first = output_dict["cls_preds"]
            return first.new_zeros(())
        occupancy = output_dict.get("occ_single_list")
        if occupancy is None:
            raise KeyError("Pyramid auxiliary loss requires occ_single_list")
        if len(occupancy) != len(self.relative_downsample):
            raise ValueError("Pyramid occupancy level count does not match loss")

        batch_size = target_dict["pos_equal_one"].shape[0]
        positives = torch.logical_or(
            target_dict["pos_equal_one"][..., 0],
            target_dict["pos_equal_one"][..., 1],
        ).unsqueeze(-1).float()
        negatives = torch.logical_and(
            target_dict["neg_equal_one"][..., 0],
            target_dict["neg_equal_one"][..., 1],
        ).unsqueeze(-1).float()

        total = occupancy[0].new_zeros(())
        for logits, downsample, weight in zip(
            occupancy,
            self.relative_downsample,
            self.pyramid_weights,
        ):
            positive_level = F.max_pool2d(
                positives.permute(0, 3, 1, 2),
                kernel_size=downsample,
                stride=downsample,
            ).permute(0, 2, 3, 1)
            negative_level = 1 - F.max_pool2d(
                (1 - negatives).permute(0, 3, 1, 2),
                kernel_size=downsample,
                stride=downsample,
            ).permute(0, 2, 3, 1)
            labels = positive_level.reshape(batch_size, -1, 1)
            negative_level = negative_level.reshape(batch_size, -1, 1)
            normalizer = labels.sum(1, keepdim=True).clamp_min(1.0)
            weights = (
                labels * self.pos_cls_weight + negative_level
            ) / normalizer
            predictions = logits.permute(0, 2, 3, 1).reshape(
                batch_size, -1, 1
            )
            level_loss = sigmoid_focal_loss(
                predictions, labels, weights=weights, **self.cls
            )
            total = total + level_loss.sum() * float(weight) / batch_size

        self.loss_dict.update(
            {"pyramid_loss": total.item(), "total_loss": total.item()}
        )
        return total


__all__ = ["PointPillarPyramidLoss"]
