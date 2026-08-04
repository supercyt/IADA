import math
import unittest

import torch

from opencood.models.sub_modules.interaction_da import (
    GradientReversal,
    InteractionDomainAdapter,
    InteractionGraphEncoder,
)


def _identity_pairwise(batch_size, max_agents, dtype=torch.float32):
    identity = torch.eye(4, dtype=dtype)
    return identity.view(1, 1, 1, 4, 4).repeat(
        batch_size,
        max_agents,
        max_agents,
        1,
        1,
    )


class GradientReversalTest(unittest.TestCase):
    def test_forward_identity_and_backward_sign(self):
        inputs = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        weights = torch.tensor([2.0, 4.0, -1.0])

        reversed_inputs = GradientReversal(0.25)(inputs)
        torch.testing.assert_close(reversed_inputs, inputs)

        (reversed_inputs * weights).sum().backward()
        torch.testing.assert_close(inputs.grad, -0.25 * weights)

    def test_call_time_coefficient_overrides_default(self):
        inputs = torch.tensor([1.0, 2.0], requires_grad=True)
        output = GradientReversal(9.0)(inputs, coefficient=0.5)
        output.sum().backward()
        torch.testing.assert_close(
            inputs.grad,
            torch.full_like(inputs, -0.5),
        )


class InteractionGraphEncoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.encoder = InteractionGraphEncoder(
            in_channels=2,
            hidden_dim=8,
            geometry_scale=10.0,
        )

    def test_shapes_masks_and_geometry(self):
        agent_features = torch.stack(
            (
                torch.full((2, 2, 3), 1.0),
                torch.full((2, 2, 3), 2.0),
                torch.full((2, 2, 3), 3.0),
            )
        )
        record_len = torch.tensor([2, 1])
        pairwise = _identity_pairwise(batch_size=2, max_agents=3)

        yaw = math.pi / 2
        pairwise[0, 0, 1, 0, 0] = math.cos(yaw)
        pairwise[0, 0, 1, 0, 1] = -math.sin(yaw)
        pairwise[0, 0, 1, 1, 0] = math.sin(yaw)
        pairwise[0, 0, 1, 1, 1] = math.cos(yaw)
        pairwise[0, 0, 1, 0, 3] = 3.0
        pairwise[0, 0, 1, 1, 3] = 4.0

        # Invalid padded transforms must be ignored even when non-identity.
        pairwise[:, 2, :, 0, 3] = 999.0
        pairwise[:, :, 2, 1, 3] = -999.0

        output = self.encoder(agent_features, record_len, pairwise)

        self.assertEqual(output["pooled_node_features"].shape, (2, 3, 2))
        self.assertEqual(output["node_embeddings"].shape, (2, 3, 8))
        self.assertEqual(output["edge_attributes"].shape, (2, 3, 3, 5))
        self.assertEqual(output["edge_hidden"].shape, (2, 3, 3, 8))
        self.assertEqual(output["graph_embedding"].shape, (2, 8))
        self.assertNotIn("interaction_logits", output)

        expected_node_mask = torch.tensor(
            [[True, True, False], [True, False, False]]
        )
        self.assertTrue(torch.equal(output["node_mask"], expected_node_mask))
        self.assertTrue(
            torch.equal(
                output["edge_mask"],
                expected_node_mask.unsqueeze(2)
                & expected_node_mask.unsqueeze(1),
            )
        )
        self.assertTrue(
            torch.equal(
                output["valid_graph_mask"],
                torch.tensor([True, False]),
            )
        )

        torch.testing.assert_close(
            output["pooled_node_features"][0, :2],
            torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        )
        torch.testing.assert_close(
            output["pooled_node_features"][1, 0],
            torch.tensor([3.0, 3.0]),
        )
        torch.testing.assert_close(
            output["edge_attributes"][0, 0, 1],
            torch.tensor([0.3, 0.4, 0.5, 1.0, 0.0]),
            atol=1e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output["edge_attributes"][~output["edge_mask"]],
            torch.zeros_like(
                output["edge_attributes"][~output["edge_mask"]]
            ),
        )

        self.assertFalse(
            any("score_head" in key for key in self.encoder.state_dict())
        )

    def test_single_agent_falls_back_to_node_embedding_without_nan(self):
        agent_features = torch.randn(1, 2, 3, 4)
        record_len = torch.tensor([1])
        pairwise = _identity_pairwise(batch_size=1, max_agents=4)

        output = self.encoder(agent_features, record_len, pairwise)

        self.assertFalse(output["valid_graph_mask"].item())
        self.assertFalse(torch.isnan(output["graph_embedding"]).any().item())
        torch.testing.assert_close(
            output["graph_embedding"][0],
            output["node_embeddings"][0, 0],
        )
        self.assertEqual(
            int(output["graph_edge_mask"].sum().item()),
            0,
        )

    def test_invalid_agent_count_is_rejected(self):
        features = torch.randn(2, 2, 1, 1)
        pairwise = _identity_pairwise(batch_size=1, max_agents=2)
        with self.assertRaisesRegex(ValueError, r"sum\(record_len\)"):
            self.encoder(features, torch.tensor([1]), pairwise)


class InteractionDomainAdapterTest(unittest.TestCase):
    def test_clear_adapter_output_and_domain_logit_shape(self):
        adapter = InteractionDomainAdapter(
            in_channels=3,
            hidden_dim=6,
            discriminator_hidden_dim=4,
            geometry_scale=20.0,
            normalize_domain_embedding=True,
        )
        features = torch.randn(3, 3, 2, 2, requires_grad=True)
        record_len = torch.tensor([2, 1])
        pairwise = _identity_pairwise(batch_size=2, max_agents=2)

        output = adapter(
            features,
            record_len,
            pairwise,
            grl_lambda=0.3,
        )

        expected_keys = {
            "pooled_node_features",
            "node_embeddings",
            "node_mask",
            "edge_attributes",
            "edge_hidden",
            "edge_mask",
            "graph_edge_mask",
            "graph_embedding",
            "domain_embedding",
            "valid_graph_mask",
            "reversed_graph_embedding",
            "domain_logits",
        }
        self.assertEqual(set(output), expected_keys)
        self.assertEqual(output["domain_logits"].shape, (2, 1))
        torch.testing.assert_close(
            output["reversed_graph_embedding"],
            output["domain_embedding"],
        )
        torch.testing.assert_close(
            output["domain_embedding"].norm(dim=1),
            torch.ones(2),
        )
        self.assertTrue(torch.isfinite(output["domain_logits"]).all().item())
        self.assertNotIn("interaction_logits", output)

        output["domain_logits"].sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all().item())
        self.assertGreater(features.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
