# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

# A model zoo for intermediate fusion.
# Please make sure your pairwise_t_matrix is normalized before using it.


import torch
import torch.nn.functional as F
from torch import nn

from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple


def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

def warp_feature(x, record_len, pairwise_t_matrix):
    _, C, H, W = x.shape
    B, L = pairwise_t_matrix.shape[:2]
    split_x = regroup(x, record_len)
    batch_node_features = split_x
    out = []
    # iterate each batch
    for b in range(B):
        N = record_len[b]
        t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
        # update each node i
        i = 0 # ego
        neighbor_feature = warp_affine_simple(batch_node_features[b],
                                        t_matrix[i, :, :, :],
                                        (H, W))
        out.append(neighbor_feature)

    out = torch.cat(out, dim=0)
    
    return out

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            out.append(torch.max(neighbor_feature, dim=0)[0])
        out = torch.stack(out)
        
        return out

class AttFusion(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)

    def forward(self, xx, record_len, normalized_affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        xx : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = xx.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
            out.append(h)

        out = torch.stack(out)
        return out

class DiscoFusion(nn.Module):
    def __init__(self, feature_dims):
        super(DiscoFusion, self).__init__()
        from opencood.models.fuse_modules.disco_fuse import PixelWeightLayer
        self.pixel_weight_layer = PixelWeightLayer(feature_dims)

    def forward(self, xx, record_len, normalized_affine_matrix):
        _, C, H, W = xx.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        out = []

        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            i = 0 # ego
            neighbor_feature = warp_affine_simple(split_x[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            # (N, C, H, W)
            ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
            # (N, 2C, H, W)
            neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # (N, 1, H, W)
            agent_weight = self.pixel_weight_layer(neighbor_feature_cat) 
            # (N, 1, H, W)
            agent_weight = F.softmax(agent_weight, dim=0)

            agent_weight = agent_weight.expand(-1, C, -1, -1)
            # (N, C, H, W)
            feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
            out.append(feature_fused)

        return torch.stack(out)

class V2VNetFusion(nn.Module):
    def __init__(self, args):
        super(V2VNetFusion, self).__init__()
        from opencood.models.sub_modules.convgru import ConvGRU
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W'] # remember to modify for v2xsim dataset
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']
        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']
        self.agg_operator = args['agg_operator']

        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels] * num_gru_layers,
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)

    def forward(self, x, record_len, normalized_affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]

        split_x = regroup(x, record_len)
        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L,1,H,W)).to(x)
                roi_mask[b,i] = warp_affine_simple(one_tensor, normalized_affine_matrix[b][i, :, :, :],(H, W))

        batch_node_features = split_x
        # iteratively update the features for num_iteration times
        for l in range(self.num_iteration):

            batch_updated_node_features = []
            # iterate each batch
            for b in range(B):

                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = normalized_affine_matrix[b][:N, :N, :, :]

                updated_node_features = []

                # update each node i
                for i in range(N):
                    # (N,1,H,W)
                    mask = roi_mask[b, i, :N, ...]
                    neighbor_feature = warp_affine_simple(batch_node_features[b],
                                                   t_matrix[i, :, :, :],
                                                   (H, W))

                    # (N,C,H,W)
                    ego_agent_feature = batch_node_features[b][i].unsqueeze(
                        0).repeat(N, 1, 1, 1)
                    #(N,2C,H,W)
                    neighbor_feature = torch.cat(
                        [neighbor_feature, ego_agent_feature], dim=1)
                    # (N,C,H,W)
                    # message contains all feature map from j to ego i.
                    message = self.msg_cnn(neighbor_feature) * mask

                    # (C,H,W)
                    if self.agg_operator=="avg":
                        agg_feature = torch.mean(message, dim=0)
                    elif self.agg_operator=="max":
                        agg_feature = torch.max(message, dim=0)[0]
                    else:
                        raise ValueError("agg_operator has wrong value")
                    # (2C, H, W)
                    cat_feature = torch.cat(
                        [batch_node_features[b][i, ...], agg_feature], dim=0)
                    # (C,H,W)
                    if self.gru_flag:
                        gru_out = \
                            self.conv_gru(cat_feature.unsqueeze(0).unsqueeze(0))[
                                0][
                                0].squeeze(0).squeeze(0)
                    else:
                        gru_out = batch_node_features[b][i, ...] + agg_feature
                    updated_node_features.append(gru_out.unsqueeze(0))
                # (N,C,H,W)
                batch_updated_node_features.append(
                    torch.cat(updated_node_features, dim=0))
            batch_node_features = batch_updated_node_features
        # (B,C,H,W)
        out = torch.cat(
            [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        # (B,C,H,W) -> (B, H, W, C) -> (B,C,H,W)
        out = self.mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out

class V2XViTFusion(nn.Module):
    def __init__(self, args):
        super(V2XViTFusion, self).__init__()
        from opencood.models.sub_modules.v2xvit_basic import V2XTransformer
        self.fusion_net = V2XTransformer(args['transformer'])
        self.prior_encoding_fallback = args.get(
            'prior_encoding_fallback', 'zeros'
        )
        supported_fallbacks = {
            'zeros', 'local_index_1_infra', 'error'
        }
        if self.prior_encoding_fallback not in supported_fallbacks:
            raise ValueError(
                'v2xvit.prior_encoding_fallback must be one of '
                f'{sorted(supported_fallbacks)}, got '
                f'{self.prior_encoding_fallback!r}'
            )

    def _prepare_prior_encoding(self, prior_encoding, record_len, mask, x):
        """Validate or explicitly construct V2X-ViT's heterogeneity prior."""
        batch_size, max_agents = mask.shape

        if prior_encoding is None:
            if self.prior_encoding_fallback == 'error':
                raise ValueError(
                    'V2X-ViT requires data_dict["prior_encoding"]; no '
                    'fallback is enabled'
                )
            prior_encoding = x.new_zeros(batch_size, max_agents, 3)
            if self.prior_encoding_fallback == 'local_index_1_infra':
                has_infrastructure = record_len > 1
                prior_encoding[has_infrastructure, 1, 2] = 1
        else:
            if not torch.is_tensor(prior_encoding):
                raise TypeError('prior_encoding must be a torch.Tensor')
            if prior_encoding.device != x.device:
                raise ValueError(
                    'prior_encoding and V2X-ViT features must be on the '
                    f'same device, got {prior_encoding.device} and {x.device}'
                )
            if not prior_encoding.is_floating_point():
                raise TypeError('prior_encoding must have a floating dtype')
            if prior_encoding.dtype != x.dtype:
                raise TypeError(
                    'prior_encoding and V2X-ViT features must have the same '
                    f'dtype, got {prior_encoding.dtype} and {x.dtype}'
                )
            compact_shape = (batch_size, max_agents, 3)
            singleton_spatial_shape = compact_shape + (1, 1)
            if tuple(prior_encoding.shape) == singleton_spatial_shape:
                prior_encoding = prior_encoding[..., 0, 0]
            elif tuple(prior_encoding.shape) != compact_shape:
                raise ValueError(
                    'prior_encoding must have shape [B, L, 3] or '
                    f'[B, L, 3, 1, 1], got {tuple(prior_encoding.shape)} '
                    f'for B={batch_size}, L={max_agents}'
                )

        if not torch.isfinite(prior_encoding).all():
            raise ValueError('prior_encoding must contain only finite values')

        padded_values = prior_encoding.masked_select(
            ~mask.bool().unsqueeze(-1)
        )
        if padded_values.numel() and torch.any(padded_values != 0):
            raise ValueError(
                'prior_encoding must be zero in positions masked as padding'
            )

        infrastructure_type = prior_encoding[..., 2]
        if not torch.all(
            (infrastructure_type == 0) | (infrastructure_type == 1)
        ):
            raise ValueError(
                'prior_encoding infrastructure_type (channel 2) must be '
                'exactly 0 or 1'
            )

        return prior_encoding[..., None, None].expand(
            -1, -1, -1, x.shape[-2], x.shape[-1]
        )

    def forward(self, x, record_len, normalized_affine_matrix,
                prior_encoding=None):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 

        prior_encoding : torch.Tensor, optional
            Per-agent [velocity, time_delay, infrastructure_type], with shape
            (B, L, 3) or (B, L, 3, 1, 1). Explicit input always takes
            precedence over ``prior_encoding_fallback``.
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        if not torch.is_tensor(x) or x.ndim != 4:
            raise ValueError('x must be a four-dimensional torch.Tensor')
        if not x.is_floating_point():
            raise TypeError('x must have a floating dtype')
        if not torch.is_tensor(record_len) or record_len.ndim != 1:
            raise ValueError('record_len must be a one-dimensional tensor')
        if record_len.device != x.device:
            raise ValueError('record_len and x must be on the same device')
        if record_len.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
        }:
            raise TypeError('record_len must have an integer dtype')
        if not torch.is_tensor(normalized_affine_matrix):
            raise TypeError('normalized_affine_matrix must be a torch.Tensor')
        if normalized_affine_matrix.device != x.device:
            raise ValueError(
                'normalized_affine_matrix and x must be on the same device'
            )
        if not normalized_affine_matrix.is_floating_point():
            raise TypeError(
                'normalized_affine_matrix must have a floating dtype'
            )

        _, C, H, W = x.shape
        if normalized_affine_matrix.ndim != 5:
            raise ValueError(
                'normalized_affine_matrix must have shape [B, L, L, 2, 3]'
            )
        B, L = normalized_affine_matrix.shape[:2]
        expected_affine_shape = (B, L, L, 2, 3)
        if tuple(normalized_affine_matrix.shape) != expected_affine_shape:
            raise ValueError(
                'normalized_affine_matrix must have shape [B, L, L, 2, 3], '
                f'got {tuple(normalized_affine_matrix.shape)}'
            )
        if record_len.numel() != B:
            raise ValueError(
                f'record_len contains {record_len.numel()} scenes but the '
                f'affine matrix contains {B}'
            )
        if torch.any(record_len < 1) or torch.any(record_len > L):
            raise ValueError(
                f'each record_len entry must be in [1, {L}]'
            )
        if int(record_len.sum().item()) != x.shape[0]:
            raise ValueError(
                'sum(record_len) must equal the flattened agent count in x'
            )
        if not torch.isfinite(normalized_affine_matrix).all():
            raise ValueError(
                'normalized_affine_matrix must contain only finite values'
            )

        regroup_feature, mask = Regroup(x, record_len, L)
        regroup_feature = regroup_feature.to(dtype=x.dtype)
        expected_mask = (
            torch.arange(L, device=x.device).unsqueeze(0)
            < record_len.to(torch.long).unsqueeze(1)
        )
        if (tuple(mask.shape) != (B, L)
                or mask.device != x.device
                or not torch.equal(mask.bool(), expected_mask)):
            raise RuntimeError(
                'V2X-ViT regroup mask is inconsistent with record_len'
            )
        prior_encoding = self._prepare_prior_encoding(
            prior_encoding, record_len, mask, x
        )

        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        regroup_feature_new = []

        for b in range(B):
            ego = 0
            regroup_feature_new.append(warp_affine_simple(regroup_feature[b], normalized_affine_matrix[b, ego], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion. In perfect setting, there is no delay. 
        # it is possible to modify the xxx_basedataset.py and intermediatefusiondataset.py to retrieve these information
        spatial_correction_matrix = torch.eye(
            4, device=x.device, dtype=x.dtype
        ).expand(B, L, 4, 4)
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        
        return fused_feature


class CoBEVT(nn.Module):
    """CoBEVT masked swap-attention fusion for homogeneous BEV features."""

    def __init__(self, args):
        super().__init__()
        from einops.layers.torch import Rearrange, Reduce
        from opencood.models.fuse_modules.swap_fusion_modules import (
            SwapFusionBlockMask,
        )

        self.agent_size = int(args["agent_size"])
        self.window_size = int(args["window_size"])
        input_dim = int(args["input_dim"])
        self.layers = nn.ModuleList(
            SwapFusionBlockMask(
                input_dim=input_dim,
                mlp_dim=int(args["mlp_dim"]),
                dim_head=int(args["dim_head"]),
                window_size=self.window_size,
                agent_size=self.agent_size,
                drop_out=float(args["drop_out"]),
            )
            for _ in range(int(args["depth"]))
        )
        self.mlp_head = nn.Sequential(
            Reduce("b m d h w -> b d h w", "mean"),
            Rearrange("b d h w -> b h w d"),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            Rearrange("b h w d -> b d h w"),
        )

    def forward(self, features, record_len, affine_matrix):
        _, channels, height, width = features.shape
        batch_size, max_agents = affine_matrix.shape[:2]
        if max_agents != self.agent_size:
            raise ValueError(
                "CoBEVT agent_size must equal pairwise matrix max agents; "
                f"got {self.agent_size} and {max_agents}"
            )
        if height % self.window_size or width % self.window_size:
            raise ValueError(
                "CoBEVT feature height and width must be divisible by "
                f"window_size={self.window_size}; got {(height, width)}"
            )
        if int(record_len.max().item()) > self.agent_size:
            raise ValueError("CoBEVT record_len exceeds configured agent_size")

        regrouped, mask = Regroup(features, record_len, self.agent_size)
        regrouped = regrouped.to(dtype=features.dtype)
        mask = mask.to(device=features.device, dtype=torch.bool)
        communication_mask = mask[:, None, None, None, :].expand(
            batch_size, height, width, 1, self.agent_size
        )

        warped = []
        for batch_index in range(batch_size):
            warped.append(
                warp_affine_simple(
                    regrouped[batch_index],
                    affine_matrix[batch_index, 0],
                    (height, width),
                )
            )
        output = torch.stack(warped)
        for layer in self.layers:
            output = layer(output, mask=communication_mask)
        output = self.mlp_head(output)
        if output.shape != (batch_size, channels, height, width):
            raise RuntimeError("CoBEVT produced an unexpected output shape")
        return output


class When2commFusion(nn.Module):
    def __init__(self, args):
        super(When2commFusion, self).__init__()
        from opencood.models.fuse_modules.when2com_fuse import policy_net4, km_generator_v2, AdditiveAttentin

        self.in_channels = args['in_channels']
        self.feat_H = args['H']
        self.feat_W = args['W']
        self.query_size = args['query_size']
        self.key_size = args['key_size']
        

        self.query_key_net = policy_net4(self.in_channels)
        self.key_net = km_generator_v2(out_size=self.key_size)
        self.query_net = km_generator_v2(out_size=self.query_size)
        # self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size)
        self.attention_net = AdditiveAttentin(self.key_size, self.query_size)

    def forward(self, x, record_len, normalized_affine_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, shape: (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        normalized_affine_matrix : torch.Tensor
            The normalized affine transformation matrix from each cav to ego, 
            shape: (B, L, L, 2, 3) 
            
        Returns
        -------
        Fused feature : torch.Tensor
            shape: (B, C, H, W)
        """
        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        updated_node_features = []
        for b in range(B):

            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]

            # update each node i
            # (N,1,H,W)
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            query_key_maps = self.query_key_net(neighbor_feature)

            keys = self.key_net(query_key_maps).unsqueeze(0) # [N, C_k]
            query = self.query_net(query_key_maps[0].unsqueeze(0)).unsqueeze(0) # [1, C_q]

            neighbor_feature = neighbor_feature.unsqueeze(0) # [1, N, C, H, W]

            feat_fuse, prob_action = self.attention_net(query, keys, neighbor_feature, sparse=False)

            updated_node_features.append(feat_fuse)

        out = torch.cat(updated_node_features, dim=0)
        
        return out


class HMMambaFusion(nn.Module):
    def __init__(self, feature_dims):
        super(HMMambaFusion, self).__init__()
        # VMamba has optional third-party dependencies that are unrelated to
        # the standard OpenCOOD fusion operators.  Import it only when this
        # experimental fusion module is actually selected.
        from opencood.models.sub_modules.vmamba import SS2D

        self.ssd = SS2D(d_model=256, channel_first=True, forward_type="v05_noz")

    def forward(self, x, record_len, normalized_affine_matrix):
        _, C, H, W = x.shape
        B, L = normalized_affine_matrix.shape[:2]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = normalized_affine_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0  # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(1, 0, 2)  # (C, cav_num, H*W), perform self attention on each pixel.
            x = x.unsqueeze(0).contiguous()
            x = self.ssd(x)
            h = x.squeeze(0)
            h = h.permute(1, 0, 2).view(cav_num, C, W, H).contiguous()[0, ...]

            out.append(h)

        return torch.stack(out)


if __name__ == '__main__':
    import thop

    model = HMMambaFusion(256).cuda()
    # model = AttFusion(256).cuda()
    # model = DiscoFusion(256).cuda()
    x = torch.randn((3, 256, 100, 252)).cuda()
    record_len = torch.tensor([2, 1], dtype=torch.long).cuda()
    normalized_affine_matrix = torch.randn((2, 5, 5, 2, 3)).cuda()

    flops, params = thop.profile(model, inputs=(x, record_len, normalized_affine_matrix))
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")

    # model = HMMambaFusion(256).cuda()
    # # model = AttFusion(256).cuda()
    # # model = DiscoFusion(256).cuda()
    # model.eval()
    #
    # x = torch.randn((3, 256, 100, 252)).cuda()
    # record_len = torch.tensor([2, 1], dtype=torch.long).cuda()
    # normalized_affine_matrix = torch.randn((2, 5, 5, 2, 3)).cuda()
    #
    # # Warm-up to avoid initial overhead
    # for _ in range(10):
    #     with torch.no_grad():
    #         model(x, record_len, normalized_affine_matrix)
    #
    # # 开始计时
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # torch.cuda.synchronize()
    # starter.record()
    #
    # with torch.no_grad():
    #     model(x, record_len, normalized_affine_matrix)
    #
    # ender.record()
    # torch.cuda.synchronize()
    # elapsed_time_ms = starter.elapsed_time(ender)
    #
    # print(f"Inference Time: {elapsed_time_ms:.3f} ms")
