"""Slim DP3 point-cloud encoder stack.

This is a minimal, self-contained copy of the point-net / DP3 encoder used by
the ARX L5 grasp experiment. It contains only the classes needed to (a) build
the DP3 BC policy and (b) load the external `epoch=0175` checkpoint state_dict
(keys: `obs_encoder.extractor.mlp.*`, `obs_encoder.extractor.final_projection.*`,
`obs_encoder.state_mlp.*`).

Imports are kept dependency-free on the rest of RL-100 on purpose so this
folder is portable.
"""
import torch
import torch.nn as nn

from typing import Dict, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud (xyz-rgb, 6 channels)."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud (xyz, 3 channels)."""

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 **kwargs
                 ):
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')

        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")

    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class VIB(nn.Module):
    """Variational Information Bottleneck (lightweight, DP3-style).

    Maps an input feature to a latent distribution (mu, logvar), samples with
    the reparameterization trick in train mode and uses the mean at eval.
    Records the per-call KL divergence (mean over the batch, summed over the
    latent dim) on ``self.kl_loss`` so a caller can add ``beta * kl`` to the
    main loss right after the encoder forward pass.
    """

    def __init__(self, in_features, out_features=None):
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.logvar = nn.Linear(in_features, out_features)
        self.kl_loss = 0.0

    def forward(self, x):
        mu = self.mu(x)
        logvar = self.logvar(x)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        self.kl_loss = -0.5 * torch.sum(
            1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        return z


class DP3Encoder(nn.Module):
    """DP3 (3D-Diffusion-Policy) observation encoder.

    Encodes the point cloud with a PointNet + agent state with a small MLP,
    and concatenates the two. Structurally identical to the encoder used to
    train the external `epoch=0175` checkpoint.

    Optional VIB (Variational Information Bottleneck, enabled by
    ``vib: true`` inside ``pointcloud_encoder_cfg``): after the pointnet /
    state features are computed, each is re-parameterized through a tiny
    mu/logvar bottleneck (sampled in train mode, mean at eval) and the KL
    divergence (weighted by ``beta_kl``) is accumulated on ``self.kl_loss``.
    """

    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 backbone=None,
                 integrate_strategy='add',  # add, concat features of point cloud and image
                 use_agent_pos=True,
                 use_vib=False,
                 beta_kl=1e-3,
                 latent_dim=None,
                 latent_state_dim=None,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        self.use_agent_pos = use_agent_pos

        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        # Handle case where point cloud might not be available (RGB-only)
        if self.point_cloud_key in observation_space:
            self.point_cloud_shape = observation_space[self.point_cloud_key]
        else:
            self.point_cloud_shape = None
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
        self.backbone_name = backbone
        self.integrate_strategy = integrate_strategy
        if backbone is not None:
            assert backbone in ["resnet50", "resnet18", "clip"]
            raise NotImplementedError(
                "2D image backbones are not included in this slim copy. "
                "Only the point-cloud DP3 path is supported.")

        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape if self.point_cloud_shape else 'None (RGB-only mode)'}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        # VIB knobs may travel inside pointcloud_encoder_cfg; pop them so the
        # PointNet sub-encoder never sees them.
        if pointcloud_encoder_cfg is not None:
            pointcloud_encoder_cfg = dict(pointcloud_encoder_cfg)
            use_vib = bool(pointcloud_encoder_cfg.pop('vib', use_vib))
            beta_kl = float(pointcloud_encoder_cfg.pop('beta_kl', beta_kl))
            latent_dim = pointcloud_encoder_cfg.pop('latent_dim', latent_dim)
            latent_state_dim = pointcloud_encoder_cfg.pop(
                'latent_state_dim', latent_state_dim)
        self.use_vib = use_vib
        self.beta_kl = beta_kl
        self.kl_loss = None

        # Only initialize point cloud extractor if point cloud is available and config is provided
        if self.point_cloud_shape is not None and pointcloud_encoder_cfg is not None:
            # Channel count follows the declared observation shape.  The
            # semantic-mask experiment appends two binary channels (1024x8);
            # legacy xyz/xyz-rgb configs keep 3/6 unchanged.
            n_pc_channels = int(self.point_cloud_shape[-1])
            if pointnet_type == "pointnet":
                if use_pc_color:
                    cfg_ch = int(pointcloud_encoder_cfg.get(
                        'in_channels', n_pc_channels))
                    if cfg_ch != n_pc_channels:
                        cprint(f"[DP3Encoder] in_channels {cfg_ch} != shape "
                               f"{n_pc_channels}; using shape", "yellow")
                    pointcloud_encoder_cfg['in_channels'] = n_pc_channels
                    self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
                else:
                    if n_pc_channels != 3:
                        raise ValueError(
                            "use_pc_color=False only supports xyz (3) point "
                            f"clouds, got {n_pc_channels} channels")
                    pointcloud_encoder_cfg['in_channels'] = 3
                    self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
            else:
                raise NotImplementedError(f"pointnet_type: {pointnet_type}")
        else:
            self.extractor = None  # No point cloud extractor for RGB-only case

        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        if use_agent_pos:
            self.n_output_channels += output_dim
        if backbone is not None and self.integrate_strategy == 'concat':
            self.n_output_channels += out_channel

        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")

        # ---- optional VIB bottleneck ----
        if self.use_vib:
            latent_pc_dim = latent_dim if latent_dim is not None else out_channel
            self.vib_pc = VIB(out_channel, latent_pc_dim)
            if use_agent_pos:
                lsd = latent_state_dim if latent_state_dim is not None else output_dim
                self.vib_state = VIB(output_dim, lsd)
                self.n_output_channels = latent_pc_dim + lsd
            else:
                self.vib_state = None
                self.n_output_channels = latent_pc_dim
            cprint(f"[DP3Encoder] VIB enabled | pc {out_channel}->{latent_pc_dim} | "
                   f"state {output_dim if use_agent_pos else '-'}->"
                   f"{lsd if use_agent_pos else '-'} | beta {beta_kl}", "cyan")
        else:
            self.vib_pc = None
            self.vib_state = None

    def forward(self, observations: Dict) -> torch.Tensor:
        # RGB-only mode: without a point cloud extractor, features come from
        # the (frozen) image backbone; otherwise from the point cloud encoder.
        pn_feat = None
        if self.extractor is not None and self.point_cloud_key in observations:
            points = observations[self.point_cloud_key]
            if len(points.shape) != 3:
                print(f"[WARNING] point cloud shape: {points.shape}, expected 3D, got {len(points.shape)}D")
            assert len(points.shape) == 3, f"point cloud shape: {points.shape}, length should be 3"
            if self.use_imagined_robot:
                img_points = observations[self.imagination_key][..., :points.shape[-1]]  # align the last dim
                points = torch.concat([points, img_points], dim=1)
            pn_feat = self.extractor(points)  # B * out_channel

        if self.use_agent_pos:
            state = observations[self.state_key]
            state_feat = self.state_mlp(state)  # B * 64
            if self.use_vib:
                pc_latent = self.vib_pc(pn_feat)
                state_latent = self.vib_state(state_feat)
                self.kl_loss = self.beta_kl * (
                    self.vib_pc.kl_loss + self.vib_state.kl_loss)
                final_feat = torch.cat([pc_latent, state_latent], dim=-1)
            else:
                final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        else:
            if self.use_vib:
                self.kl_loss = self.beta_kl * self.vib_pc.kl_loss
                final_feat = self.vib_pc(pn_feat)
            else:
                final_feat = pn_feat
        return final_feat

    def output_shape(self):
        return self.n_output_channels
