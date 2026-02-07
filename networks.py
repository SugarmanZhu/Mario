"""
Custom neural network architectures for Mario RL.

Contains IMPALA-style residual CNN for multi-level training.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ImpalaResBlock(nn.Module):
    """Residual block used in IMPALA CNN."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x  # Residual connection


class ImpalaStage(nn.Module):
    """
    One stage of the IMPALA CNN.
    Conv -> MaxPool -> ResBlock -> ResBlock
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_block1 = ImpalaResBlock(out_channels)
        self.res_block2 = ImpalaResBlock(out_channels)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class MarioImpalaExtractor(BaseFeaturesExtractor):
    """
    IMPALA-style CNN feature extractor for Mario training.

    Accepts Box observation space (C, H, W) - e.g., (12, 120, 128) for 4 stacked RGB frames.

    Architecture:
        - 3 IMPALA stages with depths [32, 64, 64]
        - Each stage: Conv3x3 -> MaxPool3x3(s=2) -> 2x ResBlock
        - Flatten -> Linear -> ReLU (features)

    This is proven to work better than NatureCNN for multi-task RL.
    The agent learns to distinguish levels from visual cues (HUD text shows "WORLD 1-1").
    """

    def __init__(
        self,
        observation_space,
        cnn_output_dim: int = 512,
        stage_depths: tuple = (16, 32, 32),
    ):
        # Initialize with dummy features_dim, we'll set it properly below
        super().__init__(observation_space, features_dim=1)

        # Build CNN stages
        in_channels = observation_space.shape[0]  # e.g., 4 for stacked frames
        stages = []
        for depth in stage_depths:
            stages.append(ImpalaStage(in_channels, depth))
            in_channels = depth

        self.cnn_body = nn.Sequential(*stages, nn.ReLU(), nn.Flatten())

        # Calculate flattened size by doing a forward pass
        with th.no_grad():
            dummy_img = th.zeros(1, *observation_space.shape)
            n_flatten = self.cnn_body(dummy_img).shape[1]

        # CNN head: flatten -> dense
        self.cnn_head = nn.Sequential(
            nn.Linear(n_flatten, cnn_output_dim),
            nn.ReLU(),
        )

        # Total features dimension
        self._features_dim = cnn_output_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations is (B, C, H, W) tensor
        return self.cnn_head(self.cnn_body(observations.float()))


class MarioCnnExtractor(BaseFeaturesExtractor):
    """
    Standard CNN feature extractor for single-level Mario training.

    Uses NatureCNN-style architecture but can be configured for more capacity.
    For single-level training where Dict observations are not needed.
    """

    def __init__(
        self,
        observation_space,
        features_dim: int = 512,
        depths: tuple = (32, 64, 64),
    ):
        super().__init__(observation_space, features_dim=features_dim)

        n_input_channels = observation_space.shape[0]

        # Build convolutional layers (NatureCNN style but configurable)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate output size
        with th.no_grad():
            n_flatten = self.cnn(th.zeros(1, *observation_space.shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def get_policy_kwargs() -> dict:
    """
    Get policy_kwargs for PPO with IMPALA CNN extractor.

    Always uses IMPALA architecture - proven to work better than NatureCNN
    for multi-task RL and complex environments.

    Returns:
        policy_kwargs dict for SB3 PPO
    """
    return dict(
        features_extractor_class=MarioImpalaExtractor,
        features_extractor_kwargs=dict(
            cnn_output_dim=512,
            stage_depths=(32, 64, 64),
        ),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.ReLU,
    )
