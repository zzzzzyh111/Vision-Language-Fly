from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from vlfly.base_model import BaseModel
from vlfly.self_attention import MultiLayerDecoder


class VLFly(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        late_fusion: Optional[bool] = False,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        super().__init__(context_size, len_traj_pred, learn_angle)
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.late_fusion = late_fusion

        if obs_encoder.split("-")[0] != "efficientnet":
            raise NotImplementedError

        self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
        self.num_obs_features = self.obs_encoder._fc.in_features
        if self.late_fusion:
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
        else:
            self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.num_goal_features = self.goal_encoder._fc.in_features

        self.compress_obs_enc = (
            nn.Linear(self.num_obs_features, self.obs_encoding_size)
            if self.num_obs_features != self.obs_encoding_size
            else nn.Identity()
        )
        self.compress_goal_enc = (
            nn.Linear(self.num_goal_features, self.goal_encoding_size)
            if self.num_goal_features != self.goal_encoding_size
            else nn.Identity()
        )

        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size + 2,
            output_layers=[256, 128, 64, 32],
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )
        self.dist_predictor = nn.Sequential(nn.Linear(32, 1))
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params)
        )

    def forward(
        self, obs_img: torch.Tensor, goal_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.late_fusion:
            goal_encoding = self.goal_encoder.extract_features(goal_img)
        else:
            obsgoal_img = torch.cat([obs_img[:, 3 * self.context_size :, :, :], goal_img], dim=1)
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)
        goal_encoding = self.compress_goal_enc(goal_encoding)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)

        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.reshape((self.context_size + 1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        final_repr = self.decoder(tokens)

        dist_pred = self.dist_predictor(final_repr)
        action_pred = self.action_predictor(final_repr)
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(action_pred[:, :, 2:].clone(), dim=-1)
        return dist_pred, action_pred
