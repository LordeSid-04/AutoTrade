import torch
import torch.nn as nn
from dataclasses import dataclass
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction
from d3rlpy.models.encoders import EncoderFactory
import d3rlpy

class SharedAssetEncoderNet(Encoder):
    def __init__(self, observation_shape, feature_size=256, num_assets=15, num_features=10):
        super().__init__()
        self.observation_shape = observation_shape
        self.feature_size = feature_size
        self.num_assets = num_assets
        self.num_features = num_features
        
        self.asset_mlp = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        other_dims = num_assets + 1 + num_assets  # weights (15) + cash (1) + caps (15)
        
        self.fc = nn.Sequential(
            nn.Linear(16 * num_assets + other_dims, 256),
            nn.ReLU(),
            nn.Linear(256, feature_size),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        asset_features = x[:, :self.num_assets * self.num_features]
        other_features = x[:, self.num_assets * self.num_features:]
        
        reshaped = asset_features.reshape(-1, self.num_features)
        encoded_assets = self.asset_mlp(reshaped)
        encoded_assets = encoded_assets.reshape(batch_size, -1)
        
        out = torch.cat([encoded_assets, other_features], dim=1)
        return self.fc(out)
        
    @property
    def get_feature_size(self):
        return self.feature_size


class SharedAssetEncoderWithActionNet(EncoderWithAction):
    def __init__(self, observation_shape, action_size, feature_size=256, num_assets=15, num_features=10):
        super().__init__()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.feature_size = feature_size
        self.num_assets = num_assets
        self.num_features = num_features
        
        self.asset_mlp = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        other_dims = num_assets + 1 + num_assets
        
        self.fc = nn.Sequential(
            nn.Linear(16 * num_assets + other_dims + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, feature_size),
            nn.ReLU()
        )

    def forward(self, x, action):
        batch_size = x.shape[0]
        asset_features = x[:, :self.num_assets * self.num_features]
        other_features = x[:, self.num_assets * self.num_features:]
        
        reshaped = asset_features.reshape(-1, self.num_features)
        encoded_assets = self.asset_mlp(reshaped)
        encoded_assets = encoded_assets.reshape(batch_size, -1)
        
        out = torch.cat([encoded_assets, other_features, action], dim=1)
        return self.fc(out)
        
    @property
    def get_feature_size(self):
        return self.feature_size


@dataclass
class SharedAssetEncoderFactory(EncoderFactory):
    TYPE: str = "shared_asset"
    feature_size: int = 256
    
    def create(self, observation_shape):
        return SharedAssetEncoderNet(observation_shape, self.feature_size)
        
    def create_with_action(self, observation_shape, action_size, discrete_action=False):
        return SharedAssetEncoderWithActionNet(observation_shape, action_size, self.feature_size)
        
    @classmethod
    def get_type(cls) -> str:
        return cls.TYPE
        
    @property
    def get_params(self):
        return {"feature_size": self.feature_size}

def register_custom_encoders():
    d3rlpy.models.encoders.register_encoder_factory(SharedAssetEncoderFactory)
