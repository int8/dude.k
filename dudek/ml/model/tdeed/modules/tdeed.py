import dataclasses
import random

import timm
import torch
from torch import nn
import torchvision.transforms as T


from collections import OrderedDict

from dudek.ml.model.tdeed.modules.layers import EDSGPMIXERLayers, FCLayers
from dudek.ml.model.tdeed.modules.shift import make_temporal_shift


@dataclasses.dataclass
class TDeedLoss:
    total_loss: float
    ce_labels_loss: float
    mse_displacement_loss: float
    bce_loss_teams: float


class TDeedModule(nn.Module):

    def __init__(
        self,
        clip_len: int,
        n_layers: int,
        sgp_ks: int,
        sgp_k: int,
        num_classes: int,
        features_model_name: str = "regnety_002",
        temporal_shift_mode: str = "gsf",
        gaussian_blur_ks: int = 3,
    ):
        super().__init__()

        self.features_model_name = features_model_name
        self.temporal_shift_mode = temporal_shift_mode
        self.sgp_k = sgp_k
        self.sgp_ks = sgp_ks
        self.n_layers = n_layers

        features = timm.create_model(
            features_model_name,
            pretrained=True,
        )

        feat_dim = features.get_classifier().in_features
        features.reset_classifier(0)

        self._d = feat_dim

        self._require_clip_len = clip_len
        make_temporal_shift(features, clip_len, mode=temporal_shift_mode)

        self._features = features
        self._feat_dim = self._d
        feat_dim = self._d

        # Positional encoding
        self.temp_enc = nn.Parameter(
            torch.normal(mean=0, std=1 / clip_len, size=(clip_len, self._d))
        )
        self._temp_fine = EDSGPMIXERLayers(
            feat_dim,
            clip_len,
            num_layers=n_layers,
            ks=sgp_ks,
            k=sgp_k,
            concat=True,
        )
        self._pred_fine = FCLayers(self._feat_dim, num_classes + 1)
        self._pred_displ = FCLayers(self._feat_dim, 1)

        # Augmentations
        self.augmentation = T.Compose(
            [
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(gaussian_blur_ks)], p=0.25),
            ]
        )

        # Standarization
        self.standarization = T.Compose(
            [
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                )  # Imagenet mean and std
            ]
        )

    def forward(self, x, y=None, inference=False):

        x = self.normalize(x)  # Normalize to 0-1
        batch_size, clip_len, channels, height, width = x.shape

        if not inference:
            x.view(-1, channels, height, width)
            x = x.view(batch_size, clip_len, channels, height, width)
            x = self.augment(x)  # augmentation per-batch
            x = self.standarize(x)  # standarization imagenet stats

        else:
            x = x.view(-1, channels, height, width)
            x = x.view(batch_size, clip_len, channels, height, width)
            x = self.standarize(x)

        # Extract features
        im_feat = self._features(x.view(-1, channels, height, width)).reshape(
            batch_size, clip_len, self._d
        )

        # Temporal encoding
        im_feat = im_feat + self.temp_enc.expand(batch_size, -1, -1)

        # Temporal module (SGP-Mixer)

        output_data = {}
        im_feat = self._temp_fine(im_feat)

        displ_feat = self._pred_displ(im_feat).squeeze(-1)
        output_data["displ_feat"] = displ_feat

        im_feat = self._pred_fine(im_feat)

        output_data["im_feat"] = im_feat

        return output_data, y

    def normalize(self, x):
        return x / 255.0

    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def load_backbone(self, model_weight_path: str):
        m = torch.load(model_weight_path, weights_only=True)
        _features_layers = OrderedDict(
            {
                k[len("_features.") :]: v
                for k, v in m.items()
                if k.startswith("_features.")
            }
        )
        self._features.load_state_dict(_features_layers)
        _temp_fine_layers = OrderedDict(
            {
                k[len("_temp_fine.") :]: v
                for k, v in m.items()
                if k.startswith("_temp_fine.")
            }
        )
        self._temp_fine.load_state_dict(_temp_fine_layers)

    def load_all(self, model_weight_path: str):
        m = torch.load(model_weight_path, weights_only=True)
        self.load_state_dict(m)
