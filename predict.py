import torch
from torch import nn
import torchvision
from torchvision import transforms
import cv2
import glob
from pathlib import Path

import torch.nn.functional as F
import lightning as L

from U_net_recreation.model import *

# Default image transform
data_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        # Resize image to match the input image size of the U_net paper
        transforms.Resize(size=(104, 104)),
        # Turn image to tensor
        transforms.ToTensor(),
    ]
)


def predict(
    model: L.LightningModule = U_net_reducedV3,  # Default to U_net_reducedV3,
    best_checkpoint_path: str = "lightning_logs/version_23/checkpoints/epoch=499-step=40000.ckpt",  # Default best checkpoint
    img_path: str = "data/test/images/00a052d822.png",  # Path to iamge
    data_transform=data_transform,  # Default data_transform
):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = data_transform(img).unsqueeze(dim=0)
    if best_checkpoint_path:
        model = model.load_from_checkpoint(best_checkpoint_path)
    else:
        model = model
    model.eval()
    with torch.inference_mode():
        pred = model(img)
    pred = pred.unsqueeze(dim=0)
    return pred
