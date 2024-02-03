import torch
from torch import nn
import torchvision
from torchvision import transforms
from pathlib import Path
from PIL import Image
import cv2
import glob

from typing import Tuple
from torch.utils.data import DataLoader

import torch.nn.functional as F
import lightning as L

from utils.data_loader import SegmentationDataSet
from U_net_recreation.model import *

# Default image transform
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    # Resize image to match the input image size of the U_net paper
    transforms.Resize(size=(104, 104)),
    # Turn image to tensor
    transforms.ToTensor()
])

def train_model(
    model: L.LightningModule=U_net_reducedV1, # Default to U_net_reducedV1
    seed: int = 42, # Default seed
    epochs: int = 2, # Default training epoch
    train_folder_path: str = "./data/train", # Path to train folder
    validation_folder_path: str = "./data/validation", # Path to validation folder
    data_transform = data_transform
):

    # Create segmentation dataset for the train and validation data
    train_data = SegmentationDataSet(
        target_directory_path=Path(train_folder_path), transform=data_transform
    )
    validation_data = SegmentationDataSet(
        target_directory_path=Path(validation_folder_path), transform=data_transform
    )

    # Create dataloaders of the train and validation segmentation dataset
    train_dataloader = DataLoader(dataset=train_data, batch_size=40, num_workers=0)
    validation_dataloader = DataLoader(
        dataset=validation_data, batch_size=40, num_workers=0
    )
    L.seed_everything(seed)
    trainer = L.Trainer(max_epochs=epochs)
    train_model = model()
    trainer.fit(
        model=train_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
    )
