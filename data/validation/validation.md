# Validation folder
Use the get_subset, make_directory, and split_images_into_files in utils.dataloader to split the train folder of TGS Salt Identification Challenge competition data into a train and validation folder. 

```
import torch
import torchvision
from torchvision import transforms
from pathlib import Path
import os
import random
import glob
import shutil
import cv2
from typing import Tuple, Dict, List
from PIL import Image

from utils.data_loader import get_subset, make_directory, split_images_into_files

# Set up path to data directory
data_directory_path = Path("U_net_recreation/data")

# Set up raw train directory path
global_train_directory = data_directory_path / "raw" / "train"

L.seed_everything(42)
subset_image_pair_paths_80 = get_subset(target_directory_path=global_train_directory,
                        amount=0.8)
target_dir = data_directory_path
split_images_into_files(subset_image_pair_paths=subset_image_pair_paths_80,
                        target_directory=target_dir)
```
