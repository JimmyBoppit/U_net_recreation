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

# Create function to get a percentage of data
def get_subset(
    target_directory_path, data_splits=["train", "validation"], amount=0.05, seed=47
):
    random.seed(seed)
    image_dictionary = {}
    print(f"[INFO] Creating image split for: {data_splits[0]}...")
    image_paths = glob.glob(os.path.join(target_directory_path, "images", "*.png"))

    # Get a random samples pool of the amount specified for the train split
    number_to_sample = round(amount * len(image_paths))
    print(
        f"[INFO] Getting random subset of {number_to_sample} image pairs for {data_splits[0]}..."
    )
    sampled_images = random.sample(image_paths, k=number_to_sample)
    sampled_masks = []
    for image_path in sampled_images:
        sampled_masks.append(
            os.path.join(target_directory_path, "masks", os.path.basename(image_path))
        )
        image_paths.remove(image_path)
    image_dictionary["train"] = sampled_images + sampled_masks

    # Get the remaining sample for the validation split
    print(f"[INFO] Creating image split for: {data_splits[1]}...")
    number_to_sample = len(image_paths)
    print(
        f"[INFO] Getting random subset of {number_to_sample} image pairs for {data_splits[1]}..."
    )
    sampled_images = image_paths
    sampled_masks = []
    for image_path in sampled_images:
        sampled_masks.append(
            os.path.join(target_directory_path, "masks", os.path.basename(image_path))
        )
    image_dictionary["validation"] = sampled_images + sampled_masks

    return image_dictionary


# Function to make a directory
def make_directory(file_destination: int, file_name: int):
    # Create target directory path
    target_dir_name = f"{file_destination}/{file_name}"
    print(f"Creating directory: '{target_dir_name}'")

    # Setup the directories
    target_dir = Path(target_dir_name)

    # Make the directories
    target_dir.mkdir(parents=True, exist_ok=True)

    return target_dir


# Function to split a dictionary of image paths into sperate files of train and validation. Within train and validation the images are split into masks and images.
def split_images_into_files(subset_image_pair_paths: Dict, target_directory: Path):
    for image_split in subset_image_pair_paths.keys():
        for image_path in subset_image_pair_paths[str(image_split)]:
            image_path = Path(image_path)
            dest_dir = (
                target_dir / image_split / image_path.parent.stem / image_path.name
            )
            if not dest_dir.parent.is_dir():
                dest_dir.parent.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Copying {image_path} to {dest_dir}...")
            shutil.copy2(image_path, dest_dir)


# 1. Create a custom dataset class to load the segmentation data folder. Subclass of torch.utils.data.Dataset
class SegmentationDataSet(torch.utils.data.Dataset):
    # 2. Initialize with a target directory path (train or validation) and transform parameter
    def __init__(self, target_directory_path: str, transform=None):
        super().__init__()
        # 3. Create class attributes
        # Get image paths from images folder
        self.image_paths = glob.glob(
            os.path.join(target_directory_path, "images", "*.png")
        )
        # self.image_paths = list(Path(target_directory_path)/"images".glob("*.png"))
        # Set up transforms
        self.transform = transform
        # Get a list of masks paths that match the image paths
        self.mask_paths = []
        for image_path in self.image_paths:
            self.mask_paths.append(
                os.path.join(
                    target_directory_path, "masks", os.path.basename(image_path)
                )
            )

    # 4. Make function to load original image and masked image
    def load_image(self, index: int) -> Image.Image:
        "Open an original image and the corresponding masked image and return it."
        original_image_path = self.image_paths[index]
        image = cv2.imread(original_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[index], 0)
        return image, mask

    # 5. Overwrite the len function to return the number of image pairs
    def __len__(self) -> int:
        return len(self.image_paths)

    # 6. Overwrite the __getitem__() method
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        original_image, masked_image = self.load_image(index)
        # Transform if necessary
        if self.transform:
            return self.transform(original_image), self.transform(
                masked_image
            )  # return transformed original and masked image pair
        else:
            return original_image, masked_image  # return original and masked image pair
