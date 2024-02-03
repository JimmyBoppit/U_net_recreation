# Train folder
Use the get_subset, make_directory, and split_images_into_files in utils.dataloader to split the train folder of TGS Salt Identification Challenge competition data into a train and validation folder. 

```
# Set up path to data directory
data_directory_path = Path("U_net_recreation/data")

# Set up train and validation directory path
global_train_directory = data_directory_path / "raw" / "train"

L.seed_everything(42)
subset_image_pair_paths_80 = get_subset(target_directory_path=global_train_directory,
                        amount=0.8)
target_dir = data_directory_path
split_images_into_files(subset_image_pair_paths=subset_image_pair_paths_80,
                        target_directory=target_dir)
```

