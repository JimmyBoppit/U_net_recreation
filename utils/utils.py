import torch
from torch import nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import random

# Plot 3 random images, mask predictions and actual mask from the validation dataset
def plot_prediction_and_actual(data_set:torch.utils.data.Dataset,
                               model:torch.nn.Module,
                               output_img_size:int):
    # Get 3 random image indexes
    random_image_index = random.sample(range(len(data_set)), k=3)

    # Set up plot
    plt.figure(figsize=(5, 4))

    # Loop through samples and display them
    for sample_image in random_image_index:
        fig, ax = plt.subplots(1, 3)
        org_img, masked_img = data_set[sample_image][0], data_set[sample_image][1]
        cropped_masked_img = transforms.CenterCrop(size=(output_img_size,output_img_size))(data_set[sample_image][1])

        # Pass the image through the model
        org_img = org_img.unsqueeze(dim=0)
        model.eval()
        with torch.inference_mode():
            prediction = model(org_img.to(device))
        prediction = prediction.to("cpu")
        prediction = torch.round(torch.sigmoid(prediction.squeeze(dim=0)))

        # Permute the image to display
        org_img = org_img.squeeze(dim=0).permute(1, 2, 0)
        permuted_prediction = prediction.permute(1, 2, 0)
        permuted_masked_img = cropped_masked_img.permute(1, 2, 0)
        
        # Plot image on the right
        ax[0].imshow(org_img)
        ax[0].axis("off")
        ax[0].set_title(f"Image")
        
        # Plot predicted mask image in the middle
        ax[1].imshow(permuted_prediction)
        ax[1].axis("off")
        ax[1].set_title(f"Prediction mask")
        
    
        # Plot actual mask image on the right
        ax[2].imshow(permuted_masked_img)
        ax[2].axis("off")
        ax[2].set_title(f"Actual mask")
