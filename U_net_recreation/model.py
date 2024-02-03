import torch
from torch import nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import lightning as L

    
# Create U_net_original class that inherit from nn.Module
class U_net_original(nn.Module):
    """Straight forward U-net architecture with hyperparameters described in the paper by default.

    Takes a gray scale image and return 2 channels image: an outline mask layer and an object mask layer
    """

    def __init__(
        self,
        input_shape: int = 1,  # Number of feature channels of the input image
        first_layer_feature_channels: int = 64,  # Number of feature channels of the first layer
        second_layer_feature_channels: int = 128,  # Number of feature channels of the second layer
        third_layer_feature_channels: int = 256,  # Number of feature channels of the third layer
        fourth_layer_feature_channels: int = 512,  # Number of feature channels of the fourth layer
        fifth_layer_feature_channels: int = 1024,  # Number of feature channels of the fifth layer
        conv_kernel_size: int = 3,  # Kernel size of the convolutional step
        up_conv_kernel_size: int = 2,  # Kernel size for the transposed convolutional step
        maxpool_kernel_size: int = 2,  # Kernel size for the maxpool step
        final_conv_kernel_size: int = 1,  # Kernel size for the final classification step
        output_feature: int = 2,  # Number of output features
    ):
        super().__init__()
        self.maxpool_layer = nn.MaxPool2d(
            kernel_size=maxpool_kernel_size, stride=maxpool_kernel_size
        )
        self.contracting_layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=first_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=first_layer_feature_channels,
                out_channels=first_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
        )
        self.contracting_layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=first_layer_feature_channels,
                out_channels=second_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=second_layer_feature_channels,
                out_channels=second_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
        )
        self.contracting_layer_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=second_layer_feature_channels,
                out_channels=third_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=third_layer_feature_channels,
                out_channels=third_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
        )
        self.contracting_layer_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=third_layer_feature_channels,
                out_channels=fourth_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=fourth_layer_feature_channels,
                out_channels=fourth_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
        )
        self.expanding_layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=fourth_layer_feature_channels,
                out_channels=fifth_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=fifth_layer_feature_channels,
                out_channels=fifth_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=fifth_layer_feature_channels,
                out_channels=fourth_layer_feature_channels,
                kernel_size=up_conv_kernel_size,
                stride=up_conv_kernel_size,
            ),
        )
        self.expanding_layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=fifth_layer_feature_channels,
                out_channels=fourth_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=fourth_layer_feature_channels,
                out_channels=fourth_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=fourth_layer_feature_channels,
                out_channels=third_layer_feature_channels,
                kernel_size=up_conv_kernel_size,
                stride=up_conv_kernel_size,
            ),
        )
        self.expanding_layer_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=fourth_layer_feature_channels,
                out_channels=third_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=third_layer_feature_channels,
                out_channels=third_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=third_layer_feature_channels,
                out_channels=second_layer_feature_channels,
                kernel_size=up_conv_kernel_size,
                stride=up_conv_kernel_size,
            ),
        )
        self.expanding_layer_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=third_layer_feature_channels,
                out_channels=second_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=second_layer_feature_channels,
                out_channels=second_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=second_layer_feature_channels,
                out_channels=first_layer_feature_channels,
                kernel_size=up_conv_kernel_size,
                stride=up_conv_kernel_size,
            ),
        )
        self.expanding_layer_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=second_layer_feature_channels,
                out_channels=first_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=first_layer_feature_channels,
                out_channels=first_layer_feature_channels,
                kernel_size=conv_kernel_size,
            ),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(
            in_channels=first_layer_feature_channels,
            out_channels=output_feature,
            kernel_size=final_conv_kernel_size,
        )

    def forward(self, x: torch.Tensor):
        image_size_before_first_up_conv = (x.shape[2] / 16) - 7.75
        x = self.contracting_layer_1(x)
        cropped_x_layer_1 = transforms.CenterCrop(
            size=image_size_before_first_up_conv * 16 - 56
        )(x)
        x = self.maxpool_layer(x)
        x = self.contracting_layer_2(x)
        cropped_x_layer_2 = transforms.CenterCrop(
            size=image_size_before_first_up_conv * 8 - 24
        )(x)
        x = self.maxpool_layer(x)
        x = self.contracting_layer_3(x)
        cropped_x_layer_3 = transforms.CenterCrop(
            size=image_size_before_first_up_conv * 4 - 8
        )(x)
        x = self.maxpool_layer(x)
        x = self.contracting_layer_4(x)
        cropped_x_layer_4 = transforms.CenterCrop(
            size=image_size_before_first_up_conv * 2
        )(x)
        x = self.maxpool_layer(x)
        x = self.expanding_layer_1(x)
        x = self.expanding_layer_2(torch.cat((x, cropped_x_layer_4), 1))
        x = self.expanding_layer_3(torch.cat((x, cropped_x_layer_3), 1))
        x = self.expanding_layer_4(torch.cat((x, cropped_x_layer_2), 1))
        x = self.expanding_layer_5(torch.cat((x, cropped_x_layer_1), 1))
        x = self.classifier(x)
        return x

class U_net_original_simplified(nn.Module):
    """More compact recreation of the U-net architecture with hyperparameters described in the paper by default.

    Takes a gray scale image and return 2 channels image: an outline mask layer and an object mask layer
    """

    def __init__(
        self,
        contracting_layer_feature_channels=(1, 64, 128, 256, 512, 1024), # Number of feature channels of each of the contracting layer
        expanding_layer_feature_channels=(1024, 512, 256, 128, 64, 2), # Number of feature channels of each of the expanding layer
        conv_kernel_size: int = 3,  # Kernel size of the convolutional step
        up_conv_kernel_size: int = 2,  # Kernel size for the transposed convolutional step
        maxpool_kernel_size: int = 2,  # Kernel size for the maxpool step
        final_conv_kernel_size: int = 1 # Kernel size for the final classification step
    ):  
        super().__init__()
        self.contracting_layer_feature_channels = contracting_layer_feature_channels
        self.expanding_layer_feature_channels = expanding_layer_feature_channels
        self.double_conv_layers_down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=contracting_layer_feature_channels[i],
                        out_channels=contracting_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=contracting_layer_feature_channels[i + 1],
                        out_channels=contracting_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.contracting_layer_feature_channels) - 1)
            ]
        )

        self.double_conv_layers_up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=expanding_layer_feature_channels[i],
                        out_channels=expanding_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=expanding_layer_feature_channels[i + 1],
                        out_channels=expanding_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.expanding_layer_feature_channels) - 2)
            ]
        )
        self.contracting_layer = nn.MaxPool2d(
            kernel_size=maxpool_kernel_size, stride=maxpool_kernel_size
        )
        self.expanding_layer = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=self.expanding_layer_feature_channels[i],
                    out_channels=self.expanding_layer_feature_channels[i + 1],
                    kernel_size=up_conv_kernel_size,
                    stride=up_conv_kernel_size,
                )
                for i in range(len(self.expanding_layer_feature_channels) - 2)
            ]
        )
        self.classifier = nn.Conv2d(
            in_channels=self.expanding_layer_feature_channels[-2],
            out_channels=self.expanding_layer_feature_channels[-1],
            kernel_size=final_conv_kernel_size,
        )

    def forward(self, x: torch.Tensor):
        images_to_be_concat = []
        for i in range(len(self.contracting_layer_feature_channels) - 1):
            if i != (len(self.contracting_layer_feature_channels) - 2):
                x = self.double_conv_layers_down[i](x)
                images_to_be_concat.append(x)
                x = self.contracting_layer(x)
            else:
                x = self.double_conv_layers_down[i](x)
                x = self.expanding_layer[0](x)
        for i in range(1, len(self.contracting_layer_feature_channels) - 1):
            if i != (len(self.contracting_layer_feature_channels) - 2):
                image_to_be_concat = transforms.CenterCrop(size=x.shape[2])(
                    images_to_be_concat[-i]
                )
                x = self.double_conv_layers_up[i - 1](
                    torch.concat((x, image_to_be_concat), 1)
                )
                x = self.expanding_layer[i](x)
            else:
                image_to_be_concat = transforms.CenterCrop(size=x.shape[2])(
                    images_to_be_concat[-i]
                )
                x = self.double_conv_layers_up[i - 1](
                    torch.concat((x, image_to_be_concat), 1)
                )
                x = self.classifier(x)
        return x

#Torch Lightning was used instead of Pytorch for its speed advantage during experimentations
    
class U_net_reducedV1(L.LightningModule):
    """A down-sized verison of the original model with 4 layers.

    The number of feature channels also was reduced to a maximum of 64 channels.
    """

    def __init__(
        self,
        contracting_layer_feature_channels=(3, 16, 32, 64),  # Number of feature channels of each of the contracting layer
        expanding_layer_feature_channels=(64, 32, 16, 1),  # Number of feature channels of each of the expanding layer
        conv_kernel_size: int = 3,  # Kernel size of the convolutional step
        up_conv_kernel_size: int = 2,  # Kernel size for the transposed convolutional step
        maxpool_kernel_size: int = 2,  # Kernel size for the maxpool step
        final_conv_kernel_size: int = 1,  # Kernel size for the final classification step
    ):
        super().__init__()
        self.contracting_layer_feature_channels = contracting_layer_feature_channels
        self.expanding_layer_feature_channels = expanding_layer_feature_channels
        self.double_conv_layers_down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=contracting_layer_feature_channels[i],
                        out_channels=contracting_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=contracting_layer_feature_channels[i + 1],
                        out_channels=contracting_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.contracting_layer_feature_channels) - 1)
            ]
        )

        self.double_conv_layers_up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=expanding_layer_feature_channels[i],
                        out_channels=expanding_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=expanding_layer_feature_channels[i + 1],
                        out_channels=expanding_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.expanding_layer_feature_channels) - 2)
            ]
        )
        self.contracting_layer = nn.MaxPool2d(
            kernel_size=maxpool_kernel_size, stride=maxpool_kernel_size
        )
        self.expanding_layer = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=self.expanding_layer_feature_channels[i],
                    out_channels=self.expanding_layer_feature_channels[i + 1],
                    kernel_size=up_conv_kernel_size,
                    stride=up_conv_kernel_size,
                )
                for i in range(len(self.expanding_layer_feature_channels) - 2)
            ]
        )
        self.classifier = nn.Conv2d(
            in_channels=self.expanding_layer_feature_channels[-2],
            out_channels=self.expanding_layer_feature_channels[-1],
            kernel_size=final_conv_kernel_size,
        )

    def forward(self, x: torch.Tensor):
        images_to_be_concat = []
        for i in range(len(self.contracting_layer_feature_channels) - 1):
            if i != (len(self.contracting_layer_feature_channels) - 2):
                x = self.double_conv_layers_down[i](x)
                images_to_be_concat.append(x)
                x = self.contracting_layer(x)
            else:
                x = self.double_conv_layers_down[i](x)
                x = self.expanding_layer[0](x)
        for i in range(1, len(self.contracting_layer_feature_channels) - 1):
            if i != (len(self.contracting_layer_feature_channels) - 2):
                image_to_be_concat = transforms.CenterCrop(size=x.shape[2])(
                    images_to_be_concat[-i]
                )
                x = self.double_conv_layers_up[i - 1](
                    torch.concat((x, image_to_be_concat), 1)
                )
                x = self.expanding_layer[i](x)
            else:
                image_to_be_concat = transforms.CenterCrop(size=x.shape[2])(
                    images_to_be_concat[-i]
                )
                x = self.double_conv_layers_up[i - 1](
                    torch.concat((x, image_to_be_concat), 1)
                )
                x = self.classifier(x)
        return x

    def binary_cross_entropy_with_logits(self, logits, labels):
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        y = transforms.CenterCrop(size=logits.shape[2])(y)
        loss = self.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        y = transforms.CenterCrop(size=logits.shape[2])(y)
        loss = self.binary_cross_entropy_with_logits(logits, y)
        self.log("val_loss", loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.1, patience=20, threshold=0.01, verbose=True
                ),
                "monitor": "train_loss",
                "frequency": 1,
            },
        }

class U_net_reducedV2(L.LightningModule):
    """A down-sized verison of the original model with 4 layers.

    The number of feature channels also was reduced to a maximum of 64 channels. Padding was added to preseve the original image size
    """

    def __init__(
        self,
        contracting_layer_feature_channels=(3, 16, 32, 64), # Number of feature channels of each of the contracting layer
        expanding_layer_feature_channels=(64, 32, 16, 1), # Number of feature channels of each of the expanding layer
        conv_kernel_size: int = 3,  # Kernel size of the convolutional step
        up_conv_kernel_size: int = 2,  # Kernel size for the transposed convolutional step
        maxpool_kernel_size: int = 2,  # Kernel size for the maxpool step
        final_conv_kernel_size: int = 1, # Kernel size for the final classification step
        padding: int=1 # Padding to preserve image size
    ):  
        super().__init__()
        self.contracting_layer_feature_channels = contracting_layer_feature_channels
        self.expanding_layer_feature_channels = expanding_layer_feature_channels
        self.double_conv_layers_down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=contracting_layer_feature_channels[i],
                        out_channels=contracting_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                        padding=padding
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=contracting_layer_feature_channels[i + 1],
                        out_channels=contracting_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                        padding=padding
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.contracting_layer_feature_channels) - 1)
            ]
        )

        self.double_conv_layers_up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=expanding_layer_feature_channels[i],
                        out_channels=expanding_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                        padding=padding
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=expanding_layer_feature_channels[i + 1],
                        out_channels=expanding_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                        padding=padding
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.expanding_layer_feature_channels) - 2)
            ]
        )
        self.contracting_layer = nn.MaxPool2d(
            kernel_size=maxpool_kernel_size, stride=maxpool_kernel_size
        )
        self.expanding_layer = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=self.expanding_layer_feature_channels[i],
                    out_channels=self.expanding_layer_feature_channels[i + 1],
                    kernel_size=up_conv_kernel_size,
                    stride=up_conv_kernel_size,
                )
                for i in range(len(self.expanding_layer_feature_channels) - 2)
            ]
        )
        self.classifier = nn.Conv2d(
            in_channels=self.expanding_layer_feature_channels[-2],
            out_channels=self.expanding_layer_feature_channels[-1],
            kernel_size=final_conv_kernel_size,
        )

    def forward(self, x: torch.Tensor):
        images_to_be_concat = []
        for i in range(len(self.contracting_layer_feature_channels) - 1):
            if i != (len(self.contracting_layer_feature_channels) - 2):
                x = self.double_conv_layers_down[i](x)
                images_to_be_concat.append(x)
                x = self.contracting_layer(x)
            else:
                x = self.double_conv_layers_down[i](x)
                x = self.expanding_layer[0](x)
        for i in range(1, len(self.contracting_layer_feature_channels) - 1):
            if i != (len(self.contracting_layer_feature_channels) - 2):
                image_to_be_concat = images_to_be_concat[-i]
            
                x = self.double_conv_layers_up[i - 1](
                    torch.concat((x, image_to_be_concat), 1)
                )
                x = self.expanding_layer[i](x)
            else:
                image_to_be_concat = images_to_be_concat[-i]

                x = self.double_conv_layers_up[i - 1](
                    torch.concat((x, image_to_be_concat), 1)
                )
                x = self.classifier(x)
        return x
    def binary_cross_entropy_with_logits(self, logits, labels):
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.binary_cross_entropy_with_logits(logits, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.binary_cross_entropy_with_logits(logits, y)
        self.log('val_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.1, patience=20, threshold=0.01, verbose=True
                ),
                "monitor": "train_loss",
                "frequency": 1,
            },
        }

class U_net_reducedV3(L.LightningModule):
    """A down-sized verison of the original model with 5 layers.

    The number of feature channels also was reduced to a maximum of 64 channels. Padding was added to preseve the original image size
    """

    def __init__(
        self,
        contracting_layer_feature_channels=(3, 16, 32, 64, 128),
        expanding_layer_feature_channels=(128, 64, 32, 16, 1),
        conv_kernel_size: int = 3,  # Kernel size of the convolutional step
        up_conv_kernel_size: int = 2,  # Kernel size for the transposed convolutional step
        maxpool_kernel_size: int = 2,  # Kernel size for the maxpool step
        final_conv_kernel_size: int = 1, # Kernel size for the final classification step
        padding: int=1 # Padding to preserve image size
    ):  
        super().__init__()
        self.contracting_layer_feature_channels = contracting_layer_feature_channels
        self.expanding_layer_feature_channels = expanding_layer_feature_channels
        self.double_conv_layers_down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=contracting_layer_feature_channels[i],
                        out_channels=contracting_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        padding_mode="reflect"
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=contracting_layer_feature_channels[i + 1],
                        out_channels=contracting_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        padding_mode="reflect"
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.contracting_layer_feature_channels) - 1)
            ]
        )

        self.double_conv_layers_up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=expanding_layer_feature_channels[i],
                        out_channels=expanding_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        padding_mode="reflect"
                    ),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_channels=expanding_layer_feature_channels[i + 1],
                        out_channels=expanding_layer_feature_channels[i + 1],
                        kernel_size=conv_kernel_size,
                        padding=padding,
                        padding_mode="reflect"
                    ),
                    nn.ReLU(),
                )
                for i in range(len(self.expanding_layer_feature_channels) - 2)
            ]
        )
        self.contracting_layer = nn.MaxPool2d(
            kernel_size=maxpool_kernel_size, stride=maxpool_kernel_size
        )
        self.expanding_layer = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=self.expanding_layer_feature_channels[i],
                    out_channels=self.expanding_layer_feature_channels[i + 1],
                    kernel_size=up_conv_kernel_size,
                    stride=up_conv_kernel_size,
                )
                for i in range(len(self.expanding_layer_feature_channels) - 2)
            ]
        )
        self.classifier = nn.Conv2d(
            in_channels=self.expanding_layer_feature_channels[-2],
            out_channels=self.expanding_layer_feature_channels[-1],
            kernel_size=final_conv_kernel_size,
        )

    def forward(self, x: torch.Tensor):
        images_to_be_concat = []
        for i in range(len(self.contracting_layer_feature_channels) - 1):
            if i != (len(self.contracting_layer_feature_channels) - 2):
                x = self.double_conv_layers_down[i](x)
                images_to_be_concat.append(x)
                x = self.contracting_layer(x)
            else:
                x = self.double_conv_layers_down[i](x)
                x = self.expanding_layer[0](x)
        for i in range(1, len(self.contracting_layer_feature_channels) - 1):
            if i != (len(self.contracting_layer_feature_channels) - 2):
                image_to_be_concat = images_to_be_concat[-i]
            
                x = self.double_conv_layers_up[i - 1](
                    torch.concat((x, image_to_be_concat), 1)
                )
                x = self.expanding_layer[i](x)
            else:
                image_to_be_concat = images_to_be_concat[-i]

                x = self.double_conv_layers_up[i - 1](
                    torch.concat((x, image_to_be_concat), 1)
                )
                x = self.classifier(x)
        return x
    def binary_cross_entropy_with_logits(self, logits, labels):
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.binary_cross_entropy_with_logits(logits, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.binary_cross_entropy_with_logits(logits, y)
        self.log('val_loss', loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0001)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, factor=0.1, patience=20, threshold=0.01, verbose=True
                ),
                "monitor": "train_loss",
                "frequency": 1,
            },
        }
