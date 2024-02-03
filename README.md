# U_net_recreation

## Description

A recreation of the U_net architecture using Pytorch. This was done both to gain a better understanding of the model and to refine experimental procedures. I had just completed the Pytorch-deep-learning course by Daniel Bourke so this was a good next step for me. Furthermore, I think this project is a good precursor to learning diffusion model. 

First, I recreated a straight foward U_net architecture as described in: 

U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597.

I used Pytorch for this first model and initialized every layer individually. The math for the image size to be cropped was also calculated by hand. This created (in my opinion) a hard to read and inefficient for scaling model but it was a good start. For the simplifed verison as I call it, I cleaned up the code by intergrating some recursive elements and automated the calculation for the image cropping. This shorten the number of lines of code from ~200 to ~100. This simplified version is what I used to scale down for training on the TGS Salt Identification Challenge. 

![u-net-architecture](https://github.com/JimmyBoppit/U_net_recreation/assets/151961878/34fee1cb-6bbe-4a37-8311-537cb87862e1)

## Usage
The architecture can be used just like any U_net model. The train and validation funtions are built into the scaled down models as they are subclass of the Lightning Module. The simplified version of the orginial is a subclass of Pytorch and can be very easily scaled up or down by adjusting the hyperparameters: "contracting_layer_feature_channels" and "expanding_feature_channels". This model can then be trained using any desired loss function, optimizer, and learning_rate scheduler. 

Example usage:
5_layers_30_mil_params_model = U_net_simplified()

4_layers_100_thousand_params_model = U_net_simplified(contracting_layer_feature_channels=(3, 16, 32, 64), expanding_layer_feature_channels=(64, 32, 16, 1)) 

## Credits
The structure of this project and some code was reused or modified from the "Zero to Mastery Learn PyTorch for Deep Learning" https://www.learnpytorch.io course by Daniel Bourke. 

I also referenced the SegmentationDataSet from "U-Net: Training Image Segmentation Models in PyTorch"
https://pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/
for my SegmentationDataSet.
## Training results
4_layers_100_thousand_params_model

![Small_0](https://github.com/JimmyBoppit/U_net_recreation/assets/151961878/97c0a840-667d-4d4b-9878-883aa3c247da)
![Small_1](https://github.com/JimmyBoppit/U_net_recreation/assets/151961878/cef653bb-b4b3-43f7-a412-9d480af0d484)
![Small_2](https://github.com/JimmyBoppit/U_net_recreation/assets/151961878/930d060e-77ba-41db-b1ae-fbcad3bb73d6)

5_layers_500_thousand_params_model_with_reflective_padding

![Large_0](https://github.com/JimmyBoppit/U_net_recreation/assets/151961878/a90f1ab9-2a9d-4cd3-86e1-5201cd256cd0)
![Large_1](https://github.com/JimmyBoppit/U_net_recreation/assets/151961878/da392e77-211d-4d71-8298-964b73a009e3)
![Large_2](https://github.com/JimmyBoppit/U_net_recreation/assets/151961878/9341a2fb-2835-44f5-9fe8-52eeb080e685)

## Lisence
MIT License

Copyright (c) 2024 JimmyBoppit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
