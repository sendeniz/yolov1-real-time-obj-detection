#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:48:38 2022

@author: sen
"""
import torch
import torch.nn as nn
test_model = False

def conv_params(input_h, n_padding, kernel_size, n_stride):
    return (input_h - kernel_size + 2*n_padding) / (n_stride) + 1

def num_paddings(kernel_size):
    return (kernel_size -1) //2

class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.darknet = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels = 3, out_channels = 64, 
                      kernel_size = (7, 7) , stride = 2, 
                      padding = 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            
            # Block 2
            nn.Conv2d(in_channels = 64, out_channels = 192, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            
            # Block 3
            nn.Conv2d(in_channels = 192, out_channels = 128, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 128, out_channels = 256, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 256, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            
            # Block 4
            nn.Conv2d(in_channels = 512, out_channels = 256, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512,
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 256,
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512,
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 256,
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 256,
                      kernel_size = (1, 1), stride = 1, 
                      padding = 0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 256, out_channels = 512,
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 512,
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 1024,
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = (2, 2), stride = 2),
            
            # Block 5
            nn.Conv2d(in_channels = 1024, out_channels = 512, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 1024, out_channels = 512, 
                      kernel_size = (1, 1), stride = 1,
                      padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

		)

        self.fc_block1 = nn.Sequential (
            nn.AvgPool2d(7),
            )
            
        self.fc_block2 = nn.Sequential (
            nn.Linear(1024, 1000),
            )
        
    def forward(self, x):
        x = self.darknet(x)
        x = self.fc_block1(x)
        x = torch.squeeze(x)
        x = self.fc_block2(x)
        return x

if __name__ == "__main__":
    if test_model == True:
        batch_size, nclasses, channels, h, w = 32, 1000, 3, 244, 244
        darknet_model = DarkNet()
        input = torch.randn(batch_size, channels, h, w)
        output = darknet_model(input)
        assert output.size() == torch.Size([batch_size, nclasses])
        print("Passed size check")
 