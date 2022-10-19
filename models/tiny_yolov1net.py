import torch
import torch.nn as nn
import torchvision.models as models

class Tiny_YoloV1(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(Tiny_YoloV1, self).__init__()
        
        # Model architecture modelled after TinyYoloV1 from
        # https://github.com/pjreddie/darknet/blob/master/cfg/yolov1-tiny.cfg
        self.tiny_backbone =  nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels = 16, out_channels = 32, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels = 32, out_channels = 64, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels = 64, out_channels = 128, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels = 128, out_channels = 256, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            
            nn.Conv2d(in_channels = 256, out_channels = 512, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            )

        
        self.yolov1head = nn.Sequential (

            nn.Conv2d(in_channels = 512, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
           
            nn.Conv2d(in_channels = 1024, out_channels = 256, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # prediction block
            nn.Flatten(),
            nn.Linear(in_features = 256 * S * S, out_features = 1470),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),

            nn.Linear(in_features = 1470, out_features = S * S * (C + B * 5)),
            # reshape in loss to be (S, S, 30) with C + B * 5 = 30
            )

        self.random_weight_init()

    def forward(self, x):
        x = self.tiny_backbone(x)
        x = self.yolov1head(x)
        return x
    
    def random_weight_init(self):
        for i in range(len(self.yolov1head)):
             if type(self.yolov1head[i]) == torch.nn.modules.conv.Conv2d:
                self.yolov1head[i].weight.data = self.yolov1head[i].weight.data.normal_(0, 0.02)
                self.yolov1head[i].bias.data = self.yolov1head[i].bias.data.zero_()
def test ():
    model = Tiny_YoloV1()
    x = torch.rand(2, 3, 448, 448)
    xshape = model(x).shape
    return x, xshape

testx, xdims = test()