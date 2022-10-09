import torch
import torch.nn as nn
import torchvision.models as models

class Tiny_YoloV1_Resnet18(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(Tiny_YoloV1_Resnet18, self).__init__()

        self.resnet18backbone =  nn.Sequential(*list(models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').children())[:-2])
        self.resnet18backbone[4] = self.resnet18backbone[4][:-1]
        self.resnet18backbone[5] = self.resnet18backbone[5][:-1]
        self.resnet18backbone[5] = nn.Sequential(*list(self.resnet18backbone[5][0].children())[:-2])
        self.resnet18backbone[6] = self.resnet18backbone[6][:-1]
        self.resnet18backbone[6] = nn.Sequential(*list(self.resnet18backbone[6][0].children())[:-2])
        self.resnet18backbone[7] = self.resnet18backbone[7][:-1]
        self.resnet18backbone[7] = nn.Sequential(*list(self.resnet18backbone[7][0].children())[:-2])
        
        self.yolov1head = nn.Sequential (
            # Block 5 (last two conv layers)
            # Since the last ResNet 18 layer consists of a (3x3, 512) conv layer
            # we adjust the input size of the yolo head from 1024 to 512.
            #nn.Conv2d(in_channels = 512, out_channels = 512, 
            #          kernel_size = (3, 3), stride = 1,
            #          padding = 1),
            #nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels = 512, out_channels = 1024, 
                      kernel_size = (3, 3), stride = 2,
                      padding = 1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
           
            # Block 6
            # We adjust the predicitiong head by removing double convolutional
            # layers to make the network shallower and faster
            #nn.Conv2d(in_channels = 1024, out_channels = 1024, 
            #          kernel_size = (3, 3), stride = 1,
            #          padding = 1),
            #nn.BatchNorm2d(1024),
            #nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels = 1024, out_channels = 256, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),

            # prediction block
            nn.Flatten(),
            # nn.Linear(in_features = 1024 * S * S, out_features = 4096),
            nn.Linear(in_features = 256 * S * S, out_features = 1470),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            # nn.Linear(in_features = 4096, out_features = S * S * (C + B * 5)),
            nn.Linear(in_features = 1470, out_features = S * S * (C + B * 5)),
            # reshape in loss to be (S, S, 30) with C + B * 5 = 30
            )

        self.random_weight_init()

    def forward(self, x):
        x = self.resnet18backbone(x)
        x = self.yolov1head(x)
        return x
    
    def random_weight_init(self):
        for i in range(len(self.yolov1head)):
             if type(self.yolov1head[i]) == torch.nn.modules.conv.Conv2d:
                self.yolov1head[i].weight.data = self.yolov1head[i].weight.data.normal_(0, 0.02)
                self.yolov1head[i].bias.data = self.yolov1head[i].bias.data.zero_()
def test ():
    model = Tiny_YoloV1_Resnet18()
    x = torch.rand(2, 3, 448, 448)
    xshape = model(x).shape
    return x, xshape

testx, xdims = test()