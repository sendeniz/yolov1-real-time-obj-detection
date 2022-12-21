import torch
import torch.nn as nn
import torchvision.models as models
test = False

class Tiny_YoloV1_SqueezeNet(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(Tiny_YoloV1_SqueezeNet, self).__init__()

        self.squeezenet_backbone =  nn.Sequential(*list(models.squeezenet1_1(weights='SqueezeNet1_1_Weights.IMAGENET1K_V1').children())[:-1])
        self.yolov1head = nn.Sequential (
            # Block 5 (last two conv layers)
            # Since the last SqueezeNet layer consists of a (1x1, 512) conv layer
            # we adjust the input size of the yolo head from 1024 to 512.

            nn.Conv2d(in_channels = 512, out_channels = 128, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1), 

            nn.Conv2d(in_channels = 128, out_channels = 64, 
                      kernel_size = (3, 3), stride = 2,
                      padding = 1),
            nn.BatchNorm2d(64),
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
        x = self.squeezenet_backbone(x)
        x = self.yolov1head(x)
        return x
    
    def random_weight_init(self):
        for i in range(len(self.yolov1head)):
             if type(self.yolov1head[i]) == torch.nn.modules.conv.Conv2d:
                self.yolov1head[i].weight.data = self.yolov1head[i].weight.data.normal_(0, 0.02)
                self.yolov1head[i].bias.data = self.yolov1head[i].bias.data.zero_()
def test ():
    model = Tiny_YoloV1_SqueezeNet()
    x = torch.rand(2, 3, 448, 448)
    xshape = model(x).shape
    return x, xshape

if test == True:
    testx, xdims = test()

