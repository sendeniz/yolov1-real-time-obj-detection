import torch
import torch.nn as nn
import torchvision.models as models
test = False

class Tiny_YoloV1_MobileNetV3_Small(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(Tiny_YoloV1_MobileNetV3_Small, self).__init__()

        self.mobilenetv3_small_backbone =  nn.Sequential(*list(models.mobilenet_v3_small(weights='MobileNet_V3_Small_Weights.IMAGENET1K_V1').children())[:-2])
        #self.squeeze_net = nn.Sequential(*list(models.squeezenet1_1(weights='IMAGENET1K_V1').children())[:-1])
        self.yolov1head = nn.Sequential (
            # Block 5 (last two conv layers)
            # Since the last MobileNetV3 small layer consists of a (1x1, 567) conv layer
            # we adjust the input size of the yolo head from 1024 to 567 

            nn.Conv2d(in_channels = 576, out_channels = 192, 
                      kernel_size = (3, 3), stride = 1,
                      padding = 1),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.1),

            nn.Conv2d(in_channels = 192, out_channels = 64, 
                      kernel_size = (3, 3), stride = 1,
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

    def forward(self, x):
        x = self.mobilenetv3_small_backbone(x)
        x = self.yolov1head(x)
        return x
    
def test ():
    model = Tiny_YoloV1_MobileNetV3_Small()
    x = torch.rand(2, 3, 448, 448)
    xshape = model(x).shape
    return x, xshape

testx, xdims = test()
