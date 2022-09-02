# Yolo: You Only Look Once Real Object Detection
 
**General:**
<br>
This repo contains a reimplementation of the original Yolo: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) paper by Joseph Redmon using PyTorch. 

**Training:**

**Python files:**
<br>
```yolov1net_backbonename.py``` : There are 3 pretrained backbones supported: Vgg19 with batch norm ```yolov1net_vgg19bn.py```, Resnet18 ```yolov1net_resnet18.py```
 and Resnet50 ```yolov1net_resnet50.py```. While the original darknet backbone is also supported yolov1net_darknet.py, there are no pretrained PyTorch weights available for this backbone. Methods that convert the original darknet weights from [Joseph Redmon's website](https://pjreddie.com/darknet/imagenet/) do not support conversion of this particular backbone. If anyone has such weights or the GPU load available to train these from scratch on ImageNet, please feel free to contact me. The darknet training files for ImageNet data have been included in this repo for that purpose. 
 
```train_yolov1.py``` : performs the entire training and testing procedure, giving progress updates after each epoch for both training and test loss in addition to the mean average precision metric. 

