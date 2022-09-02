# Yolo: You Only Look Once Real Object Detection
 
**General:**
<br>
This repo contains a reimplementation of the original Yolo: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) paper by Joseph Redmon using PyTorch. 

**Getting started:**
In order to get started, the PASCAL VOC 2007 and 2012 data-set is requiered. You can either manually download the data manually from the [PASCAL VOC homepage](http://host.robots.ox.ac.uk/pascal/VOC/) or simply call the following shell file: `get_data.sh`, which will automatically download and sort the data into the approriate folders and format. You may need to ensure that the shell file is executable by calling `chmod +x get_data.sh` and then executing it `./get_data.sh`. Note that the for the 2012 data-set test data is only available on the PASCAL test server and therefore not publicly available for download. 

**Training:**

**Python files:**
<br>
`yolov1net_backbonename.py` : There are 3 pretrained backbones supported: Vgg19 with batch norm `yolov1net_vgg19bn.py`, Resnet18 `yolov1net_resnet18.py`
 and Resnet50 `yolov1net_resnet50.py`. While the original darknet backbone is also supported yolov1net_darknet.py, there are no pretrained PyTorch weights available for this backbone. Methods that convert the original darknet weights from [Joseph Redmon's website](https://pjreddie.com/darknet/imagenet/) do not support conversion of this particular backbone. If anyone has such weights or the GPU load available to train these from scratch on ImageNet, please feel free to contact me. The darknet training files for ImageNet data have been included in this repo for that purpose. 
 
`train_yolov1.py` : performs the entire training and testing procedure, giving progress updates after each epoch for both training and test loss in addition to the mean average precision metric. 

**Acknowledgement:**
I would like to thank Aske Plaat, Micheal Lew, Wojtek Kowalczyk, Jonas Eilers, Josef Jaeger, Paul Jaeger, Daniel Klassen Zhao Yang and Paul Peters for their support, time and thoughts throughout my studies.
