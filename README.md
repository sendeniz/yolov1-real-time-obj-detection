# Yolo: You Only Look Once Real Object Detection (V1)
 
**General:**
<br>
This repo contains a reimplementation of the original Yolo: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) paper by Joseph Redmon using PyTorch. 

**Getting started:**
<br>
In order to get started first `cd` into the `./yolov1-real-time-obj-detection` dictionary. To train from scratch, the PASCAL VOC 2007 and 2012 data-set is requiered. You can either manually download the data from the [PASCAL VOC homepage](http://host.robots.ox.ac.uk/pascal/VOC/) or simply call the following shell file: `get_data.sh`, which will automatically download and sort the data into the approriate folders and format for training. You may need to ensure that the shell file is executable by calling `chmod +x get_data.sh` and then executing it `./get_data.sh`. Note that the for the PASCAL VOC 2012 data-set, test data is only available on the PASCAL test server and therefore not publicly available for download. 

**Training:**
<br>
To train the model simply call the `train_yolov1.py` from terminal. Select one of the supported pre-trained models to be initalised as a backbone for training by setting one of the following backbone tags to `True` and all others to `False`: 1) `use_vgg19bn_backbone`, 2) `use_resnet18_backbone`, 3) and `use_resnet50_backbone` and 4) `use_original_darknet_backbone`. Note that as there are no pretrained weights available for the darknet weights in pytorch, the original backbone is currently not supported. If anyone has such weights or the GPU load available to train these from scratch on ImageNet, please feel free to contact me. The darknet training files for ImageNet data have been included in this repo for this purpose. 

Loss and mean average precision (mAP) values are computed after every epoch and can be seen from the console. 

**Real time object detection**
<br>
In order to run YoloV1 in real-time on a video or webcam in real-time, please if not trained from scratch download one of the pretrained weights from the Table 1. Make sure that at least one of the pretrained checkpoint `.cpt` files is within the `cpts` folder. If you want to do real time inference on a video, move the video file into the `./video` folder. Then specify both 1) which pre-trained model to use and 2) path to the vdeo in `yolov1_watches_youtube.py` by setting the appropriate tag to `True`. This will open up a window and perform object detection in real time. If you wish to perform object detection on a webcam call the `yolov1_watches_you.py`, which will open up a window of your camera stream and perform object detecton.

**Pretrained weights**
<br>

 Backbone      |    Train mAP   |    Test mAP   |      FPS      |     Link     |
| :---         |     :---:      |     :---:     |     :---:     |         ---: |
|    Vgg19bn   |     66.12%     |     44.01%    |      233      |   [Link]()   |
|    Resnet18  |     68.39%     |     44.29%    |      212      |   [Link]()   |
|    Resnet50  |     69.51%     |     49.94%    |       96      |   [Link]()   |
|    Darknet   |       -        |       -       |       -       |      -       |
<!---|    Darknet (YoloV1 Paper)     |       63.40%  |      57.90%       |       -       |--->

**Python files:**
<br>
`yolov1net_backbonename.py` : There are 3 pretrained backbones supported: Vgg19 with batch norm `yolov1net_vgg19bn.py`, Resnet18 `yolov1net_resnet18.py`
 and Resnet50 `yolov1net_resnet50.py`. While the original darknet backbone is also supported `yolov1net_darknet.py`, there are no pretrained PyTorch weights available for this backbone. Methods that convert the original darknet weights from [Joseph Redmon's website](https://pjreddie.com/darknet/imagenet/) do not support conversion of this particular backbone. If anyone has such weights or the GPU load available to train these from scratch on ImageNet, please feel free to contact me. The darknet training files for ImageNet data have been included in this repo for this purpose. 
 
`train_yolov1.py` : performs the entire training and testing procedure, giving progress updates after each epoch for both training and test loss in addition to the mean average precision metric. 

**Acknowledgement:**
<br>
I would like to thank Aske Plaat, Micheal Lew, Wojtek Kowalczyk, Jonas Eilers, Josef Jaeger, Paul Jaeger, Daniel Klassen, Zhao Yang and Paul Peters for their support, time and thoughts throughout my studies.
