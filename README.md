# Yolo: You Only Look Once Real-Time Object Detection (V1)
 
**General:**
<br>
This repo contains a reimplementation of the original Yolo: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) paper by Joseph Redmon using PyTorch. A short demo of our detection system can be seen in Fig. 1. The full demonstration can be found [here](https://www.youtube.com/watch?v=Q30_ScFp8us). 
<p align="center">
  <img src="figures/yolov1_demo.gif" alt="animated" />
  <figcaption>Fig.1 - Real-time inference using YoloV1. </figcaption>
</p>

**Getting started:**
<br>
In order to get started first `cd` into the `./yolov1-real-time-obj-detection` dictionary and run the following lines:
```
virtualenv -p python3 venv
!source venv/bin/activate
!pip install -e .
```
Depending on what libraries you may already have, you may wish to `pip install -r requirements.txt` first. To train from scratch, the PASCAL VOC 2007 and 2012 data-set is required. You can either manually download the data from the [PASCAL VOC homepage](http://host.robots.ox.ac.uk/pascal/VOC/) or simply call the following shell file: `get_data.sh`, which will automatically download and sort the data into the approriate folders and format for training. You may need to ensure that the shell file is executable by calling `chmod +x get_data.sh` and then executing it `./get_data.sh`. Note that the for the PASCAL VOC 2012 data-set, test data is only available on the PASCAL test server and therefore not publicly available for download. 

**Training:**
<br>
To train the model simply call the `train_yolov1.py` from terminal. Select one of the supported pre-trained models to be initalised as a backbone for training by setting one of the following backbone tags to `True` and all others to `False`: 1) `use_vgg19bn_backbone`, 2) `use_resnet18_backbone`, 3) and `use_resnet50_backbone` and 4) `use_original_darknet_backbone`. Note that as there are no pretrained weights available for the darknet weights in pytorch, the original backbone is currently not supported. If anyone has such weights or the GPU load available to train these from scratch on ImageNet, please feel free to contact me. The darknet training files for ImageNet data have been included in this repo for this purpose and should only requiere some small adjustments.

**Results**
<br>
Loss and mean average precision (mAP) values are computed after every epoch and can be seen from the console. After training they can be plotted by running the calling `python fig.py`. The results for training and test loss in addition to mAP values can be seen in Fig.2 for Vgg19 with batch normalisation, in Fig.3 for Resnet18 and Fig.4 for Resnet50. 
adjustments.
<br>
A model comparison between test mAP and inference speeed can be seen in Fig.5 and Fig.6 respectively. See Table.1 for exact mAP, FPS values per model.

<p align="center">
  <img width="700" height="300" src=/figures/vgg19bn_loss_map.png?raw=true "Training Environment">
	<figcaption>Fig.2 - Vgg19bn Yolov1: Training, test loss and mAP as a function of epochs.</figcaption>
</p>


<p align="center">
  <img width="700" height="300" src=/figures/resnet18_loss_map.png?raw=true "Training Environment">
	<figcaption>Fig.3 - Resnet18 Yolov1: Training, test loss and mAP as a function of epochs.</figcaption>
</p>

<p align="center">
  <img width="700" height="300" src=/figures/resnet50_loss_map.png?raw=true "Training Environment">
	<figcaption>Fig.4 - Resnet50 Yolov1: Training, test loss and mAP as a function of epochs.</figcaption>
</p>


<p align="center">
  <img width="700" height="300" src=/figures/model_comparison_map.png?raw=true "Training Environment">
	<figcaption>Fig.5 - Test mAP across different YoloV1 backbones.</figcaption>
</p>


<p align="center">
  <img width="700" height="300" src=/figures/model_comparison_inference_speed.png?raw=true "Training Environment">
	<figcaption>Fig.6 - Inference speed across different YoloV1 backbones.</figcaption>
</p>


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
I would like to thank Aske Plaat, Michael Lew, Wojtek Kowalczyk, Jonas Eilers, Jakob Walter, Josef Jaeger, Paul Jaeger, Daniel Klassen, Zhao Yang and Paul Peters for their support, time and thoughts throughout my studies.
