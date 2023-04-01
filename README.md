# Yolo: You Only Look Once Real-Time Object Detection (V1)
 
**General:**
<br>
This repo contains a reimplementation of the original Yolo: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) paper by Joseph Redmon using PyTorch. A short demo of our detection system can be seen in Fig. 1. The full demonstration can be found [here](https://www.youtube.com/watch?v=Q30_ScFp8us). 
<p align="center">
  <img src="figures/yolov1_demo.gif" alt="animated" />
  <figcaption>Fig.1 - Real-time inference using YoloV1. </figcaption>
</p>


**Example Dictionary Structure**

<details>
<summary style="font-size:14px">View dictionary structure</summary>
<p>

```
.
├── application                		# Real time inference tools
    └── __init__.py 
    └── yolov1_watches_you.py  		# YoloV1 inference on webcam
    └── yolov1_watches_youtube.py	# YoloV1 inference on an .mp4 video file in `video/`
├── cpts				# Weights as checkpoint .cpt files
    └── vgg19bn_adj_lr_yolov1.cpt	# Pretrained YoloV1 utilizing Vgg19 backbone
    └── resnet18_adj_lr_yolov1.cpt  	# Pretrained YoloV1 utilizing Resnet18 backbone	
    └── resnet50_adj_lr_yolov1.cpt	# Pretrained YoloV1 utilizing Resnet50 backbone	
├── figures                    		# Figures and graphs
    └── ....
├── loss                       		# Custom PyTorch loss
    └── __init__.py  		
    └── yolov1_loss.py
├── models                     		# Pytorch models
    └── __init__.py  		
    └── darknet.py
    └── yolov1net_darknet.py		# Original YoloV1 backbone (not supported: no backbone weights available)
    └── tiny_yolov1net.py		# Original tiny YoloV1 backbone 
    └── tiny_yolov1net_mobilenetv3_large.py # Mobilenetv3 size M/large backbone
    └── tiny_yolov1net_mobilenetv3_small.py # Mobilenetv3 size S/small backbone
    └── tiny_yolov1net_squeezenet.py 	# Squeezenet backbone 
    └── yolov1net_resnet18.py		# Resnet18 pre-trained backbone
    └── yolov1net_resnet50.py		# Resnet50 pre-trained backbone
    └── yolov1net_vgg19bn.py		# Vgg19 with batchnormalization pre-trained backbone
├── results                    		# Result textfiles
    └── ....
├── train                      		# Training files
    └── __init__.py  
    └── train_darknet.py
    └── train_yolov1.py 
├── utils                      		# Tools and utilities
    └── __init__.py
    └── custom_transform.py		# Image transformation/augmentation
    └── darknet_utils.py		
    └── dataset.py
    └── figs.py.			# Create figures
    └── generate_csv.py			# Create training and testing csv files
    └── get_data.sh			# Fetch data and assign into appropriate folder structure
    └── get_data_macos.sh		
    └── get_inference_speed.py		# Get inference speed
    └── iou_map_tester.py		# mAP tester
    └── voc_label.py			
    └── yolov1_utils.py			
├── video                      		
    └── youtube_video.mp4		# .mp4 video from youtube
    └── yolov1_watches_youtube.mp4      # Result of `yolov1_watches_youtube.py`
├── requierments.txt           		# Python libraries
├── setup.py                   		
├── terminal.ipynb             		# If you want to run experiments on google collab
├── LICENSE
└── README.md
```

</p></details>

**Getting started:**
<br>
In order to get started first `cd` into the `./yolov1-real-time-obj-detection` dictionary and run the following lines:
```
virtualenv -p python3 venv
source venv/bin/activate
pip install -e .
```
Depending on what libraries you may already have, you may wish to `pip install -r requirements.txt`. To train from scratch, the PASCAL VOC 2007 and 2012 data-set is required. You can either manually download the data from the [PASCAL VOC homepage](http://host.robots.ox.ac.uk/pascal/VOC/) or simply call the following shell file: `utils/get_data.sh`, which will automatically download and sort the data into the approriate folders and format for training. If you are on mac use the `utils/get_data_macos.sh` file. You may need to ensure that the shell file is executable by calling `chmod +x get_data.sh` and then executing it `./get_data.sh`. Note that the for the PASCAL VOC 2012 data-set, test data is only available on the PASCAL test server and therefore not publicly available for download. 

**Training:**
<br>
To train the model simply call `python train/train_yolov1.py` from terminal. Select one of the supported pre-trained models to be initalised as a backbone for training by setting one of the following backbone tags to `True` and all others to `False`: 1) `use_vgg19bn_backbone`, 2) `use_resnet18_backbone`, 3) and `use_resnet50_backbone` and 4) `use_original_darknet_backbone`. Note that as there are no pretrained weights available for the darknet weights in pytorch, the original backbone is currently not supported. If anyone has such weights or the GPU load available to train these from scratch on ImageNet, please feel free to contact me. The darknet training files for ImageNet data have been included in this repo for this purpose and should only requiere some small adjustments.

**Results**
<br>
Loss and mean average precision (mAP) values are computed after every epoch and can be seen from the console. To obtain results regarding inference speed, call `python utils/inference_speed.py`. After training and obtaining the inference speed, plots can be created by calling `python utils/figs.py`, which are stored in the `figures/` folder. The results for training and test loss in addition to mAP values can be seen in Fig.2 for Vgg19 with batch normalisation, in Fig.3 for Resnet18 and Fig.4 for Resnet50. 
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
  <img width="700" height="300" src=/figures/tiny_yolo_loss_map.png?raw=true "Training Environment">
	<figcaption>Fig.4 - Tiny YolovV1: Training, test loss and mAP as a function of epochs.</figcaption>
</p>

<p align="center">
  <img width="900" height="300" src=/figures/model_comparison_map.png?raw=true "Training Environment">
	<figcaption>Fig.5 - Test mAP across different YoloV1 backbones.</figcaption>
</p>


<p align="center">
  <img width="700" height="300" src=/figures/gpu_model_comparison_inference_speed.png?raw=true "Training Environment">
	<figcaption>Fig.6 - Inference speed across different YoloV1 backbones.</figcaption>
</p>

<p align="center">
  <img width="700" height="300" src=/figures/cpu_model_comparison_inference_speed.png?raw=true "Training Environment">
	<figcaption>Fig.7 - Inference speed across different YoloV1 backbones.</figcaption>
</p>


**Real time object detection (GPU)**
<br>
In order to run YoloV1 in real-time on a video or webcam in real-time, please if not trained from scratch download one of the pretrained weights from the Table 1. Make sure that at least one of the pretrained checkpoint `.cpt` files is within the checkpoints `cpts` folder.  If you want to do real time inference on a video, move the video file (preferably .mp4) into the `./video` folder. Then specify both 1) which pre-trained model to use and 2) path to the video in `application/yolov1_watches_youtube.py` by setting the appropriate tag to `True`. This will open up a window and perform object detection in real time. If you wish to perform object detection on a webcam call the `application/yolov1_watches_you.py`, which will open up a window of your camera stream and perform object detecton. 

**Real time object detection (CPU)**
<br>
To run real-time object detection from your webcam feed on CPU only machines, in `application/yolov1_watches_you.py` change:
<br>
`checkpoint = torch.load(path_cpt_file)` to `checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))`.

**Pretrained weights**
<br>

 Backbone      			|    Train mAP   |    Test mAP   |      FPS      |	Params	|	Link	|
| :---         			|     :---:      |     :---:     |     :---:     |	:---:	|	---: 	|
|    Vgg19bn   			|     66.12%     |     44.01%    |      233      |	- 	|	[Link](https://drive.google.com/file/d/1-5vqoN2QxRqvFQ_KBZxD2q3hi5dBwcmq/view?usp=sharing)   |
|    Resnet18  			|     68.39%     |      44.29%   |      212      |	-	|	[Link](https://drive.google.com/file/d/1VsDFNMDYBWSy9qFGooMVNo5SFVyYToT0/view?usp=sharing)   |
|    Resnet50  			|     69.51%     |      49.94%   |       96      |   	-  	|	[Link](https://drive.google.com/file/d/1-31xnUeXpkb2AHLr9GDw0wlgn9MmUM13/view?usp=sharing)   |
|    Tiny Yolov1		|     -     |     -    |       -        |   	-  	|	[Link](-)   |
|    Tiny Yolov1 Mobilenetv3 S  |     37.81%     |     30.43%    |      -         |   	-  	|	[Link](https://drive.google.com/file/d/1-i-V_hXNPH75I-PpErn3bZRLdtDNVlFO/view?usp=sharing)   |
|    Tiny Yolov1 Mobilenetv3 L  |     50.72%     |     38.64%  	 |	- 	 |	-  	|   	[Link](https://drive.google.com/file/d/1-lYeKLf3pE2wmUb_TaNIRnrzdn8TubBZ/view?usp=sharing)   |
|    Tiny Squeezenet 1_1  	|     27.27%     |     	18.06%   |	-        | 	 - 	|[Link](https://drive.google.com/file/d/1-ZO32j6K7L41qpnwXTeRS0LvJY_bV9lL/view?usp=sharing)   |
|    Darknet   			|       -       |       -        |       -       |   -   |	-       |
<!---|    Darknet (YoloV1 Paper)     |       63.40%  |      57.90%       |       -       |--->
Download the entire `cpts`folder [here](https://drive.google.com/drive/folders/1GDj3jLBWbruhSQ7Gx01cLdkJFYW7kDwj?usp=sharing).

**Python files:**
<br>

`yolov1net_backbonename.py` : There are 6 pretrained backbones supported: for GPU inference there is Vgg19 with batch norm `yolov1net_vgg19bn.py`, Resnet18 `yolov1net_resnet18.py`, Resnet50 `yolov1net_resnet50.py` and for CPU inference there is `Mobilenetv3 Small (S)` and `Mobilenetv3 Large (L)`. For pocket models (i.e., detection that should run on a small device) we therefore encourage the use of our pretrained CPU models as they allow for real time inference on small devices.
 a. While the original darknet backbone is included `yolov1net_darknet.py`, there are no pretrained PyTorch weights available for this backbone. Methods that convert the original darknet weights from [Joseph Redmon's website](https://pjreddie.com/darknet/imagenet/) do not support conversion of this particular backbone. If anyone has such weights or the GPU load available to train these from scratch on ImageNet, please feel free to contact me. The darknet training files for ImageNet data have been included in this repo for this purpose and should only requiere small adjustments. 
 
`train_yolov1.py` : performs training and testing procedure, giving progress updates after each epoch for both training and test loss in addition to the mean average precision metric. 

`yolov1_loss.py` : defines the yolov1 loss as a custom PyTorch nn.module.

`custom_transform.py` : defines the data augmentations applied to yolov1 as per the original paper by Joseph Redmon.

`utils.py` : defines a series of utility functions, such as computation for the intersection over unions, mean average precision, converting bouding box coordinates relative to the cellboxes and from cellboxes the image.

`get_data.sh` : downloads the data and assigns them into the approriate folder structure for training and testing and converts the `train.txt` and `text.txt` to a csv using `generate_csv.py`

`get_data_macos.sh` : same as above but supported for unix macos systems.
 
`yolov1_watches_you.py` : performs object detection on webcam stream. 

`yolov1_watches_youtube.py` : performs object detection on a specified video path. 


**Acknowledgement:**
<br>
I would like to thank Michael Lew, Bryan A. Gass, Jonas Eilers, Jakob Walter and Daniel Klassen for their support, time and thoughts.
