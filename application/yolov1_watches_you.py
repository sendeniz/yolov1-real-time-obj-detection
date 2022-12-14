#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 23:14:10 2022

@author: sen
"""
import cv2 as cv
import numpy as np
import time 
import torch
import torch.optim as optim
from utils.yolov1_utils import non_max_suppression, cellboxes_to_boxes, get_bboxes
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from models.yolov1net_darknet import YoloV1Net
from models.yolov1net_vgg19bn import YoloV1_Vgg19bn
from models.yolov1net_resnet18 import YoloV1_Resnet18
from models.yolov1net_resnet50 import YoloV1_Resnet50
from models.tiny_yolov1net import Tiny_YoloV1
from models.tiny_yolov1net_mobilenetv3_large import Tiny_YoloV1_MobileNetV3_Large
from models.tiny_yolov1net_mobilenetv3_small import Tiny_YoloV1_MobileNetV3_Small
from models.tiny_yolov1net_squeezenet import Tiny_YoloV1_SqueezeNet
from utils.custom_transform import draw_bounding_box


transform = T.Compose([T.ToTensor()])
weight_decay = 0.0005
device = "cuda" if torch.cuda.is_available() else "cpu"

yolov1_darknet_pretrained = False
yolov1_vgg19bn_pretrained = False
yolov1_resnet18_pretrained = False
yolov1_resnet50_pretrained = False
tiny_yolov1_pretrained = False
tiny_yolov1_mobilenetv3_large_pretrained = False
tiny_yolov1_mobilenetv3_small_pretrained = True
tiny_yolov1_squeezenet_pretrained= False
run_on_cpu = True

model_names = ['vgg19bn_adj_lr_',
                'resnet18_adj_lr_',
                'resnet50_adj_lr_',
                'tiny_adj_lr_',
                'mobilenetv3_large_tiny_adj_lr_',
                'mobilenetv3_small_tiny_adj_lr_',
                'squeezenet_tiny_adj_lr_']

if yolov1_vgg19bn_pretrained == True:
    lr =  0.00001
    current_model = model_names[0]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif yolov1_resnet18_pretrained == True:
    lr =  0.00001
    current_model = model_names[1]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif yolov1_resnet50_pretrained == True:
    lr =  0.00001
    current_model = model_names[2]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif tiny_yolov1_pretrained == True:
    lr =  0.00001
    current_model = model_names[3]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif tiny_yolov1_mobilenetv3_large_pretrained == True:
    lr =  0.00001
    current_model = model_names[4]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif tiny_yolov1_mobilenetv3_small_pretrained == True:
    lr =  0.00001
    current_model = model_names[5]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif tiny_yolov1_squeezenet_pretrained == True:
    lr =  0.00001
    current_model = model_names[6]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'  


# init model
if yolov1_darknet_pretrained == True:
    model = YoloV1Net(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    checkpoint = torch.load(path_cpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Pretrained yolov1 darknet network initalized.")
    print("Note: This is the original backbone from the YoloV1 paper. PyTorch weights are not available.")
    print("This backbone requieres pre-training on ImageNet to obtain compareable performance to other backbones.")

elif yolov1_vgg19bn_pretrained == True:
    model = YoloV1_Vgg19bn(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    checkpoint = torch.load(path_cpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained yolov1 vgg 19 network with batch normalization initalized.")

elif yolov1_resnet18_pretrained == True:
    model = YoloV1_Resnet18(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    checkpoint = torch.load(path_cpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained yolov1 resnet 18 network initalized.")

elif yolov1_resnet50_pretrained == True:
    model = YoloV1_Resnet50(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    checkpoint = torch.load(path_cpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained yolov1 resnet 50 network initalized.")

elif tiny_yolov1_pretrained == True:
    model = Tiny_YoloV1(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    if run_on_cpu == True:
        checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path_cpt_file)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained tiny yolov1 network initalized.")

elif tiny_yolov1_mobilenetv3_large_pretrained == True:
    model = Tiny_YoloV1_MobileNetV3_Large(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    if run_on_cpu == True:
        checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path_cpt_file)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained tiny yolov1 large mobilenetv3 s network initalized.")

elif tiny_yolov1_mobilenetv3_small_pretrained == True:
    model = Tiny_YoloV1_MobileNetV3_Small(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    if run_on_cpu == True:
        checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path_cpt_file)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained tiny yolov1 small mobilenetv3 network initalized.")

elif tiny_yolov1net_squeezenet == True:
    model = Tiny_YoloV1_SqueezeNet(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    if run_on_cpu == True:
        checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path_cpt_file)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained tiny yolov1 squeezenet network initalized.")

else:
    print("No pretrained yolov1 model was specified. Please check the boolean flags and set the flag for supported pretrained models to True.")


# video captioning
cap = cv.VideoCapture(0)
fps = 0
fps_start = 0
prev = 0 

while(cap.isOpened()):
    
    ret, frame = cap.read()
    if not ret:
        break
    frame = np.array(frame)
    frame = cv.resize(frame, (448, 448))
    fps_end = time.time() 
    time_diff = fps_end - fps_start
    fps = int(1 / (time_diff - prev))
    prev = fps_end
    height, width = frame.shape[:2]
    fps_txt = "FPS: {}".format(fps)
    frame = cv.putText(frame, fps_txt, (width - 90, 20), cv.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

    frame = transform(frame)
    frame = frame.unsqueeze(0)
    preds = model(frame.to(device))
    
    get_bboxes = cellboxes_to_boxes(preds)
    frame = frame.squeeze(0)
    frame = frame.permute(1, 2, 0).numpy() * 255
    bboxes = non_max_suppression(get_bboxes[0], iou_threshold = 0.5, threshold = 0.4, boxformat = "midpoints")
    frame = draw_bounding_box(frame, bboxes, test = True)
    
    cv.imshow('Video', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print("Webcam stream cancelled.")
