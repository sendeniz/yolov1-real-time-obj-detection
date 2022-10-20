#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:27:34 2022

@author: sen
"""
import torch
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.yolov1_utils import mean_average_precision as mAP
from utils.yolov1_utils import get_bboxes
from utils.dataset import VOCData
from loss.yolov1_loss import YoloV1Loss
from torch.optim import lr_scheduler
from models.yolov1net_darknet import YoloV1Net
from models.yolov1net_vgg19bn import YoloV1_Vgg19bn
from models.yolov1net_resnet18 import YoloV1_Resnet18
from models.yolov1net_resnet50 import YoloV1_Resnet50
from models.tiny_yolov1net import Tiny_YoloV1
from models.tiny_yolov1net_mobilenetv3_large import Tiny_YoloV1_MobileNetV3_Large
from models.tiny_yolov1net_mobilenetv3_small import Tiny_YoloV1_MobileNetV3_Small
from models.tiny_yolov1net_squeezenet import Tiny_YoloV1_SqueezeNet

import cv2 as cv
import numpy as np 
from numpy import genfromtxt

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
weight_decay = 0.0005
epochs = 140
nworkers = 2
pin_memory = True

# to resume training from a previous checkpoint set to True
continue_training = False

img_dir = 'data/images/'
label_dir = 'data/labels/'
lr_sched_original = False
lr_sched_adjusted = True
use_vgg19bn_backbone = False
use_original_darknet_backbone = False
use_resnet18_backbone = False
use_resnet50_backbone = False
use_tiny_backbone = False
use_mobilenetv3_large_backbone = False
use_mobilenetv3_small_backbone = True
use_squeezenet_backbone = False
check_image_transform = False
save_model = True

model_names = ['vgg19bn_orig_lr_', 
                'vgg19bn_adj_lr_',
                'resnet18_adj_lr_',
                'resnet50_adj_lr_',
                'tiny_adj_lr_',
                'mobilenetv3_large_tiny_adj_lr_',
                'mobilenetv3_small_tiny_adj_lr_',
                'squeezenet_tiny_adj_lr_']

if lr_sched_original == True and use_vgg19bn_backbone == True:
    lr = 0.001
    current_model = model_names[0]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif lr_sched_adjusted == True and use_vgg19bn_backbone == True:
    lr =  0.00001
    current_model = model_names[1]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif lr_sched_adjusted == True and use_resnet18_backbone == True:
    lr =  0.00001
    current_model = model_names[2]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif lr_sched_adjusted == True and use_resnet50_backbone == True:
    lr =  0.00001
    current_model = model_names[3]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif lr_sched_adjusted == True and use_tiny_backbone == True:
    lr =  0.00001
    current_model = model_names[4]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif lr_sched_adjusted == True and use_mobilenetv3_large_backbone == True:
    lr =  0.00001
    current_model = model_names[5]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif lr_sched_adjusted == True and use_mobilenetv3_small_backbone == True:
    lr =  0.00001
    current_model = model_names[6]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
elif lr_sched_adjusted == True and use_squeezenet_backbone == True:
    lr =  0.00001
    current_model = model_names[7]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
        return img, bboxes
    
train_transform = Compose([T.Resize((448, 448)),
            T.ColorJitter(brightness=[0,1.5], saturation=[0,1.5]),
            T.ToTensor()])

test_transform = Compose([T.Resize((448, 448)),
            T.ToTensor()])

def train (train_loader, model, optimizer, loss_f):
    """
    Input: train loader (torch loader), model (torch model), optimizer (torch optimizer)
          loss function (torch custom yolov1 loss).
    Output: loss (torch float).
    """
    # Gradient accumulation parameter: perform gradient accumulation over 16
    # batches
    accum_iter = 16
    model.train()

    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        x_cpu = x.to("cpu")
        
        with torch.set_grad_enabled(True):
            out = model(x)
            del x
            loss_val = loss_f(out, y)
            # Maybe we need to divide loss by accum_iter. I however doubt this
            # since loss is computed within each batch, so this correction is
            # not needed. 
            loss_val = loss_val 
            del y
            del out
            
            loss_val.backward()
            
            if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
            
    return (float(loss_val.item()))
    
def test (test_loader, model, loss_f):
    """
    Input: test loader (torch loader), model (torch model), loss function 
          (torch custom yolov1 loss).
    Output: test loss (torch float).
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            del x
            test_loss_val = loss_f(out, y)
            del y
            del out

        return(float(test_loss_val))

def main():

    if use_original_darknet_backbone == True:
        model = YoloV1Net(S = 7, B = 2, C = 20).to(device)
        print("Untrained darknet network initalized as a backbone.")
        print("Note: This is the original backbone from the YoloV1 paper. PyTorch weights are not available.")
        print("This backbone requieres pre-training on ImageNet to obtain compareable performance to other backbones.")
    
    elif use_vgg19bn_backbone == True:
        model = YoloV1_Vgg19bn(S = 7, B = 2, C = 20).to(device)
        print("Petrained vgg 19 network with batch normalization initalized as a backbone.")

    elif use_resnet18_backbone == True:
        model = YoloV1_Resnet18(S = 7, B = 2, C = 20).to(device)
        print("Petrained resnet 18 network initalized as a backbone.")

    elif use_resnet50_backbone == True:
        model = YoloV1_Resnet50(S = 7, B = 2, C = 20).to(device)
        print("Petrained resnet 50 network initalized as a backbone.")
    
    elif use_tiny_backbone == True:
        model = Tiny_YoloV1(S = 7, B = 2, C = 20).to(device)
        print("Untrained tiny yolov1 network initalized as a backbone.")

    elif use_mobilenetv3_large_backbone == True:
        model = Tiny_YoloV1_MobileNetV3_Large(S = 7, B = 2, C = 20).to(device)
        print("Pretrained mobilenet v3 large network initalized as a backbone.")
    
    elif use_mobilenetv3_small_backbone == True:
        model = Tiny_YoloV1_MobileNetV3_Small(S = 7, B = 2, C = 20).to(device)
        print("Pretrained mobilenet v3 small network initalized as a backbone.")
    
    elif use_squeezenet_backbone == True:
        model = Tiny_YoloV1_SqueezeNet(S = 7, B = 2, C = 20).to(device)
        print("Pretrained squeezenet network initalized as a backbone.")
    
    else:
        print("No backbone was specified. Please check the boolean flags in train_yolov1.py and set the flag for supported backbones to True.")
    
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    loss_f = YoloV1Loss()

    train_loss_lst = []
    train_mAP_lst = []
    test_mAP_lst = []
    test_loss_lst = []
    last_epoch = 0
    
    if continue_training == True:
        checkpoint = torch.load(path_cpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        print(f"Checkpoint from epoch:{last_epoch + 1} successfully loaded.")
        train_loss_lst = list(genfromtxt(f"results/{current_model}train_loss.txt", delimiter=','))
        train_mAP_lst = list(genfromtxt(f"results/{current_model}train_mAP.txt", delimiter=','))
        test_loss_lst = list(genfromtxt(f"results/{current_model}test_loss.txt", delimiter=','))
        test_mAP_lst = list(genfromtxt(f"results/{current_model}test_mAP.txt", delimiter=','))

    train_dataset = VOCData(csv_file = 'data/train.csv',
                            img_dir = img_dir, label_dir = label_dir, 
                            transform = train_transform, transform_scale_translate = True)
    
    test_dataset = VOCData(csv_file = 'data/test.csv',
                            img_dir = img_dir, label_dir = label_dir, 
                            transform = test_transform, transform_scale_translate = False)
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, 
                              num_workers = nworkers, shuffle = True)
    
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, 
                              num_workers = nworkers, shuffle = True)

    for epoch in range(epochs - last_epoch):
        torch.cuda.empty_cache()
        # learning rate scheduler
        # 1. something like the orginal learning rate from the YoloV1 paper
        # results are terrible
        if lr_sched_original == True:
            for g in optimizer.param_groups:
                # 1. linear increase from 0.001 (10^-3) to 0.01 (10^-2) over 5 epochs
                if epoch + last_epoch > 0 and epoch + last_epoch <= 5:
                    g['lr'] = 0.001 + 0.0018 * epoch
                # train at 0.01 for 75 epochs 
                if epoch + last_epoch <=80 and epoch + last_epoch > 5:
                    g['lr'] = 0.01
                # train at 0.001 for 30 epochs
                if epoch + last_epoch <= 110 and epoch + last_epoch > 80:
                    g['lr'] = 0.001
                # continue training until done
                if epoch + last_epoch > 110:
                    g['lr'] = 0.00001

        if lr_sched_adjusted == True:
             for g in optimizer.param_groups:
                # 1. linear increase from 0.00001 to 0.0001 over 5 epochs
                if epoch + last_epoch > 0 and epoch + last_epoch <= 5:
                    g['lr'] = 0.00001 +(0.00009/5) * (epoch + last_epoch)
                # train at  0.0001 for 75 epochs 
                if epoch + last_epoch <=80 and epoch + last_epoch > 5:
                    g['lr'] = 0.0001
                # train at 0.00001 for 30 epochs
                if epoch + last_epoch <= 110 and epoch + last_epoch > 80:
                    g['lr'] = 0.00001
                # train until done
                if epoch + last_epoch > 110:
                    g['lr'] = 0.000001

        # for training data
        pred_bbox, target_bbox = get_bboxes(train_loader, model, iou_threshold = 0.5, 
                                          threshold = 0.4)

        test_pred_bbox, test_target_bbox = get_bboxes(test_loader, model, iou_threshold = 0.5, 
                                          threshold = 0.4)                                
        
        # store mAP and average mAP
        train_mAP_val = mAP(pred_bbox, target_bbox, iou_threshold = 0.5, boxformat="midpoints")
        test_mAP_val = mAP(test_pred_bbox, test_target_bbox, iou_threshold = 0.5, boxformat="midpoints")
        train_mAP_lst.append(train_mAP_val.item())
        test_mAP_lst.append(test_mAP_val.item())
        train_loss_value = train(train_loader, model, optimizer, loss_f)
        train_loss_lst.append(train_loss_value)
        test_loss_value = test(test_loader, model, loss_f)
        test_loss_lst.append(test_loss_value)
        print(f"Learning Rate:", optimizer.param_groups[0]["lr"])
        print(f"Epoch:{epoch + last_epoch + 1 }  Train[Loss:{train_loss_value} mAP:{train_mAP_val}]  Test[Loss:{test_loss_value} mAP:{test_mAP_val}]")
  
        if save_model == True and ( (epoch + last_epoch + 1 ) % 2) == 0 or epoch + last_epoch == epochs - 1 :
            torch.save({
                'epoch': epoch + last_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, path_cpt_file)
            print(f"Checkpoint at {epoch + last_epoch + 1} stored")
            with open(f'results/{current_model}train_loss.txt','w') as values:
                values.write(str(train_loss_lst))
            with open(f'results/{current_model}train_mAP.txt','w') as values:
                values.write(str(train_mAP_lst))
            with open(f'results/{current_model}test_loss.txt','w') as values:
                values.write(str(test_loss_lst))
            with open(f'results/{current_model}test_mAP.txt','w') as values:
                values.write(str(test_mAP_lst))
        
            
if __name__ == "__main__":
    main()
