#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 22:24:58 2022

@author: sen
"""
import torch
import torch.nn as nn
from yolov1_utils import intersec_over_union as IUO

class YoloV1Loss(nn.Module):
    def __init__(self, S = 7, B = 2, C = 20):
        super(YoloV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_no_obj = 0.5
        self.lambda_obj = 5

    def forward(self, preds, target):
        mse_loss = nn.MSELoss(reduction="sum")
        # reshape predictions to S by S by 30
        preds = preds.reshape(-1, self.S, self.S, self.C + self.B * 5)
        # extract 4 bounding box values for bounding box 1 and box 2
        iou_bbox1 = IUO(preds[...,21:25], target[...,21:25])
        iou_bbox2 = IUO(preds[...,26:30], target[...,21:25])
        ious = torch.cat([iou_bbox1.unsqueeze(0), iou_bbox2.unsqueeze(0)], dim = 0)
        _ , bestbox = torch.max(ious, dim = 0)
        # Determine if an object is in cell i using identity
        identity_obj_i = target[...,20].unsqueeze(3) 

        # 1. Bouding Box Loss Component 
        boxpreds = identity_obj_i * (
            (
                bestbox * preds[...,26:30] 
                + (1 - bestbox) * preds[...,21:25]
            )
        )
        
        boxtargets = identity_obj_i * target[...,21:25]

        boxpreds[...,2:4] = torch.sign(boxpreds[...,2:4]) * torch.sqrt(
            torch.abs(boxpreds[...,2:4] + 1e-6)
        )    
        boxtargets[...,2:4] = torch.sqrt(boxtargets[...,2:4])
        
        # N, S, S, 4 -> N*N*S,4

        boxloss = mse_loss(torch.flatten(boxpreds, end_dim = -2),
                           torch.flatten(boxtargets, end_dim = -2)
        )
        
        # 2. Object Loss Component
        # has shape N by S by S

        predbox = (
            bestbox * preds[...,25:26] + (1 - bestbox) * preds[...,20:21]
            )
        
        
        objloss = mse_loss(
            torch.flatten(identity_obj_i * predbox),
            torch.flatten(identity_obj_i * target[...,20:21])
        )
        
        
        # 3. No Object Loss Component 
        no_objloss = mse_loss(
            torch.flatten((1 - identity_obj_i) * preds[...,20:21], start_dim = 1),
            torch.flatten((1 - identity_obj_i) * target[...,20:21], start_dim = 1)
            )
        
        no_objloss += mse_loss(
            torch.flatten((1 - identity_obj_i) * preds[...,25:26], start_dim = 1),
            torch.flatten((1 - identity_obj_i) * target[...,20:21], start_dim = 1)
            )
        
        # 4. Class Loss Component 
        classloss = mse_loss(
            torch.flatten(identity_obj_i * preds[...,:20], end_dim = -2),
            torch.flatten(identity_obj_i * target[...,:20], end_dim = -2)
            )
        
        # 5. Combine Loss Components 
        loss = (
            self.lambda_obj * boxloss
            + objloss
            + self.lambda_no_obj * no_objloss
            + classloss)

        return loss