#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 13:28:05 2022

@author: sen
"""

import torch
from yolov1_utils import intersec_over_union as IOU
from yolov1_utils import mean_avg_precision as mAP


def iou_tester(true, preds, eps = 0.001):
    """
    input: true values and predicted values of x1,y1,x2,y2 coordinates (size: N by 4) 
    for bounding boxes.
    input: epsilon is a trehshhold value indicating the size of error we allow to
    occur. 
    output: booelians of whether predictions match true
    """
    res = IOU(preds, true)
    return (corr_iou - res < 0.001)
    
true_iou_coord = torch.tensor([
                      [0.8, 0.1, 0.2, 0.2], 
                      [0.95, 0.6, 0.5, 0.2],
                      [0.25, 0.15, 0.3, 0.1],
                      [0.7, 0.95, 0.6, 0.1],
                      [0.5, 0.5, 0.2, 0.2],
                      [2, 2, 6, 6],
                      [0, 0, 2, 2],
                      [0, 0, 2, 2],
                      [0, 0, 2, 2],
                      [0, 0, 2, 2],
                      [0, 0, 3, 2]
                      ])

preds_iou_coords = torch.tensor([
                      [0.9, 0.2, 0.2, 0.2], 
                      [0.95, 0.7, 0.3, 0.2],
                      [0.25, 0.35, 0.3, 0.1],
                      [0.5, 1.15, 0.4, 0.7],
                      [0.5, 0.5, 0.2, 0.2],
                      [4, 4, 7, 8],
                      [3, 0, 5, 2],
                      [0, 3, 2, 5],
                      [2, 0, 5, 2],
                      [1, 1, 3, 3],
                      [1, 1, 3, 3]
                      ])


corr_iou = torch.tensor([
                        [1 / 7],
                        [3 / 13],
                        [0],
                        [3 / 31],
                        [1],
                        [4 / 24],
                        [0],
                        [0],
                        [0],
                        [1 / 7],
                        [1/4]
                        ])

print("IOU Test return", iou_tester(preds_iou_coords, true_iou_coord))

map_preds1 =  [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
map_true1 = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
# map = 1
map_preds2 = [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
map_true2 = [
            [1, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]

# map = 1

map_preds3 = [
            [0, 1, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 1, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 1, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
map_true3 = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
# map = 0

map_preds4 = [
            [0, 0, 0.9, 0.15, 0.25, 0.1, 0.1],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]

map_true4 = [
            [0, 0, 0.9, 0.55, 0.2, 0.3, 0.2],
            [0, 0, 0.8, 0.35, 0.6, 0.3, 0.2],
            [0, 0, 0.7, 0.8, 0.7, 0.2, 0.2],
        ]
# map = 5 / 18

mAP(map_preds4, map_true4)


