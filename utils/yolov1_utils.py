#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:02:15 2022

@author: sen
"""

import torch 
from collections import Counter
device = "cuda" if torch.cuda.is_available() else "cpu"

def intersec_over_union(bboxes_preds, bboxes_targets, boxformat = "midpoints"):    
    """
    Calculates intersection of unions (IoU).
    Input: Boundbing box predictions (tensor) x1, x2, y1, y2 of shape (N , 4)
            with N denoting the number of bounding boxes.
            Bounding box target/ground truth (tensor) x1, x2, y1, y2 of shape (N, 4).
            box format whether midpoint location or corner location of bounding boxes
            are used.
    Output: Intersection over union (tensor).
    """
    
    if boxformat == "midpoints":
        box1_x1 = bboxes_preds[...,0:1] - bboxes_preds[...,2:3] / 2
        box1_y1 = bboxes_preds[...,1:2] - bboxes_preds[...,3:4] / 2
        box1_x2 = bboxes_preds[...,0:1] + bboxes_preds[...,2:3] / 2
        box1_y2 = bboxes_preds[...,1:2] + bboxes_preds[...,3:4] / 2
    
        box2_x1 = bboxes_targets[...,0:1] - bboxes_targets[...,2:3] / 2
        box2_y1 = bboxes_targets[...,1:2] - bboxes_targets[...,3:4] / 2
        box2_x2 = bboxes_targets[...,0:1] +  bboxes_targets[...,2:3] / 2
        box2_y2 = bboxes_targets[...,1:2] +  bboxes_targets[...,3:4] / 2
        
    if boxformat == "corners":
        box1_x1 = bboxes_preds[...,0:1]
        box1_y1 = bboxes_preds[...,1:2]
        box1_x2 = bboxes_preds[...,2:3]
        box1_y2 = bboxes_preds[...,3:4]
    
        box2_x1 = bboxes_targets[...,0:1]
        box2_y1 = bboxes_targets[...,1:2]
        box2_x2 = bboxes_targets[...,2:3]
        box2_y2 = bboxes_targets[...,3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    
    # clip intersection at zero to ensure it is never negative and equal to zero
    # if no intersection exists
    intersec = torch.clip((x2 - x1), min = 0) * torch.clip((y2 - y1), min = 0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersec + 1e-6
    iou = intersec / union
    return iou


def mean_avg_precision(bboxes_preds, bboxes_targets, iou_threshold = 0.5, 
                        boxformat ="midpoints", num_classes = 20):
    """
    Calculates mean average precision, by collecting predicted bounding boxes on the
    test set and then evaluate whether predictied boxes are TP or FP. Prediction with an 
    IOU larger than 0.5 are TP and predictions larger than 0.5 are FP. Since there can be
    more than a single bounding box for an object, TP and FP are ordered by their confidence
    score or class probability in descending order, where the precision is computed as
    precision = (TP / (TP + FP)) and recall is computed as recall = (TP /(TP + FN)).

    Input: Predicted bounding boxes (list): [training index, class prediction C,
                                              probability score p, x1, y1, x2, y2], ,[...]
            Target/True bounding boxes:
    Output: Mean average precision (float)
    """

    avg_precision = []
    
    # iterate over classes category
    for c in range(num_classes):
        # init candidate detections and ground truth as an empty list for storage
        candidate_detections = []
        ground_truths = []
        
        # iterate over candidate bouding box predictions 
        for detection in bboxes_preds:
            # index 1 is the class prediction and if equal to class c we are currently
            # looking at append
            # if the candidate detection in the bounding box predictions is equal 
            # to the class category c we are currently looking at add it to 
            # candidate list 
            if detection[1] == c:
                candidate_detections.append(detection)
                
        # iterate over true bouding boxes in the target bounding boxes
        for true_bbox in bboxes_targets:
            # if true box equal class category c we are currently looking at
            # append the ground truth list
            if true_bbox[1] == c:
                ground_truths.append(true_bbox)
        
        # first index 0 is the training index, given image zero with 3 bbox
        # and img 1 has 5 bounding boxes, Counter will count how many bboxes
        # and create a dictionary, so amoung_bbox = [0:3, 1:5]
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        
        for key, val in amount_bboxes.items():
            # fills dic with torch tensor zeors of len num_bboxes
            amount_bboxes[key] = torch.zeros(val)
            
        # sort over probability scores
        candidate_detections.sort(key=lambda x: x[2], reverse = True)
        
        # length for true positives and false positives for class based on detection
        # initalise tensors of zeros for true positives (TP) and false positives 
        # (FP) as the length of possible candidate detections for a given class C
        TP = torch.zeros((len(candidate_detections)))
        FP = torch.zeros((len(candidate_detections)))
        total_true_bboxes = len(ground_truths)
        
        if total_true_bboxes == 0:
            continue
        
        for detection_idx, detection in enumerate(candidate_detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            
            num_gts = len(ground_truth_img)
            best_iou = 0
            
            # iterate over all ground truth bbox in grout truth image
            for idx, gt in enumerate(ground_truth_img):
                iou = intersec_over_union(
                    # extract x1,x2,y1,y2 using index 3:
                    bboxes_preds = torch.unsqueeze(torch.tensor(detection[3:]),0),
                    bboxes_targets = torch.unsqueeze(torch.tensor(gt[3:]),0),
                    boxformat = boxformat)
            
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
                
                
            if best_iou > iou_threshold:
                # check if the bounding box has already been covered or examined before
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    # set it to 1 since we already covered the bounding box
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    # if bounding box already covered previously set as FP
                    FP[detection_idx] = 1
            # if the iou was not greater than the treshhold set as FP
            else:
                FP[detection_idx] = 1
    
        # compute cumulative sum of true positives (TP) and false positives (FP)
        # i.e. given [1, 1, 0, 1, 0] the cumulative sum is [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)
        recall = torch.div(TP_cumsum , (total_true_bboxes + 1e-6))
        precision = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))
        
        # compute average precision by integrating using numeric integration
        # with the trapozoid method starting at point x = 1, y = 0 
        # starting points are added to precision = x and recall = y using
        # torch cat
        precision = torch.cat((torch.tensor([1]), precision))
        recall = torch.cat((torch.tensor([0]), recall))
        integral = torch.trapz(precision, recall)
        avg_precision.append(integral)
    
    return sum(avg_precision) / len(avg_precision)

def get_bboxes(loader, model, iou_threshold, threshold, pred_format="cells", boxformat="midpoints",
    device="cuda" if torch.cuda.is_available() else "cpu"):
    
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(bboxes[idx], iou_threshold=iou_threshold, threshold=threshold,boxformat=boxformat)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    #model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. 
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat( (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0 )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    converted_preds = torch.cat( (predicted_class, best_confidence, converted_bboxes), dim=-1 )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def non_max_suppression(bboxes, iou_threshold, threshold, boxformat="corners"):
    """
    Does Non Max Suppression given bboxes.
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersec_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                boxformat=boxformat,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, boxformat="midpoints", num_classes=20
):
    """
    Calculates mean average precision 
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            # num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersec_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    boxformat=boxformat,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def strip_square_brackets(pathtotxt):
    with open(pathtotxt, 'r') as my_file:
        text = my_file.read()
        text = text.replace("[","")
        text = text.replace("]","")
    with open(pathtotxt, 'w') as my_file:
        my_file.write(text)
