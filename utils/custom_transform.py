# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 12:31:32 2022

@author: deniz
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def scale_translate(image, factor = 20): 
    """
    Input: img (path to image).
           scale_img (boolean) determining wheter to scale the image.
           translate (boolean) determining wheter to translate the image.
           factoor (int) how much to translate and/or scale the image with
           respect to the images original size. Default 20 is 20 %.
    Output: transformed image.
    """
    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    
    # Scaling variables
    x_up_bound = width
    x_low_bound = width - (width / 100 * factor)
    x_scale_to = np.random.randint(low = x_low_bound, high = x_up_bound)
    
    y_up_bound = height
    y_low_bound = height - (height / 100 * factor)
    y_scale_to = np.random.randint(low = y_low_bound, high = y_up_bound)
    
    x_ratio_percentage = x_scale_to / width * 100
    y_ratio_percentage = y_scale_to / height * 100
    
    # Translation variables
    x_upper_bound = float(width / 100 * factor) 
    x_lower_bound = float(width / 100 * factor) * -1
    y_upper_bound = float(height / 100 * factor) 
    y_lower_bound = float(height / 100 * factor) * -1
    
    # Uniform vals to translate into x coord t_x and y coord t_y
    t_x = np.random.uniform(low = x_lower_bound, high = x_upper_bound)
    t_y = np.random.uniform(low = y_lower_bound, high = y_upper_bound)
    
    # Translation matrix T
    T = np.float32([[1, 0, t_x], [0, 1, t_y]])
    
    # Scale image
    scaled_img = cv.resize(image, (x_scale_to, y_scale_to), interpolation = cv.INTER_CUBIC)
    height_scaled, width_scaled = scaled_img.shape[:2]
    blankimg = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    height_blank, width_blank = blankimg.shape[:2]
    yoff = round((height_blank - height_scaled) / 2)
    xoff = round((width_blank - width_scaled) / 2)
    result = blankimg.copy()
    result[yoff:yoff + height_scaled, xoff:xoff + width_scaled] = scaled_img
    scaled_img = result

    # Translate image after performing scaling
    img_scale_trans = cv.warpAffine(scaled_img, T, (width, height))
    # Convert from opencv format to pil
    # Opencv uses brg, pil uses rgb: convert brg to rgb
    img_scale_trans = cv.cvtColor(img_scale_trans, cv.COLOR_BGR2RGB)
    img_scale_trans = Image.fromarray(img_scale_trans)

    transform_vals = np.array([[int(height), int(width)],
                               [t_x, t_y], 
                                [xoff, yoff],
                                [x_ratio_percentage, 
                                 y_ratio_percentage]])
    
    return img_scale_trans, transform_vals

def draw_bounding_box(image, bounding_boxes, test = False):
    """
    Input: PIL image and bounding boxes (as list).
    Output: Image with drawn bounding boxes.
    """
    image = np.ascontiguousarray(image, dtype = np.uint8)
    colors = [[147,69,52], # aeroplane
                [29,178,255], # bicycle 
                [200,149,255], # bird
                [151,157, 255], # boat 
                [255,115,100], # bottle 
                [134,219,61], # bus
                [199,55,255], # car 
                [49,210,207], # cat
                [187,212, 0], # chair
                [52,147,26], # cow
                [236,24,0], # diningtable
                [168,153,44], # dog
                [56,56,255], # horse
                [10,249,72], # motorbike
                [255,194, 0], # person
                [255,56,132], # plant
                [133,0,82], # sheep
                [255,56,203], # sofa
                [31 ,112,255], # train
                [23,204,146]] # tvmonitor
    
    class_names = ["aeroplane","bicycle","bird","boat","bottle","bus","car",
        "cat","chair","cow","diningtable","dog","horse","motorbike","person",
        "pottedplant","sheep","sofa","train","tvmonitor"]

    # Extract transform_vals
    for i in range(len(bounding_boxes)):
        if test == True:
            height, width = image.shape[:2]

            class_pred = int(bounding_boxes[i][0])
            certainty = bounding_boxes[i][1]
            bounding_box = bounding_boxes[i][2:]

            # Note: width and heigh indexes are switches, somewhere, these are switched so
            # we correct for the switch by switching 
            bounding_box[2], bounding_box[3] = bounding_box[3], bounding_box[2]
            assert len(bounding_box) == 4, "Bounding box prediction exceed x, y ,w, h."
            # Extract x, midpoint, y midpoint, w width and h height
            x = bounding_box[0] 
            y = bounding_box[1] 
            w = bounding_box[2] 
            h = bounding_box[3]  
        
        else:
            height, width = image.shape[:2]
            class_pred = int(bounding_boxes[i][0])
            bounding_box = bounding_boxes[i][1:]
            
            assert len(bounding_box) == 4, "Bounding box prediction exceed x, y ,w, h."
            # Extract x midpoint, y midpoint, w width and h height
            x = bounding_box[0] 
            y = bounding_box[1] 
            w = bounding_box[2]
            h = bounding_box[3] 

        l = int((x - w / 2) * width)
        r = int((x + w / 2) * width)
        t = int((y - h / 2) * height)
        b = int((y + h / 2) * height)
        
        if l < 0:
            l = 0
        if r > width - 1:
            r = width - 1
        if t < 0:
            t = 0
        if b > height - 1:
            b = height - 1

        image = cv.rectangle(image, (l, t), (int(r), int(b)), colors[class_pred], 3)
        (txt_width, txt_height), _ = cv.getTextSize(class_names[class_pred], cv.FONT_HERSHEY_TRIPLEX, 0.6, 2)

        if t < 20:
            image = cv.rectangle(image, (l-2, t + 15), (l + txt_width, t), colors[class_pred], -1)
            image = cv.putText(image, class_names[class_pred], (l, t+12),
                    cv.FONT_HERSHEY_TRIPLEX, 0.5, [255, 255, 255], 1)
        else:
            image = cv.rectangle(image, (l-2, t - 15), (l + txt_width, t), colors[class_pred], -1)
            image = cv.putText(image, class_names[class_pred], (l, t-3),
                    cv.FONT_HERSHEY_TRIPLEX, 0.5, [255, 255, 255], 1)
   
    return image


def scale_translate_bounding_box(bounding_boxes, trans_vals):
    """
    Input: A list where each element is a list of the original (non-transformed) 
           bounding box information of length 5 ( x, y ,w, h.). 
    Output: A list where each element is a list of scaled and translated
            bounding box information of length 5 ( x, y ,w, h.).
    """
    t_x, t_y  = trans_vals[1]
    xoff, yoff = trans_vals[2]
    x_ratio_percentage, y_ratio_percentage = trans_vals[3]
    transformed_bounding_boxes = bounding_boxes.copy()
    for i in range(len(bounding_boxes)):
        height, width = trans_vals[0]
        # Extract bounding box information (x, y ,w, h.)
        bounding_box = bounding_boxes[i][1:]

        assert len(bounding_box) == 4, "Bounding box prediction exceed x, y ,w, h."
        # Extract x midpoint, y midpoint, w width and h height and transform
        # corresponding to the image transformation
        x = np.clip( ((bounding_box[0] / 100) * x_ratio_percentage) + (xoff / width) + (t_x / width), 0, 0.999)
        y = np.clip( ((bounding_box[1] / 100) * y_ratio_percentage) + (yoff / height) + (t_y / height), 0, 0.999)
        w =  np.clip( ((bounding_box[2] / 100) * x_ratio_percentage), 0, 0.999)
        h =  np.clip( ((bounding_box[3] / 100) * y_ratio_percentage), 0, 0.999)
        
        transformed_bounding_boxes[i][1:] = [x, y ,w ,h]
        
    return transformed_bounding_boxes