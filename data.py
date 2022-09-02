#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 13:23:33 2022

@author: sen
"""
import torch 
import os
import pandas as pd
from PIL import Image
from custom_transform import scale_translate, scale_translate_bounding_box


class VOCData(torch.utils.data.Dataset):
    # csv file: image.jpg and labels.txt
    def __init__(self, csv_file, img_dir, label_dir, S = 7, B = 2, C = 20,
                 transform = None, transform_scale_translate = True):
        
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self. transform_scale_translate = transform_scale_translate
             
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        # loads label txt annotation location
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        #print("Label Path Index, 1 :", label_path)
        boxes = []
        with open(label_path) as f:
            # for every line
            for label in f.readlines():
                # class is an int and x,y width and height is a float
                # converts the string class_label, x, y width, height into int and floats
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()]
                # append line of anotation to list of bounding box
                boxes.append([class_label, x, y, width, height])

        # loads image location
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        # open the image
        
        image = Image.open(img_path)

        # scale and translate image 

        if self.transform_scale_translate == True:
            image, transform_vals = scale_translate(image)
        # scale and translate bounding boxes
            boxes = scale_translate_bounding_box(boxes, transform_vals)
        
        # conver the boxes into a tensor
        boxes = torch.tensor(boxes)
        if self.transform:
            # image = self.transform(image)
            image, boxes = self.transform(image, boxes)

        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        # iterate over each box to convert it to fit the label_matrix
        # it is relative to the entire image, we want to see which cell
        # this bounding box belongs to and convert it relative to that cell
        for box in boxes:
            # class_label, x, y, width, height = box.tolist()
            class_label, x, y, height, width = box.tolist()
            #print("Class label:, X:, Y:, W:, H:", class_label, x, y, width, height)
            
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x)
            #("Index I and J:", i,j)
            x_cell, y_cell = self.S * x - j, self.S * y - i

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            width_cell, height_cell = (
                width * self.S,
                height * self.S,)

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            if label_matrix[i, j, 20] == 0:
                # Set that there exists an object
                label_matrix[i, j, 20] = 1

                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell])

                label_matrix[i, j, 21:25] = box_coordinates

                # Set one hot encoding for class_label
                label_matrix[i, j, class_label] = 1

        return image, label_matrix