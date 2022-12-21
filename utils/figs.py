import cv2 as cv
import numpy as np
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

import matplotlib.pyplot as plt
from PIL import Image
from utils.custom_transform import scale_translate, draw_bounding_box, scale_translate_bounding_box
import os
import pandas as pd
from numpy import genfromtxt

img_dir = 'data/images/'
label_dir = 'data/labels/'
results_dir = 'results/'
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
tiny_yolov1_squeezenet_pretrained = False

run_on_cpu = True

model_names = [ 'vgg19bn_adj_lr_',
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
    lr =  0.00001
    current_model = model_names[4]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
    model = Tiny_YoloV1_MobileNetV3_Large(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    if run_on_cpu == True:
        checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path_cpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained tiny yolov1 mobilenetv3 large network initalized.")

elif tiny_yolov1_mobilenetv3_small_pretrained == True:
    lr =  0.00001
    current_model = model_names[5]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
    model = Tiny_YoloV1_MobileNetV3_Small(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    if run_on_cpu == True:
        checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(path_cpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained tiny yolov1 mobilenetv3 small network initalized.")

elif tiny_yolov1_squeezenet_pretrained == True:
    lr =  0.00001
    current_model = model_names[6]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
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

# Initalize model
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
    checkpoint = torch.load(path_cpt_file, map_location=torch.device('cpu'))
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
    print("Petrained tiny yolov1 mobilenetv3 small network initalized.")

else:
    print("No pretrained yolov1 model was specified. Please check the boolean flags and set the flag for supported pretrained models to True.")

# Plot train data with ground truth bounding box
csvfile = pd.read_csv('data/train.csv', header=None, nrows = 250)
random_row = (csvfile.sample())
img_path = random_row.iloc[0, 0]
label_path  = random_row.iloc[0, 1]
img = Image.open(img_dir+img_path)
transformed_img, transform_vals = scale_translate(img)
transformed_img = np.array(transformed_img)
transformed_img = cv.resize(transformed_img, (448, 448))
cv.imwrite('figures/train_img.jpg', transformed_img)


boxes = []
with open(label_dir+label_path) as f:
    # for every line
    for label in f.readlines():
        # class is an int and x,y width and height is a float
        # converts the string class_label, x, y width, height into int and floats
        class_label, x, y, width, height = [
            float(x) if float(x) != int(float(x)) else int(x)
            for x in label.replace("\n", "").split()]
            # append line of anotation to list of bounding box
        boxes.append([class_label, x, y, width, height])

boxes = scale_translate_bounding_box(boxes, transform_vals)
train_img_bboxes = draw_bounding_box(transformed_img, boxes)
cv.imwrite('figures/train_img_bboxes.jpg', train_img_bboxes)


# Plot test data with predicted bounding box
csvfile = pd.read_csv('data/test.csv', header=None, nrows = 250)
random_row = (csvfile.sample())
img_path = random_row.iloc[0, 0]
test_img = Image.open(img_dir+img_path)
test_img = np.array(test_img)
test_img = cv.resize(test_img, (448, 448))
cv.imwrite('figures/test_img.jpg', test_img)

test_img = transform(test_img)
test_img = test_img.unsqueeze(0).to(device)
preds = model(test_img)
get_bboxes = cellboxes_to_boxes(preds)
test_img = test_img.squeeze(0)
test_img = test_img.permute(1, 2, 0).cpu().numpy() * 255
bboxes = non_max_suppression(get_bboxes[0], iou_threshold=0.5, threshold=0.4, boxformat="midpoints")
test_img_bboxes = draw_bounding_box(test_img, bboxes, test = True)
cv.imwrite('figures/test_img_bboxes.jpg', test_img_bboxes)


def strip_square_brackets(pathtotxt):    
    with open(pathtotxt, 'r') as my_file:
        text = my_file.read()
        text = text.replace("[", "")
        text = text.replace("]", "")
    with open(pathtotxt, 'w') as my_file:
        my_file.write(text)

# strip square brackets, which results in first and last element to be NaN
strip_square_brackets("results/vgg19bn_adj_lr_train_loss.txt")
strip_square_brackets("results/vgg19bn_adj_lr_train_mAP.txt")
strip_square_brackets("results/vgg19bn_adj_lr_test_loss.txt")
strip_square_brackets("results/vgg19bn_adj_lr_test_mAP.txt")

strip_square_brackets("results/resnet18_adj_lr_train_loss.txt")
strip_square_brackets("results/resnet18_adj_lr_train_mAP.txt")
strip_square_brackets("results/resnet18_adj_lr_test_loss.txt")
strip_square_brackets("results/resnet18_adj_lr_test_mAP.txt")

strip_square_brackets("results/resnet50_adj_lr_train_loss.txt")
strip_square_brackets("results/resnet50_adj_lr_train_mAP.txt")
strip_square_brackets("results/resnet50_adj_lr_test_loss.txt")
strip_square_brackets("results/resnet50_adj_lr_test_mAP.txt")

strip_square_brackets("results/tiny_adj_lr_train_loss.txt")
strip_square_brackets("results/tiny_adj_lr_train_mAP.txt")
strip_square_brackets("results/tiny_adj_lr_test_loss.txt")
strip_square_brackets("results/tiny_adj_lr_test_mAP.txt")

strip_square_brackets("results/mobilenetv3_large_tiny_adj_lr_train_loss.txt")
strip_square_brackets("results/mobilenetv3_large_tiny_adj_lr_train_mAP.txt")
strip_square_brackets("results/mobilenetv3_large_tiny_adj_lr_test_loss.txt")
strip_square_brackets("results/mobilenetv3_large_tiny_adj_lr_test_mAP.txt")


strip_square_brackets("results/mobilenetv3_small_tiny_adj_lr_train_loss.txt")
strip_square_brackets("results/mobilenetv3_small_tiny_adj_lr_train_mAP.txt")
strip_square_brackets("results/mobilenetv3_small_tiny_adj_lr_test_loss.txt")
strip_square_brackets("results/mobilenetv3_small_tiny_adj_lr_test_mAP.txt")

strip_square_brackets("results/squeezenet_tiny_adj_lr_train_loss.txt")
strip_square_brackets("results/squeezenet_tiny_adj_lr_train_mAP.txt")
strip_square_brackets("results/squeezenet_tiny_adj_lr_test_loss.txt")
strip_square_brackets("results/squeezenet_tiny_adj_lr_test_mAP.txt")

strip_square_brackets("results/gpu_vgg19bn_adj_lr_inference_speed.txt")
strip_square_brackets("results/gpu_resnet18_adj_lr_inference_speed.txt")
strip_square_brackets("results/gpu_resnet50_adj_lr_inference_speed.txt")
strip_square_brackets("results/cpu_tiny_adj_lr_inference_speed.txt")
strip_square_brackets("results/cpu_mobilenetv3_large_tiny_adj_lr_inference_speed.txt")
strip_square_brackets("results/cpu_mobilenetv3_small_tiny_adj_lr_inference_speed.txt")
strip_square_brackets("results/cpu_squeezenet_tiny_adj_lr_inference_speed.txt")

# Load values from txt file
train_vgg19_loss = genfromtxt("results/vgg19bn_adj_lr_train_loss.txt", delimiter=',')[:135]
train_vgg19_map = genfromtxt("results/vgg19bn_adj_lr_train_mAP.txt", delimiter=',')[:135]
test_vgg19_loss = genfromtxt("results/vgg19bn_adj_lr_test_loss.txt", delimiter=',')[:135]
test_vgg19_map = genfromtxt("results/vgg19bn_adj_lr_test_mAP.txt", delimiter=',')[:135]

train_resnet18_loss = genfromtxt("results/resnet18_adj_lr_train_loss.txt", delimiter=',')[:135]
train_resnet18_map = genfromtxt("results/resnet18_adj_lr_train_mAP.txt", delimiter=',')[:135]
test_resnet18_loss = genfromtxt("results/resnet18_adj_lr_test_loss.txt", delimiter=',')[:135]
test_resnet18_map = genfromtxt("results/resnet18_adj_lr_test_mAP.txt", delimiter=',')[:135]

train_resnet50_loss = genfromtxt("results/resnet50_adj_lr_train_loss.txt", delimiter=',')[:135]
train_resnet50_map = genfromtxt("results/resnet50_adj_lr_train_mAP.txt", delimiter=',')[:135]
test_resnet50_loss = genfromtxt("results/resnet50_adj_lr_test_loss.txt", delimiter=',')[:135]
test_resnet50_map = genfromtxt("results/resnet50_adj_lr_test_mAP.txt", delimiter=',')[:135]

train_tiny_loss = genfromtxt("results/tiny_adj_lr_train_loss.txt", delimiter=',')[:135]
train_tiny_map = genfromtxt("results/tiny_adj_lr_train_mAP.txt", delimiter=',')[:135]
test_tiny_loss = genfromtxt("results/tiny_adj_lr_test_loss.txt", delimiter=',')[:135]
test_tiny_map = genfromtxt("results/tiny_adj_lr_test_mAP.txt", delimiter=',')[:135]

train_tiny_mobile_large_loss = genfromtxt("results/mobilenetv3_large_tiny_adj_lr_train_loss.txt", delimiter=',')[:135]
train_tiny_mobile_large_map = genfromtxt("results/mobilenetv3_large_tiny_adj_lr_train_mAP.txt", delimiter=',')[:135]
test_tiny_mobile_large_loss = genfromtxt("results/mobilenetv3_large_tiny_adj_lr_test_loss.txt", delimiter=',')[:135]
test_tiny_mobile_large_map = genfromtxt("results/mobilenetv3_large_tiny_adj_lr_test_mAP.txt", delimiter=',')[:135]

train_tiny_mobile_small_loss = genfromtxt("results/mobilenetv3_small_tiny_adj_lr_train_loss.txt", delimiter=',')[:135]
train_tiny_mobile_small_map = genfromtxt("results/mobilenetv3_small_tiny_adj_lr_train_mAP.txt", delimiter=',')[:135]
test_tiny_mobile_small_loss = genfromtxt("results/mobilenetv3_small_tiny_adj_lr_test_loss.txt", delimiter=',')[:135]
test_tiny_mobile_small_map = genfromtxt("results/mobilenetv3_small_tiny_adj_lr_test_mAP.txt", delimiter=',')[:135]

train_tiny_squeeze_loss = genfromtxt("results/squeezenet_tiny_adj_lr_train_loss.txt", delimiter=',')[:135]
train_tiny_squeeze_map = genfromtxt("results/squeezenet_tiny_adj_lr_train_mAP.txt", delimiter=',')[:135]
test_tiny_squeeze_loss = genfromtxt("results/squeezenet_tiny_adj_lr_test_loss.txt", delimiter=',')[:135]
test_tiny_squeeze_map = genfromtxt("results/squeezenet_tiny_adj_lr_test_mAP.txt", delimiter=',')[:135]

speed_vgg19bn = genfromtxt("results/gpu_vgg19bn_adj_lr_inference_speed.txt", delimiter=',')[:135]
speed_resnet18 = genfromtxt("results/gpu_resnet18_adj_lr_inference_speed.txt", delimiter=',')[:135]
speed_resnet50 = genfromtxt("results/gpu_resnet50_adj_lr_inference_speed.txt", delimiter=',')[:135]

speed_tiny = genfromtxt("results/cpu_tiny_adj_lr_inference_speed.txt", delimiter=',')[:135]
speed_tiny_mobile_large = genfromtxt("results/cpu_mobilenetv3_large_tiny_adj_lr_inference_speed.txt", delimiter=',')[:135]
speed_tiny_mobile_small = genfromtxt("results/cpu_mobilenetv3_small_tiny_adj_lr_inference_speed.txt", delimiter=',')[:135]
speed_tiny_squeeze = genfromtxt("results/cpu_squeezenet_tiny_adj_lr_inference_speed.txt", delimiter=',')[:135]

# First inference pass takes longer than subsequent passes. As such we substract
# each element by that constant.
speed_vgg19bn = np.array(speed_vgg19bn) - speed_vgg19bn[0]
speed_resnet18 = speed_resnet18 - speed_resnet18[0]
speed_resnet50 = speed_resnet50 - speed_resnet50[0]
speed_tiny = speed_tiny - speed_tiny[0]
speed_tiny_mobile_large = speed_tiny_mobile_large - speed_tiny_mobile_large[0]
speed_tiny_mobile_small = speed_tiny_mobile_small - speed_tiny_mobile_small[0]
speed_tiny_squeeze = speed_tiny_squeeze - speed_tiny_squeeze[0]

# Plot vgg loss 
plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.plot(train_vgg19_loss, linewidth=1.5)
plt.plot(test_vgg19_loss, linewidth=1.5)
plt.title('Vgg19bn: Loss per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('Loss value', fontsize = 16)
plt.legend(['Train loss', 'Test loss'],
           prop={'size': 14},           
            frameon=False)

# Plot vgg mean average precision
plt.subplot(1, 2, 2)
plt.plot(train_vgg19_map, linewidth = 1.5)
plt.plot(test_vgg19_map, linewidth = 1.5)
plt.title('Vgg19bn: Mean average precision per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
plt.savefig('figures/vgg19bn_loss_map.png')


# Plot resnet18 loss
plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.plot(train_resnet18_loss, linewidth=1.5)
plt.plot(test_resnet18_loss, linewidth=1.5)
plt.title('ResNet18: Loss per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('Loss value', fontsize = 16)
plt.legend(['Train loss', 'Test loss'],
           prop={'size': 14},           
            frameon=False)

# Plot resnet18 mean average precision
plt.subplot(1, 2, 2)
plt.plot(train_resnet18_map, linewidth = 1.5)
plt.plot(test_resnet18_map, linewidth = 1.5)
plt.title('ResNet18: Mean average precision per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
plt.savefig('figures/resnet18_loss_map.png')

# Plot resnet50 loss
plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.plot(train_resnet50_loss, linewidth=1.5)
plt.plot(test_resnet50_loss, linewidth=1.5)
plt.title('ResNet50: Loss per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('Loss value', fontsize = 16)
plt.legend(['Train loss', 'Test loss'],
           prop={'size': 14},           
            frameon=False)

# Plot resnet50 mean average precision
plt.subplot(1, 2, 2)
plt.plot(train_resnet50_map, linewidth = 1.5)
plt.plot(test_resnet50_map, linewidth = 1.5)
plt.title('ResNet50: Mean average precision per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
plt.savefig('figures/resnet50_loss_map.png')

# Plot tiny yolo loss
plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
plt.plot(train_tiny_loss, linewidth=1.5)
plt.plot(test_tiny_loss, linewidth=1.5)
plt.title('Tiny ResNet18: Loss per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('Loss value', fontsize = 16)
plt.legend(['Train loss', 'Test loss'],
           prop={'size': 14},           
            frameon=False)

# Plot tiny yolo mean average precision
plt.subplot(1, 2, 2)
plt.plot(train_tiny_map, linewidth = 1.5)
plt.plot(test_tiny_map, linewidth = 1.5)
plt.title('Tiny YoloV1: Mean average precision per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
plt.savefig('figures/tiny_resnet18_loss_map.png')

# Plot mobilev3 M mean loss
plt.figure(figsize=(13,5)) 
plt.subplot(1, 2, 1)
plt.plot(train_tiny_mobile_large_loss, linewidth = 1.5)
plt.plot(test_tiny_mobile_large_loss, linewidth = 1.5)
plt.title('MobileNetV3 M: Loss per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
# Plot mobilev3 M mean mAP 
plt.subplot(1, 2, 2)
plt.plot(train_tiny_mobile_large_map, linewidth = 1.5)
plt.plot(test_tiny_mobile_large_map, linewidth = 1.5)
plt.title('MobileNetV3 M: Mean average precision per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
plt.savefig('figures/mnetv3_m_loss_map.png')


# Plot mobilev3 S mean loss 
plt.figure(figsize=(13,5))
plt.subplot(1, 2, 1)
plt.plot(train_tiny_mobile_small_loss, linewidth = 1.5)
plt.plot(test_tiny_mobile_small_loss, linewidth = 1.5)
plt.title('MobileNetV3 S: Loss per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
# Plot mobilev3 M mean mAP 
plt.subplot(1, 2, 2)
plt.plot(train_tiny_mobile_large_map, linewidth = 1.5)
plt.plot(test_tiny_mobile_large_map, linewidth = 1.5)
plt.title('MobileNetV3 S: Mean average precision per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
plt.savefig('figures/mnetv3_s_loss_map.png')


# Plot squeezenet mean loss 
plt.figure(figsize=(13,5))
plt.subplot(1, 2, 1)
plt.plot(train_tiny_squeeze_loss, linewidth = 1.5)
plt.plot(test_tiny_squeeze_loss, linewidth = 1.5)
plt.title('SqueezeNet: Loss per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
# Plot squeeze M mean mAP 
plt.subplot(1, 2, 2)
plt.plot(train_tiny_squeeze_map, linewidth = 1.5)
plt.plot(test_tiny_squeeze_map, linewidth = 1.5)
plt.title('SqueezeNet: Mean average precision per epoch', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(['Train mAP', 'Test mAP'], prop={'size': 14},           
            frameon=False)
plt.savefig('figures/squeezenet_loss_map.png')




# Plot train map across models
plt.figure(figsize=(18,5))
plt.subplot(1,2,1)
plt.plot(train_vgg19_map, linewidth=1.5)
plt.plot(train_resnet18_map, linewidth=1.5)
plt.plot(train_resnet50_map, linewidth=1.5)
plt.plot(train_tiny_map, linewidth=1.5)
plt.plot(train_tiny_mobile_large_map, linewidth=1.5)
plt.plot(train_tiny_mobile_small_map, linewidth=1.5)
plt.plot(train_tiny_squeeze_map, linewidth=1.5)
plt.title('Model comparison: Train mAP', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value', fontsize = 16)
#plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5),
#labels = ['Vgg19bn', 'Resnet18', 'Resnet50', 'Tiny', 'MnetV3M', 'MnetV3S', 
#'SNet'], prop={'size': 14},                     
#            frameon=False)

# Plot test map across models
plt.subplot(1, 2, 2)
plt.plot(test_vgg19_map, linewidth=1.5)
plt.plot(test_resnet18_map, linewidth=1.5)
plt.plot(test_resnet50_map, linewidth=1.5)
plt.plot(test_tiny_map, linewidth=1.5)
plt.plot(test_tiny_mobile_large_map, linewidth=1.5)
plt.plot(test_tiny_mobile_small_map, linewidth=1.5)
plt.plot(test_tiny_squeeze_map, linewidth=1.5)
plt.title('Model comparison: Test mAP', fontsize = 16)
plt.xlabel('Number of epochs', fontsize = 16)
plt.ylabel('mAP value ', fontsize = 16)
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5),
labels = ['Vgg19bn', 'Resnet18', 'Resnet50', 'Tiny', 'MnetV3M', 'MnetV3S', 
'SNet'], prop={'size': 14},           
            frameon=False)
plt.savefig('figures/model_comparison_map.png')

# Plot inference speed per image/frame
plt.figure(figsize=(13,5))
plt.plot(speed_vgg19bn, linewidth=1.5)
plt.plot(speed_resnet18, linewidth=1.5)
plt.plot(speed_resnet50, linewidth=1.5)
plt.title('GPU Model comparison: Inference speed', fontsize = 16)
plt.xlabel('Number of frames/images processed', fontsize = 16)
plt.ylabel('Time in seconds', fontsize = 16)
plt.legend(['Vgg19bn', 'Resnet18', 'Resnet50'],
           prop={'size': 14},           
            frameon=False)
plt.savefig('figures/gpu_model_comparison_inference_speed.png')

plt.figure(figsize=(13,5))
plt.plot(speed_tiny, linewidth=1.5)
plt.plot(speed_tiny_mobile_large, linewidth=1.5)
plt.plot(speed_tiny_mobile_small, linewidth=1.5)
plt.plot(speed_tiny_squeeze, linewidth=1.5)
plt.title('CPU Model comparison: Inference speed', fontsize = 16)
plt.xlabel('Number of frames/images processed', fontsize = 16)
plt.ylabel('Time in seconds', fontsize = 16)
plt.legend(['Tiny', 'MnetV3M', 'MnetV3S', 'Snet'],
           prop={'size': 14},           
            frameon=False)
plt.savefig('figures/cpu_model_comparison_inference_speed.png')
