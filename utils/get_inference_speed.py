from pickle import TRUE
import numpy as np
import time 
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from models.yolov1net_darknet import YoloV1Net
from models.yolov1net_vgg19bn import YoloV1_Vgg19bn
from models.yolov1net_resnet18 import YoloV1_Resnet18
from models.yolov1net_resnet50 import YoloV1_Resnet50
from models.tiny_yolov1net import Tiny_YoloV1
from models.tiny_yolov1net_mobilenetv3_large import Tiny_YoloV1_MobileNetV3_Large
from models.tiny_yolov1net_mobilenetv3_small import Tiny_YoloV1_MobileNetV3_Small
from models.tiny_yolov1net_squeezenet import Tiny_YoloV1_SqueezeNet

import torchvision.transforms as T
from PIL import Image
import pandas as pd
import cv2 as cv

img_dir = 'data/images/'
label_dir = 'data/labels/'
results_dir = 'results/'
transform = T.Compose([T.ToTensor()])
weight_decay = 0.0005
device = "cuda" if torch.cuda.is_available() else "cpu"

yolov1_darknet_pretrained = False
yolov1_vgg19bn_pretrained = True
yolov1_resnet18_pretrained = False
yolov1_resnet50_pretrained = False
tiny_yolov1_pretrained = False
tiny_yolov1_mobilenetv3_large_pretrained = False
tiny_yolov1_mobilenetv3_small_pretrained = False
tiny_yolov1_squeezenet_pretrained = False

# On cpu flag is only for our tiny models. By setting it to false, the models will
# run inference on gpu instead of cpu. 

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
    model = YoloV1_Vgg19bn(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    checkpoint = torch.load(path_cpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained tiny yolov1 vgg19 network initalized.")
elif yolov1_resnet18_pretrained == True:
    lr =  0.00001
    current_model = model_names[1]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
    model = YoloV1_Resnet18(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    checkpoint = torch.load(path_cpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained tiny yolov1 resnet18 network initalized.")
elif yolov1_resnet50_pretrained == True:
    lr =  0.00001
    current_model = model_names[2]
    path_cpt_file = f'cpts/{current_model}yolov1.cpt'
    model = YoloV1_Resnet50(S = 7, B = 2, C = 20).to(device)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    checkpoint = torch.load(path_cpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    print("Petrained tiny yolov1 resnet50 network initalized.")
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


csvfile = pd.read_csv('data/test.csv', header=None)

lst = []
prev = 0
for i in range(300):
    random_row = (csvfile.sample())
    img_path = random_row.iloc[0, 0]
    img_path = random_row.iloc[0, 0]
    test_img = Image.open(img_dir+img_path)
    test_img = np.array(test_img)
    test_img = cv.resize(test_img, (448, 448))
    test_img = transform(test_img)
    test_img = test_img.unsqueeze(0).to(device)
    start = time.time()
    model(test_img)
    end = time.time()
    elapsed = (end-start)
    lst.append(elapsed + prev)
    prev = lst[-1]

if yolov1_darknet_pretrained == True:
    with open(f'results/gpu_{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif yolov1_vgg19bn_pretrained == True:
    with open(f'results/gpu_{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif yolov1_resnet18_pretrained == True:
    with open(f'results/gpu_{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif yolov1_resnet50_pretrained == True:
    with open(f'results/gpu_{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif tiny_yolov1_pretrained == True:
    with open(f'results/cpu_{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif tiny_yolov1_mobilenetv3_large_pretrained == True:
    with open(f'results/cpu_{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif tiny_yolov1_mobilenetv3_small_pretrained == True:
    with open(f'results/cpu_{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif tiny_yolov1_squeezenet_pretrained == True:
    with open(f'results/cpu_{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))