import numpy as np
import time 
import torch
import torch.optim as optim
from yolov1net import YoloV1Net
from yolov1net_vgg19bn import YoloV1_Vgg19bn
from yolov1net_resnet18 import YoloV1_Resnet18
from yolov1net_resnet50 import YoloV1_Resnet50
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
yolov1_vgg19bn_pretrained = False
yolov1_resnet18_pretrained = False
yolov1_resnet50_pretrained = True

model_names = ['vgg19bn_adj_lr_',
                'resnet18_adj_lr_',
                'resnet50_adj_lr_']

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


# # init model
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
    with open(f'results/{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif yolov1_vgg19bn_pretrained == True:
    with open(f'results/{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif yolov1_resnet18_pretrained == True:
    with open(f'results/{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))
elif yolov1_resnet50_pretrained == True:
    with open(f'results/{current_model}inference_speed.txt','w') as values:
        values.write(str(lst))