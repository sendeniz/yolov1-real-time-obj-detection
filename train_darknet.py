#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 13:27:34 2022

@author: sen
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from darknet import DarkNet
from darknet_utils import top1accuracy, top5accuracy
from numpy import genfromtxt
import math

lr = 0.0003
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
weight_decay = 0.0005
epochs = 250
num_workers = 2
pin_memory = True
continue_training = False
path_cpt_file = 'cpts/darknet.cpt'
path_train_data = 'data/ILSVRC2012_img_train'
path_val_data = 'data/ILSVRC2012_img_val'

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def step_wise_lr_schedule(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # TensorBoard
    writer.add_scalar('lr', lr, epoch)

train_transform = Compose([T.Resize((244, 244)),
            #T.RandomResizedCrop((224, 224)),
            T.ColorJitter(brightness=[0, 0.75], 
            saturation=[0, 0.75],
            hue = [0, 0.1]),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ])

val_transform = Compose([T.Resize((244, 244)),
		    #T.CenterCrop((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
            ])

def train (train_loader, model, optimizer, loss_f):
    loop = tqdm(train_loader, leave = True)
    model.train()
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)  
        pred_class = torch.argmax(out, dim = 1)
        top1_acc = top1accuracy(out, y, batch_size)
        top5_acc = top5accuracy(out, y, batch_size)
        del x
        loss_val = loss_f(out, y)
        del y
        del out
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()  
        # update progress bar
        loop.set_postfix(loss = loss_val.item())
    return (float(loss_val.item()), top1_acc, top5_acc)
    
def test(test_loader, model, loss_f):
    loop = tqdm(test_loader, leave = True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loop):
            x, y = x.to(device), y.to(device)
            out = model(x)
            top1_acc = top1accuracy(out, y, batch_size)
            top5_acc = top5accuracy(out, y, batch_size)
            del x
            test_loss_val = loss_f(out, y)
            del y
            del out
            # update progress bar
            loop.set_postfix(loss = test_loss_val.item())
        return(float(test_loss_val), top1_acc, top5_acc)

def main():
    save_model = True
    model = DarkNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    loss_f = nn.CrossEntropyLoss()
    train_data = ImageFolder(path_train_data, transform = train_transform)

    train_loader = DataLoader(train_data, shuffle = True, batch_size = batch_size, 
                                num_workers = num_workers, pin_memory=True)
    
    val_data = ImageFolder(path_val_data, transform = val_transform)
    val_loader = DataLoader(val_data, shuffle = True, batch_size = batch_size, 
                            num_workers = num_workers, pin_memory=True)
    
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.94)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=4, 
                                                                T_mult=2,
                                                                eta_min=0.00001,
                                                                last_epoch=-1)

    train_loss_lst = []
    val_loss_lst = []
    train_top1_acc_lst = []
    train_top5_acc_lst = []
    val_top1_acc_lst = []
    val_top5_acc_lst = []
    last_epoch = 0
    if (continue_training == True):
        checkpoint = torch.load(path_cpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        last_epoch = checkpoint['epoch']
        print(f"Checkpoint from epoch:{last_epoch} successfully loaded.")
        train_loss_lst = list(genfromtxt("results/darknet_trainloss.txt", delimiter=','))
        val_loss_lst = list(genfromtxt("results/darknet_valloss.txt", delimiter=','))
        train_top1_acc_lst = list(genfromtxt("results/darknet_traintop1acc.txt", delimiter=','))
        train_top5_acc_lst = list(genfromtxt("results/darknet_traintop5acc.txt", delimiter=','))
        val_top1_acc_lst = list(genfromtxt("results/darknet_valtop1acc.txt", delimiter=','))
        val_top5_acc_lst = list(genfromtxt("results/darknet_valtop5acc.txt", delimiter=','))


    for epoch in range(epochs - last_epoch):
        torch.cuda.empty_cache()
        train_loss, train_top1_acc, train_top5_acc = train(train_loader, model, optimizer, loss_f)
        train_loss_lst.append(train_loss)
        train_top1_acc_lst.append(train_top1_acc)
        train_top5_acc_lst.append(train_top5_acc)
        
        val_loss, val_top1_acc, val_top5_acc = test(val_loader, model, loss_f)
        val_loss_lst.append(val_loss)
        val_top1_acc_lst.append(val_top1_acc)
        val_top5_acc_lst.append(val_top5_acc)
        print(f"Learning Rate:", optimizer.param_groups[0]["lr"])
        print(f"Epoch:{epoch+last_epoch}  Train[Loss:{train_loss}   Top-1-Accuracy:{train_top1_acc}   Top-5-Accuracy:{train_top5_acc}]")
        print(f"Epoch:{epoch+last_epoch}  Test[Loss:{val_loss}   Top-1-Accuracy:{val_top1_acc}   Top-5-Accuracy:{val_top5_acc}]")

        scheduler.step()
        #step_wise_lr_schedule(optimizer, epoch + last_epoch)
        
        if save_model == True and (epoch % 10) == 0:
            torch.save({
                'epoch': epoch+last_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, path_cpt_file)
            print(f"Checkpoint at {epoch+last_epoch} stored")
            with open('results/darknet_trainloss.txt','w') as values:
                values.write(str(train_loss_lst))
            with open('results/darknet_traintop1acc.txt','w') as values:
                values.write(str(train_top1_acc_lst))    
            with open('results/darknet_traintop5acc.txt','w') as values:
                values.write(str(train_top5_acc_lst))
            with open('results/darknet_valloss.txt','w') as values:
                values.write(str(val_loss_lst))
            with open('results/darknet_valtop1acc.txt','w') as values:
                values.write(str(val_top1_acc_lst))    
            with open('results/darknet_valtop5acc.txt','w') as values:
                values.write(str(val_top5_acc_lst))
        
        if ((save_model == True and (val_top5_acc_lst[-1] >= 0.88)) or (save_model == True and (epoch+last_epoch == epochs))):
            torch.save({
                'epoch': epoch+last_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
                }, path_cpt_file)
            save_model = False
            print(f"Stopping condition top 5 accuracy or max epoch reached.")
            print(f"Model has been stored.")
            with open('results/darknet_trainloss.txt','w') as values:
                values.write(str(train_loss_lst))  
            with open('results/darknet_top1acc.txt','w') as values:
                values.write(str(train_top1_acc_lst))    
            with open('results/darknet_top5acc.txt','w') as values:
                values.write(str(train_top5_acc_lst))
            with open('results/darknet_testloss.txt','w') as values:
                values.write(str(val_loss_lst))
            with open('results/darknet_valtop1acc.txt','w') as values:
                values.write(str(val_top1_acc_lst))    
            with open('results/darknet_valtop5acc.txt','w') as values:
                values.write(str(val_top5_acc_lst))
            break

if __name__ == "__main__":
    main()