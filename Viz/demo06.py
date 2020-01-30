#自然图像分割结果可视化
#https://blog.csdn.net/baidu_36669549/article/details/95047859
#主要用的模型用的是FCN，参考的是https://github.com/bat67/pytorch-FCN-easiest-demo
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from onehot import onehot
import torch
import torch.nn as nn
from FCN import FCN8s, FCN16s, FCNs, VGGNet
from torchvision import transforms
import pdb
 
os.environ["CUDA_VISION_DEVICES"] = "2"
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
def img_2_tensor(imgfile):
    a = troch.Tensor(1,3,160,160)
    imgA = cv2.imread(imgfile)
    imgA = cv2.resize(imgA, (160,160))
    imgA = transform(imgA)
    imgA = torch.FloatTensor(imgA)
    a[0,:] = imgA
    return a
def cpr_2_array(array_1,array_2):
    array_3 = np.ones(array_1.shape, np.uint8)*255
    for h in range(160):
        for w in range(160):
            if(array_1[h][w]==0 and array_2[h][w]==0):
                array_3[h][w]=0
    for hh in range(160):
        for ww in range(160):
            if(array_3[hh][ww] == 0):
                array_2[hh][ww] =255
    return array_3,array_2
img_dir = 'test_data/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fcn_model_person = torch.load('checkpoints/fcn_model_person.pt')
fcn_model_person = fcn_model_person.to(device)
 
fcn_model_motor = torch.load('checkpoints/fcn_model_motor.pt')
fcn_model_motor = fcn_model_motor.to(device)
 
 
for i, j, k in os.walk(img_dir):
    for im in k:
        img_file = i + '/' +im 
        imput_im = cv2.imread(img_file)
        input_im = cv2.resize(imput_im, (160,160))
        img_template = img_2_tensor(img_file)
        output_person = fcn_model_person(img_template.cuda())
        output0_person = output_person[0,0,:]
        output1_person = output_person[0,1,:]
        output_np0_person = output0_person.cpu().detach().numpy().copy()
        output_np1_person = output1_person.cpu().detach().numpy().copy()
        output_np1_person = -output_np1_person
        output_np1_person[output_np1_person>0] = 255
        output_np1_person[output_np1_person<=0] = 0
        plt.subplot(1,5,1)
        plt.title('Image')
        plt.imshow(input_im)
 
        plt.subplot(1,5,2)
        plt.title('Person')
        plt.imshow(np.squeeze(output_np1_person),'gray')
 
        output_motor = fcn_model_person(img_template.cuda())
        output0_motor = output_motor[0,0,:]
        output1_motor = output_motor[0,1,:]
        output_np0_motor = output0_motor.cpu().detach().numpy().copy()
        output_np1_motor = output1_motor.cpu().detach().numpy().copy()
        output_np1_motor = -output_np1_motor
        output_np1_motor[output_np1_motor<0.5] = 0
        output_np1_motor[output_np1_motor>=0.5] = 255
 
        plt.subplot(1,5,3)
        plt.title('Motor')
        plt.imshow(np.squeeze(output_np1_motor),'gray')
 
 
        plt.subplot(1,5,4)
        plt.title('&&')
        cpr_arr,apr_a = cpr_2_array(output_np1_person, output_np1_motor)
        plt.imshow(np.squeeze(cpr_arr),'gray')
 
 
        plt.subplot(1,5,5)
        plt.title('--')
        plt.imshow(np.squeeze(cpr_arr),'gray')
 
 
 
 