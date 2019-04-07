# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:08:29 2019

@author: imlab_Amin_Ullah
"""


import cv2

import numpy as np
import caffe
import time 

img00 = '/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Action Recognition uing GRU/LiteFlowNet-master/models/testing/1.PNG'
img11 = '/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Action Recognition uing GRU/LiteFlowNet-master/models/testing/2.PNG'


caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Net('/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Action Recognition uing GRU/LiteFlowNet-master/models/testing/My_deploy.prototxt',
            '/media/imlab/IMLab Server Data/Ubuntu/AminUllah/Action Recognition uing GRU/LiteFlowNet-master/models/trained/liteflownet.caffemodel',
            caffe.TEST)
            
num_blobs = 2
input_data = []


vidcap = cv2.VideoCapture('/media/imlab/IMLab Server Data/Datasets/Activity Net/Anet_tools2.0-master/Videos/v_1XElQ8DoDMU.mp4')
videolength = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
videoFeatures=[]
frame_no=-1;

while (frame_no < videolength-1):  #(videolength%30)
    frame_no = frame_no + 1
    vidcap.set(1,frame_no)
    ret0,img0 = vidcap.read()
    
    frame_no = frame_no + 1
    vidcap.set(1,frame_no)
    ret1,img1 = vidcap.read()
    if(ret0 == 1 and ret1 == 1):
        img1 = cv2.resize(img1, (320, 256))
        img0 = cv2.resize(img0, (320, 256)) 
        if len(img0.shape) < 3: input_data.append(img0[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
        
        if len(img1.shape) < 3: input_data.append(img1[np.newaxis, np.newaxis, :, :])
        else:                   input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    
    
        input_dict = {}
        for blob_idx in range(num_blobs):
            input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
    
  
        net.forward(**input_dict)

        flow = np.squeeze(net.blobs['final_flow'].data).transpose(1, 2, 0)
        
        
        
        
