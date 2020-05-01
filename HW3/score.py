# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 12:04:23 2020

@author: ZhiFang
"""


import tensorflow as tf
import skimage.transform
import numpy as np
import scipy

def scale_images(images,shape):
    images_list = []
    for img in images:
        new_img = skimage.transform.resize(img,shape,0)
        images_list.append(new_img)
    return np.array(images_list)

def calculate_inception(images,batch_size):
    print("Calculating Inception Score...")
    model = tf.keras.applications.inception_v3.InceptionV3()
    
    is_score = []
    for i in range(0,images.shape[0],batch_size):
        if i+batch_size < images.shape[0]:
            processed_images = scale_images(images[i:i+batch_size],(299,299,3))
        else:
            processed_images = scale_images(images[i:],(299,299,3))
        p_yx = model.predict(processed_images)
        p_y = np.expand_dims(p_yx.mean(axis=0),0)
        kl_d = p_yx * (np.log(p_yx + np.finfo(np.float32).eps) - np.log(p_y + np.finfo(np.float32).eps))
        score = np.exp(np.mean(np.sum(kl_d,axis=1)))
        print(i,"/",images.shape[0],", Calculating IS Score: ",score)
        is_score.append(score)
    is_avg,is_std = np.mean(is_score),np.std(is_score)
    return is_avg,is_std

def calculate_FID(image1,image2):
    print("Calculating FID Score...")
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=False,pooling='avg',input_shape=(299,299,3))
    processed_img1 = scale_images(image1,(299,299,3))
    processed_img2 = scale_images(image2,(299,299,3))
    activation1 = model.predict(processed_img1)
    activation2 = model.predict(processed_img2)
    mean1 = np.mean(activation1,axis=0)
    cov1 = np.cov(activation1,rowvar=False)
    mean2 = np.mean(activation2,axis=0)
    cov2 = np.cov(activation2,rowvar=False)
    diff = np.sum((mean1-mean2)**2.0)
    covmean = scipy.linalg.sqrtm(np.dot(cov1,cov2))
    if(np.iscomplexobj(covmean)):
        covmean = covmean.real
    score = diff + np.trace(cov1 + cov2 - 2*covmean)
    return score