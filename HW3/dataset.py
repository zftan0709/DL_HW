# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:53:13 2020

@author: ZhiFang
"""
import tensorflow as tf
import numpy as np

class dataset():
    def __init__(self):
        self.images = None
        self.order = None
        self.data_idx = None
        self.batch_idx = 0
        self.data_size = 0
        
    def load_CIFAR10(self):
        (train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()
        self.images = np.concatenate([train_images,test_images])
        self.images = self.images.astype('float32')
        self.images = self.images/127.5-1
        self.data_size = len(self.images)
        self.data_idx = np.arange(self.data_size,dtype=np.int)
        
    # def next_batch(self,batch_size):
    #     idx = np.random.randint(0,self.data_num,batch_size)
    #     return self.images[idx]
    
    def shuffle(self):
        np.random.shuffle(self.data_idx)
    
    def next_batch(self,batch_size):
        if self.batch_idx + batch_size <= self.data_size:
            idx = self.data_idx[self.batch_idx:(self.batch_idx+batch_size)]
            self.batch_idx += batch_size
        else:
            idx = self.data_idx[self.batch_idx:]
            self.batch_idx = 0
        return self.images[idx]