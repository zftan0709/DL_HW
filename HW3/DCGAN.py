# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:17:59 2020

@author: ZhiFang
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from dataset import dataset
import score
import os


class DCGAN():
    def __init__(self,lr = 0.0005):
        self.lr = lr


    def generator(self,input_layer,isTrain=False,reuse=False):
        init = tf.truncated_normal_initializer(stddev=0.02)
        with tf.variable_scope('generator',reuse=reuse):
            input_dense = tf.layers.dense(input_layer,2*2*512,use_bias=False)
            input_reshape = tf.reshape(input_dense,[-1,2,2,512])
            gen_ct1 = tf.layers.batch_normalization(input_reshape,training=isTrain)
            gen_ct1 = tf.maximum(gen_ct1,gen_ct1*0.2)
            
            gen_ct2 = tf.layers.conv2d_transpose(gen_ct1,filters=256,kernel_size=(5,5),strides=2,padding='same',kernel_initializer=init,use_bias=False)
            gen_ct2 = tf.layers.batch_normalization(gen_ct2,training=isTrain)
            gen_ct2 = tf.maximum(gen_ct2,gen_ct2*0.2)
            
            gen_ct3 = tf.layers.conv2d_transpose(gen_ct2,filters=128,kernel_size=(5,5),strides=2,padding='same',kernel_initializer=init,use_bias=False)
            gen_ct3 = tf.layers.batch_normalization(gen_ct3,training=isTrain)
            gen_ct3 = tf.maximum(gen_ct3,gen_ct3*0.2)
             
            gen_ct4 = tf.layers.conv2d_transpose(gen_ct3,filters=64,kernel_size=(5,5),strides=2,padding='same',kernel_initializer=init,use_bias=False)
            gen_ct4 = tf.layers.batch_normalization(gen_ct4,training=isTrain)
            gen_ct4 = tf.maximum(gen_ct4,gen_ct4*0.2)
             
            gen_ct5 = tf.layers.conv2d_transpose(gen_ct4,filters=3,kernel_size=(5,5),strides=2,padding='same',kernel_initializer=init,use_bias=False)
            gen_output = tf.tanh(gen_ct5)
            
            return gen_output
        
    def discriminator(self,input_layer,isTrain=True,reuse=False):
        init = tf.truncated_normal_initializer(stddev=0.02)
        with tf.variable_scope('discriminator',reuse=reuse):
            disc_conv1 = tf.layers.conv2d(input_layer,filters=64,kernel_size=(5,5),strides=2,padding='same',kernel_initializer=init)
            disc_conv1 = tf.maximum(disc_conv1,disc_conv1*0.2)
            disc_conv1 = tf.nn.dropout(disc_conv1,0.5)
            
            disc_conv2 = tf.layers.conv2d(disc_conv1,filters=128,kernel_size=(5,5),strides=2,padding='same',kernel_initializer=init)
            disc_conv2 = tf.layers.batch_normalization(disc_conv2,training=isTrain)
            disc_conv2 = tf.maximum(disc_conv2,disc_conv2*0.2)
            disc_conv2 = tf.nn.dropout(disc_conv2,0.5)
            
            disc_conv3 = tf.layers.conv2d(disc_conv2,filters=256,kernel_size=(5,5),strides=2,padding='same',kernel_initializer=init)
            disc_conv3 = tf.layers.batch_normalization(disc_conv3,training=isTrain)
            disc_conv3 = tf.maximum(disc_conv3,disc_conv3*0.2)
            
            disc_conv4 = tf.layers.conv2d(disc_conv3,filters=512,kernel_size=(5,5),strides=2,padding='same',kernel_initializer=init)
            disc_conv4 = tf.layers.batch_normalization(disc_conv4,training=isTrain)
            disc_conv4 = tf.maximum(disc_conv4,disc_conv4*0.2)
            
            disc_flat = tf.layers.flatten(disc_conv4)
            disc_output = tf.layers.dense(disc_flat,1,kernel_initializer=init)
            
            return disc_output
        
    def build_model(self):
        real_img = tf.placeholder(tf.float32, [None,32,32,3],name='image_input')
        noise_input = tf.placeholder(tf.float32,[None,100],name='noise_input')
    
        gen_img = self.generator(noise_input,isTrain=True,reuse=False)
        disc_real_output = self.discriminator(real_img,reuse=False)
        disc_gen_output = self.discriminator(gen_img,reuse=True)
        
        gen_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=disc_gen_output,multi_class_labels=tf.ones_like(disc_gen_output)))
        
        disc_real_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=disc_real_output,multi_class_labels=0.9*tf.ones_like(disc_real_output)))
        disc_gen_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits=disc_gen_output,multi_class_labels=tf.zeros_like(disc_gen_output)))
        disc_loss = tf.add(disc_real_loss,disc_gen_loss)
        
        gen_var = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        disc_var = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_train_op = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.5,beta2=0.9).minimize(gen_loss,var_list=gen_var)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=self.lr,beta1=0.5,beta2=0.9).minimize(disc_loss,var_list=disc_var) 
        
        return real_img,noise_input,gen_train_op,disc_train_op,gen_loss,disc_real_loss,disc_gen_loss
    
  

def train(epoch_num,batch_size=64,learning_rate=0.0002):
    data = dataset()
    data.load_CIFAR10()
    graph_DCGAN = tf.Graph()
    with graph_DCGAN.as_default():
        model = DCGAN(lr = learning_rate)
        real_img,noise_input,gen_train_op,disc_train_op,gen_loss_op,disc_real_loss_op,disc_gen_loss_op = model.build_model()
        gen_infer_op = model.generator(noise_input,isTrain=False,reuse=True)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=3)
    config = tf.ConfigProto()   
    config.gpu_options.allow_growth = True
    sess_train = tf.Session(graph=graph_DCGAN,config=config)
     
    sess_train.run(init)
    loss = []
    sample_noise = np.random.normal(0,1,[49, 100])
    for epoch in tqdm(range(epoch_num)):
        batch_num = int(np.ceil(data.data_size/batch_size))
        data.shuffle()
        for i in range(batch_num):
            img_batch = data.next_batch(batch_size)
            noise = np.random.normal(0,1,[len(img_batch), 100])
            feed_dict={noise_input:noise,real_img:img_batch}
            sess_train.run(disc_train_op,feed_dict=feed_dict)
            sess_train.run(gen_train_op,feed_dict=feed_dict)
            gen_loss, disc_real_loss,disc_gen_loss = sess_train.run([gen_loss_op, disc_real_loss_op, disc_gen_loss_op], feed_dict=feed_dict)
        loss.append((gen_loss,disc_real_loss,disc_gen_loss))
        tqdm.write("Epoch %i, Gen Loss: %f, Disc Real Loss: %f, Disc Gen Loss: %f" %(epoch,gen_loss,disc_real_loss,disc_gen_loss))
        if (epoch+1)%50 == 0:
            model_path = saver.save(sess_train, './save/DCGAN/DCGAN', global_step=epoch_num*batch_num)
        if (epoch+1) % 20 == 0:
            gen_samples = sess_train.run(gen_infer_op,feed_dict={noise_input:sample_noise})
            gen_samples = (gen_samples+1)/2
            fig,axs=plt.subplots(7,7) 
            fig.suptitle('Epoch: %d' %epoch)
            for i in range(len(gen_samples)):
                axs[int(i/7),i%7].imshow(gen_samples[i,:,:,:])
                axs[int(i/7),i%7].axis('off')
                axs[int(i/7),i%7].set_aspect("auto")
    return loss
    
def test():
    print("Testing on DCGAN")
    graph_DCGAN = tf.Graph()
    with graph_DCGAN.as_default():
        model = DCGAN(lr = 0.0002)
        real_img,noise_input,_,_,_,_,_ = model.build_model()
        gen_infer_op = model.generator(noise_input,isTrain=False,reuse=True)
        saver = tf.train.Saver(max_to_keep=3)
    config = tf.ConfigProto()   
    config.gpu_options.allow_growth = True
    sess_test = tf.Session(graph=graph_DCGAN,config=config)
    
    cur_dict = os.getcwd()
    print('Restoring model from:')
    print(cur_dict,'/save/\n')
    latest_checkpoint = tf.train.latest_checkpoint(cur_dict+'/save/DCGAN/')
    saver.restore(sess_test, latest_checkpoint)
    
    data_size = 5000
    print("Generating sample...")
    sample_noise = np.random.normal(0,1,[data_size, 100]) 
    gen_samples = sess_test.run(gen_infer_op,feed_dict={noise_input:sample_noise})
    print("Finished generating sample...")
    output_samples = (gen_samples[0:49,:,:,:]+1)/2
    fig,axs=plt.subplots(7,7)
    fig.set_figwidth(20)
    fig.set_figheight(20)
    for i in range(len(output_samples)):
        axs[int(i/7),i%7].imshow(output_samples[i,:,:,:])
        axs[int(i/7),i%7].axis('off')
        axs[int(i/7),i%7].set_aspect("auto")
    return output_samples
    is_avg,is_std = score.calculate_inception(gen_samples,500)
    print("Final IS Score, avg: ",is_avg,", std: ",is_std)
    data = dataset()
    data.load_CIFAR10()
    data.shuffle()
    fid_score = score.calculate_FID(gen_samples,data.next_batch(data_size))
    print("Final FID Score, score: ",fid_score)
# loss = train(200)
pic = test()