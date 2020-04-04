# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:01:40 2020

@author: ZhiFang
"""
from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence
import numpy as np
import json
import pickle
import os

filters = '`","?!/.()'

class dataset():
    
    def __init__(self,batch_size,feat_dir,label_dir=None,max_caption_len=50):
        self.batch_size = batch_size
        self.max_caption_len = max_caption_len
        self.feat_dir = feat_dir
        self.label_dir = label_dir
        self.batch_idx = 0
        self.vocab_size = 0
        self.data_size = None
        self.data_idx = None
        self.id_list = None
        self.cap_list = None
        self.cap_length_list = None
    
    def generate_token(self):
        self.tokenizer = Tokenizer(filters=filters,split=" ")
        total_list = []
        with open(self.label_dir) as f:
            raw_data = json.load(f)
        for vid in raw_data:
            for cap in vid['caption']:
                total_list.append(cap)
        self.tokenizer.fit_on_texts(total_list)
        self.vocab_size = len(self.tokenizer.word_index)
        self.tokenizer.fit_on_texts(['<PAD>','<BOS>','<EOS>','<UNK>'])
    
    def load_token(self):
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.vocab_size = len(self.tokenizer.word_index) - 4
        
    def process_data(self):
        pad = self.tokenizer.texts_to_sequences(['<PAD>'])[0]
        id_list = []
        cap_list = []
        cap_length_list = []
        self.feat_data = {}
        with open(self.label_dir) as f:
            raw_data = json.load(f)
        for vid in raw_data:
            vid_id = vid['id']
            self.feat_data[vid_id] = np.load(self.feat_dir + vid_id + '.npy')

            for caption in vid['caption']:
                words = text_to_word_sequence(caption)
                for i in range(len(words)):
                    if words[i] not in self.tokenizer.word_index:
                        words[i] = '<UNK>'
                words.append('<EOS>')
                one_hot = self.tokenizer.texts_to_sequences([words])[0]
                cap_length = len(one_hot)
                one_hot += pad * (self.max_caption_len - cap_length)
                id_list.append(vid_id)
                cap_list.append(one_hot)
                cap_length_list.append(cap_length)
                
        self.id_list = np.array(id_list)
        self.cap_list = np.array(cap_list)
        self.cap_length_list = np.array(cap_length_list)
        self.data_size = len(self.cap_list)
        self.data_idx = np.arange(self.data_size,dtype=np.int)
        
    def save_vocab(self):
        with open('tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def shuffle(self):
        np.random.shuffle(self.data_idx)
        
    def next_batch(self):
        if self.batch_idx + self.batch_size <= self.data_size:
            idx = self.data_idx[self.batch_idx:(self.batch_idx+self.batch_size)]
            self.batch_idx += self.batch_size
        else:
            idx = self.data_idx[self.batch_idx:]
            self.batch_idx = 0
        id_batch = self.id_list[idx]
        cap_batch = self.cap_list[idx]
        cap_length_batch = self.cap_length_list[idx]
        feat_batch = []
        for vid_id in id_batch:
            feat_batch.append(self.feat_data[vid_id])
        feat_batch = np.array(feat_batch)
        return id_batch,feat_batch,cap_batch,cap_length_batch
    
    def process_feature_data(self):
        id_list = []
        self.feat_data = {}
        for filename in os.listdir(self.feat_dir):
            if filename.endswith('.npy'):
                vid_id = os.path.splitext(filename)[0]
                self.feat_data[vid_id] = np.load(self.feat_dir + filename)
                id_list.append(vid_id)
            self.id_list = np.array(id_list)
        self.data_size = len(self.id_list)
        self.data_idx = np.arange(self.data_size,dtype=np.int)
        
    def next_feature_batch(self):
        if self.batch_idx + self.batch_size <= self.data_size:
            idx = self.data_idx[self.batch_idx:(self.batch_idx+self.batch_size)]
            self.batch_idx += self.batch_size
        else:
            idx = self.data_idx[self.batch_idx:]
            self.batch_idx = 0
        id_batch = self.id_list[idx]
        feat_batch = []
        for vid_id in id_batch:
            feat_batch.append(self.feat_data[vid_id])
        feat_batch = np.array(feat_batch)
        return id_batch,feat_batch
