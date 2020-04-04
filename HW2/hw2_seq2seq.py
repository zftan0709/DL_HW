"""
Created on Thu Apr  2 02:27:47 2020

@author: ZhiFang
"""


import tensorflow as tf
import numpy as np
from dataset import dataset
import sys
import math
import pickle
import os

input_num = 4096
hidden_num = 256
frame_num = 80
batch_size = 128
epoch_num = 200
class S2VT:
    def __init__(self, input_num, hidden_num, frame_num = 0, max_caption_len = 50, lr = 1e-4, sampling = 0.8):
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.frame_num = frame_num
        self.max_caption_len = max_caption_len
        self.learning_rate = lr
        self.sampling_prob = sampling
        self.saver = None
        self.vocab_num = None
        self.token = None
        
    def load_vocab(self):
        with open('tokenizer.pickle', 'rb') as handle:
            self.token = pickle.load(handle)
        self.vocab_num = len(self.token.word_index)
            
    def build_model(self, feat, cap=None, cap_len=None, isTrain=True):
        W_top = tf.Variable(tf.random_uniform([self.input_num, self.hidden_num],-0.1,0.1), name='W_top')
        b_top = tf.Variable(tf.zeros([self.hidden_num]), name='b_top')
        W_btm = tf.Variable(tf.random_uniform([self.hidden_num,self.vocab_num],-0.1,0.1), name='W_btm')
        b_btm = tf.Variable(tf.zeros([self.vocab_num]),name='b_btm')
        embedding = tf.Variable(tf.random_uniform([self.vocab_num,self.hidden_num],-0.1,0.1), name='Embedding')
        batch_size = tf.shape(feat)[0]
        
        with tf.variable_scope('LSTMTop'):
            lstm_top = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_num, forget_bias=1.0, state_is_tuple=True)
            if isTrain:
                lstm_top = tf.contrib.rnn.DropoutWrapper(lstm_top, output_keep_prob=0.5)    
        with tf.variable_scope('LSTMBottom'):
            lstm_btm = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_num, forget_bias=1.0, state_is_tuple=True)
            if isTrain:
                lstm_btm = tf.contrib.rnn.DropoutWrapper(lstm_btm, output_keep_prob=0.5)
                
        if isTrain:
            feat = tf.nn.dropout(feat,0.5)
            cap_mask = tf.sequence_mask(cap_len,self.max_caption_len, dtype=tf.float32)
        feat = tf.reshape(feat,[-1,self.input_num])
        img_emb = tf.add(tf.matmul(feat,W_top),b_top)
        img_emb = tf.transpose(tf.reshape(img_emb,[-1, self.frame_num, self.hidden_num]),perm=[1,0,2])
        # [batch, frame_num, hidden_num] -> [frame_num, batch, hidden_num] 
                
        h_top = lstm_top.zero_state(batch_size, dtype=tf.float32)
        h_btm = lstm_top.zero_state(batch_size, dtype=tf.float32)
        
        pad = tf.ones([batch_size, self.hidden_num])*self.token.texts_to_sequences(['<PAD>'])[0][0]
        
        for i in range(frame_num):
            with tf.variable_scope('LSTMTop'):
                output_top, h_top = lstm_top(img_emb[i,:,:],h_top)
            with tf.variable_scope('LSTMBottom'):
                output_btm, h_btm = lstm_btm(tf.concat([pad,output_top],axis=1),h_top)
                
        logit = None
        logit_list = []
        cross_entropy_list = []
        
        
        for i in range(0, self.max_caption_len):
            with tf.variable_scope('LSTMTop'):
                output_top, h_top = lstm_top(pad, h_top)

            if i == 0:
                with tf.variable_scope('LSTMBottom'):
                    bos = tf.ones([batch_size, self.hidden_num])*self.token.texts_to_sequences(['<BOS>'])[0][0]
                    bos_btm_input = tf.concat([bos, output_top], axis=1)
                    output_btm, h_btm = lstm_btm(bos_btm_input, h_btm)
            else:
                if isTrain:
                    if np.random.uniform(0,1,1) < self.sampling_prob:
                        input_btm = cap[:,i-1]
                    else:
                        input_btm = tf.argmax(logit, 1)
                else:
                    input_btm = tf.argmax(logit, 1)
                btm_emb = tf.nn.embedding_lookup(embedding, input_btm)
                with tf.variable_scope('LSTMBottom'):
                    input_btm_emb = tf.concat([btm_emb, output_top], axis=1)
                    output_btm, h_btm = lstm_btm(input_btm_emb, h_btm)
                    
            logit = tf.add(tf.matmul(output_btm, W_btm), b_btm)
            logit_list.append(logit)
            
            if isTrain:
                labels = cap[:, i]
                one_hot_labels = tf.one_hot(labels, self.vocab_num, on_value = 1, off_value = None, axis = 1) 
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=one_hot_labels)
                cross_entropy = cross_entropy * cap_mask[:, i]
                cross_entropy_list.append(cross_entropy)
        
        if isTrain:
            cross_entropy_list = tf.stack(cross_entropy_list, 1)
            loss = tf.reduce_sum(cross_entropy_list, axis=1)
            loss = tf.divide(loss, tf.cast(cap_len, tf.float32))
            loss = tf.reduce_mean(loss, axis=0)

        logit_list = tf.stack(logit_list, axis = 0)
        logit_list = tf.reshape(logit_list, (self.max_caption_len, batch_size, self.vocab_num))
        logit_list = tf.transpose(logit_list, [1, 0, 2])
        # [max_cap, batch, vocab_num] -> [batch, max_cap, vocab_num]
        
        if isTrain:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = optimizer.minimize(loss)
        else:
            train_op = None
            loss = None
            
        pred_op = tf.argmax(logit_list, axis=2)
        return train_op, loss, pred_op, logit_list

def train():
    ## Loading training data and build vocabulary
    data_train = dataset(batch_size,'./training_data/feat/','./training_label.json')
    data_train.generate_token()
    data_train.process_data()
    data_train.save_vocab()
    
    graph_train = tf.Graph()
    with graph_train.as_default():
        model = S2VT(input_num,hidden_num,frame_num)
        model.load_vocab()
        feat = tf.placeholder(tf.float32, [None, frame_num, input_num], name='features')
        cap = tf.placeholder(tf.int32, [None, 50], name='caption')
        cap_len = tf.placeholder(tf.int32, [None], name='captionLength')
        train_op, loss_op, pred_op, logit_list_op = model.build_model(feat, cap, cap_len, True)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=3)
    sess_train = tf.Session(graph=graph_train)
    
    training_loss = []
    batch_num = int(math.ceil(data_train.data_size/batch_size))
    sess_train.run(init)
    for epoch in range(epoch_num):
        data_train.shuffle()
        for i in range(batch_num):
            id_batch, feat_batch, cap_batch, cap_len_batch,  = data_train.next_batch()
            sess_train.run(train_op,feed_dict={feat:feat_batch,cap:cap_batch,cap_len:cap_len_batch})
        loss = sess_train.run(loss_op,feed_dict={feat:feat_batch,cap:cap_batch,cap_len:cap_len_batch})
        training_loss.append(loss)
        print("Epoch: ",epoch," Loss: ",loss)
    model_path = saver.save(sess_train, './save/model', global_step=epoch_num*batch_num)
    print("Model saved: ",model_path)
    
def test(test_dict):
    print('Loading testing data in provided directory:')
    print(test_dict,'\n')
    data_test = dataset(batch_size,test_dict)
    data_test.load_token()
    data_test.process_feature_data() 
    print('Test data loaded successfully.')
    
    graph_test = tf.Graph()
    with graph_test.as_default():
        model = S2VT(input_num,hidden_num,frame_num)
        model.load_vocab()
        feat = tf.placeholder(tf.float32, [None, frame_num, input_num], name='features')
        _, _, pred_op, logit_list_op = model.build_model(feat, isTrain=False)
        saver = tf.train.Saver(max_to_keep=3)
    sess_test = tf.Session(graph=graph_test)
    
    cur_dict = os.getcwd()
    print('Restoring model from:')
    print(cur_dict,'/save/\n')
    latest_checkpoint = tf.train.latest_checkpoint(cur_dict+'/save/')
    saver.restore(sess_test, latest_checkpoint)
    print('Model restoration completed.')
    
    txt = open(cur_dict+'output_testset.txt', 'w')
    batch_num = int(math.ceil(data_test.data_size/batch_size))
    eos = model.token.texts_to_sequences(['<EOS>'])[0][0]
    eos_idx = model.max_caption_len
    for i in range(batch_num):
        id_batch, feat_batch = data_test.next_feature_batch()
        prediction = sess_test.run(pred_op,feed_dict={feat:feat_batch})
        for j in range(len(feat_batch)):
            for k in range(model.max_caption_len):
                if prediction[j][k]== eos:
                    eos_idx = k
                    break
            cap_output = model.token.sequences_to_texts([prediction[j][0:eos_idx]])[0]
            txt.write(id_batch[j] + "," + str(cap_output) + "\n")
    txt.close()
    print('Testing Output Generated.')
    
if __name__ == '__main__':
    test_dict = sys.argv[1]
    #test_dict = './testing_data/feat/'
    test(test_dict)