
# coding: utf-8

# # ** CPSC 8810 Deep Learning - HW1-3 **
# ---
# 
# ## Introduction
# _**Note:** This assignment makes use of the MNIST dataset_
# 
# The main objective of this assignments:
# * Fit network with random labels
# * Compare number of parameters vs generalization
# * Compare flatness vs generalization

# # Flatness vs Generalization

# In[2]:


import tensorflow as tf
import cv2
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA

tf.__version__
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ## MNIST Dataset Preparation and Visualization

# In[3]:


data = input_data.read_data_sets('data/MNIST/', one_hot=True);

train_num = data.train.num_examples;
valid_num = data.validation.num_examples;
test_num = data.test.num_examples;
img_flatten = 784
img_size = 28
num_classes = 10
print("Training Dataset Size:",train_num)
print("Validation Dataset Size:",valid_num)
print("Testing Dataset Size:",test_num)


# ### Functions to acquire weights

# In[4]:


def get_weights_variable(layer_name):
    # Retrieve an existing variable named 'kernel' in the scope
    # with the given layer_name.
    # This is awkward because the TensorFlow function was
    # really intended for another purpose.
    with tf.variable_scope(layer_name, reuse=True):
        weights = tf.get_variable('kernel')
    return weights

def get_bias_variable(layer_name):
    with tf.variable_scope(layer_name,reuse=True):
        bias = tf.get_variable('bias')
    return bias


# ___
# # Batch Size Comparison

# ## Model 1&2 Architecture

# In[5]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

### Model 1 Architecture
m1_conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv1');
m1_pool1 = tf.layers.max_pooling2d(inputs=m1_conv1,pool_size=2,strides=2);
m1_conv2 = tf.layers.conv2d(inputs=m1_pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv2');
m1_pool2 = tf.layers.max_pooling2d(inputs=m1_conv2,pool_size=2,strides=2);
m1_flat1 = tf.layers.flatten(m1_pool2);
m1_fc1 = tf.layers.dense(inputs=m1_flat1,units=128,activation=tf.nn.relu,name='m1_fc1');
m1_logits = tf.layers.dense(inputs=m1_fc1,units=num_classes,activation=None,name='m1_fc_out');
m1_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m1_logits);
m1_loss = tf.reduce_mean(m1_cross_entropy);
m1_softmax = tf.nn.softmax(logits=m1_logits);
m1_pred_op = tf.argmax(m1_softmax,dimension=1);
m1_acc_op = tf.reduce_mean(tf.cast(tf.equal(m1_pred_op, y_cls), tf.float32));
m1_optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
m1_train_op = m1_optimizer.minimize(m1_loss);

### Model 2 Architecture
m2_conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu,name='m2_conv1');
m2_pool1 = tf.layers.max_pooling2d(inputs=m2_conv1,pool_size=2,strides=2);
m2_conv2 = tf.layers.conv2d(inputs=m2_pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m2_conv2');
m2_pool2 = tf.layers.max_pooling2d(inputs=m2_conv2,pool_size=2,strides=2);
m2_flat1 = tf.layers.flatten(m2_pool2);
m2_fc1 = tf.layers.dense(inputs=m2_flat1,units=128,activation=tf.nn.relu,name='m2_fc1');
m2_logits = tf.layers.dense(inputs=m2_fc1,units=num_classes,activation=None,name='m2_fc_out');
m2_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m2_logits);
m2_loss = tf.reduce_mean(m2_cross_entropy);
m2_softmax = tf.nn.softmax(logits=m2_logits);
m2_pred_op = tf.argmax(m2_softmax,dimension=1);
m2_acc_op = tf.reduce_mean(tf.cast(tf.equal(m2_pred_op, y_cls), tf.float32));
m2_optimizer = tf.train.AdamOptimizer(learning_rate=0.001);
m2_train_op = m2_optimizer.minimize(m2_loss);


# ### Weights and Bias

# In[6]:


m1_weights_conv1 = get_weights_variable('m1_conv1')
m1_weights_conv2 = get_weights_variable('m1_conv2')
m1_weights_fc1 = get_weights_variable('m1_fc1')
m1_weights_fc_out = get_weights_variable('m1_fc_out')

m1_bias_conv1 = get_bias_variable('m1_conv1')
m1_bias_conv2 = get_bias_variable('m1_conv2')
m1_bias_fc1 = get_bias_variable('m1_fc1')
m1_bias_fc_out = get_bias_variable('m1_fc_out')

m2_weights_conv1 = get_weights_variable('m2_conv1')
m2_weights_conv2 = get_weights_variable('m2_conv2')
m2_weights_fc1 = get_weights_variable('m2_fc1')
m2_weights_fc_out = get_weights_variable('m2_fc_out')

m2_bias_conv1 = get_bias_variable('m2_conv1')
m2_bias_conv2 = get_bias_variable('m2_conv2')
m2_bias_fc1 = get_bias_variable('m2_fc1')
m2_bias_fc_out = get_bias_variable('m2_fc_out')


# ### Training Model 1

# In[7]:


session = tf.Session()
session.run(tf.global_variables_initializer())

train_loss_list1 = []
train_acc_list1 = []
test_loss_list1 = []
test_acc_list1 = []

BATCH_SIZE = 64
EPOCH = 1
for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(m1_train_op, feed_dict={x: x_batch,y: y_true_batch})
    train_loss, train_acc = session.run([m1_loss,m1_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list1.append(train_loss)
    train_acc_list1.append(train_acc)
    test_loss, test_acc = session.run([m1_loss,m1_acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list1.append(test_loss)
    test_acc_list1.append(test_acc)
    msg = "Epoch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))

m1_w_conv1,m1_w_conv2,m1_w_fc1,m1_w_fc_out = session.run([m1_weights_conv1,m1_weights_conv2,m1_weights_fc1,m1_weights_fc_out])
m1_b_conv1,m1_b_conv2,m1_b_fc1,m1_b_fc_out = session.run([m1_bias_conv1,m1_bias_conv2,m1_bias_fc1,m1_bias_fc_out])


# ### Training Model 2

# In[8]:


train_loss_list2 = []
train_acc_list2 = []
test_loss_list2 = []
test_acc_list2 = []

BATCH_SIZE = 1024
EPOCH = 1
for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(m2_train_op, feed_dict={x: x_batch,y: y_true_batch})
    train_loss, train_acc = session.run([m2_loss,m2_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list1.append(train_loss)
    train_acc_list1.append(train_acc)
    test_loss, test_acc = session.run([m2_loss,m2_acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list1.append(test_loss)
    test_acc_list1.append(test_acc)
    msg = "Epoch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))
    
m2_w_conv1,m2_w_conv2,m2_w_fc1,m2_w_fc_out = session.run([m2_weights_conv1,m2_weights_conv2,m2_weights_fc1,m2_weights_fc_out])
m2_b_conv1,m2_b_conv2,m2_b_fc1,m2_b_fc_out = session.run([m2_bias_conv1,m2_bias_conv2,m2_bias_fc1,m2_bias_fc_out])


# ### Run model 3

# In[9]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

m3_w_conv1 = tf.placeholder(tf.float32,shape=[5,5,1,8])
m3_b_conv1 = tf.placeholder(tf.float32,shape=[8])
m3_w_conv2 = tf.placeholder(tf.float32,shape=[5,5,8,16])
m3_b_conv2 = tf.placeholder(tf.float32,shape=[16])
m3_w_fc1 = tf.placeholder(tf.float32,shape=[784,128])
m3_b_fc1 = tf.placeholder(tf.float32,shape=[128])
m3_w_fc_out = tf.placeholder(tf.float32,shape=[128,10])
m3_b_fc_out = tf.placeholder(tf.float32,shape=[10])

m3_conv1 = tf.nn.conv2d(input_x,filters=m3_w_conv1,padding='SAME')
m3_conv1 = tf.nn.relu(tf.nn.bias_add(m3_conv1,m3_b_conv1))
m3_pool1 = tf.nn.max_pool2d(m3_conv1,2,2,'SAME')
m3_conv2 = tf.nn.conv2d(m3_pool1,m3_w_conv2,padding='SAME')
m3_conv2 = tf.nn.relu(tf.nn.bias_add(m3_conv2,m3_b_conv2))
m3_pool2 = tf.nn.max_pool2d(m3_conv2,2,2,'SAME')
m3_flat1 = tf.layers.flatten(m3_pool2)
m3_fc1 = tf.nn.relu(tf.add(tf.matmul(m3_flat1,m3_w_fc1),m3_b_fc1))
m3_logits = tf.add(tf.matmul(m3_fc1,m3_w_fc_out),m3_b_fc_out)

m3_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m3_logits)
m3_loss = tf.reduce_mean(m3_cross_entropy)

# Accuracy
m3_softmax = tf.nn.softmax(logits=m3_logits)
m3_pred_op = tf.argmax(m3_softmax,dimension=1)
m3_acc_op = tf.reduce_mean(tf.cast(tf.equal(m3_pred_op, y_cls), tf.float32))

session = tf.Session()
for alpha in np.arange(-1,2,0.05):
    w_conv1 = (1-alpha)*m1_w_conv1 + alpha*m2_w_conv1
    w_conv2 = (1-alpha)*m1_w_conv2 + alpha*m2_w_conv2
    w_fc1 = (1-alpha)*m1_w_fc1 + alpha*m2_w_fc1
    w_fc_out = (1-alpha)*m1_w_fc_out + alpha*m2_w_fc_out
    
    b_conv1 = (1-alpha)*m1_b_conv1 + alpha*m2_b_conv1
    b_conv2 = (1-alpha)*m1_b_conv2 + alpha*m2_b_conv2
    b_fc1 = (1-alpha)*m1_b_fc1 + alpha*m2_b_fc1
    b_fc_out = (1-alpha)*m1_b_fc_out + alpha*m2_b_fc_out
    
    train_loss,train_acc = session.run([m3_loss,m3_acc_op],feed_dict={x:data.train.images,y:data.train.labels,m3_w_conv1:w_conv1, m3_w_conv2:w_conv2, m3_w_fc1:w_fc1, m3_w_fc_out:w_fc_out,m3_b_conv1:b_conv1,m3_b_conv2:b_conv2,m3_b_fc1:b_fc1,m3_b_fc_out:b_fc_out})
    test_loss,test_acc = session.run([m3_loss,m3_acc_op],feed_dict={x:data.test.images,y:data.test.labels,m3_w_conv1:w_conv1, m3_w_conv2:w_conv2, m3_w_fc1:w_fc1, m3_w_fc_out:w_fc_out,m3_b_conv1:b_conv1,m3_b_conv2:b_conv2,m3_b_fc1:b_fc1,m3_b_fc_out:b_fc_out})
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


# ### Result

# In[10]:


fig,axs1 = plt.subplots()
fig.suptitle('Batch Size: 64 vs 1024')
alpha_list = np.arange(-1,2,0.05)
axs1.plot(alpha_list,test_acc_list,'r')
axs1.plot(alpha_list,train_acc_list,'r--')
axs1.legend(['test','train'])
axs1.set_ylabel('Accuracy')
axs1.yaxis.label.set_color('red')
axs2 = axs1.twinx()
axs2.plot(alpha_list,test_loss_list,'b')
axs2.plot(alpha_list,train_loss_list,'b--')
axs2.set_yscale('log')
axs2.set_ylabel('Cross Entropy Loss (Log Scale)')
axs2.yaxis.label.set_color('blue')


# ___
# # Learning Rate Comparison

# ## Model 1&2 Architecture

# In[11]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

### Model 1 Architecture
m1_conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv1');
m1_pool1 = tf.layers.max_pooling2d(inputs=m1_conv1,pool_size=2,strides=2);
m1_conv2 = tf.layers.conv2d(inputs=m1_pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m1_conv2');
m1_pool2 = tf.layers.max_pooling2d(inputs=m1_conv2,pool_size=2,strides=2);
m1_flat1 = tf.layers.flatten(m1_pool2);
m1_fc1 = tf.layers.dense(inputs=m1_flat1,units=128,activation=tf.nn.relu,name='m1_fc1');
m1_logits = tf.layers.dense(inputs=m1_fc1,units=num_classes,activation=None,name='m1_fc_out');
m1_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m1_logits);
m1_loss = tf.reduce_mean(m1_cross_entropy);
m1_softmax = tf.nn.softmax(logits=m1_logits);
m1_pred_op = tf.argmax(m1_softmax,dimension=1);
m1_acc_op = tf.reduce_mean(tf.cast(tf.equal(m1_pred_op, y_cls), tf.float32));
m1_optimizer = tf.train.AdamOptimizer(learning_rate=0.01);
m1_train_op = m1_optimizer.minimize(m1_loss);

### Model 2 Architecture
m2_conv1 = tf.layers.conv2d(inputs=input_x,filters=8,kernel_size=5,padding="same",activation=tf.nn.relu,name='m2_conv1');
m2_pool1 = tf.layers.max_pooling2d(inputs=m2_conv1,pool_size=2,strides=2);
m2_conv2 = tf.layers.conv2d(inputs=m2_pool1,filters=16,kernel_size=5,padding="same",activation=tf.nn.relu,name='m2_conv2');
m2_pool2 = tf.layers.max_pooling2d(inputs=m2_conv2,pool_size=2,strides=2);
m2_flat1 = tf.layers.flatten(m2_pool2);
m2_fc1 = tf.layers.dense(inputs=m2_flat1,units=128,activation=tf.nn.relu,name='m2_fc1');
m2_logits = tf.layers.dense(inputs=m2_fc1,units=num_classes,activation=None,name='m2_fc_out');
m2_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m2_logits);
m2_loss = tf.reduce_mean(m2_cross_entropy);
m2_softmax = tf.nn.softmax(logits=m2_logits);
m2_pred_op = tf.argmax(m2_softmax,dimension=1);
m2_acc_op = tf.reduce_mean(tf.cast(tf.equal(m2_pred_op, y_cls), tf.float32));
m2_optimizer = tf.train.AdamOptimizer(learning_rate=0.00001);
m2_train_op = m2_optimizer.minimize(m2_loss);


# ### Weights & Bias

# In[12]:


m1_weights_conv1 = get_weights_variable('m1_conv1')
m1_weights_conv2 = get_weights_variable('m1_conv2')
m1_weights_fc1 = get_weights_variable('m1_fc1')
m1_weights_fc_out = get_weights_variable('m1_fc_out')

m1_bias_conv1 = get_bias_variable('m1_conv1')
m1_bias_conv2 = get_bias_variable('m1_conv2')
m1_bias_fc1 = get_bias_variable('m1_fc1')
m1_bias_fc_out = get_bias_variable('m1_fc_out')

m2_weights_conv1 = get_weights_variable('m2_conv1')
m2_weights_conv2 = get_weights_variable('m2_conv2')
m2_weights_fc1 = get_weights_variable('m2_fc1')
m2_weights_fc_out = get_weights_variable('m2_fc_out')

m2_bias_conv1 = get_bias_variable('m2_conv1')
m2_bias_conv2 = get_bias_variable('m2_conv2')
m2_bias_fc1 = get_bias_variable('m2_fc1')
m2_bias_fc_out = get_bias_variable('m2_fc_out')


# ### Training model 1

# In[13]:


session = tf.Session()
session.run(tf.global_variables_initializer())

train_loss_list1 = []
train_acc_list1 = []
test_loss_list1 = []
test_acc_list1 = []

BATCH_SIZE = 64
EPOCH = 1
for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(m1_train_op, feed_dict={x: x_batch,y: y_true_batch})
    train_loss, train_acc = session.run([m1_loss,m1_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list1.append(train_loss)
    train_acc_list1.append(train_acc)
    test_loss, test_acc = session.run([m1_loss,m1_acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list1.append(test_loss)
    test_acc_list1.append(test_acc)
    msg = "Epoch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))

m1_w_conv1,m1_w_conv2,m1_w_fc1,m1_w_fc_out = session.run([m1_weights_conv1,m1_weights_conv2,m1_weights_fc1,m1_weights_fc_out])
m1_b_conv1,m1_b_conv2,m1_b_fc1,m1_b_fc_out = session.run([m1_bias_conv1,m1_bias_conv2,m1_bias_fc1,m1_bias_fc_out])


# ### Training model 2

# In[14]:


train_loss_list2 = []
train_acc_list2 = []
test_loss_list2 = []
test_acc_list2 = []

BATCH_SIZE = 64
EPOCH = 1
for i in range(EPOCH):
    for j in range(int(data.train.num_examples/BATCH_SIZE)):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        session.run(m2_train_op, feed_dict={x: x_batch,y: y_true_batch})
    train_loss, train_acc = session.run([m2_loss,m2_acc_op],feed_dict={x:x_batch,y:y_true_batch})
    train_loss_list1.append(train_loss)
    train_acc_list1.append(train_acc)
    test_loss, test_acc = session.run([m2_loss,m2_acc_op],feed_dict={x:data.test.images,y:data.test.labels})
    test_loss_list1.append(test_loss)
    test_acc_list1.append(test_acc)
    msg = "Epoch: {0:>6}, Training Loss: {1:>1.6}, Training Accuracy: {2:>6.1%}, Test Loss: {3:>1.6}, Test Accuracy: {4:>6.1%}"
    print(msg.format(i, train_loss, train_acc, test_loss, test_acc))
    
m2_w_conv1,m2_w_conv2,m2_w_fc1,m2_w_fc_out = session.run([m2_weights_conv1,m2_weights_conv2,m2_weights_fc1,m2_weights_fc_out])
m2_b_conv1,m2_b_conv2,m2_b_fc1,m2_b_fc_out = session.run([m2_bias_conv1,m2_bias_conv2,m2_bias_fc1,m2_bias_fc_out])


# ### Run Model 3

# In[15]:


tf.reset_default_graph()
x = tf.placeholder(tf.float32, shape=[None, img_flatten], name='x')
input_x = tf.reshape(x,[-1,img_size,img_size,1])
y = tf.placeholder(tf.float32, shape=[None, num_classes], name='y')
y_cls = tf.argmax(y,dimension=1)

train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

m3_w_conv1 = tf.placeholder(tf.float32,shape=[5,5,1,8])
m3_b_conv1 = tf.placeholder(tf.float32,shape=[8])
m3_w_conv2 = tf.placeholder(tf.float32,shape=[5,5,8,16])
m3_b_conv2 = tf.placeholder(tf.float32,shape=[16])
m3_w_fc1 = tf.placeholder(tf.float32,shape=[784,128])
m3_b_fc1 = tf.placeholder(tf.float32,shape=[128])
m3_w_fc_out = tf.placeholder(tf.float32,shape=[128,10])
m3_b_fc_out = tf.placeholder(tf.float32,shape=[10])

m3_conv1 = tf.nn.conv2d(input_x,filters=m3_w_conv1,padding='SAME')
m3_conv1 = tf.nn.relu(tf.nn.bias_add(m3_conv1,m3_b_conv1))
m3_pool1 = tf.nn.max_pool2d(m3_conv1,2,2,'SAME')
m3_conv2 = tf.nn.conv2d(m3_pool1,m3_w_conv2,padding='SAME')
m3_conv2 = tf.nn.relu(tf.nn.bias_add(m3_conv2,m3_b_conv2))
m3_pool2 = tf.nn.max_pool2d(m3_conv2,2,2,'SAME')
m3_flat1 = tf.layers.flatten(m3_pool2)
m3_fc1 = tf.nn.relu(tf.add(tf.matmul(m3_flat1,m3_w_fc1),m3_b_fc1))
m3_logits = tf.add(tf.matmul(m3_fc1,m3_w_fc_out),m3_b_fc_out)

m3_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=m3_logits)
m3_loss = tf.reduce_mean(m3_cross_entropy)

# Accuracy
m3_softmax = tf.nn.softmax(logits=m3_logits)
m3_pred_op = tf.argmax(m3_softmax,dimension=1)
m3_acc_op = tf.reduce_mean(tf.cast(tf.equal(m3_pred_op, y_cls), tf.float32))

session = tf.Session()
for alpha in np.arange(-1,2,0.05):
    w_conv1 = (1-alpha)*m1_w_conv1 + alpha*m2_w_conv1
    w_conv2 = (1-alpha)*m1_w_conv2 + alpha*m2_w_conv2
    w_fc1 = (1-alpha)*m1_w_fc1 + alpha*m2_w_fc1
    w_fc_out = (1-alpha)*m1_w_fc_out + alpha*m2_w_fc_out
    
    b_conv1 = (1-alpha)*m1_b_conv1 + alpha*m2_b_conv1
    b_conv2 = (1-alpha)*m1_b_conv2 + alpha*m2_b_conv2
    b_fc1 = (1-alpha)*m1_b_fc1 + alpha*m2_b_fc1
    b_fc_out = (1-alpha)*m1_b_fc_out + alpha*m2_b_fc_out
    
    train_loss,train_acc = session.run([m3_loss,m3_acc_op],feed_dict={x:data.train.images,y:data.train.labels,m3_w_conv1:w_conv1, m3_w_conv2:w_conv2, m3_w_fc1:w_fc1, m3_w_fc_out:w_fc_out,m3_b_conv1:b_conv1,m3_b_conv2:b_conv2,m3_b_fc1:b_fc1,m3_b_fc_out:b_fc_out})
    test_loss,test_acc = session.run([m3_loss,m3_acc_op],feed_dict={x:data.test.images,y:data.test.labels,m3_w_conv1:w_conv1, m3_w_conv2:w_conv2, m3_w_fc1:w_fc1, m3_w_fc_out:w_fc_out,m3_b_conv1:b_conv1,m3_b_conv2:b_conv2,m3_b_fc1:b_fc1,m3_b_fc_out:b_fc_out})
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


# ### Result

# In[16]:


fig,axs1 = plt.subplots()
fig.suptitle('Learning Rate: 0.01 vs 0.00001')
alpha_list = np.arange(-1,2,0.05)
axs1.plot(alpha_list,test_acc_list,'r')
axs1.plot(alpha_list,train_acc_list,'r--')
axs1.legend(['test','train'])
axs1.set_ylabel('Accuracy')
axs1.yaxis.label.set_color('red')
axs2 = axs1.twinx()
axs2.plot(alpha_list,test_loss_list,'b')
axs2.plot(alpha_list,train_loss_list,'b--')
axs2.set_yscale('log')
axs2.set_ylabel('Cross Entropy Loss (Log Scale)')
axs2.yaxis.label.set_color('blue')

