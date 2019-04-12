import tensorflow as tf 
import numpy as np 
import pickle as pkl
from sklearn.manifold import TSNE
from scipy.misc import imsave
import sys
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

tf.set_random_seed(0);
random.seed(0);

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 3], name="input")
X_target = tf.placeholder(tf.float32,[None,28,28,3]);
domain = tf.placeholder(tf.float32, shape=[None, 1], name="domain")
src_labels = tf.placeholder(tf.float32,shape=[None,10]);
target_labels = tf.placeholder(tf.float32,shape=[None,10]);
epochs = 10;
batch_size = 128

#DCGAN 
#initializer = tf.truncated_normal_initializer(mean=0, stddev=0.02);#
initializer = tf.contrib.layers.xavier_initializer()

#tuning knobs
z_dim = 128;
enc1_z = enc2_z = int(z_dim/2.0);
enc_dec_lr = 0.001;
disc_lr = 0.0005;
cls_lr = 0.0001


def load_MNIST_data():
    # Process MNIST
    mnist_train = (mnist.train.images > 0).reshape(55000, 28, 28, 1).astype(np.uint8) * 255
    mnist_train = np.concatenate([mnist_train, mnist_train, mnist_train], 3)/255.0
    mnist_test = (mnist.test.images > 0).reshape(10000, 28, 28, 1).astype(np.uint8) * 255
    mnist_test = np.concatenate([mnist_test, mnist_test, mnist_test], 3)/255.0
    # print("MNIST meta data");
    # print("mnist_train : ",mnist_train.shape);
    # print("mnist_test : ",mnist_test.shape);
    return mnist_train, mnist_test;

def load_m_mnist_data():
    # Load MNIST-M
    mnistm = pkl.load(open('mnistm_data.pkl', 'rb'))
    mnistm_train = mnistm['train']/255.0
    mnistm_test = mnistm['test']/255.0
    mnistm_valid = mnistm['valid']/255.0

    # print("MNIST-M meta data");
    # print("mnist_m_train : ",mnistm_train.shape);
    # print("mnist_m_test : ",mnistm_test.shape);
    # print("mnist_m_val : ",mnistm_valid.shape);
    return mnistm_train, mnistm_test, mnistm_valid;



def encoder(x,isTrainable=True,reuse=False):
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables();

        #28x28x3
        conv1 = tf.layers.conv2d(x, 64, 5, strides = 2, padding="SAME", 
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv1_layer');
        conv1 = tf.layers.batch_normalization(conv1,name='conv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv1 = tf.nn.relu(conv1);

        #14x14x64
        conv2 = tf.layers.conv2d(conv1, 128, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv2_layer')
        conv2 = tf.layers.batch_normalization(conv2,name='conv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv2 = tf.nn.relu(conv2);
        #7x7xx128 
        conv3 = tf.layers.conv2d(conv2, 256, 5, strides = 2, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv3_layer')
        conv3 = tf.layers.batch_normalization(conv3,name='conv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv3 = tf.nn.relu(conv3);

        conv4 = tf.layers.conv2d(conv3, 512, 5, strides = 1, padding="SAME",
            kernel_initializer=initializer, activation=None,trainable=isTrainable,reuse=reuse,name='conv4_layer')
        conv4 = tf.layers.batch_normalization(conv4,name='conv4_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        conv4 = tf.nn.relu(conv4);

        #4x4x256
        fc = tf.contrib.layers.flatten(conv4)
        fc1 = tf.layers.dense(fc, units = enc1_z, 
            kernel_initializer= initializer, activation = tf.nn.leaky_relu,trainable=isTrainable,reuse=reuse,name='fc_layer')

        return fc1



def decoder(z,isTrainable=True,reuse=False):
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables();

        fc = tf.layers.dense(z, units = 4*4*512, 
            kernel_initializer = initializer, activation = None,trainable=isTrainable,reuse=reuse)
        fc = tf.layers.batch_normalization(fc,name='fc_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        fc = tf.nn.relu(fc);
        fl = tf.reshape(fc, [-1, 4, 4, 512]);

        #4x4x512
        deconv1 = tf.layers.conv2d_transpose(fl, 256, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv1_layer')
        deconv1 = tf.layers.batch_normalization(deconv1,name='deconv1_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        deconv1 = tf.nn.relu(deconv1);

        #print("dec_deconv1.shape : ",deconv1.shape);

        #8x8x256
        deconv2 = tf.layers.conv2d_transpose(deconv1, 128, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv2_layer')
        deconv2 = tf.layers.batch_normalization(deconv2,name='deconv2_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        deconv2 = tf.nn.relu(deconv2);


        #16x16x128
        deconv3 = tf.layers.conv2d_transpose(deconv2, 64, 5, strides=2, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv3_layer')
        deconv3 = tf.layers.batch_normalization(deconv3,name='deconv3_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        deconv3 = tf.nn.relu(deconv3);


        #32x32x64
        deconv4 = tf.layers.conv2d_transpose(deconv3, 16, 5, strides=1, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv4_layer')
        deconv4 = tf.layers.batch_normalization(deconv4,name='deconv4_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);
        deconv4 = tf.nn.relu(deconv4);

        #32x32x16
        deconv5 = tf.layers.conv2d_transpose(deconv4, 3, 5, strides=1, padding="SAME", 
            kernel_initializer=initializer, activation = None,trainable=isTrainable,reuse=reuse,name='deconv5_layer')
        deconv5 = tf.layers.batch_normalization(deconv5,name='deconv5_layer_batchnorm',
            trainable=isTrainable,reuse=reuse);

        deconv5 = tf.nn.relu(deconv5);
        #32x32x3
        deconv5 = tf.layers.flatten(deconv5);
        deconv_fc = tf.layers.dense(deconv5,28*28*3,activation=tf.nn.sigmoid,trainable=isTrainable,reuse=reuse,name='dec_dense_fc_last_layer');

        deconv_fc_reshaped = tf.reshape(deconv_fc,[-1,28,28,3]); 

        return deconv_fc_reshaped;



latent_code = encoder(X)
X_tilde = decoder(latent_code)

recon_loss = tf.reduce_mean(tf.square(X - X_tilde));


enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='encoder');
dec_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='decoder');


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS);
with tf.control_dependencies(update_ops):
    recon_op = tf.train.AdamOptimizer(learning_rate = enc_dec_lr).minimize(recon_loss, var_list = enc_params+dec_params);
     
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

def train():
    with tf.Session() as sess:
        sess.run(init)
        source_train,source_test = load_MNIST_data();
        target_train,target_test,target_val = load_m_mnist_data();

        print('source_train.shape : ',source_train.shape);
        print('target_train.shape : ',target_train.shape);

        source_train_labels = mnist.train.labels
        source_test_labels = mnist.test.labels

        target_train_labels = mnist.train.labels
        target_test_labels = mnist.test.labels


        recon = []

        target_len = len(target_train);
        source_len = len(source_train);

        source_len_test = len(source_test);
        target_len_val = len(target_val);

        print('-'*80);
        print('target_len : ',target_len);
        print('source_len : ',source_len);
        print('-'*80);

        iterations = 0;

        _stin = [];

        n_batches = int(1.0*source_len/batch_size);
        print('n_batches : ',n_batches);
        iterations = 0;
        for epoch in range(epochs):
        	print(epoch)
        	for batch in range(n_batches):
        		random_indexes = np.random.choice(source_len, batch_size)
        		src_x = source_train[random_indexes]
        		src_y = source_train_labels[random_indexes]

        		fd = {X:src_x}

        		_, slc = sess.run([recon_op, latent_code], feed_dict = fd)

        		random_indexes = np.random.choice(target_len, batch_size)
        		tar_x = target_train[random_indexes]
        		tar_y = target_train_labels[random_indexes]

        		fd = {X: tar_x}

        		_, tlc = sess.run([recon_op, latent_code], feed_dict = fd)



        random_indexes = np.random.choice(source_len, batch_size)
        src_x = source_train[random_indexes]
        src_y = source_train_labels[random_indexes]

        fd = {X:src_x}

        slc = sess.run(latent_code, feed_dict = fd)

        random_indexes = np.random.choice(target_len, batch_size)
        tar_x = target_train[random_indexes]
        tar_y = target_train_labels[random_indexes]

        fd = {X: tar_x}

        tlc = sess.run(latent_code, feed_dict = fd)

        vectors = np.concatenate((slc, tlc), axis=0)
        labels = np.concatenate((np.ones(batch_size), np.zeros(batch_size)), axis=0)

        np.save("vectors.npy", vectors)
        np.save("labels.npy", labels)


train()