"""
!/usr/bin/env python
-*- coding:utf-8 -*-
Author: eric.lai
Time: 2019/2/25 13:43
"""

import tensorflow as tf
import numpy as np
import h5py,random,time


class LeNet_Mode():
    """ create LeNet network use tensorflow
        LeNet network structure:
        (conv 5x5 32 ,pool/2)
        (conv 5x5 64, pool/2)
        (fc 100)=>=>(fc classes)
    """
    def conv_layer(self, data, ksize, stride, name, w_biases = False,padding = "SAME"):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            w_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable(name= name,shape= ksize, initializer= w_init)
            biases = tf.Variable(tf.constant(0.0, shape=[ksize[3]], dtype=tf.float32), 'biases')
        if w_biases == False:
            cov = tf.nn.conv2d(input= data, filter= w, strides= stride, padding= padding)
        else:
            cov = tf.nn.conv2d(input= data,filter= w, stride= stride,padding= padding) + biases
        return cov

    def pool_layer(self, data, ksize, stride, name, padding= 'VALID'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            max_pool =  tf.nn.max_pool(value= data, ksize= ksize, strides= stride,padding= padding)
        return max_pool

    def flatten(self,data):
        [a,b,c,d] = data.get_shape().as_list()
        ft = tf.reshape(data,[-1,b*c*d])
        return ft

    def fc_layer(self,data,name,fc_dims):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            data_shape = data.get_shape().as_list()
            w_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable(shape=[data_shape[1],fc_dims],name= 'w',initializer=w_init)
            # w = tf.Variable(tf.truncated_normal([data_shape[1], fc_dims], stddev=0.01),'w')
            biases = tf.Variable(tf.constant(0.0, shape=[fc_dims], dtype=tf.float32), 'biases')
            fc = tf.nn.relu(tf.matmul(data,w)+ biases)
        return fc

    def finlaout_layer(self,data,name,fc_dims):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            w_init = tf.contrib.layers.xavier_initializer()
            w = tf.get_variable(shape=[data.shape[1],fc_dims],name= 'w',initializer=w_init)
            biases = tf.Variable(tf.constant(0.0, shape=[fc_dims], dtype=tf.float32), 'biases')
            # fc = tf.nn.softmax(tf.matmul(data,w)+ biases)
            fc = tf.matmul(data,w)+biases
        return fc

    def model_bulid(self, height, width, channel,classes):
        x = tf.placeholder(dtype= tf.float32, shape = [None,height,width,channel])
        y = tf.placeholder(dtype= tf.float32 ,shape=[None,classes])

        # conv 1 ,if image Nx19x12x1 ,(conv 3x3 8 ,pool/2)
        conv1_1 = tf.nn.relu(self.conv_layer(x,ksize=[3,3,channel,16],stride=[1,1,1,1],padding="SAME",name="conv1_1")) # Nx19x12x1 ==>   Nx19x12x8
        pool1_1 = self.pool_layer(conv1_1,ksize=[1,2,2,1],stride=[1,2,2,1],name="pool1_1") # Nx9x6x8

        # conv 2,(conv 3x3 8)=>(conv 3x3 16, pool/2)
        conv2_1 = tf.nn.relu(self.conv_layer(pool1_1,ksize=[3,3,16,32],stride=[1,1,1,1],padding="SAME",name="conv2_1"))
        pool2_1 = self.pool_layer(conv2_1,ksize=[1,2,2,1],stride=[1,2,2,1],name="pool2_1") # Nx9x6x16

        # Flatten
        ft = self.flatten(pool2_1)

        # Dense layer,(fc 100)=>=>(fc classes)
        fc1 = self.fc_layer(ft,fc_dims=10,name="fc1")
        finaloutput = self.finlaout_layer(fc1,fc_dims=4,name="final")

        # cost
        loss = tf.losses.softmax_cross_entropy(y,finaloutput)
        # loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(finaloutput),reduction_indices=[1]))

        # optimize
        LEARNING_RATE_BASE = 0.001
        LEARNING_RATE_DECAY = 0.1
        LEARNING_RATE_STEP = 300
        gloabl_steps = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE
                                                   , gloabl_steps,
                                                   LEARNING_RATE_STEP,
                                                   LEARNING_RATE_DECAY,
                                                   staircase=True)
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            optimize = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        # prediction
        prediction_label = finaloutput
        correct_prediction = tf.equal(tf.argmax(prediction_label,1),tf.argmax(y,1))
        accurary = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.float32))
        correct_times_in_batch = tf.reduce_mean(tf.cast(correct_prediction,dtype=tf.int32))
        return dict(
            x=x,
            y=y,
            optimize=optimize,
            correct_prediction=prediction_label,
            correct_times_in_batch=correct_times_in_batch,
            cost=loss,
            accurary = accurary
        )

    def init_sess(self):
        init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)

    def train_network(self,graph,x_train,y_train):
        # Tensorfolw Adding more and more nodes to the previous graph results in a larger and larger memory footprint
        # reset graph
        tf.reset_default_graph()
        self.sess.run(graph['optimize'],feed_dict={graph['x']:x_train, graph['y']:y_train})
        # print("cost: ",self.sess.run(graph['cost'],feed_dict={graph['x']:x_train, graph['y']:y_train}))
        # print("accurary: ",self.sess.run(graph['accurary'],feed_dict={graph['x']:x_train, graph['y']:y_train}))

    def read_h5py(self,filename):
        with h5py.File(filename, 'r') as f:
            # def prtname(name):
            #     print(name)
            # f.visit(prtname)
            subgroup = f['subgroup']
            data1 = subgroup['feature']
            data2 = subgroup['label']
            feature = data1.value
            label = data2.value
            # print("data1 feature:", feature)
            print("data1 label:", label)
        return feature, label

    def load_data(self):
        feature, label = self.read_h5py('feature.h5')
        print(feature.shape, label.shape)
        feature_all = []
        label_all = []
        for i in range(len(label)):
            if label[i] == 0:
                feature_all.append(feature[i])
                label_all.append(label[i])
            if label[i] == 1:
                feature_all.append(feature[i])
                label_all.append(label[i])
            if label[i] == 2:
                feature_all.append(feature[i])
                label_all.append(label[i])
            if label[i] == 9:
                feature_all.append(feature[i])
                label_all.append(label[i])
        # print(np.shape(feature_all))
        print(len(label_all))
        id = np.arange(0, len(label_all))
        random.shuffle(id)
        in_feature = []
        in_label = []
        for i in range(len(id)):
            in_feature.append(feature_all[id[i]])
            in_label.append(label_all[id[i]])

        # one-hot
        one_label = np.zeros(shape=(len(in_label),4))
        for i in range(len(in_label)):
            if in_label[i]==9:
                one_label[i][3] = 1
            else:
                one_label[i][in_label[i]] = 1


        g = self.model_bulid(19, 12, 1, 4)
        self.init_sess()
        # Build the model first, then initialize it, just once
        start = time.time()
        N = int(len(in_label)/1000)
        for j in range(100):
            for i in range(N):
                batch_xs,batch_ys = in_feature[i*1000:(i+1)*1000],one_label[i*1000:(i+1)*1000]
                batch_xs = np.reshape(batch_xs,[-1,19,12,1])
                self.train_network(g,batch_xs,batch_ys)
                if i%10==0:
                    print("cost: ",self.sess.run(g['cost'],feed_dict={g['x']:batch_xs, g['y']:batch_ys}),"accurary: ",self.sess.run(g['accurary'],feed_dict={g['x']:batch_xs, g['y']:batch_ys}))
                    # print("correct_prediction",self.sess.run(g['correct_prediction'],feed_dict={g['x']:batch_xs,g['y']:batch_ys}))
        end = time.time()
        print(end-start,"min times")


LeNet = LeNet_Mode()
LeNet.load_data()
# LeNet.load_CIFAR_data()