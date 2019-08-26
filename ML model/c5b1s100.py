# Code modified from https://github.com/taki0112/SENet-Tensorflow/blob/master/SE_ResNeXt.py

import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from s100 import *
import numpy as np

weight_decay = 0.0005
momentum = 0.9

init_learning_rate = 0.1
cardinality = 5 # how many split ? 8
blocks = 1 # res_block ! (split + transition) 3
depth = 64 # out channel 64

"""
So, the total number of layers is (3*blokcs)*residual_layer_num + 2
because, blocks = split(conv 2) + transition(conv 1) = 3 layer
and, first conv layer 1, last dense layer 1
thus, total number of layers = (3*blocks)*residual_layer_num + 2
"""

reduction_ratio = 4

batch_size = 50
iteration = 11
# training data size is 543

test_iteration = 10

total_epochs = 100

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x) :
    return tf.nn.sigmoid(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Fully_connected(x, units=class_num, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=False, units=units)

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 6

    for it in range(test_iteration):
        if it < test_iteration - 1:
            test_batch_x = test_x[test_pre_index: test_pre_index + add]
            test_batch_y = test_y[test_pre_index: test_pre_index + add]
        else:
            test_batch_x = test_x[test_pre_index:]
            test_batch_y = test_y[test_pre_index:]

        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration # average loss
    test_acc /= test_iteration # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary

class SE_ResNeXt():
    def __init__(self, x, training):
        self.training = training
        self.model = self.Build_SEnet(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=depth, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[3,3], stride=stride, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(cardinality) :
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name) :


            squeeze = Global_Average_Pooling(input_x)

            excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
            excitation = Relu(excitation)
            excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
            excitation = Sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1,1,1,out_dim])
            scale = input_x * excitation

            return scale

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge
        # input_dim = input_x.get_shape().as_list()[-1]

        for i in range(res_block):
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))
            x = self.squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio, layer_name='squeeze_layer_'+layer_num+'_'+str(i))

            if flag is True :
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else :
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x


    def Build_SEnet(self, input_x):
        # only cifar10 architecture

        input_x = self.first_layer(input_x, scope='first_layer')

        x = self.residual_layer(input_x, out_dim=64, layer_num='1')
        x = self.residual_layer(x, out_dim=128, layer_num='2')
        x = self.residual_layer(x, out_dim=256, layer_num='3')

        x = Global_Average_Pooling(x)
        x = flatten(x)

        x = Fully_connected(x, layer_name='final_fully_connected')
        return x


# train_x, train_y, test_x, test_y = prepare_data()
# train_x, test_x = color_preprocessing(train_x, test_x)


# Use load_data() to get all data and labels
data, labels = load_data()
# for i in range(0, 600, 10):
#     print(labels[i])
data = spectral_preprocessing(data)

# image_size = 100, img_channels = 45, class_num = 5 in image_data.py
x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, img_channels])
label = tf.placeholder(tf.float32, shape=[None, class_num])

training_flag = tf.placeholder(tf.bool)

learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = SE_ResNeXt(x, training=training_flag).model

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(tf.global_variables())

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8

with tf.Session(config=config) as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    epoch_learning_rate = init_learning_rate

    # divide data into 90% training + 10% testing
    # repeat for 10 times, cross-validation
    division = 10 # percent of the data used for validation
    num_imgs_division = {} # division size for each class
    for c in range(class_num):
        st, end = imgs_index_in_classes[c]
        n = (end - st + 1) / division
        num_imgs_division[c] = np.ceil(n)

    for i in range(1, division + 1):
        train_x, train_y = np.zeros((1, image_size, image_size, img_channels)), np.zeros((1, class_num))
        test_x, test_y = np.zeros((1, image_size, image_size, img_channels)), np.zeros((1, class_num))
        # prepare train and test data during each division
        for c in range(class_num):
            n = int(num_imgs_division[c])
            # start and end index of data of imgs in curr class
            c_st, c_end = imgs_index_in_classes[c]
            if i < division - 1:
                test_x = np.append(test_x, data[c_st+i*n:c_st+(i+1)*n], axis=0)
                test_y = np.append(test_y, labels[c_st:c_st+n], axis=0)

                train_x = np.append(train_x, data[c_st:c_st+i*n], axis=0)
                train_x = np.append(train_x, data[c_st+(i+1)*n:c_end+1], axis=0)
                train_y = np.append(train_y, labels[c_st+n:c_end+1], axis=0)
            else:
                test_x = np.append(test_x, data[c_st+i*n:c_end+1], axis=0)
                test_y = np.append(test_y, labels[c_st+i*n:c_end+1], axis=0)
                train_x = np.append(train_x, data[c_st:c_st+i*n], axis=0)
                train_y = np.append(train_y, labels[c_st:c_st+i*n], axis=0)

        train_x, train_y = train_x[1:], train_y[1:]
        test_x, test_y = test_x[1:], test_y[1:]
        # shuffle the divided data before starting training
        train_x, train_y = shuffle_data(train_x, train_y)
        test_x, test_y = shuffle_data(test_x, test_y)

        for epoch in range(1, total_epochs + 1):
            if epoch % 30 == 0 :
                epoch_learning_rate = epoch_learning_rate / 10

            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0

            for step in range(1, iteration + 1):
                if pre_index + batch_size < 550:
                    batch_x = train_x[pre_index: pre_index + batch_size]
                    batch_y = train_y[pre_index: pre_index + batch_size]
                else:
                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]

                batch_x = data_augmentation(batch_x)

                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag: True
                }

                _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size


            train_loss /= iteration # average loss
            train_acc /= iteration # average accuracy

            train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                              tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

            test_acc, test_loss, test_summary = Evaluate(sess)

            summary_writer.add_summary(summary=train_summary, global_step=epoch)
            summary_writer.add_summary(summary=test_summary, global_step=epoch)
            summary_writer.flush()

            line = "data division: %d/%d, epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                i, division, epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
            print(line)

            with open('logs-c3b1s100.txt', 'a') as f:
                f.write(line)

            saver.save(sess=sess, save_path='./model/ResNeXt.ckpt')