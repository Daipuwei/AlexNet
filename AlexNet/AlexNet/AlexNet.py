# -*- coding: utf-8 -*-
# @Time    : 2018/11/8 21:00
# @Author  : Daipuwei
# @Blog    ：https://blog.csdn.net/qq_30091945
# @EMail   ：771830171@qq.com
# @Site    ：中国民航大学北教25实验室506
# @FileName: AlexNet.py
# @Software: PyCharm

"""
    这是AlexNet的类代码
"""

import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.examples.tutorials.mnist import input_data

IMAGE_SIZE = 28                             # MNIST图片尺寸
IMAGE_CHANNELS = 1                          # MNIST图像通道个数
NUM_LABELS = 10                             # MNIST数据集分类个数
DROPOUT = 0.8                               # dropout概率
BATCH_SIZE = 64                             # 小样本规模
REGULARIZER_LEARNING_RATE = 0.0005          # 正则化洗漱
EXPONENTIAL_MOVING_AVERAGE_DECAY = 0.99     # 动量因子
LEARNING_RATE_BASE = 0.001                  # 学习率衰初始值
LEARNING_RATE_DECAY = 0.99                  # 学习率衰减率
TRAINING_STEP = 10000                       # 迭代次数
MODEL_SAVE_PATH = "./Models"               # 模型文件存放文件夹地址
MODEL_NAME = "AlexNet.ckpt"               # 模型名称

class AlexNet(object):
    def __init__(self):
        # 将上述变量定义为全局变量
        global IMAGE_SIZE,IMAGE_CHANNELS,NUM_LABELS,DROPOUT
        global BATCH_SIZE,REGULARIZER_LEARNING_RATE,EXPONENTIAL_MOVING_AVERAGE_DECAY
        global LEARNING_RATE_BASE,TRAINING_STEP,LEARNING_RATE_DECAY
        global MODEL_SAVE_PATH,MODEL_NAME
        # 初始化各层的权重
        with tf.variable_scope("weights"):
            self.weights ={
                'conv1': tf.Variable(tf.random_normal([11, 11, 1, 64]),trainable=True),
                'conv2': tf.Variable(tf.random_normal([5, 5, 64, 192]),trainable=True),
                'conv3': tf.Variable(tf.random_normal([3, 3, 192, 384]),trainable=True),
                'conv4':tf.Variable(tf.random_normal([3, 3, 384, 384]),trainable=True),
                'conv5': tf.Variable(tf.random_normal([3, 3, 384, 256]),trainable=True),
                'fc1': tf.Variable(tf.random_normal([4*4*256,4096]),trainable=True),
                'fc2': tf.Variable(tf.random_normal([4096,4096]),trainable=True),
                'fc3': tf.Variable(tf.random_normal([4096,NUM_LABELS]),trainable=True)
            }
        # 初始化各层的偏置
        with tf.variable_scope("biases"):
            self.biases ={
                'conv1': tf.Variable(tf.random_normal([64]),trainable=True),
                'conv2': tf.Variable(tf.random_normal([192]),trainable=True),
                'conv3': tf.Variable(tf.random_normal([384]),trainable=True),
                'conv4': tf.Variable(tf.random_normal([384]),trainable=True),
                'conv5': tf.Variable(tf.random_normal([256]),trainable=True),
                'fc1': tf.Variable(tf.random_normal([4096]),trainable=True),
                'fc2': tf.Variable(tf.random_normal([4096]),trainable=True),
                'fc3': tf.Variable(tf.random_normal([10]),trainable=True)
            }

    def con2d(self,name,input,weights,biases,strides):
        """
        这是AlexNet内的卷积操作的函数
        :param name: 名称
        :param input: 输入
        :param weights: 权重
        :param biases: 偏置
        :param strides: 步长
        """
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input,weights,strides=[1,strides,strides,1],padding="SAME"),biases),name=name)

    def max_pool(self,name,input,ksize,strides):
        """
        这是AlexNet内的最大池化操作函数
        :param name: 名称
        :param input: 输入
        :param ksize: 内核大小
        :param strides: 步长
        """
        return tf.nn.max_pool(input,ksize=[1,ksize,ksize,1],strides=[1,strides,strides,1],padding='SAME',name=name)

    def LocalResponseNormlization(self,name,input):
        """
        这是AlexNet内的LRN操作
        :param name: 名称
        :param input: 输入
        """
        return tf.nn.lrn(input,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name=name)

    def inferecne(self,images,train_flag,regularizer=None):
        """
        这是前向传播的函数
        :param images: 输入图像
        :param train_flag: 训练测试标志
        :param regularizer: 正则化函数
        """
        # 第一层，定义卷积权重、偏置和下采样,计算卷积结果
        conv1 = self.con2d('conv1',images,self.weights['conv1'],self.biases['conv1'],strides=1)
        # 最大池化
        pool1 = self.max_pool('pool1',conv1,ksize=2,strides=2)
        # LRN
        lrn1 = self.LocalResponseNormlization('lrn1',pool1)

        # 第二层，定义卷积权重、偏置和下采样,计算卷积结果
        conv2 = self.con2d('conv2', lrn1, self.weights['conv2'], self.biases['conv2'], strides=1)
        # 最大池化
        pool2 = self.max_pool('pool2', conv2, ksize=2, strides=2)
        # LRN
        lrn2 = self.LocalResponseNormlization('lrn2', pool2)

        # 第三层，定义卷积权重、偏置和下采样,计算卷积结果
        conv3 = self.con2d('conv3', lrn2, self.weights['conv3'], self.biases['conv3'], strides=1)
        # LRN
        lrn3 = self.LocalResponseNormlization('lrn3',conv3)

        # 第四层，定义卷积权重、偏置和下采样,计算卷积结果
        conv4 = self.con2d('conv4', lrn3, self.weights['conv4'], self.biases['conv4'], strides=1)
        # LRN
        lrn4 = self.LocalResponseNormlization('lrn4',conv4)

        # 第五层，定义卷积权重、偏置和下采样,计算卷积结果
        conv5 = self.con2d('conv4', lrn4, self.weights['conv5'], self.biases['conv5'], strides=1)
        # 最大池化
        pool5 = self.max_pool('pool5', conv5, ksize=2, strides=2)
        # LRN
        lrn5 = self.LocalResponseNormlization('lrn4', pool5)

        # 第六层为全连接层,fc1
        fc1_input = tf.reshape(lrn5,[-1,self.weights['fc1'].get_shape().as_list()[0]])
        fc1 = tf.nn.relu(tf.matmul(fc1_input,self.weights['fc1'])+self.biases['fc1'],name='fc1')
        # 训练阶段，则进行dropout
        if train_flag == True:
            fc1 = tf.nn.dropout(fc1,DROPOUT)
        # 对权重进行l2正则化
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(self.weights['fc1']))

        # 第七层为全连接层,fc2
        fc2_input = tf.reshape(fc1,[-1,self.weights['fc2'].get_shape().as_list()[0]])
        fc2 = tf.nn.relu(tf.matmul(fc2_input,self.weights['fc2'])+self.biases['fc2'],name='fc2')
        # 训练阶段，则进行dropout
        if train_flag == True:
            fc2 = tf.nn.dropout(fc2,DROPOUT)
        # 对权重进行l2正则化
        if regularizer != None:
            tf.add_to_collection("losses",regularizer(self.weights['fc2']))

        # 第八层为全连接层,fc3
        fc3_input = tf.reshape(fc2,[-1,self.weights['fc3'].get_shape().as_list()[0]])
        fc3 = tf.matmul(fc3_input,self.weights['fc3'])+self.biases['fc3']
        # 对权重进行l2正则化
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(self.weights['fc3']))
        return fc3

    def train(self,mnist):
        """
        这是AlexNet的训练函数
        :param Train_Data: 训练数据集
        :param Train_Label: 训练标签集
        """
        # 定义AlexNet的输入和输出
        x = tf.placeholder(tf.float32,shape=[None,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS],name="x_input")
        y_ = tf.placeholder(tf.float32,shape=[None,NUM_LABELS],name="y_input")
        # 定义L2正则化
        regularizer = tf.contrib.layers.l2_regularizer(REGULARIZER_LEARNING_RATE)
        # 利用AlexNet进行前向传播得到预测分类
        y = self.inferecne(x,True,regularizer)
        global_step = tf.Variable(0,trainable=False)
        # 定义滑动平均操作以及训练过程
        variable_average = tf.train.ExponentialMovingAverage(EXPONENTIAL_MOVING_AVERAGE_DECAY, global_step)
        variable_average_op = variable_average.apply(tf.trainable_variables())
        # 定义cross_entropy交叉熵和损失函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection("losses",cross_entropy_mean)
        loss = tf.add_n(tf.get_collection("losses"))
        # 定义精度
        correct = tf.equal(tf.arg_max(y_,1),tf.arg_max(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
        # 定义指数衰减学习率
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,
                                                   mnist.train.labels.shape[0]/BATCH_SIZE,LEARNING_RATE_DECAY)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step)
        with tf.control_dependencies([train_step, variable_average_op]):
            train_op = tf.no_op(name='train')

        # 初始化持久化类
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 初始化所有变量
            tf.global_variables_initializer().run()
            for i in range(TRAINING_STEP):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                xs = np.reshape(xs, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
                ys = np.reshape(ys,(BATCH_SIZE,NUM_LABELS))
                '''random_sequence_batch = self.Shuffle_Sequence(len(Train_Data), BATCH_SIZE)
                accuracy_train = []
                loss_train = []
                step_train = 0
                for batch in random_sequence_batch:
                    Train_Data_Batch,Train_Label_Batch = Train_Data[batch],Train_Label[batch]
                    Train_Data_Batch = np.shape(Train_Data_Batch,(BATCH_SZIE,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS))
                    Train_Label_Batch = np.shape(Train_Label_Batch,(BATCH_SZIE,NUM_LABELS))
                    #xs = np.reshape(xs,(BATCH_SIZE, Inference.IMAGE_SIZE, Inference.IMAGE_SIZE, Inference.NUM_CHANNELS))
                    _,loss_train,acc_train,step = sess.run([train_op,loss,accuracy,global_step],
                                                       feed_dict={x:Train_Data_Batch,y_:Train_Label_Batch})
                    accuracy_train.append(acc_train)
                    loss_train.append(loss_train)
                    step_train = step'''
                _,step = sess.run([train_op,global_step],feed_dict={x: xs, y_: ys})
                if i % 1000 == 0:
                    # 计算精度
                    acc_train,step = sess.run([accuracy,global_step], feed_dict={x: xs, y_: ys})
                    acc_val = []
                    iter = int(mnist.validation.labels.shape[0]/100)
                    #print(mnist.validation.labels.shape)
                    for i in np.arange(0,iter):
                        xs, ys = mnist.validation.next_batch(100)
                        xs = np.reshape(xs, (100, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
                        acc = sess.run(accuracy, feed_dict={x: xs, y_: ys})
                        acc_val.append(acc)
                    acc_val = np.sum(np.array(acc_val)*100)/mnist.validation.labels.shape[0]
                    print("After %d training step(s),accuracy on train batch is:%.5f,accuracy on validation is:%.5f"%(step,acc_train,acc_val))
                    saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=step)

    def test(self,mnist):
        """
        这是AlexNet的测试函数
        :param mnist:MNIST数据集
        """
        # 定义AlexNet的输入和输出
        x = tf.placeholder(tf.float32,[None,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS],name="x_input")
        y_ = tf.placeholder(tf.float32,[None,NUM_LABELS],name="y_input")
        # 利用AlexNet进行前向传播得到预测分类
        y = self.inferecne(x,False)
        # 定义精度
        correct = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平局值了。
        # 这样就可以完全共用mnist_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(EXPONENTIAL_MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        #每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while True:
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    xs,ys = mnist.test.images,mnist.test.labels
                    accuracy_test = []
                    for start in np.arange(0,len(ys),100):
                        end = np.min([start+100,len(ys)])
                        test_data,test_label = xs[start:end],ys[start:end]
                        test_data = np.reshape(test_data,(100,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS))
                        acc_test = sess.run(accuracy, feed_dict={x: test_data, y_: test_label})
                        accuracy_test.append(acc_test)
                    accuracy_test = np.sum(np.array(accuracy_test)*100)/len(ys)
                    print("After %s training step(s), test accuracy = %f" % (global_step, accuracy_test))
                else:
                    print("No checkpoint file found")
            time.sleep(10)

def run_main():
    """
    这是主函数
    """
    mnist = input_data.read_data_sets("E:\\DPW\\MNIST",one_hot=True)
    alexnet = AlexNet()
    alexnet.train(mnist)
    #alexnet.test(mnist)


if __name__ == '__main__':
    run_main()