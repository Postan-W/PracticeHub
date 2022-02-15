from 图片处理 import get_images_labels
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
images,labels = get_images_labels()
print("图片为:{}".format(images[0]))
print("标签为:{}".format(labels[0]))
#99%训练，1%测试
sep = int(len(images)*0.99)
train_x,train_y = images[:sep],labels[:sep]
test_x,test_y = images[sep:],labels[sep:]
print("训练集大小:{},测试集大小:{}".format(sep,len(images)-sep))
input_data = tf.placeholder("float", shape=[None, 320,320,3])
input_label = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],
                          strides=[1,2,2,1],padding='SAME')

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])#一个卷积核的卷积结果加上一个偏置值(偏置维度扩展为与卷积结果相适应)
h_conv1 = tf.nn.relu(conv2d(input_data, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
W_conv3 = weight_variable([5,5,64,64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
#通过下面的方法可以计算出全连接层神经元的个数,这样动态增加卷积池化层后也不用手动去修改全连接大小
full_connection_numbers = 0
gpu_options = tf.GPUOptions(allow_growth=True)#动态请求占用的显存
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    total = sess.run(tf.size(h_pool3),feed_dict={input_data:test_x[:2]})
    batch = sess.run(tf.shape(h_pool3),feed_dict={input_data:test_x[:2]})[0]
    print("总的batch是:{}".format(batch))
    full_connection_numbers = int(total / batch)
    print("单个图片神经元个数是:{}".format(full_connection_numbers))

W_fc1 = weight_variable([full_connection_numbers,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1,full_connection_numbers])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder('float')#定义dropout比例
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#全连接层1进行dropout处理
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
predictions = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(input_label * tf.log(predictions))#加负号才是正值,这个正值越小越接近真实情况
train_step = tf.train.AdamOptimizer().minimize(cross_entropy)#这里优化器使用默认参数
right_position = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(input_label,axis=1))#返回一维张量,长度为batch大小。二者相等的位置为1，不等的位置为0
accuracy = tf.reduce_mean(tf.cast(right_position, tf.float32))#显然,平均值就是准确率

epochs = 10#总的数据轮次
batch_size = 32#batch_size和训练数据量共同决定了一轮经过多少次迭代
iterations = int(len(train_x)/batch_size)
print("===============开始训练===================")
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(epochs):
        for iteration in range(iterations):
            #优化
            sess.run(train_step,feed_dict={input_data:train_x[iteration*batch:(iteration+1)*batch],input_label:train_y[iteration*batch:(iteration+1)*batch],keep_prob:0.5})
            if iteration %  20 == 0:
                train_accuracy = accuracy.eval(
                    feed_dict={input_data: train_x[iteration * batch:(iteration + 1) * batch],
                               input_label: train_y[iteration * batch:(iteration + 1) * batch], keep_prob: 1})
                print("训练准确度:{}".format(train_accuracy))

        print("==================================================================================")
        test_accuracy = accuracy.eval(feed_dict={input_data: test_x, input_label: test_y, keep_prob: 1.0})
        print("测试准确度:{}".format(test_accuracy))

