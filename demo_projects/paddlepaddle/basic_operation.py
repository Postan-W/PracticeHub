import os
import shutil
import paddle as paddle
import paddle.fluid as fluid
import time
from utility.图片获取 import get_images_labels
paddle.enable_static()
# 定义输入层
images,labels = get_images_labels()
images = images.reshape([images.shape[0],images.shape[3],images.shape[1],images.shape[2]])
print(images.shape)
sep = int(len(images)*0.95)
train_x,train_y = images[:sep],labels[:sep]
test_x,test_y = images[sep:],labels[sep:]
image = fluid.layers.data(name='image', shape=[None,3, 320, 320], dtype='float32')
label = fluid.layers.data(name='label', shape=[None,5], dtype='float')

base_model_program = fluid.default_main_program().clone()
model = fluid.layers.fc(input=image, size=5, act='softmax')
cost = fluid.layers.cross_entropy(input=model, label=label)
avg_cost = fluid.layers.mean(cost)
acc = fluid.layers.accuracy(input=model, label=label)

# 获取训练和测试程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化方法
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=1e-3,
                                          regularization=fluid.regularizer.L2DecayRegularizer(1e-4))
opts = optimizer.minimize(avg_cost)



# 定义一个使用GPU的执行器
# place = fluid.CUDAPlace(0)
# 使用CPU训练
place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

# 训练100次
since = time.time()
for pass_id in range(100):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed([train_x,train_y]),
                                        fetch_list=[avg_cost, acc])


    # # 进行测试
    # test_accs = []
    # test_costs = []
    # for batch_id, data in enumerate(test_reader()):
    #     test_cost, test_acc = exe.run(program=test_program,
    #                                   feed=feeder.feed(data),
    #                                   fetch_list=[avg_cost, acc])
    #     test_accs.append(test_acc[0])
    #     test_costs.append(test_cost[0])
    # # 求测试结果的平均值
    # test_cost = (sum(test_costs) / len(test_costs))
    # test_acc = (sum(test_accs) / len(test_accs))
    # print('Test:%d, Cost:%0.5f, Accuracy:%0.5f' % (pass_id, test_cost, test_acc))
    # # 保存预测模型
    # save_path = 'infer_model/'
    # # 删除旧的模型文件
    # shutil.rmtree(save_path, ignore_errors=True)
    # # 创建保持模型文件目录
    # os.makedirs(save_path)
    # # 保存预测模型
    # fluid.io.save_inference_model(save_path, feeded_var_names=[image.name], target_vars=[model], executor=exe)