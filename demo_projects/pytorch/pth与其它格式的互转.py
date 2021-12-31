#Pytorch中tensor维度顺序为B,C,H,W
import torch
import torchvision.models as buildinmodels
from onnx_tf.backend import prepare
import onnx
from tensorflow.python import keras
import onnxruntime
import numpy as np
import tensorflow.compat.v1 as tf
import os
import keras2onnx
from onnx2pytorch import ConvertModel
import onnx2keras
import tf2onnx
import torch.nn.functional as functional
from utility.model_process import h5_input_shape
#Pytorch加载自定义模型时，由于pickle的操作，自定义模型类需要能在当前模块找到，要么导入要么在当前模块给出
class MnistClassificationDynamicInput(torch.nn.Module):
    def __init__(self,h,w):
        super(MnistClassificationDynamicInput,self).__init__()
        h1 = int((h-5+1)/2)
        w1 = int((w-5+1)/2)
        h2 = int((h1-4)/2)
        w2 = int((w1-4)/2)
        self.convolution1 = torch.nn.Conv2d(1,10,kernel_size=5)#输入单通道，输出10通道，卷积核大小(5,5)
        self.convolution2 = torch.nn.Conv2d(10,20,kernel_size=5)#输入10通道，输出20通道，卷积核大小(5,5)
        self.convolution2_drop = torch.nn.Dropout2d()
        self.k = 20*h2*w2
        print("全连接层的输入大小为:",self.k)
        self.full_connection1 = torch.nn.Linear(self.k,60)
        self.full_connection2 = torch.nn.Linear(60,10)


    def forward(self,x):
        x = functional.relu(functional.max_pool2d(self.convolution1(x),2))#此时尺寸为(10,12,12)
        x = functional.relu(functional.max_pool2d(self.convolution2_drop(self.convolution2(x)),2))#此时尺寸为(20,4,4)，所以共320个神经元
        x = x.view(-1,self.k)#展平操作
        x = functional.relu(self.full_connection1(x))
        x = functional.dropout(x,training=self.training)#training=True,训练的时候才dropout
        x = self.full_connection2(x)
        return functional.log_softmax(x)

def use_pth(modelpath,inputshape):
    model = torch.load(modelpath)
    data = torch.randn(*inputshape)
    predictions = model(data)
    print(predictions)

# use_pth("./models/netG.pth",[10, 3, 256, 256])

def pth_to_onnx(sourcepath,destinationpath,shape):
    print("pth====>>onnx")
    model = torch.load(sourcepath)
    print("模型加载完成")
    # print(linear_model.eval())
    # model = buildinmodels.resnet18()
    # print(resnet18.eval())
    x = torch.randn(*shape)
    # resnet18_predictions = model(x)
    torch.onnx.export(model,x, destinationpath,input_names=["inputtest"],output_names=["outputtest"],dynamic_axes=
    {"inputtest": {0: "batch_size"},"outputtest": {0: "batch_size"}})#必须要说明首个维度为batch维,否则第一个维度的数值就被固定了
    #可视化网络可以看到这里设定的输入输出名称即便在转为其它格式的模型后，其它模型依然采用该名称
# pth_to_onnx('./models/mnist_classification_epoch2.pth',"./models/mnist_classification_epoch2_pth2onnx.onnx",[1,1,28,28])

def load_onnx(path):
    print("加载onnx模型")
    onnx_model = onnx.load(path)
    inference_model = onnxruntime.InferenceSession(path)
    onnx.checker.check_model(onnx_model)
    inuput_tensor = onnx_model.graph.input
    print(inuput_tensor)
    output_tensor = onnx_model.graph.output
    print(output_tensor)
    print(onnx.helper.printable_graph(onnx_model.graph))
    input = inference_model.get_inputs()[0].name
    output = inference_model.get_outputs()[0].name
    print(input,output)
    print("get_inputs:",inference_model.get_inputs()[0])
    shape_without_dimension = inference_model.get_inputs()[0].shape[1:]
    shape = [3]
    shape.extend(shape_without_dimension)
    print(shape)
    x = torch.randn(*shape)
    # print("测试数据为:{}".format(np.array(x)))
    result = inference_model.run([output],{input:np.array(x)})
    print(result[0])

# load_onnx("./models/mnist_classification_epoch2_pth2onnx.onnx")

def onnx_to_pb(sourcepath,destinationpath):
    print("onnx====>>pb")
    onnx_model = onnx.load(sourcepath)
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(destinationpath)

# onnx_to_pb("./models/netG.onnx","./models/netG_pth2onnx2pb.pb")

def use_pbmodel(path,input,output):#默认一个输入输出
    print("加载pb模型")
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = path
        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.graph.get_tensor_by_name(output)  # ”output“ 也是
            input = sess.graph.get_tensor_by_name(input)# "input" 是在pth文件转为onnx文件时定义好的，名字要一致
            shape = [3]
            shape.extend(list(input.shape)[1:])
            print(shape)
            input_data = np.random.randint(1,10,shape,'int32')  # 输入要测试的数据，格式要一致
            # print("测试数据:{}".format(input_data))
            predictions = sess.run(output, feed_dict={input: input_data})
            print("predictions:", predictions)

# use_pbmodel("./models/netG_pth2onnx2pb.pb","inputtest:0","outputtest:0")

def  onnx_to_h5(sourcepath,destinationpath):
    print("onnx====>>h5")
    onnx_model = onnx.load(sourcepath)
    keras_model = onnx2keras.onnx_to_keras(onnx_model,["inputtest"],input_shapes=[[32,32,1]])
    #用tf2集成的Keras，以免版本不符导致错误
    keras.models.save_model(keras_model,destinationpath,include_optimizer=True)
# onnx_to_h5("./models/LeNet_onnx2pth2onnx.onnx","./models/LeNet_onnx2pth2onnx2h5.h5")

def use_h5model(path):
    print("加载h5模型")
    model = keras.models.load_model(path)
    shape_withoutdimension = h5_input_shape(model.to_json())
    shape = [3]
    shape.extend(shape_withoutdimension)
    print("输入数据形状:{}".format(shape))
    data = np.random.randint(1,10,shape,'int32')
    # print("测试数据:{}".format(data))
    # print(model.summary())
    prediction = model.predict(data)
    print(np.sum(prediction,axis=1))
    print("预测结果:{}".format(prediction))

#pytorch的resnet18模型Keras不支持，无法解析一些层
# use_h5model("./models/model_MobileNet_v1.h5")

def onnx_to_pth(sourcepath,destinationpath):
    print("onnx====>>pth")
    onnx_model = onnx.load(sourcepath)
    inference_model = onnxruntime.InferenceSession(sourcepath)
    shape_without_dimension = inference_model.get_inputs()[0].shape[1:]
    shape = [3]
    shape.extend(shape_without_dimension)
    print("输入数据形状:{}".format(shape))
    pytorch_model = ConvertModel(onnx_model)#转换后的pth模型只能接收一个batch的数据(与上面pth转onnx定的[1,1]无关。经使用多个模型验证，是这个转换方法的问题
    data = torch.randn(*shape)
    predictions = pytorch_model(data).data.numpy()
    print(predictions)
    print(np.sum(predictions,axis=1))
    # result = np.array(pytorch_model(data))
    # print("预测结果:{}".format(result))
    torch.save(pytorch_model,destinationpath)

# onnx_to_pth("./models/model_MobileNet_h5toonnx.onnx","./models/MobileNet_h5toonnx2pth.pth")

def convert_h5_into_onnx(modelpath,destinationpath):
    # h5model = tf.keras.models.load_model(modelpath)#低版本的tensorflow(这里是1.11)继承的keras使用该句会出错KeyError: 'weighted_metrics'。这里换做keras自身，而不是继承的
    print("h5====>>onnx")
    h5model = keras.models.load_model(modelpath)
    print("h5模型名称:",h5model.name)
    onnxmodel = keras2onnx.convert_keras(h5model,h5model.name)
    onnx.save_model(onnxmodel,destinationpath)

# convert_h5_into_onnx("./models/model_MobileNet_v1.h5","./models/model_MobileNet_h5toonnx.onnx")

def convert_h5_to_savedmodel(sourcepath,destinationpath):
    print("h5====>>savedmodel")
    h5model = keras.models.load_model(sourcepath)
    tf.saved_model.save(h5model,destinationpath)

# convert_h5_to_savedmodel("resnet18.h5","resnet18h5_to_savedmodel")
def convert_saved_model_to_onnx(sourcepath,destinationpath):
    print("savedmodel====>>onnx")
    print(os.system("conda activate tf2.0 && python -m tf2onnx.convert --saved-model %s --output %s" % (
    sourcepath, destinationpath)))

# convert_saved_model_to_onnx("resnet18h5_to_savedmodel","resnet18savedmodel_to_onnx.onnx")



