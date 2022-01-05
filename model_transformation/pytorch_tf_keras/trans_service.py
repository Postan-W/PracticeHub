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
from self_build import MnistClassificationDynamicInput
from PIL import Image
import requests
from flask import Flask, jsonify, request,make_response,send_file
from trans_utility import logger,remove_model,dir_dict,model_dict
import os
"""
Tensorflow: BHWC;Pytorch: BCHW;尽管不同框架处理的数据维度顺序不同，但转换的模型始终保持同一种数据格式输入，不会影响结果，
这说明处理原始模型foward过程在转换为其他框架后对数据的维度处理仍然保持原始框架的顺序
"""

"""
代码逻辑:
1.上传模型接口:被请求时，会先删除models目录下现有的一个模型，然后保存上传的模型
2.模型转换接口：被请求时，先会创建ModelTrans对象(初始化参数是目标框架类型)，该对象根据models目录下的模型来判断源模型是哪种框架的，用source属性记录源框架类型，
用destination属性记录目标框架类型,并将该值赋给全局变量target，作为在预测接口中判断模型类型的依据；之后开始转换，
首先判断source和destination是否相等，如果相等则不调用转换函数，不进行任何动作(
所以前端展示一段提示语句为益，比如源框架和目标框架不要为同一种),不相等则进行转换，转换是从源类型到中间类型(一般是onnx)再到目标类型，
从源类型到中间类型转换成功则删除intermediate_models下的模型并将中间模型保存在该目录，不成功则不执行下阶段的转换,从中间类型到目标类型的
转换成功后则删除transformed_models下面的模型并将目标模型保存在该目录，不成功则不进行任何动作，至少上次转换成功的模型还能用于下载
3.下载模型接口：被请求时，如果transformed_models下面没有模型，那么返回一个file_not_exist.txt文件，如果有则直接返回模型
4.预测接口：从transformed_models文件夹下加载模型，如果不存在模型文件则返回提示信息，否则返回预测结果，并附带上模型文件名称，因为本次预测可能使用
的是上次的模型
"""
class ModelTrans:
    def __init__(self,destination):
        self.destination = destination
        self.inputname = "inputtest"
        self.outputname = "outputtest"
        self.shape = [1,28,28]#默认CHW
        self.__get_frame_name()
        logger.info("将{}转为{}".format(self.source, self.destination))

    def __get_frame_name(self):
        try:
            model = os.listdir(dir_dict[1])[0]
        except Exception as e:
            logger.info("{}:源模型不存在".format(e))
        file_type = model[model.rfind(".")+1:]
        frame_name = None
        for key in model_dict.keys():
            if file_type in model_dict[key]:
                frame_name = key
                break
        self.source = frame_name


    @staticmethod
    def pth2onnx(shape:list,inputname:str,outputname:str):
        file = os.listdir(dir_dict[1])[0]
        remove_model(2)
        filename = file[:file.rfind(".")]
        target_name = filename + "_pth2onnx.onnx"
        logger.info("转换{}-->{}".format(file, target_name))
        try:
            model = torch.load(os.path.join(dir_dict[1], os.listdir(dir_dict[1])[0]))
        except Exception as e:
            logger.info("加载pth模型失败:{}".format(e))

        shape_with_batch = [3] #随意指定一个batch维度
        shape_with_batch.extend(shape)
        x = torch.randn(*shape_with_batch)#需要通过一个张量指定模型的输入形状
        try:
            torch.onnx.export(model, x, dir_dict[2]+target_name, input_names=[inputname], output_names=[outputname],
                              dynamic_axes=
                              {inputname: {0: "batch_size"}, outputname: {0: "batch_size"}})#要说明第一个维度是batch维
            logger.info("pth转为onnx成功")
        except Exception as e:
            logger.info("由pth转到onnx出错:{}".format(e))

    @staticmethod
    def onnx2pb():
        if len(os.listdir(dir_dict[2])) != 0:
            remove_model(3)
            file = os.listdir(dir_dict[2])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2pb.pb"
            try:
                onnx_model = onnx.load(os.path.join(dir_dict[2], os.listdir(dir_dict[2])[0]))
                logger.info("加载onnx成功")
            except Exception as e:
                logger.info("加载onnx失败:{}".format(e))
            try:
                tf_exp = prepare(onnx_model)  # prepare tf representation
                tf_exp.export_graph(dir_dict[3] + target_name)
                logger.info("onnx到pb成功")
            except Exception as e:
                logger.info("onnx到pb失败:{}".format(e))

    def transformer(self):
        if self.source == "Pytorch":
            if self.destination == "Tensorflow":
                ModelTrans.pth2onnx(self.shape,self.inputname,self.outputname)
                ModelTrans.onnx2pb()



# trans = ModelTrans("Tensorflow")
# trans.transformer()
# #供用户请求的模型下载接口
# from flask import Flask, jsonify, request,make_response,send_file,send_from_directory
# app = Flask(__name__)
# @app.route('/download', methods=['GET'])
# def download_model():
#     return send_from_directory("./",filename="mnist_classification_epoch2.zip",as_attachment=True)
#
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=False)

def test():
    pth_model = torch.load("./models/mnist_classification_epoch2.pth")
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    p0, p2, p8, p9, p4, p6 = x_train[1], x_train[12000], x_train[24000], x_train[36000], x_train[48000], x_train[59000]
    images = np.array(
        [p0.reshape((1, 28, 28)), p2.reshape((1, 28, 28)), p8.reshape((1, 28, 28)), p9.reshape((1, 28, 28)),
         p4.reshape((1, 28, 28)), p6.reshape((1, 28, 28))])
    print(images.shape)
    pth_predictions = pth_model(torch.Tensor(images))
    print("pth的预测结果:",pth_predictions.detach().numpy().argmax(axis=1))
    onnx_model = onnxruntime.InferenceSession("./intermediate_models/mnist_classification_epoch2_pth2onnx.onnx")
    onnx_predictions = onnx_model.run(["outputtest"], {"inputtest": images.astype("float32")})
    print("onnx预测结果:", np.array(onnx_predictions[0]).argmax(axis=1))
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = "./transformed_models/mnist_classification_epoch2_pth2onnx2pb.pb"
        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.graph.get_tensor_by_name("outputtest:0")  # ”output“ 也是
            input = sess.graph.get_tensor_by_name("inputtest:0")  # "input" 是在pth文件转为onnx文件时定义好的，名字要一致

            # print("测试数据:{}".format(input_data))
            predictions = sess.run(output, feed_dict={input: images.astype("float32")})
            print("pb预测结果:", np.array(predictions).argmax(axis=1))

test()