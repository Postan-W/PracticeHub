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
from PIL import Image
from pytorch2keras import pytorch_to_keras
from trans_utility import h5_input_shape
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
转换成功后则删除transformed_models下面的模型并将目标模型保存在该目录，不成功则不进行任何动作，至少上次转换成功的模型还能用于下载。
服务模块trans_server中设置一个·全局对象trans,每次模型转换接口被请求时生成的ModelTrans对象赋给trans，所以下游要用转换后的模型预测时可以
利用trans对象来提供预测所需的输入输出信息等
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
    def pb2onnx(inputs,outputs):
        model_path = os.path.join(dir_dict[1], os.listdir(dir_dict[1])[0])
        remove_model(2)
        file = os.listdir(dir_dict[1])[0]
        filename = file[:file.rfind(".")]
        target_name = filename + "2onnx.onnx"
        logger.info("转换返回状态(0代表正常):", os.system(
            "conda activate tf2.0 && python -m tf2onnx.convert --input %s --inputs %s --outputs %s --output %s" %(
             model_path, inputs, outputs, dir_dict[2]+target_name)))

    @staticmethod
    def h52onnx():
        try:
            model_path = os.path.join(dir_dict[1], os.listdir(dir_dict[1])[0])
            h5model = keras.models.load_model(model_path)
            logger.info("加载h5模型成功")
            onnxmodel = keras2onnx.convert_keras(h5model, h5model.name)
            remove_model(2)
            file = os.listdir(dir_dict[1])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2onnx.onnx"
            onnx.save_model(onnxmodel, dir_dict[2]+target_name)
        except Exception as e:
            logger.info("h5_to_onnx出错:{}".format(e))

    @staticmethod
    def pytorch2onnxfunction(shape:list, inputname:str, outputname:str):
        try:
            model_path = os.path.join(dir_dict[1], os.listdir(dir_dict[1])[0])
            model = torch.load(model_path)
            logger.info("pytorch模型加载完成")
            shape_with_batch = [2]
            shape_with_batch.extend(shape)
            x = torch.randn(*shape_with_batch)#样例数据
            remove_model(2)
            file = os.listdir(dir_dict[1])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2onnx.onnx"
            torch.onnx.export(model, x, dir_dict[2]+target_name, input_names=[inputname], output_names=[inputname],
                              dynamic_axes=
                              {inputname: {0: "batch_size"}, outputname: {0: "batch_size"}})
        except Exception as e:
            logger.info(e)
    @staticmethod
    def pytorch2kerash5(shape:list):
        try:
            model_path = os.path.join(dir_dict[1], os.listdir(dir_dict[1])[0])
            torch_model = torch.load(model_path)
            logger.info("加载Pytorch模型成功")
            input_shape = [1]  # 第一维是batch
            input_shape.extend(shape)
            logger.info("Pytorch转Keras，输入形状为:{}".format(input_shape))
            keras_model = pytorch_to_keras(model=torch_model, args=torch.autograd.Variable(
                torch.FloatTensor(np.random.uniform(0, 1, input_shape))), input_shapes=[shape],
                                           change_ordering=True, verbose=True)
            logger.info("Pytorch2Keras成功")
            remove_model(3)
            file = os.listdir(dir_dict[1])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2h5.h5"
            keras.models.save_model(keras_model, dir_dict[3] + target_name)
        except Exception as e:
            logger.info("Pytorch2Keras失败:{}".format(e))

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

    @staticmethod
    def onnx2h5(inputname:str,inputshape:list=None):
        if len(os.listdir(dir_dict[2])) != 0:
            remove_model(3)
            file = os.listdir(dir_dict[2])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2h5.h5"
            try:
                onnx_model = onnx.load(os.path.join(dir_dict[2], os.listdir(dir_dict[2])[0]))
                logger.info("加载onnx成功")
            except Exception as e:
                logger.info("加载onnx失败:{}".format(e))
            try:
                keras_model = onnx2keras.onnx_to_keras(onnx_model,[inputname])
                keras.models.save_model(keras_model, dir_dict[3]+target_name, include_optimizer=True)
                logger.info("h5模型生成成功")
            except Exception as e:
                logger.info("onnx到h5失败:{}".format(e))

    @staticmethod
    def onnx2pytorch():
        try:
            onnx_model_path = os.path.join(dir_dict[2], os.listdir(dir_dict[2])[0])
            onnx_model = onnx.load(onnx_model_path)
            inference_model = onnxruntime.InferenceSession(onnx_model_path)#推理
            logger.info("onnx模型加载成功")
            shape_without_dimension = inference_model.get_inputs()[0].shape[1:]
            shape = [3]
            shape.extend(shape_without_dimension)
            logger.info("输入数据形状:{}".format(shape))
            pytorch_model = ConvertModel(onnx_model)
            logger.info("onnx2pytorch转换成功")
            data = torch.randn(*shape)
            predictions = pytorch_model(data).data.numpy()
            logger.info("用onnx2pytorch得到的模型预测:{}".format(predictions))
            remove_model(3)
            file = os.listdir(dir_dict[2])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2pth.pth"
            torch.save(pytorch_model, dir_dict[3]+target_name)
        except Exception as e:
            logger.info("onnx2pytorch出错:{}".format(e))

    def transformer(self):
        if self.source == "Pytorch":
            if self.destination == "Tensorflow":
                ModelTrans.pytorch2onnxfunction(self.shape, self.inputname, self.outputname)
                ModelTrans.onnx2pb()
            if self.destination == "Keras":
                ModelTrans.pytorch2kerash5(self.shape)
        elif self.source == "Keras":
            if self.destination == "Pytorch":
                ModelTrans.h52onnx()
                ModelTrans.onnx2pytorch()
            if self.destination == "Tensorflow":
                ModelTrans.h52onnx()
                ModelTrans.onnx2pb()
        elif self.source == "Tensorflow":
            if self.destination == "Pytorch":
                ModelTrans.pb2onnx(self.inputname,self.outputname)
                ModelTrans.onnx2pytorch()
            if self.destination == "Keras":
                ModelTrans.pb2onnx(self.inputname,self.outputname)
                ModelTrans.onnx2h5(self.inputname)

    @staticmethod
    def h5_predictor():
        try:
            model = keras.models.load_model(os.path.join(dir_dict[3]+os.listdir(dir_dict[3])[0]))
            model_input = model.input
            model_output = model.output
            shape_withoutdimension = h5_input_shape(model.to_json())
            logger.info("模型输入的形状是{}".format(shape_withoutdimension))
            image = Image.open(os.path.join("./image_for_predict",os.listdir("./image_for_predict")[0]))
            logger.info("图片的尺寸是:{}".format(image.size))
            image.resize((shape_withoutdimension[1],shape_withoutdimension[0]))#h5模型就按HWC来看待
        except Exception as e:
            logger.info("使用h5模型预测失败:{}".format(e))

    def predict(self):
        ModelTrans.h5_predictor()

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
            output = sess.graph.get_tensor_by_name("outputtest:0")
            input = sess.graph.get_tensor_by_name("inputtest:0")
            predictions = sess.run(output, feed_dict={input: images.astype("float32")})
            print("pb预测结果:", np.array(predictions).argmax(axis=1))

