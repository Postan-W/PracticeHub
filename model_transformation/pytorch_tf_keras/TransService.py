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
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image
import requests
from flask import Flask, jsonify, request,make_response,send_file
"""
1.服务需要的信息有:输入的模型类型、输出的模型类型、输入的模型下载路径、预测的图片或数据集
2.生成的模型保存在镜像的一个已转换的模型目录中，转换后的模型的命名规则是"from_原模型名称"；上传的模型保存在镜像的一个上传的模型目录中
3.基于2,转换后的模型下载，以及使用模型预测(无论是转换前还是转换后的),都以列表形式展示，二者的列表均为转换前后的所有模型
4.转换是pth;h5;pb三种格式的转换，但是完成转换后还会隐式地往其他格式转，如果失败就算了
"""
class ModelTrans:
    def __init__(self):
        pass


