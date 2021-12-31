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
from  model_process import h5_input_shape,universal_image_process
import logging
from logging.handlers import RotatingFileHandler
from PIL import Image

class ModelTrans:
    def __init__(self):
        pass

origin_image = Image.open("test_images/animal.jpg")
origin_image.show()
image_numpy = universal_image_process("test_images/animal.jpg",[200,200,3]).reshape([200,200,3])
resized_image = Image.fromarray(image_numpy)
resized_image.show()