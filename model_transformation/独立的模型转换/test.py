import shutil
import onnx
from trans_utility import dir_dict
import os
from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf
import onnxruntime
model = onnxruntime.InferenceSession(dir_dict[2]+os.listdir(dir_dict[2])[0])
inputs = model.get_inputs()[0].name
outputs = model.get_outputs()[0].name
shape = model.get_inputs()[0].shape[1:]
print(inputs,outputs,shape)
image = Image.open(os.path.join("./image_for_predict", os.listdir("./image_for_predict")[0]))
image = image.resize((shape[1], shape[0]))
with_batch = [1]
with_batch.extend(shape)
image_array = np.array(image).reshape(with_batch).astype("float32")
result = model.run([outputs],input_feed={inputs:image_array})
print(result[0])



