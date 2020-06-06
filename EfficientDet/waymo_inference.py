import math
import random
import os
import tensorflow.compat.v1 as tf
import numpy as np
import itertools
import model_inspect

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2


inspector = model_inspect.ModelInspector('efficientdet-d0', 'logdir')
filepath='/mnt/pool0-100/shape-it/validation_data/segment-14486517341017504003_3406_349_3426_349_with_camera_labels.tfrecord'
dataset=tf.data.TFRecordDataset(filepath, compression_type='')

numFrames=0
config_dict = {}
config_dict['line_thickness']=3
config_dict['max_boxes_to_draw']=15
config_dict['min_score_thresh']=0.5
for data in dataset:
    if numFrames >=3:
        break
    numFrames = numFrames+1
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    numImages = 0
    for index, img in enumerate(frame.images):
        if numImages>=5:
            break
        numImages = numImages+1
        f=open(str(img.name)+'.jpg', 'wb+')
        f.write(img.image)
        
        inspector.inference_single_image(str(img.name) + '.jpg', '/mnt/pool0-100/filip/images/EfficientDet/04003' + str(numFrames) + '_' + str(numImages) + '.jpg', **config_dict)
        f.close()
        os.remove(str(img.name)+'.jpg')
