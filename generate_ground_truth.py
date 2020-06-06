import os
import tensorflow.compat.v1 as tf
import numpy as np
import itertools

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

dataset = tf.data.TFRecordDataset("/mnt/pool0-100/shape-it/validation_data/segment-967082162553397800_5102_900_5122_900_with_camera_labels.tfrecord", compression_type='')
ground = open('ground.bin', 'wb')
objects = metrics_pb2.Objects()
for data in dataset:
    print("enter first loop")
    frame = dataset_pb2.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    for camera_labels in frame.camera_labels:
        print("enter second loop")
        o = metrics_pb2.Object()
        o.camera_name = camera_labels.name
        o.context_name = frame.context.name
        o.frame_timestamp_micros = frame.timestamp_micros
        for label in camera_labels.labels:
            box = label.box
            o.object.box.CopyFrom(box)
            o.object.type = label.type
            o.object.id = ""
            o.object.detection_difficulty_level = label.detection_difficulty_level
            o.object.num_lidar_points_in_box: 6
        objects.objects.append(o)
    break
ground.write(objects.SerializeToString())

