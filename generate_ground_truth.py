########################################################
#This file serves to generate a ground truth binary file
#for use with the compute_detection_metrics tool
########################################################

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
#opens individual record file
#for the entire dataset, this should be made to iterate through all files
dataset = tf.data.TFRecordDataset("REPLACE_WITH_FILEPATH", compression_type='')


ground = open('ground.bin', 'wb')   #create file to store annotations
objects = metrics_pb2.Objects()     #create Objects() iterable to store all detected objects
for data in dataset:                #iterates through all frames in the record file
    frame = dataset_pb2.Frame()     #initialize frame object
    frame.ParseFromString(bytearray(data.numpy()))      #frame object must first be parsed to be used
    
    #iterates through all camera labels
    #This refers to the cameras at the front and different sides
    for camera_labels in frame.camera_labels:
        o = metrics_pb2.Object()    #initialize Object() to store object annotations
        o.camera_name = camera_labels.name              #copy camera name into object metadata
        o.context_name = frame.context.name             #copy name of context into object metadata
        o.frame_timestamp_micros = frame.timestamp_micros       #copy timestamp of frame into object metadata
        
        #iterate through all labeled objects in image
        for label in camera_labels.labels:
            box = label.box         #intialize box format
            o.object.box.CopyFrom(box)      #copy box information - coordinates and dimensions
            o.object.type = label.type      #copy type of object - car, bicycle, sign or pedestrian
            o.object.id = ""                #was unsure how to extract this information
            o.object.detection_difficulty_level = label.detection_difficulty_level      #difficulty level of object - used when dividing results into difficulty classes on eval server
            o.object.num_lidar_points_in_box: 6                 #num of lidar points - not used in 2d detection - should still be populated by value >5
        objects.objects.append(o)
    break
ground.write(objects.SerializeToString())

