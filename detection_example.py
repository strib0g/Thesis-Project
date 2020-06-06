import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import tensorflow.compat.v1 as tf

# Make sure that caffe is on the python path:
caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2
from waymo_open_dataset import dataset_pb2

#Open record file, insert own filepath
filepath='PATH'
dataset=tf.data.TFRecordDataset(filepath, compression_type='')

#Iterate through record file
for data in dataset:

    #Initialize frame ref to Frame object 
    frame=dataset_pb2.Frame()
    
    #Parse frame from record
    frame.ParseFromString(bytearray(data.numpy()))
    
    #Initialize array of detection objects
    objects = metrics_pb2.Objects()
    
    #iterate through images in frame
    for index, img in enumerate(frame.images):
    
        #Initialize reference to object
        o = metrics_pb2.Object()
        
        # The following 3 fields are used to uniquely identify a frame a prediction
        # is predicted at. Make sure you set them to values exactly the same as what
        # we provided in the raw data. Otherwise your prediction is considered as a
        # false negative.
        o.context_name =  frame.context.name
        # The frame timestamp for the prediction. See Frame::timestamp_micros in
        # dataset.proto.
        o.frame_timestamp_micros = frame.timestamp_micros
        # This is only needed for 2D detection or tracking tasks.
        # Set it to the camera name the prediction is for.
          if camera_labels.name == 0:
              o.camera_name = dataset_pb2.CameraName.UNKNOWN
          elif camera_labels.name == 1:
              o.camera_name = dataset_pb2.CameraName.FRONT
          elif camera_labels.name == 2:
              o.camera_name = dataset_pb2.CameraName.FRONT_LEFT
          elif camera_labels.name == 3:
              o.camera_name = dataset_pb2.CameraName.FRONT_RIGHT
          elif camera_labels.name == 4:
              o.camera_name = dataset_pb2.CameraName.SIDE_LEFT
          elif camera_labels.name == 5:
              o.camera_name = dataset_pb2.CameraName.SIDE_RIGHT
              
          #Initialize box reference to store detection location information
          #All box values should be replaced from values gathered from the model's detections
          box = label_pb2.Label.Box()
          
          box.center_x = 0
          box.center_y = 0
          box.length = 0
          box.width = 0
          
          #Copy values from box object into detection object
          o.object.box.CopyFrom(box)
          
          #value that stores confidence of detection
          o.score = 0
          
          #ID only needs to be unique for the object, a random number generator can be used
          o.object.id = 0
          
          #check name of object detected and assign appropriate type
          #all types are static enums inside the label object
          if nameTag == 'person':
              o.object.type = label_pb2.Label.TYPE_PEDESTRIAN
          elif nameTag == 'car':
              o.object.type = label_pb2.Label.TYPE_VEHICLE
          elif nameTag == 'bicycle':
              o.object.type = label_pb2.Label.TYPE_CYCLIST
          elif nameTag == 'stop sign':
              o.object.type = label_pb2.Label.TYPE_SIGN
              
          #add object to list of objects
          objects.objects.append(o)

  # Write list of objects to a file.
  f = open('/tmp/your_preds.bin', 'wb')
  f.write(objects.SerializeToString())
  f.close()
