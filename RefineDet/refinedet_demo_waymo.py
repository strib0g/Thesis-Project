'''
In this example, we will load a RefineDet model and use it to detect objects.
'''

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

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in range(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def ShowResults(img, image_file, results, labelmap, threshold=0.6, save_fig=True):
    plt.clf()
    plt.imshow(img)
    plt.axis('off')
    ax = plt.gca()

    num_classes = len(labelmap.item) - 1
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

    for i in range(0, results.shape[0]):
        score = results[i, -2]
        if threshold and score < threshold:
            continue

        label = int(results[i, -1])
        name = get_labelname(labelmap, label)[0]
        color = colors[label % num_classes]

        xmin = int(round(results[i, 0]))
        ymin = int(round(results[i, 1]))
        xmax = int(round(results[i, 2]))
        ymax = int(round(results[i, 3]))
        coords = (xmin, ymin), xmax - xmin, ymax - ymin
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=3))
        display_text = '%s: %.2f' % (name, score)
        ax.text(xmin, ymin, display_text, bbox={'facecolor':color, 'alpha':0.5})
    if save_fig:
        plt.savefig(image_file, bbox_inches="tight")
        print('Saved: ' + image_file[:-4] + '_dets.jpg')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save_fig', action='store_true')
    args = parser.parse_args()

    # gpu preparation
    if args.gpu_id >= 0:
        caffe.set_device(args.gpu_id)
        caffe.set_mode_gpu()

    # load labelmap
    labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
    file = open(labelmap_file, 'r')
    labelmap = caffe_pb2.LabelMap()
    text_format.Merge(str(file.read()), labelmap)

    # load model
    model_def = '/home/fatic/RefineDet/models/VGGNet/VOC0712Plus/refinedet_vgg16_512x512_ft/deploy.prototxt'
    model_weights = '/home/fatic/RefineDet/models/VGGNet/VOC0712Plus/refinedet_vgg16_512x512_ft/VOC0712Plus_refinedet_vgg16_512x512_ft_final.caffemodel'
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # image preprocessing
    if '320' in model_def:
        img_resize = 320
    else:
        img_resize = 512
    net.blobs['data'].reshape(1, 3, img_resize, img_resize)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array([104, 117, 123]))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB

    # im_names = os.listdir('examples/images')
    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg', '001763.jpg', '004545.jpg']
    
    #TODO add file
    filepath='/mnt/pool0-100/shape-it/validation_data/segment-17136314889476348164_979_560_999_560_with_camera_labels.tfrecord'
    dataset=tf.data.TFRecordDataset(filepath, compression_type='')
    numFrames = 0
    for data in dataset:
        if numFrames >= 3:
            break
        numFrames = numFrames+1
        print("enter first loop")
        frame=dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        numImages = 0
        for index, img in enumerate(frame.images):
            if numImages >= 5:
                break
            numImages = numImages+1
            print("enter second loop")
            f=open(str(img.name)+'.jpg', 'wb+')
            f.write(img.image)
            image = caffe.io.load_image(str(img.name)+'.jpg')
            f.close()
            transformed_image = transformer.preprocess('data', image)
            net.blobs['data'].data[...] = transformed_image

            detections = net.forward()['detection_out']
            print("performing detection")
            det_label = detections[0, 0, :, 1]
            det_conf = detections[0, 0, :, 2]
            det_xmin = detections[0, 0, :, 3] * image.shape[1]
            det_ymin = detections[0, 0, :, 4] * image.shape[0]
            det_xmax = detections[0, 0, :, 5] * image.shape[1]
            det_ymax = detections[0, 0, :, 6] * image.shape[0]
            result = np.column_stack([det_xmin, det_ymin, det_xmax, det_ymax, det_conf, det_label])
            num_classes = len(labelmap.item) - 1
            #print(result[0])
            for i in range(0, result.shape[0]):
               # print(result[i, 4])
                if result[i, 4]  < 0.6:
                    continue
                print("score accepted")

                label = int(result[i, -1])
                print(label)
                name = get_labelname(labelmap, label)[0]

                xmin = int(round(result[i, 0]))
                ymin = int(round(result[i, 1]))
                xmax = int(round(result[i, 2]))
                ymax = int(round(result[i, 3]))

                print(name)
                print(result[i, 4])
            f.close()
            os.remove(str(img.name)+'.jpg')

        # show result
            ShowResults(image, "/mnt/pool0-100/filip/images/RefineDet/48164_" +str(numFrames)+"_"+ str(numImages) + ".jpg" , result, labelmap, 0.6, save_fig=True)
