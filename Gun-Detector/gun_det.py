import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
#import zerosms

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
picadd='test/test2.jpg'
cap = cv2.VideoCapture(picadd)
if picadd[-1]=='4': w=5
else: w=50000

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280,720))


sys.path.append("..")


from utils import label_map_util

from utils import visualization_utils as vis_util


MODEL_NAME = 'gun_graph'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)



with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      if ret == True:
       image_np_expanded = np.expand_dims(image_np, axis=0)
       image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
       boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
       scores = detection_graph.get_tensor_by_name('detection_scores:0')
       classes = detection_graph.get_tensor_by_name('detection_classes:0')
       num_detections = detection_graph.get_tensor_by_name('num_detections:0')
       # Actual detection.
       (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
		  
       
        	
	
	
       # Visualization of the results of a detection.
       vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
		  
       
	   
       # write the frame
       out.write(image_np)
       

       cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
       if cv2.waitKey(w) & 0xFF == ord('q'):
       
         break
      else:
        break
	  

cap.release()
out.release()
cv2.destroyAllWindows()
