import os
from utils.app_utils import *
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'model/mscoco_label_map.pbtxt'

NUM_CLASSES = 90

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
INPUT_SIZE = 513
FROZEN_GRAPH_NAME = 'frozen_inference_graph'

def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap
colormap = create_pascal_label_colormap()

def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def detect_objects(image_np, sess, detection_graph):
    # # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims(image_np, axis=0)
    # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # # Each box represents a part of the image where a particular object was detected.
    # boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # # Each score represent how level of confidence for each of the objects.
    # # Score is shown on the result image, together with the class label.
    # scores = detection_graph.get_tensor_by_name('detection_scores:0')
    # classes = detection_graph.get_tensor_by_name('detection_classes:0')
    # num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # # Actual detection.
    # (boxes, scores, classes, num_detections) = sess.run(
    #     [boxes, scores, classes, num_detections],
    #     feed_dict={image_tensor: image_np_expanded})


    width, height = image_np.shape[:2]
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    #resized_image = image_np.convert('RGB').resize(target_size, Image.ANTIALIAS)
    resized_image = cv2.resize(image_np, target_size, interpolation = cv2.INTER_CUBIC)  ###TODO: Colorspace!
    batch_seg_map = sess.run(
        OUTPUT_TENSOR_NAME,
        feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    seg_image = label_to_color_image(seg_map).astype(np.uint8)

    # # Visualization of the results of a detection.
    # vis_util.visualize_boxes_and_labels_on_image_array(
    #     image_np,
    #     np.squeeze(boxes),
    #     np.squeeze(classes).astype(np.int32),
    #     np.squeeze(scores),
    #     category_index,
    #     use_normalized_coordinates=True,
    #     line_thickness=4)

    return seg_image



def worker(input_q, output_q):
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)

    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()

        # Check frame object is a 2-D array (video) or 1-D (webcam)
        if len(frame) == 2:
            frame_rgb = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
            output_q.put((frame[0], detect_objects(frame_rgb, sess, detection_graph)))
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_q.put(detect_objects(frame_rgb, sess, detection_graph))
    fps.stop()
    sess.close()
