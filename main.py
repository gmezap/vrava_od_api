import os
import tensorflow as tf
import getpass
import cv2 
import numpy as np
import matplotlib.pyplot as plt
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


CUSTOM_MODEL_NAME = 'faster_rcnn_resnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}
# for path in paths.values():
#     if not os.path.exists(path):
#         !mkdir -p {path}
        
# Install Tensorflow Object Detection 

# !apt-get install protobuf-compiler
# !cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install . 

# if not os.path.exists(os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection')):
#     !git clone https://github.com/tensorflow/models {paths['APIMODEL_PATH']}

labels = [{'name':'dent', 'id':1}, {'name':'scratch', 'id':2}, {'name':'broken', 'id':3}]

# with open(files['LABELMAP'], 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')
        
import object_detection

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

def json_danios(image):
    json = '' 
    tensor_image = tf.convert_to_tensor(np.expand_dims(image, 0), dtype=tf.float32)
    detections = detect_fn(tensor_image)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_with_detections = boxes(image, detections)
    danos = list()
    for i,score in enumerate(detections['detection_scores']):
        if score > 0.5:
            danos.append({'tipo':int(detections['detection_classes'][i]),'certeza':str(round(score,3))})
    return danos, image_with_detections

def boxes(image, detections):
    category_index = {1: {'id': 1, 'name': 'dent'}, 2: {'id': 2, 'name': 'scratch'}, 3: {'id': 3, 'name': 'broken'}}
    image_np_with_detections = image.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+1,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.5,
            agnostic_mode=False)
    return cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB).reshape(image.shape)

from fastapi import FastAPI, File, UploadFile, Response
from io import BytesIO
from PIL import Image
import json
from matplotlib.image import imsave
import requests
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
app = FastAPI()

@app.post('/pred')
async def predi( file: UploadFile = File(...) ):
    image = np.array(Image.open(BytesIO(await file.read())))
    danos, imagen = json_danios(image)
    res, im_png = cv2.imencode(".png", imagen)
    return StreamingResponse(BytesIO(im_png.tobytes()), media_type="image/png", headers={'danos':danos})

@app.get('/predict/{id}')
async def predict(id, response: Response):
    url = "https://strapi-malayapps.s3.amazonaws.com/"+str(id)
    res = requests.get(url)
    image = np.array(Image.open(BytesIO(res.content)))
    danos, imagen = json_danios(image)
    res, im_png = cv2.imencode(".png", imagen)
    filename = (id)
    url = await upload(filename,BytesIO(im_png.tobytes()))
    # return StreamingResponse(BytesIO(im_png.tobytes()), media_type="image/jpg", headers={'danos':json.dumps(danos), 'url': url})
    # prediccion = {'predict': {'danos':json.dumps(danos), 'url': url}}
    response.headers['danos'] = json.dumps(danos)
    response.headers['url'] = url
    return 'holi'

@app.get('/')
def home():
    return {'message': 'wena compare'} 

import requests

async def upload(filename, files):
    upload = 'https://vravabackahorasi-production.up.railway.app/upload'
    files={'files' : (filename, files, 'image/jpg')}
    r = requests.post(upload, files=files)
    print(r.content)
    return r.json()[0]['url']
