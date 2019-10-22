#!/usr/bin/python3 
import sys
sys.path.append("./assets")

from termcolor import cprint
import numpy as np
import assets.onnx_ml_pb2 as onnx_ml_pb2
import assets.predict_pb2 as predict_pb2
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# load input image
input_shape = (1, 3, 1200, 1200)
img = Image.open("assets/BlueAngels.jpg")
img = img.resize((1200, 1200), Image.BILINEAR)
# preprocess image
img_data = np.array(img)
img_data = np.transpose(img_data, [2, 0, 1])
img_data = np.expand_dims(img_data, 0)
mean_vec = np.array([0.485, 0.456, 0.406])
stddev_vec = np.array([0.229, 0.224, 0.225])
norm_img_data = np.zeros(img_data.shape).astype('float32')
for i in range(img_data.shape[1]):
    norm_img_data[:,i,:,:] = (img_data[:,i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
# prepare http request
input_tensor = onnx_ml_pb2.TensorProto()
input_tensor.dims.extend(norm_img_data.shape)
input_tensor.data_type = 1
input_tensor.raw_data = norm_img_data.tobytes()

request_message = predict_pb2.PredictRequest()
request_message.inputs["image"].data_type = input_tensor.data_type
request_message.inputs["image"].dims.extend(input_tensor.dims)
request_message.inputs["image"].raw_data = input_tensor.raw_data

content_type_headers = ['application/x-protobuf', 'application/octet-stream', 'application/vnd.google.protobuf']

for h in content_type_headers:
    request_headers = {
        'Content-Type': h,
        'Accept': 'application/x-protobuf'
    }
# issue a http request 
PORT_NUMBER = 9001
inference_url = "http://127.0.0.1:" + str(PORT_NUMBER) + "/v1/models/default/versions/1:predict"
cprint("the URI is: " + inference_url, "red")
import time
start = time.time()
response = requests.post(inference_url, headers=request_headers, data=request_message.SerializeToString())
time_elapse = time.time() - start
cprint("latency of one request is: %s seconds"%time_elapse, "red")
# parse http response
response_message = predict_pb2.PredictResponse()
response_message.ParseFromString(response.content)
bboxes = np.frombuffer(response_message.outputs['bboxes'].raw_data, dtype=np.float32)
labels = np.frombuffer(response_message.outputs['labels'].raw_data, dtype=np.int64)
scores = np.frombuffer(response_message.outputs['scores'].raw_data, dtype=np.float32)
print('Boxes shape:', response_message.outputs['bboxes'].dims)
print('Labels shape:', response_message.outputs['labels'].dims)
print('Scores shape:', response_message.outputs['scores'].dims)

# Plot the bounding boxes on the image
plt.figure()
fig, ax = plt.subplots(1, figsize=(12,9))
ax.imshow(img)

resized_width = 1200 
resized_height = 1200
num_boxes = 4
    
for c in range(num_boxes):    
    base_index = c * 4
    y1, x1, y2, x2 = bboxes[base_index] * resized_height, bboxes[base_index + 1] * resized_width, bboxes[base_index + 2] * resized_height, bboxes[base_index + 3] * resized_width 
    color = 'blue'
    box_h = (y2 - y1)
    box_w = (x2 - x1)
    bbox = patches.Rectangle((y1, x1), box_h, box_w, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(bbox)
    plt.text(y1, x1, s="aeroplane", color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})
plt.axis('off')

# Save image
plt.savefig("output/ssd_result.jpg", bbox_inches='tight', pad_inches=0.0)
plt.show()
