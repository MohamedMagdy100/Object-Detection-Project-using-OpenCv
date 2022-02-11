import cv2
import numpy as np

thresh = 0.5
nmsThresh = 0.4

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5, 127.5))
net.setInputSwapRB(True)

# ____________________________________________________________Photo__________________________________________________________

# img = cv2.imread('orange.jfif')
#
# classIds , confs , bbox = net.detect(img,confThreshold=thresh)
# indices = cv2.dnn.NMSBoxes(bbox,confs,thresh,nmsThresh)
#
# for i in indices:
#     box = bbox[i]
#     x, y, w, h = box[0], box[1], box[2], box[3]
#     cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
#     cv2.putText(img, classNames[classIds[i] - 1], (box[0], box[1] + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6,
#                 (0,255,0), 2)
#     cv2.putText(img, str(round(confs[i] * 100, 2)) + '%', (box[0], box[1] + 50), cv2.FONT_HERSHEY_COMPLEX, 0.6,
#                 (0,255,0), 2)
#
# cv2.imshow('Output',img)
# cv2.waitKey(0)

# ____________________________________________________________Camera_________________________________________________________

# cap = cv2.VideoCapture(1)
#
# while True:
#     success,img = cap.read()
#     img = cv2.flip(img, 1)
#     classIds , confs , bbox = net.detect(img,confThreshold=thresh)
#     bbox = list(bbox)
#     confs = list(np.array(confs).reshape(1,-1)[0])
#     confs = list(map(float,confs))
#
#     indices = cv2.dnn.NMSBoxes(bbox,confs,thresh,nmsThresh)
#
#     for i in indices:
#         box = bbox[i]
#         x,y,w,h = box[0],box[1],box[2],box[3]
#         cv2.rectangle(img, (x,y),(x+w,y+h), color=(0, 255, 0), thickness=2)
#         cv2.putText(img, classNames[classIds[i]-1], (box[0], box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.7,
#                     (0, 255, 0), 2)
#         cv2.putText(img, str(round(confs[i] * 100, 2)) + '%', (box[0], box[1] + 60), cv2.FONT_HERSHEY_COMPLEX, 0.7,
#                     (0, 255, 0), 2)
#
#     cv2.imshow('Output',img)
#     cv2.waitKey(1)