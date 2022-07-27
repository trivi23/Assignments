import cv2
import numpy as np
from utils import proccess_frame, draw_prediction

confidence = 0.3
nms = 0.4
with open('coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg", "darknet")
layernames = net.getUnconnectedOutLayersNames()
while True:
    image = cv2.imread("car.jpg")
    blob = cv2.dnn.blobFromImage(
        image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layernames)
    proccess_frame(image, outs, classes, confidence, nms)
    cv2.imshow("result", image)
    cv2.waitKey(1)
