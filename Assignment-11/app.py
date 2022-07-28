import numpy as np
import time
import cv2
import math

from itertools import chain 
from constants import *

LABELS = open('coco.names','rt').read().strip().split('\n')

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

print('Loading YOLO from disk...')

neural_net = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
layer_names = neural_net.getLayerNames()
layer_names = [layer_names[i-1] for i in neural_net.getUnconnectedOutLayers()]

vs = cv2.VideoCapture(0)
writer = None
(W, H) = (None, None)

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break
    
    if W is None or H is None:
        H, W = (frame.shape[0], frame.shape[1])

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    neural_net.setInput(blob)

    start_time = time.time()
    layer_outputs = neural_net.forward(layer_names)
    end_time = time.time()
    
    boxes = []
    confidences = []
    classIDs = []
    lines = []
    box_centers = []

    for output in layer_outputs:
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.5 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                box_centers = [centerX, centerY]

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    if len(idxs) > 0:
        unsafe = []
        count = 0
        
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centeriX = boxes[i][0] + (boxes[i][2] // 2)
            centeriY = boxes[i][1] + (boxes[i][3] // 2)

            color = [int(c) for c in COLORS[classIDs[i]]]
            text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])

            idxs_copy = list(idxs.flatten())
            idxs_copy.remove(i)

            for j in np.array(idxs_copy):
                centerjX = boxes[j][0] + (boxes[j][2] // 2)
                centerjY = boxes[j][1] + (boxes[j][3] // 2)

                distance = math.sqrt(math.pow(centerjX - centeriX, 2) + math.pow(centerjY - centeriY, 2))
                print(distance)
                if distance <= 300:
                    cv2.line(frame, (boxes[i][0] + (boxes[i][2] // 2), boxes[i][1]  + (boxes[i][3] // 2)), (boxes[j][0] + (boxes[j][2] // 2), boxes[j][1] + (boxes[j][3] // 2)), (0, 0, 255), 2)
                    unsafe.append([centerjX, centerjY])
                    unsafe.append([centeriX, centeriY])

            if centeriX in chain(*unsafe) and centeriY in chain(*unsafe):
                count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, 'No Social Distancing', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (50, 50), (450, 90), (0, 0, 0), -1)
                cv2.putText(frame, 'No people are unsafe', (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)            


    if writer is None:
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter('output.avi', fourcc, 30,(frame.shape[1], frame.shape[0]), True)

    cv2.imshow('output',frame)    
    writer.write(frame)
    cv2.waitKey(1)

print('Cleaning up...')
writer.release()
vs.release()    