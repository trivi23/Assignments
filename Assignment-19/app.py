import cv2
from deepface import DeepFace
import time

cam = cv2.VideoCapture(0)
happy = cv2.imread('happy.jpg')
sad = cv2.imread('sad.jpg')
angry = cv2.imread('angry.jpg')
default = cv2.imread('grey.jpg')
img=default
while True:
    status, image = cam.read()
    if status:
        try:
            result = DeepFace.analyze(image, actions=['emotion'])
            emotion = max(result['emotion'], key=result['emotion'].get)
            if emotion == "happy":
                img = happy
            elif emotion == 'sad':
                img=sad
            elif emotion == 'angry':
                img = angry
            cv2.imshow("Parent", img)
        except:
            img=default
    if cv2.waitKey(5000) & 0xFF == ord('q'):
        break
    cv2.destroyAllWindows()
