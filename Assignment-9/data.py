import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


class HandDetector:
    def __init__(self,min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def storedata(self,f,a):
        f1 = open(f, 'a')
        f1.write(str(a))
        f1.close()

    def findPoseLandmarks(self, image,draw= False):
        OriginalImage=image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        landMarkList = []
        if results.:
           for hand_landmarks in results.multi_hands_landmarks:
            for point in mpHands.handLandmark:
                normalizeLandmark = hand_landmarks.landmark[point]
                landMarkList.append(normalizeLandmark.x)
                landMarkList.append(normalizeLandmark.y)
                landMarkList.append(normalizeLandmark.z)
            print(len(landMarkList))
            data = str(landMarkList)[1:-1]
            self.storedata('gesture1.csv', data)

        if draw:
            mpDraw.draw_landmarks(
                originalImage, hand_landmarks, mpHands.HAND_CONNECTIONS)
            return landMarkList
