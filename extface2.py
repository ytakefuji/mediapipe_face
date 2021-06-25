# original: https://pysource.com/blur-faces-in-real-time-with-opencv-mediapipe-and-python
# single person face extraction modified by takefuji

import mediapipe as mp
import cv2,sys
import numpy as np
import dlib

class FaceLandmarks:
    def __init__(self):
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh()

    def get_facial_landmarks(self, frame):
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)
        facelandmarks = []
        try:
         for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])
         return np.array(facelandmarks, np.int32)
        except TypeError:
         return np.array([])
cam = cv2.VideoCapture(0)
while True:
 r,img = cam.read()
 height, width, _ = img.shape
 img_copy=img.copy()
 fl = FaceLandmarks()
 landmarks = fl.get_facial_landmarks(img)
 if len(landmarks)==0:
  rr='not detected'
  cv2.putText(img,rr, (74, 74), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 8)
  cv2.imshow('result',img)
  cv2.waitKey(1)
 else:
  rr='detected'
  convexhull = cv2.convexHull(landmarks)
  mask = np.zeros((height, width), np.uint8)
  cv2.polylines(mask, [convexhull], True, 255, 3)
  cv2.fillConvexPoly(mask, convexhull, 255)
  face_ext = cv2.bitwise_and(img_copy, img_copy, mask=mask)
  cv2.putText(face_ext, rr, (74, 74), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 8)
  cv2.imshow('result', face_ext )
  cv2.waitKey(1)
cv2.destroyAllWindows()
