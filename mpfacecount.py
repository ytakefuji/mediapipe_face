import cv2
import mediapipe as mp
import time

mpface = mp.solutions.face_mesh
face= mpface.FaceMesh()
mpdraw = mp.solutions.drawing_utils
drawspec=mpdraw.DrawingSpec(thickness=1,circle_radius=1)
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    if results.multi_face_landmarks:
     for i in results.multi_face_landmarks:
      mpdraw.draw_landmarks(img, i, mpface.FACE_CONNECTIONS,drawspec,drawspec)
    else:
     continue
    cv2.putText(img, str(len(results.multi_face_landmarks)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
