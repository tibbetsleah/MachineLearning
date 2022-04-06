#!/usr/bin/env python3
import cv2
import mediapipe as mp
import time

"""I am not putting a framerate down because 
I made this with my webcam, if you need to the formula is below """
# pTime = 0
# cTime = time.time()
# fps = 1 / (cTime - pTime)
# Then you can display the text on the screen if you want with
# The OpenCV text display feature.

# Sets capture mode to webcam
vidCap = cv2.VideoCapture(0)

# Draws the mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=5, min_detection_confidence=0.5, static_image_mode=False, min_tracking_confidence=0.5, refine_landmarks=True)


while True:
    # reads image
    success, img = vidCap.read()
    # Converts image from BGR to RBG
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRBG)
    # Display
    if results.multi_face_landmarks:
        # Loops through all faces detected
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS)


    cv2.imshow("image", img)
    cv2.waitKey(1)

