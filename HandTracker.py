#!/usr/bin/env python3

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0) #Capturing webcam
# Setting up hand tracking with Mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0 # Past Time
cTime = 0 # Current Time

while True:
    # Captures the image / reads input (What it sees)
    success, img = cap.read()
    # This program can only read RGB images, so this is the conversion
    imgRBG = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Results / Processes the info
    results = hands.process(imgRBG)
    # Draws hand landmarks!
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)  
                # All of this helps get the ID of the mapped points for further 
                # Programming.
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)
                # Draws circle on mapped point to ID for further programming
                if id == 0:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)


            # Draws landmarks 
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    # Checks FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # Draws the FPS counter
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 2)


    # Opens the webcam
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
