import numpy as np
import cv2
import sys
import datetime

# Get user supplied values
video_capture = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
filename = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")[:-7] + ".avi"
out = cv2.VideoWriter(filename, fourcc, 25.0, (640, 480))

recording = False
movement_detected = False
noMoveDelay = datetime.timedelta(0, 3)
last_movement = datetime.datetime.now()

# tryb pracy - ktory algorytm ma byc uzywany
programMode = 2

# parametry redukcji drzenia okularow
noiseReactionDelay = 10
noiseTolerantion = 10
oldGlassesPosX = 0
oldGlassesPosY = 0
oldGlassesLiveTime = 0

# parametry wykrywania poprzez ruch
possibleColorDiff = 10
boundaryFound = -1
boundaryX = -1
boundaryY = -1
boundaryW = -1
boundaryH = -1

positionChangeParam = 0.85
scaleChangeParam = 0.97

ret, oldFrame = video_capture.read()
oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)

# Read the image
while True:
    if programMode == 2:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        foundObject = cv2.absdiff(oldGray, gray)
        thresh = cv2.threshold(foundObject, possibleColorDiff, 255, cv2.THRESH_BINARY)[1]
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=2)
        thresh = cv2.dilate(thresh, kernel, iterations=35)
        thresh = cv2.erode(thresh, kernel, iterations=20)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            if w / h > 1.5 or h / w > 1.5:
                continue
            if boundaryFound == -1:
                boundaryX = x
                boundaryY = y
                boundaryW = w
                boundaryH = h
                boundaryFound = 0
            else:
                boundaryX = np.int32(boundaryX * positionChangeParam + x * (1 - positionChangeParam))
                boundaryY = np.int32(boundaryY * positionChangeParam + y * (1 - positionChangeParam))
                boundaryW = np.int32(boundaryW * scaleChangeParam + w * (1 - scaleChangeParam))
                boundaryH = np.int32(boundaryH * scaleChangeParam + h * (1 - scaleChangeParam))

        cv2.rectangle(frame, (boundaryX, boundaryY), (boundaryX + boundaryW, boundaryY + boundaryH), (0, 255, 0), 2)
        # saving video
        if movement_detected:
            last_movement = datetime.datetime.now()
            if not recording:
                filename = str(datetime.datetime.now()).replace(" ", "_").replace(":", "_")[:-7] + ".avi"
                out = cv2.VideoWriter(filename, fourcc, 25.0, (640, 480))
                recording = True
            out.write(frame)
        else:
            if recording:
                if datetime.datetime.now() - last_movement < noMoveDelay:  # dokoncz
                    out.write(frame)
                else:
                    recording = False
                    out.release()

        # out.write(frame)
        cv2.imshow("Faces", frame)
        oldGray = gray

    key = cv2.waitKey(10)
    if key == 27:  # ESC
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
