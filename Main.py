import numpy as np
import cv2
import sys

# Get user supplied values
#imagePath = "drdoktor.jpg"
facePath = "haarcascade_frontalface_default.xml"
eyesPath = "haarcascade_eye.xml"
video_capture = cv2.VideoCapture(0)
gl = cv2.imread("glasses.png", -1)

#video_capture.set(3, 640)
#video_capture.set(4, 480)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(facePath)
eyesCascade = cv2.CascadeClassifier(eyesPath)

#czy pokazywac prostokaty wokol wykrytej twarzy
showRectangles=1

#tryb pracy - ktory algorytm ma byc uzywany
programMode = 1

#parametry redukcji drzenia okularow
noiseReactionDelay = 10
noiseTolerantion = 10
oldGlassesPosX = 0
oldGlassesPosY = 0
oldGlassesLiveTime = 0

#parametry wykrywania poprzez ruch
possibleColorDiff = 10
boundaryFound = -1
boundaryX = -1
boundaryY = -1
boundaryW = -1
boundaryH = -1

positionChangeParam = 0.85
scaleChangeParam = 0.97

#video_capture.set()
ret, oldFrame = video_capture.read()
oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY);

# Read the image
while True:
    if programMode == 1:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        glasses = gl

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            if showRectangles:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            #region of interest
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eyesCascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=8, minSize=(3,3))
            sglasses = cv2.resize(glasses, (w, w/3))
            leftEye = 1280
            g = 1000
            for (ex, ey, ew, eh) in eyes:
                if showRectangles:
                    cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0, 255, 0), 2)
                if ex < leftEye:
                    leftEye = ex
                    g = ey
            if leftEye < 1280:
                if abs(x - oldGlassesPosX) < noiseTolerantion and abs(g - oldGlassesPosY) < noiseTolerantion and oldGlassesLiveTime < noiseReactionDelay:
                    x = oldGlassesPosX
                    g = oldGlassesPosY
                    oldGlassesLiveTime = oldGlassesLiveTime + 1
                else:
                    oldGlassesLiveTime = 0
                    oldGlassesPosX = x
                    oldGlassesPosY = g

                y1, y2 = g, g + sglasses.shape[0]
                x1, x2 = x, x + sglasses.shape[1]

                alpha_s = sglasses[:,:,3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(0, 3):
                    frame[y+y1:y+y2, x1:x2, c] = (alpha_s * sglasses[:,:, c] +
                                                  alpha_l * frame[y+y1:y+y2, x1:x2, c])


        cv2.imshow("Faces", frame)

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
        cv2.imshow("Faces", frame)
        oldGray = gray


    key = cv2.waitKey(10)
    if key == 115: # s
        showRectangles = 1 - showRectangles
    elif key == 27: # ESC
        break
    elif key == 49: # 1
        programMode = 1
    elif key == 50: # 2
        programMode = 2

video_capture.release()
cv2.destroyAllWindows()