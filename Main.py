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

showRectangles=1

programMode = 1

noiseReactionDelay = 10
noiseTolerantion = 10
oldGlassesPosX = 0
oldGlassesPosY = 0
oldGlassesLiveTime = 0

#video_capture.set()

# Read the image
while True:
    if programMode == 1:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        glasses = gl

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Faces", frame)


    if cv2.waitKey(5) & 0xFF == ord('s'):
        showRectangles = 1 - showRectangles
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
    if cv2.waitKey(5) & 0xFF == ord('1'):
        programMode = 1
    if cv2.waitKey(5) & 0xFF == ord('2'):
        programMode = 2

video_capture.release()
cv2.destroyAllWindows()