import cv2
import numpy as np
#Camera
cap = cv2.VideoCapture(0)

while(1):
    res, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Resultaat', img_gray)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()