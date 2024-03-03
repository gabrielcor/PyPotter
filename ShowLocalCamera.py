import cv2
import numpy as np
import sys

print(len(sys.argv))
# Check for required number of arguments
if (len(sys.argv) < 2):
    #Camera
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(sys.argv[1])



while(1):
    res, frame = cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Resultaat', img_gray)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()