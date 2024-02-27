# TODO:
# Smaller FOV for the camera helped a lot with the tracking
# 1-Try to get a smooth output from the wand tracking (maybe use threading to process the data)

# after
# USE a different source code (rpotter)


import sys
import cv2
from cv2 import *
import numpy as np
import math
import os
from os import listdir
from os.path import isfile, join, isdir
import time
import datetime
import threading
from threading import Thread
from statistics import mean 
from CountsPerSec import CountsPerSec
from HassApi import HassApi

# Check for required number of arguments
if (len(sys.argv) < 2):
    print("Incorrect number of arguments. Required Arguments: [video source url] ")
    sys.exit(0)

# Parse Required Arguments
videoSource = sys.argv[1]


# Constants
DesiredFps = 22
DefaultFps = 22 # Original constants trained for 42 FPS
MicroSecondsBetweenFrames = (1 / DesiredFps) * 1000000

TrainingResolution = 50
TrainingNumPixels = TrainingResolution * TrainingResolution
TrainingFolderName = "Training"
SpellEndMovement = 0.5 * (DefaultFps / DesiredFps )
MinSpellLength = 15 * (DesiredFps / DefaultFps)
MinSpellDistance = 100
NumDistancesToAverage = int(round( 20 * (DesiredFps / DefaultFps)))


IsTraining = False
IsDebugFps = True
IsShowOriginal = False
IsShowBackgroundRemoved = False
IsShowThreshold = False
IsShowOutput = True
IsProcessData = True
calculateDistance = False

# Create Windows
if (IsShowOriginal):
    cv2.namedWindow("Original")
    cv2.moveWindow("Original", 0, 0)
if (IsShowBackgroundRemoved):
    cv2.namedWindow("BackgroundRemoved")
    cv2.moveWindow("BackgroundRemoved", 640, 0)
if (IsShowThreshold):
    cv2.namedWindow("Threshold")
    cv2.moveWindow("Threshold", 0, 480+30)
if (IsShowOutput):
    cv2.namedWindow("Output")
    cv2.moveWindow("Output", 640, 480+30)

# Init Global Variables
IsNewFrame = False
nameLookup = {}
LastSpell = "None"

originalCps = CountsPerSec()
noBackgroundCps = CountsPerSec()
thresholdCps = CountsPerSec()
outputCps = CountsPerSec()

lk_params = dict( winSize  = (25,25),
                  maxLevel = 7,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

IsNewFrame = False
frame = None

IsNewFrameNoBackground = False
frame_no_background = None

IsNewFrameThreshold = False
frameThresh = None

findNewWands = True
trackedPoints = None
wandTracks = []

def InitClassificationAlgo() :
    """
    Create and Train k-Nearest Neighbor Algorithm
    """
    global knn, nameLookup
    labelNames = []
    labelIndexes = []
    trainingSet = []
    numPics = 0
    dirCount = 0
    scriptpath = os.path.realpath(__file__)
    trainingDirectory = join(os.path.dirname(scriptpath), TrainingFolderName)

    # Every folder in the training directory contains a set of images corresponding to a single spell.
    # Loop through all folders to train all spells.
    for d in listdir(trainingDirectory):
        if isdir(join(trainingDirectory, d)):
            nameLookup[dirCount] = d
            dirCount = dirCount + 1
            for f in listdir(join(trainingDirectory,d)):
                if isfile(join(trainingDirectory,d,f)):
                    labelNames.append(d)
                    labelIndexes.append(dirCount-1)
                    trainingSet.append(join(trainingDirectory,d,f));
                    numPics = numPics + 1

    print ("Trained Spells: ")
    print (nameLookup)

    samples = []
    for i in range(0, numPics):
        img = cv2.imread(trainingSet[i])
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        samples.append(gray);
        npArray = np.array(samples)
        shapedArray = npArray.reshape(-1,TrainingNumPixels).astype(np.float32);

    # Create KNN and Train
    knn = cv2.ml.KNearest_create()
    knn.train(shapedArray, cv2.ml.ROW_SAMPLE, np.array(labelIndexes))

def ClassifyImage(img):
    """
    Classify input image based on previously trained k-Nearest Neighbor Algorithm
    """
    global knn, nameLookup, args

    if (img.size  <= 0):
        return "Error"

    size = (TrainingResolution, TrainingResolution)
    test_gray = cv2.resize(img,size,interpolation=cv2.INTER_LINEAR)
    
    imgArr = np.array(test_gray).astype(np.float32)
    sample = imgArr.reshape(-1, TrainingNumPixels).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(sample,k=5)
    print(ret, result, neighbours, dist)

    if IsTraining:
        filename = "char" + str(time.time()) + nameLookup[ret] + ".png"
        cv2.imwrite(join(TrainingFolderName, filename), test_gray)

    if nameLookup[ret] is not None:
        print("Match: " + nameLookup[ret])
        return nameLookup[ret]
    else:
        return "error"

def CheckForPattern(wandTracks, exampleFrame):
    """
    Check the given wandTracks to see if is is complete, and if it matches a trained spell
    """
    global find_new_wands, LastSpell

    if (wandTracks == None or len(wandTracks) == 0):
        return

    thickness = 10
    croppedMax =  TrainingResolution - thickness

    distances = []
    wand_path_frame = np.zeros_like(exampleFrame)
    prevTrack = wandTracks[0]

    for track in wandTracks:
        x1 = prevTrack[0]
        x2 = track[0]
        y1 = prevTrack[1]
        y2 = track[1]

        # Calculate the distance
        distance = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        distances.append(distance)

        cv2.line(wand_path_frame, (int(x1), int(y1)),(int(x2), int(y2)), (255,255,255), thickness)
        prevTrack = track

    if (calculateDistance):
        mostRecentDistances = distances[-NumDistancesToAverage:]
        avgMostRecentDistances = mean(mostRecentDistances)
        sumDistances = sum(distances)

        contours, hierarchy = cv2.findContours(wand_path_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # Determine if wand stopped moving by looking at recent movement (avgMostRecentDistances), and check the length of distances to make sure the spell is reasonably long
        if (avgMostRecentDistances < SpellEndMovement and len(distances) > MinSpellLength):
            # Make sure wand path is valid and is over the defined minimum distance
            if (len(contours) > 0) and sumDistances > MinSpellDistance:
                cnt = contours[0]
                x,y,w,h = cv2.boundingRect(cnt)
                crop = wand_path_frame[y-10:y+h+10,x-30:x+w+30]
                result = ClassifyImage(crop);
                cv2.putText(wand_path_frame, result, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))

                print("Result: ", result, " Most Recent avg: ", avgMostRecentDistances, " Length Distances: ", len(distances), " Sum Distances: ", sumDistances)
                print("")

                # PerformSpell(result)
                LastSpell = result
            find_new_wands = True
            wandTracks.clear()

    if wand_path_frame is not None:
        if (IsShowOutput):
            wandPathFrameWithText = AddIterationsPerSecText(wand_path_frame, outputCps.countsPerSec())
            cv2.putText(wandPathFrameWithText, "Last Spell: " + LastSpell, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
            cv2.imshow("Output", wandPathFrameWithText)

    return wandTracks

def RemoveBackground():
    """
    Thread for removing background
    """
    global frame, frame_no_background, IsNewFrame, IsNewFrameNoBackground
    print("Removing Background initialized")

    fgbg = cv2.createBackgroundSubtractorMOG2()
    t = threading.current_thread()
    while getattr(t, "do_run", True):
        if (IsNewFrame):
            IsNewFrame = False
            frameCopy = frame.copy()

            # Subtract Background
            fgmask = fgbg.apply(frameCopy, learningRate=0.001)
            frame_no_background = cv2.bitwise_and(frameCopy, frameCopy, mask = fgmask)
            IsNewFrameNoBackground = True


            #if (IsShowBackgroundRemoved):
            if False:
                    frameNoBackgroundWithCounts = AddIterationsPerSecText(frame_no_background.copy(), noBackgroundCps.countsPerSec())
                    cv2.imshow("BackgroundRemoved", frameNoBackgroundWithCounts)
        else:
            time.sleep(0.001)


def ProcessData():
    """
    Thread for processing final frame
    """
    global frameThresh, IsNewFrameThreshold, findNewWands, wandTracks, outputFrameCount

    oldFrameThresh = None
    trackedPoints = None
    t = threading.currentThread()

    while getattr(t, "do_run", True):
        if (IsNewFrameThreshold):
            if (IsDebugFps):
                outputFrameCount = outputFrameCount + 1

            IsNewFrameThreshold = False
            localFrameThresh = frameThresh.copy()

            if (findNewWands):
                # Identify Potential Wand Tips using GoodFeaturesToTrack
                trackedPoints = cv2.goodFeaturesToTrack(localFrameThresh, 5, .01, 30)
                if trackedPoints is not None:
                    findNewWands = False
            else:
                # calculate optical flow
                nextPoints, statusArray, err = cv2.calcOpticalFlowPyrLK(oldFrameThresh, localFrameThresh, trackedPoints, None, **lk_params)
           
                # Select good points
                good_new = nextPoints[statusArray==1]
                good_old = trackedPoints[statusArray==1]

                if (len(good_new) > 0):
                    # draw the tracks
                    for i,(new,old) in enumerate(zip(good_new,good_old)):
                        a,b = new.ravel()
                        c,d = old.ravel()
           
                        wandTracks.append([a, b])
           
                    # Update which points are tracked
                    trackedPoints = good_new.copy().reshape(-1,1,2)
           
                    wandTracks = CheckForPattern(wandTracks, localFrameThresh)
           
                else:
                    # No Points were tracked, check for a pattern and start searching for wands again
                    #wandTracks = CheckForPattern(wandTracks, localFrameThresh)
                    wandTracks = []
                    findNewWands = True
            
            # Store Previous Threshold Frame
            oldFrameThresh = localFrameThresh

            
        else:
            time.sleep(0.001)

def AddIterationsPerSecText(frame, iterations_per_sec):
    """
    Add iterations per second text to lower-left corner of a frame.
    """
    cv2.putText(frame, "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
    return frame



timeLastPrintedFps = datetime.datetime.now()

inputFrameCount = 0
outputFrameCount = 0

# Initialize and traing the spell classification algorithm
InitClassificationAlgo()


# Set OpenCV video capture source
videoCapture = cv2.VideoCapture(videoSource)
# BackgroundSubtractor
# Default values: history=500, varThreshold=16, detectShadows=False
bgHistory = 500
bgThreshold = 500
bgShadows = True
fgbg = cv2.createBackgroundSubtractorMOG2(history=bgHistory, varThreshold=bgThreshold, detectShadows=bgShadows)


thresholdValue = 230
oldFrameThresh = None


# Main Loop
while True:
    # Get most recent frame
    ret, localFrame = videoCapture.read()

    if not ret:
        # If an error occurred, try initializing the video capture agai
        print("Frame error")
        videoCapture = cv2.VideoCapture(videoSource)
    else:

        # Flip the frame so the spells look like what we expect, instead of the mirror image
        # ONLY if you can't do it on the camera
        # cv2.flip(localFrame, 1, localFrame) 

        # SHOW ORIGINAL FRAME
        inputFrameCount = inputFrameCount + 1
        # Print FPS Debug info every second
        if ((datetime.datetime.now() - timeLastPrintedFps).seconds >= 1 ):
            timeLastPrintedFps = datetime.datetime.now()
            print("FPS: %d/%d" %(inputFrameCount, outputFrameCount))
            inputFrameCount = 0
            outputFrameCount = 0
        # Update Windows
        if (IsShowOriginal):
            frameWithCounts = AddIterationsPerSecText(localFrame.copy(), originalCps.countsPerSec())
            cv2.imshow("Original", frameWithCounts)

        # REMOVE BACKGROUND
        fgmask = fgbg.apply(localFrame, learningRate=0.001)
        frame_no_background = cv2.bitwise_and(localFrame, localFrame, mask = fgmask)

        if (IsShowBackgroundRemoved):
            frameNoBackgroundWithCounts = AddIterationsPerSecText(frame_no_background.copy(), noBackgroundCps.countsPerSec())
            cv2.imshow("BackgroundRemoved", frameNoBackgroundWithCounts)

        # THRESHOLD
        frame_gray = cv2.cvtColor(frame_no_background, cv2.COLOR_BGR2GRAY)
        ret, frameThresh = cv2.threshold(frame_gray, thresholdValue, 255, cv2.THRESH_BINARY);

        if (IsShowThreshold):
            frameThreshWithCounts = AddIterationsPerSecText(frameThresh.copy(), thresholdCps.countsPerSec())
            cv2.imshow("Threshold", frameThreshWithCounts)


        # PROCESS DATA
        if (IsProcessData):
            if (findNewWands):
                # Identify Potential Wand Tips using GoodFeaturesToTrack
                trackedPoints = cv2.goodFeaturesToTrack(frameThresh, 5, .01, 30)
                if trackedPoints is not None:
                    print("Found New Wands")
                    findNewWands = False
            else:
                # calculate optical flow
                nextPoints, statusArray, err = cv2.calcOpticalFlowPyrLK(oldFrameThresh, frameThresh, trackedPoints, None, **lk_params)
            
                # Select good points
                good_new = nextPoints[statusArray==1]
                good_old = trackedPoints[statusArray==1]

                if (len(good_new) > 0):
                    # draw the tracks
                    for i,(new,old) in enumerate(zip(good_new,good_old)):
                        a,b = new.ravel()
                        c,d = old.ravel()
            
                        wandTracks.append([a, b])
            
                    # Update which points are tracked
                    trackedPoints = good_new.copy().reshape(-1,1,2)
                    wandTracks = CheckForPattern(wandTracks, frameThresh)
            
                else:
                    # No Points were tracked, check for a pattern and start searching for wands again
                    #wandTracks = CheckForPattern(wandTracks, localFrameThresh)
                    print("No points tracked - reseting wand search.")
                    wandTracks = []
                    findNewWands = True
            
            # Store Previous Threshold Frame
            oldFrameThresh = frameThresh
            

        # Check for ESC key, if pressed shut everything down
        if (cv2.waitKey(1) is 27):
            break


cv2.destroyAllWindows()