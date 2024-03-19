import sys, numpy, os
import urllib
import numpy as np
import time
from subprocess import call
import glob
import base64
import random
import cv2
import face_recognition
import dlib
import math
import argparse
from playsound import playsound

landmarkModelFile = 'shape_predictor_68_face_landmarks.dat'



landmarkModelFile = 'shape_predictor_68_face_landmarks.dat'

argParser = argparse.ArgumentParser()
argParser.add_argument('--landmark_model', default=landmarkModelFile,
                    help='Location of the Landmark File.')
argParser.add_argument('--video_file', default='/dev/video0',
                    help='Video Source File. Default Web Cam')
args = argParser.parse_args()
landmarkModelFile = args.landmark_model
videoSource = args.video_file

faceBBoxDetector = dlib.get_frontal_face_detector()
landmarksPredictor = dlib.shape_predictor(landmarkModelFile)

# 3D landmarks (From LearnOpenCV.com)
landmarks3d = np.array([
                        (0.0, 0.0, 0.0),             # Nose tip
                        (0.0, -330.0, -65.0),        # Chin
                        (-225.0, 170.0, -135.0),     # Left eye left corner
                        (225.0, 170.0, -135.0),      # Right eye right corner
                        (-150.0, -150.0, -125.0),    # Left Mouth corner
                        (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
# Corresponding 2D landmarks
# Corresponding 2D landmarks
landmarks2dIdx = np.array((33, 8, 36, 45, 48, 54), dtype=np.uint32)

def getCameraMatrix(imgSize):
    focalLength = imgSize[1]
    center = (imgSize[1]/2, imgSize[0]/2)
    cameraMat = np.array([
                        (focalLength, 0, center[0]),
                        (0, focalLength, center[1]),
                        (0, 0, 1)
                        ], dtype=np.float64)
    return cameraMat

def rotationMatrixToEulerAngles(R):
    def isRotationMatrix(R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    x *= 180.0/np.pi
    y *= 180.0/np.pi
    z *= 180.0/np.pi
    return x, y, z

def getLandmarks(img):
    global faceBBoxDetector
    global landmarksPredictor

    def shapeToLandmarks(shape):
        landmarks = np.zeros((2,68), dtype=np.uint32)
        for i in  range(68):
            landmarks[0,i], landmarks[1,i] = shape.part(i).x, shape.part(i).y
        return landmarks

    rects = faceBBoxDetector(img, 0)
    if len(rects) == 0:
        success = False
        return success, None
    else: success = True
    # rect = getMaxAreaRect(rects)
    rect = rects[0]
    shape = landmarksPredictor(img, rect)
    return success, shapeToLandmarks(shape)

def plotLandmarks(img, landmarks):
    for (x,y) in landmarks.transpose():
        cv2.circle(img, (x,y), 3, (0,0,255), -1)

def plotTextonImg(img, text):
    cv2.putText(img, text, (15,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1)

#step3:face recognition

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.

input1 = face_recognition.load_image_file("22.jpg")
raveendra_face_encoding = face_recognition.face_encodings(input1)[0]

input2 = face_recognition.load_image_file("33.jpg")
sreekanth_face_encoding = face_recognition.face_encodings(input2)[0]



# Create arrays of known face encodings and their names
known_face_encodings = [
    raveendra_face_encoding,sreekanth_face_encoding
    
]
known_face_names = [
    "Raveendra","sreekanth"
    
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    success, frame = video_capture.read()
    cameraMat = getCameraMatrix(frame.shape)
    distCoeffs = np.zeros((4,1)) # Assuming no distortions.
    success, frame = video_capture.read()
    print('Thank you for concentrating')
    if not success:
        print('Error: Could not read the frame')
        break
    success, landmarks = getLandmarks(frame)
    if not success:
        plotTextonImg(frame, 'Face not detected.')
        cv2.imshow('Video Feed', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue
    landmarks2d = landmarks[:, landmarks2dIdx]
    plotLandmarks(frame, landmarks2d)
    landmarks2d = landmarks2d.astype(np.float64)
    success, rotationVec, translationVec = cv2.solvePnP(landmarks3d, landmarks2d.T, cameraMat,
                                            distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rotationMat = cv2.Rodrigues(rotationVec)[0]
    pitch, yaw, roll = rotationMatrixToEulerAngles(rotationMat)
    plotTextonImg(frame, "Pitch:{:3.2}, Yaw:{:3.2}, Roll:{:3.2} ".format(pitch, yaw, roll))
    noseEndPoint = cv2.projectPoints(np.array([(0,0,1000.0)]), rotationVec, translationVec,
                                            cameraMat, distCoeffs)[0]

    if yaw>30:
        print('please dont turn Right')
        cv2.imwrite("2.jpg",frame)
        playsound('alaram.mp3')
        print('alarm')
    if yaw<-30:
        print('please dont turn left')
        cv2.imwrite("3.jpg",frame)
        playsound('beeb.mp3')
        print('alarm')
    point1 = (int(landmarks2d[0,0]), int(landmarks2d[1,0]))
    point2 = (int(noseEndPoint[0][0][0]), int(noseEndPoint[0][0][1]))
    #cv2.imshow("resized output",frame)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
            
    process_this_frame = not process_this_frame
    if face_names==known_face_names:
        print("you are allowed")

        
        
    else:
        print('unknown')
        print("you are not allowed")
        

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
    # Display the resulting image
    cv2.imshow('output videoVideo', frame)
   
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()










    

