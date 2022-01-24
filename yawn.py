# # # #Importing OpenCV Library for basic image processing functions
# # # import cv2
# # # # Numpy for array related functions
# # # import numpy as np
# # # # Dlib for deep learning based Modules and face landmark detection
# # # import dlib

# # # camera_port=0
# # # camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
# # # detector = dlib.get_frontal_face_detector()
# # # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# # # def getting_landmarks(im):
# # #     rects = detector(im,1)

# # #     if len(rects)>1:
# # #         return "error"
# # #     if len(rects) == 0:
# # #         return "error"
# # #     return np.matrix([[p.x,p.y] for p in predictor(im, rects[0]).parts()])

# # #     def annotate_landmarks (im, landmarks):
# # #         im=im.copy()
# # #         for idx, point in enumerate(landmarks):
# # #             cv2.putText(im,str(idx), pos,
# # #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
# # #                 fontScale=0.4,
# # #                 color=(1,2,255))
# # #             cv.circle(im, pos, 3, color(2,2,2))
        


# # # def top_lip(landmarks):
# # #     top_lip_pts=[]
# # #     for i in range(50,53):
# # #         top_lip_pts.append(landmarks[i])
# # #     for i in range(61,64):
# # #         top_lip_pts.append(landmarks[i])
# # #     top_lip_all_pts=np.squeeze(np.asarray(top_lip_pts))
# # #     top_lip_mean=np.mean(top_lip_pts, axis=0)
# # #     return int(top_lip_mean[:,1])

# # # def bottom_lip(landmarks):
# # #     bottom_lips_pts = []
# # #     for i in range(65,68):
# # #         bottom_lip_pts.append(landmarks[i])
# # #     for i in range(56,59):
# # #         bottom_lip_pts.append(landmarks[i])
# # #     bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
# # #     bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
# # #     return int(bottom_lip_mean[:,1])

# # #     def mouth_open(image):
# # #         landmarks = getting_landmarks(image)

# # #         if landmarks == "error":
# # #             return image, 0

# # #         image_with_landmarks = annotate_landmarks(image, landmarks)
# # #         top_lip_center = top_lip(landmarks)
# # #         bottom_lip_center = top_lip(landmarks)
# # #         lip_distance = abs(top_lip_center - bottom_lip_center)
# # #         return image_with_landmarks, lip_distance

# # #         cap = cv2.VideoCapture(0)
# # #         yawms = 0
# # #         yawn_status = False

# # #         while True:
# # #             ret, frame = cap.read()
# # #             image_landmarks, lip_distance = mouth_open(frame)

# # #             prev_yawn_status = yawn_status

# # #             if lip_distance>25:
# # #                 yawn_status = True 

# # #                 cv.putText(frame, "employee is yawning", (50,450),
# # #                             cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)

# # #                 from pygame import mixer
# # #                 mixer.init()
# # #                 mixer.music.load('Yawn.mp3')
# # #                 micer.music.play()

# # #                 output_text = "yawn count: " + str(yawns+1)

# # #                 cv2.putText(frame, output_text, (50,50),
# # #                     cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)

# # #             else:
# # #                 yawn_status = False

# # #             if prev_yawn_status == True and yawn_status==False:
# # #                 yawns += 1

# # #             cv2.imshow('Live Landmarks', image_landmarks)
# # #             cv2.imshow('Yawn Detection', frame)

# # #             if cv2.waitKey(1) == 13:
# # #                 break

# # #         cap.release()
# # #         cv2.destroyAllWindows()
# # #     


# # # Import the necessary packages 
# # import datetime as dt
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animation
# # from EAR_calculator import *
# # from imutils import face_utils 
# # from imutils.video import VideoStream
# # import matplotlib.pyplot as plt
# # import matplotlib.animation as animate
# # from matplotlib import style 
# # import imutils 
# # import dlib
# # import time 
# # import argparse 
# # import cv2 
# # from playsound import playsound
# # from scipy.spatial import distance as dist
# # import os 
# # import csv
# # import numpy as np
# # import pandas as pd
# # from datetime import datetime

# # style.use('fivethirtyeight')
# # # Creating the dataset 
# # def assure_path_exists(path):
# #     dir = os.path.dirname(path)
# #     if not os.path.exists(dir):
# #         os.makedirs(dir)


# # #all eye  and mouth aspect ratio with time
# # ear_list=[]
# # total_ear=[]
# # mar_list=[]
# # total_mar=[]
# # ts=[]
# # total_ts=[]
# # # Construct the argument parser and parse the arguments 
# # ap = argparse.ArgumentParser() 
# # ap.add_argument("-p", "--shape_predictor", required = True, help = "path to dlib's facial landmark predictor")
# # ap.add_argument("-r", "--picamera", type = int, default = -1, help = "whether raspberry pi camera shall be used or not")
# # args = vars(ap.parse_args())

# # # Declare a constant which will work as the threshold for EAR value, below which it will be regared as a blink 
# # EAR_THRESHOLD = 0.3
# # # Declare another costant to hold the consecutive number of frames to consider for a blink 
# # CONSECUTIVE_FRAMES = 20 
# # # Another constant which will work as a threshold for MAR value
# # MAR_THRESHOLD = 14

# # # Initialize two counters 
# # BLINK_COUNT = 0 
# # FRAME_COUNT = 0 

# # # Now, intialize the dlib's face detector model as 'detector' and the landmark predictor model as 'predictor'
# # print("[INFO]Loading the predictor.....")
# # detector = dlib.get_frontal_face_detector() 
# # predictor = dlib.shape_predictor(args["shape_predictor"])

# # # Grab the indexes of the facial landamarks for the left and right eye respectively 
# # (lstart, lend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
# # (rstart, rend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# # (mstart, mend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# # # Now start the video stream and allow the camera to warm-up
# # print("[INFO]Loading Camera.....")
# # vs = VideoStream(usePiCamera = args["picamera"] > 0).start()
# # time.sleep(2) 

# # assure_path_exists("dataset/")
# # count_sleep = 0
# # count_yawn = 0 

 
# # # Now, loop over all the frames and detect the faces
# # while True: 
# #     # Extract a frame 
# #     frame = vs.read()
# #     cv2.putText(frame, "PRESS 'q' TO EXIT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3) 
# #     # Resize the frame 
# #     frame = imutils.resize(frame, width = 500)
# #     # Convert the frame to grayscale 
# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     # Detect faces 
# #     rects = detector(frame, 1)

# #     # Now loop over all the face detections and apply the predictor 
# #     for (i, rect) in enumerate(rects): 
# #         shape = predictor(gray, rect)
# #         # Convert it to a (68, 2) size numpy array 
# #         shape = face_utils.shape_to_np(shape)

# #         # Draw a rectangle over the detected face 
# #         (x, y, w, h) = face_utils.rect_to_bb(rect) 
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)    
# #         # Put a number 
# #         cv2.putText(frame, "Driver", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# #         leftEye = shape[lstart:lend]
# #         rightEye = shape[rstart:rend] 
# #         mouth = shape[mstart:mend]
# #         # Compute the EAR for both the eyes 
# #         leftEAR = eye_aspect_ratio(leftEye)
# #         rightEAR = eye_aspect_ratio(rightEye)

# #         # Take the average of both the EAR
# #         EAR = (leftEAR + rightEAR) / 2.0
# #         #live datawrite in csv
# #         ear_list.append(EAR)
# #         #print(ear_list)
        

# #         ts.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
# #         # Compute the convex hull for both the eyes and then visualize it
# #         leftEyeHull = cv2.convexHull(leftEye)
# #         rightEyeHull = cv2.convexHull(rightEye)
# #         # Draw the contours 
# #         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
# #         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
# #         cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

# #         MAR = mouth_aspect_ratio(mouth)
# #         mar_list.append(MAR/10)
# #         # Check if EAR < EAR_THRESHOLD, if so then it indicates that a blink is taking place 
# #         # Thus, count the number of frames for which the eye remains closed 
# #         if EAR < EAR_THRESHOLD: 
# #             FRAME_COUNT += 1

# #             cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
# #             cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

# #             if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
# #                 count_sleep += 1
# #                 # Add the frame to the dataset ar a proof of drowsy driving
# #                 cv2.imwrite("dataset/frame_sleep%d.jpg" % count_sleep, frame)
# #                 playsound('sound files/alarm.mp3')
# #                 cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# #         else: 
# #             if FRAME_COUNT >= CONSECUTIVE_FRAMES: 
# #                 playsound('sound files/warning.mp3')
# #             FRAME_COUNT = 0
# #         #cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# #         # Check if the person is yawning
# #         if MAR > MAR_THRESHOLD:
# #             count_yawn += 1
# #             cv2.drawContours(frame, [mouth], -1, (0, 0, 255), 1) 
# #             cv2.putText(frame, "DROWSINESS ALERT!", (270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
# #             # Add the frame to the dataset ar a proof of drowsy driving
# #             cv2.imwrite("dataset/frame_yawn%d.jpg" % count_yawn, frame)
# #             playsound('sound files/alarm.mp3')
# #             playsound('sound files/warning_yawn.mp3')
# #     #total data collection for plotting
# #     for i in ear_list:
# #         total_ear.append(i)
# #     for i in mar_list:
# #         total_mar.append(i)         
# #     for i in ts:
# #         total_ts.append(i)
# #     #display the frame 
# #     cv2.imshow("Output", frame)
# #     key = cv2.waitKey(1) & 0xFF 
    
    

# #     if key == ord('q'):
# #         break

# # a = total_ear
# # b=total_mar
# # c = total_ts

# # df = pd.DataFrame({"EAR" : a, "MAR":b,"TIME" : c})
# # df.to_csv("op_webcam.csv", index=False)
# # df=pd.read_csv("op_webcam.csv")

# # df.plot(x='TIME',y=['EAR','MAR'])
# # #plt.xticks(rotation=45, ha='right')

# # plt.subplots_adjust(bottom=0.30)
# # plt.title('EAR & MAR calculation over time of webcam')
# # plt.ylabel('EAR & MAR')
# # plt.gca().axes.get_xaxis().set_visible(False)
# # plt.show()
# # cv2.destroyAllWindows()
# # vs.stop()

# #Importing OpenCV Library for basic image processing functions
# import cv2
# # Numpy for array related functions
# import numpy as np
# # Dlib for deep learning based Modules and face landmark detection
# import dlib
# #face_utils for basic operations of conversion
# from imutils import face_utils


# #Initializing the camera and taking the instance
# cap = cv2.VideoCapture(0)

# #Initializing the face detector and landmark detector
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# #status marking for current state
# sleep = 0
# drowsy = 0
# active = 0
# status=""
# color=(0,0,0)
# count_yawn=0

# def compute(ptA,ptB):
#     dist = np.linalg.norm(ptA - ptB)
#     return dist

# def blinked(a,b,c,d,e,f):
#     up = compute(b,d) + compute(c,e)
#     down = compute(a,f)
#     ratio = up/(2.0*down)

#     #Checking if it is blinked
#     if(ratio>0.25):
#         return 2
#     elif(ratio>0.21 and ratio<=0.25):
#         return 1
#     else:
#         return 0

# def mouth_aspect_ratio(landmarks): 
#             A = np.compute(landmarks[13], landmarks[19])
#             B = np.compute(landmarks[14], landmarks[18])
#             C = np.compute(landmarks[15], landmarks[17])

#             MAR = (A + B + C) / 3.0
#             return MAR

# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = detector(gray)
#     #detected face in faces array
#     for face in faces:
#         x1 = face.left()
#         y1 = face.top()
#         x2 = face.right()
#         y2 = face.bottom()

#         face_frame = frame.copy()
#         cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#         landmarks = predictor(gray, face)
#         landmarks = face_utils.shape_to_np(landmarks)

#         #The numbers are actually the landmarks which will show eye
#         left_blink = blinked(landmarks[36],landmarks[37], 
#             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
#         right_blink = blinked(landmarks[42],landmarks[43], 
#             landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        
#         #Now judge what to do for the eye blinks
#         if(left_blink==0 or right_blink==0):
#             sleep+=1
#             drowsy=0
#             active=0
#             if(sleep>6):
#                 status="SLEEPING !!!"
#                 color = (255,0,0)

#         elif(left_blink==1 or right_blink==1):
#             sleep=0
#             active=0
#             drowsy+=1
#             if(drowsy>6):
#                 status="Drowsy !"
#                 color = (0,0,255)

#         else:
#             drowsy=0
#             sleep=0
#             active+=1
#             if(active>6):
#                 status="Active :)"
#                 color = (0,255,0)

        
#         mar_list=[]
#         MAR_THRESHOLD = 14
#         MAR = mouth_aspect_ratio(landmarks)
#         mar_list.append(MAR/10)

#         if MAR > MAR_THRESHOLD:
#             count_yawn += 1
            
#         cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

#         for n in range(0, 68):
#             (x,y) = landmarks[n]
#             cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)
#     face_frame = frame.copy()
#     cv2.imshow("Frame", frame)
#     cv2.imshow("Result of detector", face_frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break


# coding: utf-8

# In[ ]:

import cv2
import dlib
import numpy as np


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
#cascade_path='haarcascade_frontalface_default.xml'
#cascade = cv2.CascadeClassifier(cascade_path)
detector = dlib.get_frontal_face_detector()


def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    
    if landmarks == "error":
        return image, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return image_with_landmarks, lip_distance

    #cv2.imshow('Result', image_with_landmarks)
    #cv2.imwrite('image_with_landmarks.jpg',image_with_landmarks)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False 

while True:
    ret, frame = cap.read()   
    image_landmarks, lip_distance = mouth_open(frame)
    
    prev_yawn_status = yawn_status  
    
    if lip_distance > 25:
        yawn_status = True 
        
        cv2.putText(frame, "Subject is Yawning", (50,450), 
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255),2)
        

        output_text = " Yawn Count: " + str(yawns + 1)

        cv2.putText(frame, output_text, (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
        
    else:
        yawn_status = False 
         
    if prev_yawn_status == True and yawn_status == False:
        yawns += 1

    cv2.imshow('Live Landmarks', image_landmarks )
    cv2.imshow('Yawn Detection', frame )
    
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
        
cap.release()
cv2.destroyAllWindows() 
