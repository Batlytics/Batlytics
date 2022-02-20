import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import torch

ANGLE_BGR_COLOR = (0,165,255)
POSE_BGR_COLOR = (255,0,0)
FPS_BGR_COLOR = (0,0,255)

# Class -> bounding box color
DETECTED_CLASSES = {
    'baseball bat': (0,255,0),
    'sports ball': (0,0,255)
}

BB_THICKNESS = 3
    
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    
# Model
# Select model version - https://github.com/ultralytics/yolov5
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # or yolov5m, yolov5l, yolov5x, custom


class poseDetector:
    def __init__(
        self,
        mode=False,
        complex=1,
        smooth_landmarks=True,
        segmentation=True,
        smooth_segmentation=True,
        detectionCon=0.5,
        trackCon=0.5,
    ):

        self.mode = mode
        self.complex = complex
        self.smooth_landmarks = smooth_landmarks
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpDrawStyle = mp.solutions.drawing_styles
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode,
            self.complex,
            self.smooth_landmarks,
            self.segmentation,
            self.smooth_segmentation,
            self.detectionCon,
            self.trackCon,
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def findPose(self, img, draw = True):
        '''
        Pose estimation and drawing of the pose estimation to given frame.
        '''
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)    
          
        # Code to capture all body points and map it to a video
        if self.results.pose_landmarks:
            if draw:
                self.mp_drawing.draw_landmarks(
                    img,
                    self.results.pose_landmarks,
                    self.mpPose.POSE_CONNECTIONS,
                    self.mpDrawStyle.get_default_pose_landmarks_style())
        return img

    def findPosition(self, img, draw=False):
        '''
        Finds position of all the landmarks from previous pose estimation.
        '''
        self.lmList = []
        if self.results.pose_landmarks:
            # Save landmarks
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                x, y, z = lm.x, lm.y, lm.z
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = radians * 180.0 / np.pi

        if angle > 180.0:
            angle = 360 - angle
        elif angle < -180:
            angle += 360

        # Draw
        if draw:
            cv2.putText(
                img,
                str(int(angle)) + "",
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                ANGLE_BGR_COLOR,# (255, 0, 0),
                2,
            )
        return angle

def detect_objects(image):
    '''
    Detects objects and returns information for the bounding boxes.
    '''

    results = yolo(image)
    detected_df = results.pandas().xyxy[0]

    # Desired classes
    bounding_boxes = [] # (start_point, end_point, color)
    for class_name, bounding_box_color in DETECTED_CLASSES.items():
        # TODO: !!! Assume at most one ocurrence
        filtered_df = detected_df[detected_df['name'] == class_name]
        if len(filtered_df) > 0:
            detected_ocurrence = filtered_df.iloc[0].drop('name').astype(int)
            
            bb_start_point = (detected_ocurrence['xmin'], detected_ocurrence['ymin'])
            bb_end_point = (detected_ocurrence['xmax'], detected_ocurrence['ymax'])
            bounding_boxes.append((bb_start_point, bb_end_point, bounding_box_color))

    return bounding_boxes

def analyse_video(input_path, output_path):
    # Input
    cap = cv2.VideoCapture(input_path)

    # Output
    size = (1920, 1080)
    fourcc = cv2.VideoWriter_fourcc(*'x264')
    FPS = 25
    out = cv2.VideoWriter(output_path, fourcc, FPS, size)
    
    milliseconds = 1000
    detector = poseDetector()
    cnt = 0
    success = True
    
    while success:
        success, img = cap.read()
        
        if not success:
            break
        
        # Object detection
        detected_bounding_boxes = detect_objects(img)

        # Pose estimation + pose drawing
        img = detector.findPose(img, draw=True)
        lmList = detector.findPosition(img, draw=False)#, #draw=True)
        if len(lmList) != 0:
            detector.findAngle(img, 13, 11, 23)
            detector.findAngle(img, 24, 12, 14)

        # Drawing of object detection bounding boxes
        for bb_start_point, bb_end_point, bounding_box_color in detected_bounding_boxes:
            img = cv2.rectangle(img, bb_start_point, bb_end_point, bounding_box_color, BB_THICKNESS)
        
        # Output frame to the video
        out.write(img)

        cnt += 1

    out.release()
