import cv2
import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# ---------------------------------------------------------------------
filter = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0,-1, 0]
    ]
)

def sharpen(img):
    sharpen_img = cv2.filter2D(img, -1, filter)
    return sharpen_img

# ---------------------------------------------------------------
dim = (720, 385)

cap = cv2.VideoCapture('../video_file/Hackathon_high_home_1_Trim.mp4')

ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, dim)
pts1 = np.float32([[502,57], [218,57], [690,320], [30,320]])
pts2 = np.float32([[0,0], [dim[0], 0],[0, dim[1]], [dim[0], dim[1]]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
frame1 = cv2.warpPerspective(frame1, matrix, dim)

ret, frame2 = cap.read()
frame2 = cv2.resize(frame2, dim)
matrix = cv2.getPerspectiveTransform(pts1, pts2)
frame2 = cv2.warpPerspective(frame2, matrix, dim)

frame1 = sharpen(frame1)
frame2 = sharpen(frame2)

while True:

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 10, 510, 50)

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        print(x,y)
        if cv2.contourArea(contour) > 100 and cv2.contourArea(contour) < 450:
            cv2.rectangle(frame1, (x,y), (x+w, y+h), (255, 255, 0), 1)
        # elif cv2.contourArea(contour) < 30:
        #     cv2.rectangle(frame1, (x,y), (x+w, y+h), (0, 255, 0), 2)
        # else:
        #     cv2.rectangle(frame1, (x,y), (x+w, y+h), (255, 255, 0), 2)


    cv2.imshow('video', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    frame2 = cv2.resize(frame2, dim)

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    frame2 = cv2.warpPerspective(frame2, matrix, dim)
    frame2 = sharpen(frame2)
    if cv2.waitKey(27) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# import cv2
# import sys

# (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# if __name__ == '__main__' :

#     # Set up tracker.
#     # Instead of MIL, you can also use

#     tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
#     tracker_type = tracker_types[-1]

#     if int(minor_ver) < 3:
#         tracker = cv2.Tracker_create(tracker_type)
#     else:
#         if tracker_type == 'BOOSTING':
#             tracker = cv2.TrackerBoosting_create()
#         if tracker_type == 'MIL':
#             tracker = cv2.TrackerMIL_create()
#         if tracker_type == 'KCF':
#             tracker = cv2.TrackerKCF_create()
#         if tracker_type == 'TLD':
#             tracker = cv2.TrackerTLD_create()
#         if tracker_type == 'MEDIANFLOW':
#             tracker = cv2.TrackerMedianFlow_create()
#         if tracker_type == 'GOTURN':
#             tracker = cv2.TrackerGOTURN_create()
#         if tracker_type == 'MOSSE':
#             tracker = cv2.TrackerMOSSE_create()
#         if tracker_type == "CSRT":
#             tracker = cv2.TrackerCSRT_create()

#     # Read video
#     video = cv2.VideoCapture("../video_file/Hackathon_high_home_1_Trim.mp4")

#     # Exit if video not opened.
#     if not video.isOpened():
#         print("Could not open video")
#         sys.exit()

#     # Read first frame.
#     ok, frame = video.read()
#     if not ok:
#         print('Cannot read video file')
#         sys.exit()
    
#     # Define an initial bounding box
#     bbox = (287, 23, 86, 320)

#     # Uncomment the line below to select a different bounding box
#     bbox = cv2.selectROI(frame)

#     # Initialize tracker with first frame and bounding box
#     ok = tracker.init(frame, bbox)

#     while True:
#         # Read a new frame
#         ok, frame = video.read()
#         if not ok:
#             break
        
#         # Start timer
#         timer = cv2.getTickCount()

#         # Update tracker
#         ok, bbox = tracker.update(frame)

#         # Calculate Frames per second (FPS)
#         fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

#         # Draw bounding box
#         if ok:
#             # Tracking success
#             p1 = (int(bbox[0]), int(bbox[1]))
#             p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
#             cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
#         else :
#             # Tracking failure
#             cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

#         # Display tracker type on frame
#         cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
#         # Display FPS on frame
#         cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

#         # Display result
#         cv2.imshow("Tracking", frame)

#         # Exit if ESC pressed
#         k = cv2.waitKey(1) & 0xff
#         if k == 27 : break