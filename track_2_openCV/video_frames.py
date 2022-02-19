import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


count = 0
VIDEO_PATH = './Hackathon_1st_Hitter.mp4'
# vidcap = cv2.VideoCapture(VIDEO_PATH)
# def processFrame(sec):
    # vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_MSEC,120000)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    cnt = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue
        # hasFrames,image = vidcap.read()
        if success:
            img = image
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # height_part = int(image.shape[0] / 3)
            # width_part = int(image.shape[1] / 3)
            # image = image[height_part:2*height_part, width_part:2*width_part]
            height_size = 300
            width_size = 400
            width_offset = 200

            height_center = int(image.shape[0] / 2)
            width_center = int(image.shape[1] / 2)

            image = image[height_center - height_size:height_center + height_size,
                          width_center - width_size + width_offset: width_center + width_size]
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break

            cv2.imwrite("./out/image_"+str(cnt)+".jpg", image)     # save frame as JPG file
            cnt += 1
            if cnt == 500:
                break