import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp
import plotly.express as px
import plotly.graph_objects as go

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

    def findPose(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)    
        # self.plotly_fig(self.results.pose_landmarks)   
        print(self.results.pose_landmarks)
        print('-----------------------------------------------------------------------------------------------------------')     
          
        # if self.results.pose_landmarks:
        #     if draw:
        #         self.mp_drawing.draw_landmarks(
        #             img,
        #             self.results.pose_landmarks,
        #             self.mpPose.POSE_CONNECTIONS,
        #             # self.mpDrawStyle.get_default_pose_landmarks_style())
        #             self.mpDraw.DrawingSpec(
        #                 color=(0, 0, 255), thickness=2, circle_radius=2
        #             ),
        #             self.mpDraw.DrawingSpec(
        #                 color=(0, 255, 0), thickness=2, circle_radius=2
        #             ),
        #         )
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
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
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        print(int(angle))

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.putText(
                img,
                str(int(angle)) + "",
                (x2 - 50, y2 + 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (255, 0, 0),
                2,
            )
        return angle
    
    
    def plotly_fig(self, results):
        if not results:
            return
        plotted_landmarks = {}
        _PRESENCE_THRESHOLD = 0.5
        _VISIBILITY_THRESHOLD = 0.5
        for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
            if (
                    landmark.HasField("visibility")
                    and landmark.visibility < _VISIBILITY_THRESHOLD
            ) or (
                    landmark.HasField("presence") and landmark.presence < _PRESENCE_THRESHOLD
            ):
                continue
            plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
        if self.results.pose_landmarks.landmark:
            out_cn = []
            num_landmarks = len(self.results.pose_landmarks.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in self.mpPose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(
                        f"Landmark index is out of range. Invalid connection "
                        f"from landmark #{start_idx} to landmark #{end_idx}."
                    )
                if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                    landmark_pair = [
                        plotted_landmarks[start_idx],
                        plotted_landmarks[end_idx],
                    ]
                    out_cn.append(
                        dict(
                            xs=[landmark_pair[0][0], landmark_pair[1][0]],
                            ys=[landmark_pair[0][1], landmark_pair[1][1]],
                            zs=[landmark_pair[0][2], landmark_pair[1][2]],
                        )
                    )
            cn2 = {"xs": [], "ys": [], "zs": []}
            for pair in out_cn:
                for k in pair.keys():
                    cn2[k].append(pair[k][0])
                    cn2[k].append(pair[k][1])
                    cn2[k].append(None)
    
        df = pd.DataFrame(plotted_landmarks).T.rename(columns={0: "z", 1: "x", 2: "y"})
        df["lm"] = df.index.map(lambda s: self.mpPose.PoseLandmark(s).name).values
        fig = (
            px.scatter_3d(df, x="z", y="x", z="y", hover_name="lm")
                .update_traces(marker={"color": "red"})
                .update_layout(
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                scene={"camera": {"eye": {"x": 2.1, "y": 0, "z": 0}}},
            )
        )
        fig.add_traces(
            [
                go.Scatter3d(
                    x=cn2["xs"],
                    y=cn2["ys"],
                    z=cn2["zs"],
                    mode="lines",
                    line={"color": "black", "width": 5},
                    name="connections",
                )
            ]
        )
        return fig
        

def main():
    cap = cv2.VideoCapture('./Hackathon_1st_Hitter.mp4')
    milliseconds = 1000
    start_time = int(input("Enter Start time: "))
    end_time = int(input("Enter Length: "))
    end_time = start_time + end_time
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * milliseconds)
    pTime = 0
    detector = poseDetector()
    while True and cap.get(cv2.CAP_PROP_POS_MSEC) <= end_time * milliseconds:
        success, img = cap.read()
        img = detector.findPose(img)       
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            detector.findAngle(img, 11, 13, 15)
            detector.findAngle(img, 24, 12, 14)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # show fps count
        cv2.putText(
            img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
        )

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()