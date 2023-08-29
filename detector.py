import mediapipe as mp
import numpy as np
import cv2
import collections
from processing import Processing

# It is used to capture the human motion.
class Detector:
    def __init__(self, origin_motion, fps = 30, black_background = False, static_image = False):
        self.MP = Mediapipe(mode=static_image)
        self.processing = Processing()
        self.black_background = black_background
        self.origin_motion = origin_motion
        self.static_image = static_image
        self.fps = fps

    def detector(self, video_path="", save_path="results/output"):
        # streaming video
        if not self.static_image:
            self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        # static video
        else:
            print(f"loading file... => {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            self.out = cv2.VideoWriter(
                save_path + ".mp4",
                fps=self.fps,
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                frameSize=(int(self.cap.get(3)), int(self.cap.get(4))),
            )

        while self.cap.isOpened():
            success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break

            image, p_landmarks, p_connections = self.MP.findPose(image, False)
            # use black background
            if self.black_background:
                image = image * 0

            # draw points
            mp.solutions.drawing_utils.draw_landmarks(image, p_landmarks, p_connections)
            lmList = self.MP.getPosition(image)

            if len(lmList.keys()):
                joints_list = self.processing.Convert(lmList)
                self.origin_motion.append(joints_list)

            # Flip the image horizontally for a selfie-view display.
            if video_path != "":
                self.out.write(image)
                cv2.waitKey(10)
            cv2.imshow("MediaPipe Pose", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        if video_path != "":
            self.out.release()
        cv2.destroyAllWindows()

# Using Mediapipe model that release by google to get the human joints coordinate.
class Mediapipe:
    def __init__(self, mode = False, complexity = 1, smooth_landmark = True, enable_seg = False, smooth_seg = True, min_conf = 0.7, min_tra = 0.6):
        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(
            mode, 
            complexity,
            smooth_landmark,
            smooth_seg,
            enable_seg,
            min_conf,
            min_tra
        )
        self.joint_index = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 28, 31, 32]

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS
    
    def getPosition(self, img, draw=True):
        """
        Inference Mediapipe Model to get the joints coordinate, then return predict value(world coordinate)
        lmList[index]: [x, y, z](np.array), https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md
        """
        lmList= collections.defaultdict(np.array)
        if self.results.pose_world_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                cx, cy = int(lm.x), int(lm.y)
                if id in self.joint_index:
                    # the coordinate axes between the model and Mediapipe are opposite.
                    lmList[id] = np.array([lm.x, -lm.y, lm.z])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return lmList

if __name__ == "__main__":
    Detector = Detector(list)
    Detector.detector()