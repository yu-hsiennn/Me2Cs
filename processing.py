import pickle
import numpy as np
from joint_def import JointDef


class Processing:
    def __init__(self):
        self.TPose_path = "T-pos/T-pos-normalize.pkl"
        self.JointDef = JointDef()
        self.joints = self.JointDef.get_joint()
        self.jointConnect = self.JointDef.get_joints_connect()

    def load_TPose(self):
        with open(self.TPose_path, "rb") as f:
            TPose = pickle.load(f)[0]
        
        return TPose
    
    def get_angle(self, v):
        axis_x = np.array([1,0,0])
        axis_y = np.array([0,1,0])
        axis_z = np.array([0,0,1])

        thetax = axis_x.dot(v)/(np.linalg.norm(axis_x) * np.linalg.norm(v))
        thetay = axis_y.dot(v)/(np.linalg.norm(axis_y) * np.linalg.norm(v))
        thetaz = axis_z.dot(v)/(np.linalg.norm(axis_z) * np.linalg.norm(v))

        return thetax, thetay, thetaz

    def get_position(self, v, angles):
        r = np.linalg.norm(v)
        x = r*angles[0]
        y = r*angles[1]
        z = r*angles[2]
        return  x, y, z
    
    def normalize(self, data):
        data = data.reshape(data.shape[0], int(data.shape[1]/3), 3)
        normal_data = []
        for i, frame in enumerate(data):
            root = (frame[self.joints['RightThigh']]+frame[self.joints['LeftThigh']])/2
            data[i, self.joints['Pelvis']] = root
            normal_data.append([])
            for node in frame:
                normal_data[-1].extend(node - root)
        return np.array(normal_data)
    
    def calculate_angle(self, fullbody):
        AngleList = np.zeros_like(fullbody)
        for i, frame in enumerate(fullbody):
            for joint in self.jointConnect:
                v = frame[joint[0] : joint[0]+3]-frame[joint[1] : joint[1]+3]
                AngleList[i][joint[0] : joint[0]+3] = list(self.get_angle(v))
        return AngleList

    def calculate_position(self, fullbody, TP):
        PosList = np.zeros_like(fullbody)
        for i, frame in enumerate(fullbody):
            for joint in self.jointConnect:
                v = TP[joint[0] : joint[0]+3] - TP[joint[1] : joint[1]+3]
                angles = frame[joint[0] : joint[0]+3]
                root = PosList[i][joint[1] : joint[1]+3]
                PosList[i][joint[0] : joint[0]+3] = np.array(list(self.get_position(v, angles))) + root

        return PosList
    
    # Convert the input data(Mediapipe) into a custom format(lab joints definition).
    def Convert(self, lmlist):
        joint_list = []
        joints_index = {
            "Head": 0,
            "RightShoulder":12, "RightArm":14, "RightHand":16, 
            "LeftShoulder":11, "LeftArm":13, "LeftHand":15, 
            "RightThigh":24, "RightKnee":26,"RightAnkle":28,
            "LeftThigh":23, "LeftKnee":25, "LeftAnkle":27,
            "LeftToeBase": 31, "RightToeBase": 32,
            "LeftHandIndex1": 19, "LeftHandPinky1": 17,
            "RightHandIndex1": 20, "RightHandPinky1": 18
        }
        shoulder_middle = (lmlist[11] + lmlist[12]) / 2
        hip = (lmlist[23] + lmlist[24]) / 2
        for joint, _ in self.joints.items():
            if (joint == "Neck"):
                joint_list.append(shoulder_middle + (lmlist[0] - shoulder_middle) / 3)
            elif (joint == "Pelvis"):
                joint_list.append(hip)
            else:
                joint_list.append(lmlist[joints_index[joint]])

        return np.array(joint_list).reshape(-1)
