import numpy as np
import torch

"""
Definition:
- torso: head, neck, rshoulder, lshoulder, pelvis, rthigh, lthigh (7 joints)
- left hand: lwrist, lelbow, lshoulder, neck, pelvis, rshoulder (6 joints)
- right hand: rwrist, relbow, rshoulder, neck, pelvis,  lshoulder (6 joints)
- left leg: lthigh, lknee, lankle, neck, pelvis,  rthigh (6 joints)
- right leg: rthigh, rknee, rankle, neck, pelvis, lthigh (6 joints)

- left hand with finger: lelbow, lwrist, lindex, lpinky (4 joints)
- right hand with finger: relbow, rwrist, rindex, rpinky (4 joints)
- left foot: lknee, lfoot, ltoestart (3 joints)
- right foot: rknee, rfoot, rtoestart (3 joints)
"""
class JointDef:
    def __init__(self):
        # Define constants for joint counts
        self.N_JOINTS_TORSO = 21
        self.N_JOINTS_LIMB = 18
        self.N_JOINTS_HAND = 12
        self.N_JOINTS_FOOT = 9

        # Initialize the number of joints for each part
        self.n_joints_part = {
            'torso': self.N_JOINTS_TORSO,
            'leftarm': self.N_JOINTS_LIMB,
            'rightarm': self.N_JOINTS_LIMB,
            'leftleg': self.N_JOINTS_LIMB,
            'rightleg': self.N_JOINTS_LIMB,
            'lefthand': self.N_JOINTS_HAND,
            'righthand': self.N_JOINTS_HAND,
            'leftfoot': self.N_JOINTS_FOOT,
            'rightfoot': self.N_JOINTS_FOOT
        }

        self.part_list = ['leftarm', 'rightarm', 'leftleg', 'rightleg', 'torso', 'lefthand', 'righthand', 'leftfoot', 'rightfoot']

        self.joint = {
            "Head":0, "Neck":1, 
            "RightShoulder":2, "RightArm":3, "RightHand":4, 
            "LeftShoulder":5, "LeftArm":6, "LeftHand":7, 
            "Pelvis":8, 
            "RightThigh":9, "RightKnee":10,"RightAnkle":11,
            "LeftThigh":12, "LeftKnee":13, "LeftAnkle":14,
            "LeftToeBase": 15, "RightToeBase": 16,
            "LeftHandIndex1": 17, "LeftHandPinky1": 18,
            "RightHandIndex1": 19, "RightHandPinky1": 20
        }

        self.jointsChain = [
            ["Neck","Pelvis"], ["Head","Neck"],  
            ["RightShoulder", "Neck"], ["RightArm", "RightShoulder"], ["RightHand", "RightArm"],
            ["RightThigh", "Pelvis"], ["RightKnee", "RightThigh"], ["RightAnkle", "RightKnee"],
            ["LeftShoulder", "Neck"], ["LeftArm", "LeftShoulder"], ["LeftHand", "LeftArm"], 
            ["LeftThigh", "Pelvis"], ["LeftKnee", "LeftThigh"], ["LeftAnkle", "LeftKnee"],
            ["LeftToeBase", "LeftAnkle"], ["RightToeBase", "RightAnkle"],
            ["LeftHandIndex1", "LeftHand"], ["LeftHandPinky1", "LeftHand"],
            ["RightHandIndex1", "RightHand"], ["RightHandPinky1", "RightHand"]
        ]

        self.jointIndex = {}
        for joint, idx in self.joint.items():
            self.jointIndex[joint] = (idx * 3)
        
        self.jointsConnect = [(self.jointIndex[joint[0]], self.jointIndex[joint[1]]) for joint in self.jointsChain]

    def get_joint(self):
        return self.joint

    def get_joints_connect(self):
        return self.jointsConnect
    
    def get_joints_index(self):
        return self.jointIndex
    
    def get_joints_chain(self):
        return self.jointsChain

    def cat_torch(self, part, data):
        if part == 'torso':
            part_data = torch.cat((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = torch.cat((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = torch.cat((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        elif part == 'lefthand':
            part_data = torch.cat((data[:, 18:24], data[:, 51:57]), axis=1)
        elif part == 'righthand':
            part_data = torch.cat((data[:, 9:15], data[:, 57:63]), axis=1)
        elif part == 'leftfoot':
            part_data = torch.cat((data[:, 39:45], data[:, 45:48]), axis=1)
        elif part == 'rightfoot':
            part_data = torch.cat((data[:, 30:36], data[:, 48:51]), axis=1)
        return part_data

    def cat_numpy(self, part, data):
        if part == 'torso':
            part_data = np.concatenate((data[:, 0:9], data[:, 15:18], data[:, 24:30], data[:, 36:39]), axis=1)
        elif part == 'leftarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 18:24]), axis=1)
        elif part =='rightarm':
            part_data = np.concatenate((data[:, 3:9], data[:, 15:18], data[:, 24:27], data[:, 9:15]), axis=1)
        elif part == 'leftleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 39:45]), axis=1)
        elif part == 'rightleg':
            part_data = np.concatenate((data[:, 3:6], data[:, 24:30], data[:, 36:39], data[:, 30:36]), axis=1)
        elif part == 'lefthand':
            part_data = np.concatenate((data[:, 18:24], data[:, 51:57]), axis=1)
        elif part == 'righthand':
            part_data = np.concatenate((data[:, 9:15], data[:, 57:63]), axis=1)
        elif part == 'leftfoot':
            part_data = np.concatenate((data[:, 39:45], data[:, 45:48]), axis=1)
        elif part == 'rightfoot':
            part_data = np.concatenate((data[:, 30:36], data[:, 48:51]), axis=1)
        return part_data

    def combine_numpy(self, part_datas):
        torso = part_datas['torso']
        larm = part_datas['leftarm']
        lleg = part_datas['leftleg']
        rarm = part_datas['rightarm']
        rleg = part_datas['rightleg']
        l_hand = part_datas['lefthand']
        r_hand = part_datas['righthand']
        l_foot = part_datas['leftfoot']
        r_foot = part_datas['rightfoot']
        result = np.concatenate((
            torso[:, 0:9], rarm[:, -6:], torso[:, 9:12], larm[:, -6:], 
            torso[:, 12:18], rleg[:, -6:], torso[:, 18:21], lleg[:, -6:],
            l_foot[:, -3:], r_foot[:, -3:],
            l_hand[:, -6:], r_hand[:, -6:]
        ), 1)
        
        return result