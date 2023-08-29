import numpy as np   
import torch
from processing import Processing
from joint_def import JointDef
from model_loader import ModelLoader

class Inference:
    def __init__(self, origin_motion, smooth_motion):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processing = Processing()
        self.joint_def = JointDef()
        self.smooth_motion = smooth_motion
        self.origin_motion = origin_motion
        self.part_list = self.joint_def.part_list
        self.model_name = "test_V4_Choreomaster_train_angle_01_2010"
        self.models = ModelLoader(self.model_name, self.DEVICE, self.part_list).load_model()
        self.TPose = self.processing.load_TPose()

    # Using model to synthesis the motion data.
    # We default the input data have the same length(30 frames).
    def smooth(self, dim, model, data):
        test = data.to(self.DEVICE)
        test = test.view((1,-1,dim))
        # set the range for model generation
        start_generation, end_generation = 10, 20
        size_generation = end_generation - start_generation
        result = test[:, :start_generation, :]

        # synthesis the motion
        # inp was separated to three parts:
        # pre_frame: 0~9, need to generate: 10~19,  post_frame: 20~
        missing_data = torch.ones_like(test[:, :size_generation, :])
        inp = torch.cat((result[:, :start_generation, :], missing_data, test[:, end_generation:, :]), 1)
        out, _, _ = model(inp, size_generation, size_generation)                 
        result = torch.cat((result, out[:, start_generation:end_generation, :], test[:, end_generation:, :]), 1) .view((-1,dim))

        return result.detach().cpu().numpy()

    def main(self, motion_index):
        if (len(self.smooth_motion) < 10):
            data = self.processing.normalize(np.array(self.origin_motion[:10]))
            data = self.processing.calculate_angle(data)
            data = self.processing.calculate_position(data, self.TPose)
            self.smooth_motion.extend(data)
            return
        else:
            data = np.concatenate(( self.smooth_motion[-10 : ],
                                    self.origin_motion[motion_index - 10 : motion_index + 10]), axis=0)
        
        data = self.processing.normalize(data)
        data = self.processing.calculate_angle(data)
        data = torch.tensor(data.astype("float32"))
        # print(f"smooth index: {motion_index}")

        part_datas = {}
        for part in self.part_list:
            dim = self.joint_def.n_joints_part[part]
            part_data = self.joint_def.cat_torch(part, data)
            part_datas[part] = self.smooth(dim, self.models[part], part_data)
            
        pred = self.joint_def.combine_numpy(part_datas)
        pred = self.processing.calculate_position(pred, self.TPose)
        self.smooth_motion.extend(pred[10:20])

