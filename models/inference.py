import numpy as np   
import torch
from scipy.interpolate import CubicSpline
from utils.processing import Processing
from models.joint_def import JointDef
from models.model_loader import ModelLoader
from pykalman import KalmanFilter

class Inference:
    def __init__(self, origin_motion, smooth_motion):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.processing = Processing()
        self.joint_def = JointDef()
        self.origin_motion = origin_motion
        self.smooth_motion = smooth_motion
        self.part_list = self.joint_def.part_list
        self.total_size = 30
        self.model_name = "21version_V4_Choreomaster_train_angle_01_2010"
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
        out, _, _ = model(inp, self.total_size, self.total_size)                 
        result = torch.cat((result, out[:, start_generation:end_generation, :], test[:, end_generation:, :]), 1).view((-1,dim))

        return result.detach().cpu().numpy()

    def smooth_remain_frames(self, motion_index):
        data = np.array(self.origin_motion[motion_index:])
        data = self.processing.normalize(data)
        data = self.processing.calculate_angle(data)
        data = self.kalman_filter(data)
        data = self.processing.calculate_position(data, self.TPose)
        smooth_datas = self.crossfading(np.array(self.smooth_motion[-10:]), data, 1)
        self.smooth_motion.extend(data)
        self.smooth_motion[motion_index - 10:] = smooth_datas
            
    def smooth_next_30_frames(self, motion_index):
        data = np.array(self.origin_motion[motion_index: motion_index + 30])
        data = self.processing.normalize(data)
        data = self.processing.calculate_angle(data)
        data = self.kalman_filter(data)
        data = torch.tensor(data.astype("float32"))

        part_datas = {}
        for part in self.part_list:
            dim = self.joint_def.n_joints_part[part]
            part_data = self.joint_def.cat_torch(part, data)
            part_datas[part] = self.smooth(dim, self.models[part], part_data)

        pred = self.joint_def.combine_numpy(part_datas)
        pred = self.processing.calculate_position(pred, self.TPose)
        return pred

    def kalman_1D(self, observations, damping=1):
        """
        Apply a 1D Kalman filter to a sequence of observations.

        Parameters:
            observations (list or numpy array): The sequence of observations.
            damping (float): The damping factor for the Kalman filter.

        Returns:
            numpy array: The smoothed time series data.
        """
        observation_covariance = damping
        initial_value_guess = observations[0]
        transition_matrix = 1
        transition_covariance = 0.1

        kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
        pred_state, _ = kf.smooth(observations)
        return pred_state

    def kalman_filter(self, data):
        """
        Apply the Kalman filter to a matrix of data.

        Parameters:
            data (numpy array): A matrix of time series data.

        Returns:
            numpy array: The smoothed data after applying the Kalman filter.
        """
        kalman = [self.kalman_1D(joint, damping=0.01) for joint in data.T]
        kalman = np.array(kalman).T[0]
        return kalman
    
    def crossfading(self, motion1, motion2, cross_frame = 5):
        motion1 = motion1[:-cross_frame, :]
        motion2 = motion2[cross_frame:, :]
        last_data_motion1 = motion1[-1, :]
        first_data_motion2 = motion2[0, :]

        # Define the number of interpolation steps
        num_steps = cross_frame * 2

        # Create an array of interpolation points
        alpha_values = np.linspace(0, 1, num_steps + 2)[1:-1]  # Exclude 0 and 1

        # Create a cubic spline interpolation function for each dimension
        interpolated_data = np.zeros((num_steps, motion1.shape[1]))
        for dim in range(motion1.shape[1]):
            cs = CubicSpline([0, 1], [last_data_motion1[dim], first_data_motion2[dim]])
            interpolated_data[:, dim] = cs(alpha_values)

        # Concatenate motion1, interpolated_data, and motion2 to create the final data
        final_data = np.vstack((motion1, interpolated_data, motion2))
        
        return final_data
    
    def main(self, motion_index, mode):
        if motion_index == 0:
            # Case 1: Starting from the beginning
            pred_smooth = self.smooth_next_30_frames(motion_index)
            self.smooth_motion.extend(pred_smooth)
        elif mode == 0:
            # Case 2: Smooth transition between frames
            # Total frames of original data are larger or equal to 30
            pred_smooth = self.smooth_next_30_frames(motion_index)
            smooth_datas = self.crossfading(np.array(self.smooth_motion[-10:]), pred_smooth, 1)
            self.smooth_motion.extend(pred_smooth)
            self.smooth_motion[motion_index - 10:] = smooth_datas
        else:
            # Case 3: Handling remaining frames
            self.smooth_remain_frames(motion_index)