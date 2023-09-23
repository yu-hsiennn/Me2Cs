from inference import Inference
from detector import Detector
from visualize import AnimePlot
import numpy as np
import time, threading, argparse, os

class MediapipeSmooth2CustomJoints:
    def __init__(self, save_path, mode, input_path, black_background):
        self.origin_motion = []
        self.smooth_motion = []
        self.model = Inference(self.origin_motion, self.smooth_motion)
        self.step = 0
        self.finished = False
        self.save_path = save_path
        self.mode = mode
        self.input_path = input_path
        self.black_background = black_background

    def main(self):
        detector_thread = threading.Thread(target=self.run_detector)
        reader_thread = threading.Thread(target=self.smooth_origin_motion)

        detector_thread.start()
        reader_thread.start()

        detector_thread.join()
        reader_thread.join()

        self.visualize()

    def run_detector(self):
        detector = Detector(self.origin_motion, self.mode, self.input_path, self.black_background, self.save_path)
        detector.detector()
        self.finished = True

    def smooth_origin_motion(self):
        while True:
            while (len(self.origin_motion) - self.step < 20 and not self.finished):
                # prevent loop from hogging the CPU and giving a chance for other threads to execute.
                time.sleep(0)

            # inference model
            if (len(self.origin_motion) - self.step >= 20):
                self.model.main(self.step, 0)
                self.step += 10
            elif (self.finished and len(self.origin_motion) - self.step < 20):
                self.model.main(self.step, 1)
                break

    def visualize(self):
        print("save image...")
        print(f"origin frame: {len(self.origin_motion)}, smooth frame: {len(self.smooth_motion)}")
        figure = AnimePlot()
        labels = ['Mediapipe', 'Smooth']
        figure.set_fig(labels, self.save_path)
        figure.set_data([np.array(self.origin_motion), np.array(self.smooth_motion)], len(self.origin_motion))
        figure.animate()
        print(f"image was stored at: {self.save_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--save_path", type = str, help = "save path", default = "results/test")
    parser.add_argument("-v", "--static_video", help = "using static video", action="store_true")
    parser.add_argument("-i", "--input", type = str, help = "video path", default = "video_data/video.mp4")
    parser.add_argument("-b", "--black", help="set black background", action="store_true")
    args = parser.parse_args()
    
    if args.static_video == "video" and not os.path.isfile(args.input):
        print(f"File can't not found: {args.input}")
        exit()

    mediapipe_smooth = MediapipeSmooth2CustomJoints(args.save_path, args.static_video, args.input, args.black)
    mediapipe_smooth.main()