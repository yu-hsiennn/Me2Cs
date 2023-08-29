from inference import Inference
from detector import Detector
from visualize import AnimePlot
import numpy as np
import time, threading

class MediapipeSmooth2CustomJoints:
    def __init__(self):
        self.origin_motion = []
        self.smooth_motion = []
        self.model = Inference(self.origin_motion, self.smooth_motion)
        self.lock = threading.Lock()
        self.step = 0
        self.finished = False
        self.save_path = "results/test"

    def main(self):
        detector_thread = threading.Thread(target=self.run_detector)
        reader_thread = threading.Thread(target=self.smooth_origin_motion)

        detector_thread.start()
        reader_thread.start()

        detector_thread.join()
        reader_thread.join()

        self.visualize()

    def run_detector(self):
        detector = Detector(self.origin_motion)
        detector.detector()
        self.finished = True

    def smooth_origin_motion(self):
        while True:
            while (len(self.origin_motion) - self.step < 20 and not self.finished):
                # prevent loop from hogging the CPU and giving a chance for other threads to execute.
                time.sleep(0)

            # inference model
            if (len(self.origin_motion) - self.step >= 20):
                self.model.main(self.step)
                self.step += 10
            if (self.finished and len(self.origin_motion) - self.step < 20):
                print("--------------------------------")
                for motion in self.origin_motion[self.step:]:
                    self.smooth_motion.append(motion)
                break

    def visualize(self):
        print("save image...")
        print(f"origin frame: {len(self.origin_motion)}, smooth frame: {len(self.smooth_motion)}")
        figure = AnimePlot()
        labels = ['Origin', 'Smooth']
        figure.set_fig(labels, self.save_path)
        figure.set_data([np.array(self.origin_motion), np.array(self.smooth_motion)], len(self.origin_motion))
        figure.animate()
        print(f"image was stored at: {self.save_path}")



if __name__ == "__main__":
    mediapipe_smooth = MediapipeSmooth2CustomJoints()
    mediapipe_smooth.main()