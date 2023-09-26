import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from models.joint_def import JointDef

class AnimePlot():
    def __init__(self):
        self.fig = plt.figure()
        self.ax = []
        self.Lab_joints = JointDef()
        self.joint = self.Lab_joints.get_joint()
        self.jointChain = self.Lab_joints.get_joints_chain()
    
    def set_fig(self, labels, save_path, scale = 2.5):
        self.scale = scale
        self.save_path = save_path
        for i in range(len(labels)):
            self.ax.append(self.fig.add_subplot(1, len(labels), i+1, projection="3d"))
            self.ax[i].set_title(labels[i])
        self.time_text = self.fig.text(.5, .3, "0", ha="center")
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

    def set_data(self, data, frame_num=300):
        self.frame_num = frame_num
        for i, _ in enumerate(data):
            data[i] = data[i].reshape(data[i].shape[0], int(data[i].shape[1]/3), 3)
            data[i] = data[i]*self.scale
        self.data = data

    def ani_init(self):
        for figure in self.ax:
            figure.set_xlabel('x')
            figure.set_ylabel('y')
            figure.set_zlabel('z')
            figure.set_xlim(-.8*self.scale, .8*self.scale)
            figure.set_ylim(-.8*self.scale, .8*self.scale)
            figure.set_zlim(-.8*self.scale, .8*self.scale)
            figure.axis('off') #hide axes
            figure.view_init(elev=300,azim=-90)
    
    def ani_update(self, i):
        for figure in self.ax:
            figure.lines.clear()
            figure.collections.clear()
        for f, motion in enumerate(self.data):
            for idx, chain in enumerate(self.jointChain):
                pre_node = self.joint[chain[0]]
                next_node = self.joint[chain[1]]
                # mirror image
                x = -np.array([motion[i, pre_node, 0], motion[i, next_node, 0]])
                y = np.array([motion[i, pre_node, 1], motion[i, next_node, 1]])
                z = np.array([motion[i, pre_node, 2], motion[i, next_node, 2]])
                if idx < 8:
                    # right, red
                    self.ax[f].plot(x, y, z, color="#e74c3c")
                elif 8 <= idx < 14:
                    # left ,blue
                    self.ax[f].plot(x, y, z, color="#3498db")
                elif idx == 14 or idx == 15:
                    # foot 
                    self.ax[f].plot(x, y, z, color='cyan')
                else:
                    # finger
                    self.ax[f].plot(x, y, z, color="magenta")

        self.time_text.set_text(str(i))

    def animate(self):
        self.anime = animation.FuncAnimation(self.fig, self.ani_update, self.frame_num, interval=1,init_func=self.ani_init)
        # Write a gif
        f = f"{self.save_path}.gif"
        writergif = animation.PillowWriter(fps = 5)
        self.anime.save(f, writer=writergif)
        # Write a video
        # writervideo = animation.FFMpegWriter(fps = 10)
        # f = f"{self.save_path}.mp4"
        # self.anime.save(f, writer=writervideo)