# Mediapipe to Custom Skeleton(Me2Cs)

## About
Based on [Google Mediapipe's](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) motion capture, the captured motions are processed through our pre-trained model to reduce motion jitter or restore obscured movements. Afterward, they are uniformly outputted in a custom skeleton format.

![](/results/demo.gif)

## Prerequisites
This project is developed using **Python 3.8**, so please ensure that [Python](https://www.python.org/) is installed on your computer.

# Getting started
## Installation
- Cloning this repo on your local machine:
    ```shell
    $ git clone git@github.com:yu-hsiennn/Me2Cs.git
    ```
    Or download [zip](https://github.com/yu-hsiennn/Me2Cs/archive/refs/heads/master.zip) file.
    
- change current directory
    ```shell
    $ cd me2cs
    ```

- Create model folder
    ```shell
    $ mkdir model_files
    ```
    - Download [pre-traind model](https://drive.google.com/file/d/12v-eaHdC7tia8KiKGvEs-DCEtdge8dYY/view?usp=sharing) and Unzip it, 
    - put it into model_files

- Install [pytorch](https://pytorch.org/)

- Install others modules
    ```shell
    $ pip install -r requierments.txt
    ```
    Or [Anaconda](https://www.anaconda.com/) is recommended to create the virtual environment
    ```shell
    $ conda create -f environment.yml
    $ conda activate me2cs
    ```

## Custom Skeleton
- Here, you have the option to either convert an existing FBX file or create a custom character skeleton based on the specified skeleton definition.
    - If you wish to use a character skeleton in the **FBX** file format, please use [MotionData2Lab-skeleton](https://github.com/yu-hsiennn/MotionData2Lab-skeleton) to convert it into the **pickle** file format. 
    - Alternatively, you can create your own character skeleton based on the **skeleton definition** provided below.

- Please refer to the pickle files inside the "T-pos" folder as a reference. The final **custom skeleton format must match theirs**, with the **only difference being the values** contained within the files.
- Put your T-pos files into T-pos folder and modify:
    - **processing.py** (Path: utils/processing.py)
    ```python=8
        self.TPose_path = "T-pos/YOUR T-POS FILE PATH"
    ```

## Running
- **Argument**:
  -  `-s` or `--save_path`: save path
  -  `-v` or `--static_video`: using static video
  -  `-i` or `--input`: video path
  -  `-b` or `--black`: set black background
- **Run**:
    ```shell
      # default streaming
    $ python main.py
    
      # static video
    $ python main.py -i "video path"
    ```
    
## Others
### Custom Skeleton definition
![](/T-pos/Tpose.jpg)
```typescript
0 - Head
1 - Neck
2 - Right shoulder
3 - Right elbow
4 - Right wrist
5 - Left shoulder
6 - Left elbow
7 - Left wrist
8 - Hip
9 - Right upperleg
10 - Right knee
11 - Right ankle
12 - Left upperleg
13 - Left knee
14 - Left ankle
15 - Left toe
16 - Right toe
17 - Left index
18 - Left pinky
19 - Right index
20 - Right pinky
```

## License
This project is licensed under the MIT. See the [LICENSE](/LICENSE) file for details
