# 🫘 Grain material tracking
A demo for grain matertial tracking using point tracker.
Please refer to [CoTracker](https://github.com/facebookresearch/co-tracker) for more details.

## 🔧 Installation
- Install dependency as follows:
    ```bash
    pip install torch # better installed with cuda
    pip install opencv-python
    pip install git+https://github.com/facebookresearch/co-tracker.git
    ```

- Install model checkpoint as follows:
    ```bash
    mkdir weights && cd weights
    mkdir cotracker && cd cotracker
    wget https://dl.fbaipublicfiles.com/cotracker/cotracker_stride_4_wind_8.pth
    cd ../..
    ```

- Others
    ```bash
    mkdir data/demo_gm
    ```

## ✈️ Get started
- Prepare your data in ```./data/demo_gm/```, use the structure as follows:
  ```bash
    ├── cassia_seed
    │   ├── eye in hand
    │   │   ├── linear
    │   │   │   ├── ***.MOV
    │   │   │   └── ...
  ```
- If you would like to track in a region of interest (ROI), please set ```--roi_center [x, y]```, where ```[x,y]``` is the center of the ROI.
  ```
  python demo_gm.py --grid_size 30
  ```
  Please note the default parameters in the ```demo_gm.py``` file if you are interested.

