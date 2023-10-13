# This script is used to run CoTracker on the demo videos of the GM demo dataset.
# The vis results seem good.

import os
import cv2
import torch
import argparse
import numpy as np
import time
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor

DEFAULT_DEVICE = ('cuda' if torch.cuda.is_available() else
                  'mps' if torch.backends.mps.is_available() else
                  'cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./data/demo_gm/",
        help="path to a video or an image sequence",
    )
    parser.add_argument(
        "--eye_hand_mode",
        default=['eye in hand'],
        help="eye hand mode",
    )
    parser.add_argument(
        "--probe_move_mode",
        default=['linear'],
        help="probe move mode",
    )
    parser.add_argument(
        "--stack",
        default=128,
        help="Update points every 128 frames",
    )
    parser.add_argument(
        "--checkpoint",
        default="./weights/cotracker/cotracker_stride_4_wind_8.pth",
        help="cotracker model",
    )
    parser.add_argument("--grid_size", type=int, default=50, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame ",
    )
    parser.add_argument(
        "--roi_center",
        type=list,
        default=None,
        help="Center of ROI",
    )
    parser.add_argument(
        "--roi_size",
        type=list,
        default=[512, 512],
        help="Center of ROI",
    )
    
    args = parser.parse_args()
    gm_categories = os.listdir(args.data_path)
    gm_categories.sort()
    for gm_cat in gm_categories:
        gm_cat_path = os.path.join(args.data_path, gm_cat)
        for eye_hand in args.eye_hand_mode:
            for probe_move in args.probe_move_mode:
                video_dir = os.path.join(gm_cat_path, eye_hand, probe_move)
                video_names = os.listdir(video_dir)
                for filename in video_names:
                    video_path = os.path.join(video_dir, filename)
                    seq_name = video_path.split("/")[-1]
                    save_folder_relative = os.path.join("output", "vis", "demo_gm", gm_cat, eye_hand, probe_move)
                    save_path = os.path.join(save_folder_relative, seq_name+'_pred_track.mp4')
                    if os.path.exists(save_path):
                        print('Skipping existing file: ', save_path)
                        continue
                    if video_path[-3:] in ['avi', 'mp4', 'mov', 'flv', 'mkv', 'MOV']:
                        # load the input video frame by frame
                        video = read_video_from_path(video_path)
                        stack_num = len(video) // args.stack + 1
                        sub_videos = []
                        for stack_id in range(stack_num):
                            try:
                                sub_video = video[stack_id*args.stack:(stack_id+1)*args.stack]
                            except: # end of video
                                sub_video = video[stack_id*args.stack:]
                            sub_video = torch.from_numpy(sub_video).permute(0, 3, 1, 2)[None].float()
                            sub_videos.append(sub_video)
                    else:
                        # raise Exception('Only video and image sequence are supported!')
                        print('Skipping non-video file: ', video_path)
                        continue
                    for split_idx, sub_video in enumerate(sub_videos):
                        model = CoTrackerPredictor(checkpoint=args.checkpoint)
                        model = model.to(DEFAULT_DEVICE)
                        model.eval()
                        sub_video = sub_video.to(DEFAULT_DEVICE)

                        # create segment mask
                        if args.roi_center is not None:
                            video_resolution = sub_video.shape[-2:]
                            segm_mask = torch.zeros((1, 1, *video_resolution)).to(sub_video.device)
                            segm_mask[:, :, 
                                      args.roi_center[1]-args.roi_size[1]//2:args.roi_center[1]+args.roi_size[1]//2, 
                                      args.roi_center[0]-args.roi_size[0]//2:args.roi_center[0]+args.roi_size[0]//2] = 1
                        start_time = time.time()
                        with torch.no_grad():
                            pred_tracks, pred_visibility = model(
                                sub_video,
                                queries=None,
                                segm_mask=segm_mask,
                                grid_size=args.grid_size,
                                grid_query_frame=args.grid_query_frame,
                            )
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Computed in {elapsed_time} s")

                        # save a video with predicted tracks
                        os.makedirs(save_folder_relative, exist_ok=True)
                        vis = Visualizer(save_dir=save_folder_relative, pad_value=0, linewidth=3)
                        vis.visualize(sub_video, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=filename[:-4]+f'_{split_idx}')