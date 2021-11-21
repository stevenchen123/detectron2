# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from predictor import VisualizationDemo
from demo import setup_cfg

# constants
WINDOW_NAME = "COCO detections"
MAX_FRAMES = 1_000_000
input_folder = "/home/chenyuf2/Downloads/challenge_data/challenge_0317/focal300/1400"

output_video_filename = (
    "/home/chenyuf2/Downloads/challenge_data/challenge_0317/focal300/2880_linear.mp4"
)

frames_per_second = 10


if __name__ == "__main__":
    assert os.path.isdir(input_folder), input_folder

    assert not os.path.isfile(output_video_filename), output_video_filename
    output_file = None

    for i in range(MAX_FRAMES):
        if i % 10 == 0:
            print(f"iteration {i}")
        image_filename = f"{input_folder}/{i}.jpg"

        if not os.path.isfile(image_filename):
            break
        vis_frame = cv2.imread(image_filename)
        if output_file is None:
            width, height = vis_frame.shape[0:2]
            output_file = cv2.VideoWriter(
                filename=output_video_filename,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"XVID"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        output_file.write(vis_frame)

    output_file.release()
