import random
import json
from typing import List, Tuple, Dict

def split_by_video(
    annotation_data: Dict[str, List],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    window_size: int = 5,
    seed: int = 42
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Splits video dataset at the video level for training and validation.

    Parameters
    ----------
    annotation_data : dict
        Mapping from video filename to list of keypoint arrays per frame.
    train_ratio : float
        Proportion of videos to use for training.
    val_ratio : float
        Proportion of videos to use for validation.
    window_size : int
        Temporal window size. Must be odd.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_pairs, val_pairs : list of (video_name, center_frame_idx)
        Frame pairs from disjoint sets of videos.
    """
    assert window_size % 2 == 1, "Window size must be odd"
    half_window = window_size // 2
    random.seed(seed)

    video_names = list(annotation_data.keys())
    random.shuffle(video_names)

    split_idx = int(len(video_names) * train_ratio)
    train_videos = video_names[:split_idx]
    val_videos = video_names[split_idx:]

    train_pairs = []
    for vid in train_videos:
        num_frames = len(annotation_data[vid])
        for idx in range(half_window, num_frames - half_window):
            train_pairs.append((vid, idx))

    val_pairs = []
    for vid in val_videos:
        num_frames = len(annotation_data[vid])
        for idx in range(half_window, num_frames - half_window):
            val_pairs.append((vid, idx))

    return train_pairs, val_pairs

def write_used_video_list(
    train_pairs: List[Tuple[str, int]],
    val_pairs: List[Tuple[str, int]],
    test_pairs: List[Tuple[str, int]],
    out_path: str
):
    """
    Writes a JSON list of unique video filenames used in the dataset splits.
    """
    used_videos = sorted(set(vid for vid, _ in train_pairs + val_pairs + test_pairs))
    with open(out_path, 'w') as f:
        json.dump(used_videos, f, indent=2) and import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms

IMAGE_SIZE = 224
NUM_KEYPOINTS = 7
ORIG_W, ORIG_H = 1024, 570  # CalMS21 original resolution

class PoseDataset(Dataset):
    def __init__(self, video_dir, annotation_data, window_size=5, frame_pairs=None):
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = window_size
        self.half_window = window_size // 2
        self.samples = []
        self.to_tensor = transforms.ToTensor()

        if frame_pairs is None:
            raise ValueError("frame_pairs must be provided")

        for vidname, center_idx in frame_pairs:
            video_file = os.path.join(video_dir, vidname)
            keypoint_seq = annotation_data[vidname]
            self.samples.append((video_file, center_idx, keypoint_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_file, center_idx, kp_seq = self.samples[idx]
        cap = cv2.VideoCapture(video_file)
        frames_tensor = []
        scaled_kps = []

        for offset in range(-self.half_window, self.half_window + 1):
            frame_idx = center_idx + offset
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx} from {video_file}")

            frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
            frames_tensor.append(self.to_tensor(frame))

            if offset == 0:
                # Scale keypoints to match resized image dimensions
                scale_x = IMAGE_SIZE / ORIG_W
                scale_y = IMAGE_SIZE / ORIG_H
                kps = kp_seq[frame_idx]
                for (x, y) in kps:
                    if x < 0 or y < 0:
                        scaled_kps.append([-1.0, -1.0])
                    else:
                        scaled_kps.append([x * scale_x, y * scale_y])

        cap.release()
        images_tensor = torch.stack(frames_tensor, dim=0)  # [T, 3, 224, 224]
        coords_tensor = torch.tensor(scaled_kps, dtype=torch.float32)  # [7, 2]

        return {
            'images': images_tensor,
            'targets': coords_tensor,
            'video_file': video_file,
            'center_idx': center_idx
        }
