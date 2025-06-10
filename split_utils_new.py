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
        json.dump(used_videos, f, indent=2) 
