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
