import os
import numpy as np

def load_annotations(npy_path):
    seq_all = np.load(npy_path, allow_pickle=True).item()
    seq = seq_all['annotator-id_0']
    annotation_data = {}
    for uid, entry in seq.items():
        fname = os.path.basename(uid) + ".mp4"
        keypoints = [frame[0].T for frame in entry['keypoints']]  # [7, 2] format
        annotation_data[fname] = keypoints
    return annotation_data
