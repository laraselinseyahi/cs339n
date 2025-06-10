import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from tqdm import trange
import cv2
import random

from model import PoseTransformer
from pose_dataset import PoseDataset
from data_utils import load_annotations
from split_utils_new import split_by_video

# ----------------------------
# UTILS
# ----------------------------
def softargmax_2d(logits, orig_w=1024, orig_h=570):
    B, K, H, W = logits.shape
    probs = torch.softmax(logits.view(B, K, -1), dim=-1).view(B, K, H, W)
    device = logits.device
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=device),
        torch.linspace(0, W - 1, W, device=device),
        indexing='ij'
    )
    x = (probs * grid_x).view(B, K, -1).sum(-1)
    y = (probs * grid_y).view(B, K, -1).sum(-1)
    x = x / W * orig_w
    y = y / H * orig_h
    return torch.stack([x, y], dim=-1)

def kl_heatmap_loss(logits, targets):
    B, K, H, W = logits.shape
    logits = logits.view(B, K, -1)
    targets = targets.view(B, K, -1)
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, targets, reduction='batchmean')

def compute_l2_error(model, dataset, annotation_data, max_samples=100):
    model.eval()
    total_dist = 0.0
    count = 0
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            sample = dataset[i]
            image_seq = sample['images'].unsqueeze(0).to(DEVICE)
            logits = model(image_seq)
            pred_coords = softargmax_2d(logits)[0].cpu().numpy()
            vidname = os.path.basename(sample['video_file'])
            frame_idx = sample['center_idx']
            gt_coords = np.array(annotation_data[vidname][frame_idx])
            gt = gt_coords[0]
            pred = pred_coords[0]
            if gt[0] >= 0 and pred[0] >= 0:
                total_dist += np.linalg.norm(gt - pred)
                count += 1
    return total_dist / count if count > 0 else float('nan')

def visualize_predictions(model, dataset, annotation_data, save_dir, num_samples=5):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            input_tensor = sample['images'].unsqueeze(0).to(DEVICE)
            logits = model(input_tensor)
            pred_coords = softargmax_2d(logits)[0].cpu().numpy()
            gt_coords = np.array(annotation_data[os.path.basename(sample['video_file'])][sample['center_idx']])
            cap = cv2.VideoCapture(sample['video_file'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample['center_idx'])
            ret, frame = cap.read()
            cap.release()
            if not ret: continue
            plt.figure(figsize=(6, 4))
            plt.imshow(frame[..., ::-1])
            plt.scatter(gt_coords[0, 0], gt_coords[0, 1], c='lime', marker='o', label='GT')
            plt.scatter(pred_coords[0, 0], pred_coords[0, 1], c='red', marker='x', label='Pred')
            plt.legend()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"viz_{i:02d}.png"))
            plt.close()

# ----------------------------
# CONFIG
# ----------------------------
TRAIN_ANNOTATION_PATH = "/home/ubuntu/stats-320-file/calms21_task1_train.npy"
TEST_ANNOTATION_PATH = "/home/ubuntu/stats-320-file/calms21_task1_test.npy"
VIDEO_DIR = "/home/ubuntu/stats-320-file/task1_videos_mp4/train"
VIDEO_DIR_T = "/home/ubuntu/stats-320-file/task1_videos_mp4/test"
SAVE_DIR = "./experiment_a_again"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 7
WINDOW_SIZE = 5
LEARNING_RATE = 1e-4
# ---------------------------- DATA ----------------------------
print("🔄 Loading annotations...")
train_annots = load_annotations(TRAIN_ANNOTATION_PATH)
test_annots_full = load_annotations(TEST_ANNOTATION_PATH)

# Split test videos into validation and final test sets (video-wise)
test_videos = list(test_annots_full.keys())
random.seed(42)
random.shuffle(test_videos)
split_index = len(test_videos) // 2
val_videos = test_videos[:split_index]
final_test_videos = test_videos[split_index:]

val_annots = {vid: test_annots_full[vid] for vid in val_videos}
test_annots = {vid: test_annots_full[vid] for vid in final_test_videos}

# Limit train and val pairs to 1000 and 200 respectively
train_pairs, _ = split_by_video(train_annots, window_size=WINDOW_SIZE)
train_pairs = train_pairs[:1000]
val_pairs, _ = split_by_video(val_annots, window_size=WINDOW_SIZE)
val_pairs = val_pairs[:200]

# Print actual sample counts for debug
print(f"✅ Training on {len(train_pairs)} frames")
print(f"✅ Validating on {len(val_pairs)} frames")

train_dataset = PoseDataset(VIDEO_DIR, train_annots, WINDOW_SIZE, train_pairs, only_keypoint=0)
val_dataset = PoseDataset(VIDEO_DIR_T, val_annots, WINDOW_SIZE, val_pairs, only_keypoint=0)
test_pairs = []
half_window = WINDOW_SIZE // 2
for vid, kps in test_annots.items():
    for idx in range(half_window, len(kps) - half_window):
        test_pairs.append((vid, idx))
test_pairs = test_pairs[:200]
test_dataset = PoseDataset(VIDEO_DIR_T, test_annots, WINDOW_SIZE, test_pairs, only_keypoint=0)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)


# ----------------------------
# MODEL
# ----------------------------
model = PoseTransformer(num_keypoints=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---------------------------- TRAIN LOOP ----------------------------
print("🚀 Starting training (experiment A: 1kp)...")
best_l2 = float('inf')
patience_counter = 0
loss_history, l2_history = [], []

csv_path = os.path.join(SAVE_DIR, "metrics.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Epoch', 'KL_Loss', 'Val_L2'])

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = trange(len(train_loader), desc=f"Train {epoch+1}/{EPOCHS}", leave=False)

        for batch in train_loader:
            images = batch['images'].to(DEVICE)
            heatmaps = batch['heatmaps'][:, 0:1].to(DEVICE)
            logits = model(images)
            loss = kl_heatmap_loss(logits, heatmaps) * 10
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.update(1)

        pbar.close()
        avg_loss = total_loss / len(train_loader)
        avg_l2 = compute_l2_error(model, val_dataset, val_annots)
        loss_history.append(avg_loss)
        l2_history.append(avg_l2)
        print(f"📘 Epoch {epoch+1:2d} | KL Loss: {avg_loss:.4f} | Val L2 Error: {avg_l2:.2f} px")

        writer.writerow([epoch+1, avg_loss, avg_l2])

        if avg_l2 < best_l2:
            best_l2 = avg_l2
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pose_transformer_exp_a_best.pt"))
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0:
            epoch_viz_dir = os.path.join(SAVE_DIR, f"viz_epoch_{epoch+1:02d}")
            visualize_predictions(model, val_dataset, val_annots, epoch_viz_dir)

        if patience_counter >= PATIENCE:
            print(f"⏹️ Early stopping at epoch {epoch+1}.")
            break


# Save final model
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pose_transformer_exp_a_last.pt"))
print("✅ Model saved.")

# Plot
plt.figure()
plt.plot(loss_history, label="KL Div Loss")
plt.plot(l2_history, label="Val L2 Error (px)")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.title("Training Curve (Experiment A)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "loss_l2_curve.png"))
plt.close()

# ----------------------------
# FINAL TEST EVAL
# ----------------------------
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "pose_transformer_exp_a_best.pt")))
model.eval()

test_csv_path = os.path.join(SAVE_DIR, "test_eval_l2_per_frame.csv")
with open(test_csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Frame", "Video", "GT_X", "GT_Y", "Pred_X", "Pred_Y", "L2"])
    total_dist = 0.0
    count = 0
    for i, sample in enumerate(test_dataset):
        input_tensor = sample['images'].unsqueeze(0).to(DEVICE)
        logits = model(input_tensor)
        pred_coords = softargmax_2d(logits)[0].cpu().numpy()
        gt_coords = np.array(test_annots[os.path.basename(sample['video_file'])][sample['center_idx']])
        gt = gt_coords[0]
        pred = pred_coords[0]
        if gt[0] >= 0 and pred[0] >= 0:
            dist = np.linalg.norm(gt - pred)
            total_dist += dist
            count += 1
            writer.writerow([sample['center_idx'], os.path.basename(sample['video_file']), gt[0], gt[1], pred[0], pred[1], dist])

visualize_predictions(model, test_dataset, test_annots, os.path.join(SAVE_DIR, "test_eval_viz"), num_samples=10)

print(f"📊 Final Test L2 Error over {count} frames: {total_dist / count:.2f} px")

# visualize metrics 
