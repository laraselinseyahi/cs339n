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
    targets = targets.view(B, K, -1)  # ✅ This line was missing or incorrect
    log_probs = F.log_softmax(logits, dim=-1)
    return F.kl_div(log_probs, targets, reduction='batchmean')




def coordinate_l2_loss(logits, targets, orig_w=1024, orig_h=570):
    pred_coords = softargmax_2d(logits, orig_w, orig_h)      # (B, K, 2)
    target_coords = softargmax_2d(targets, orig_w, orig_h)    # (B, K, 2)
    mse = F.mse_loss(pred_coords, target_coords)
    return mse  # now returns unnormalized L2 in pixels





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
def save_heatmap_visualizations(model, dataset, save_dir, num_samples=5):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            input_tensor = sample['images'].unsqueeze(0).to(DEVICE)
            logits = model(input_tensor)[0, 0].cpu().numpy()
            gt_heatmap = sample['heatmaps'].cpu().numpy()
            if gt_heatmap.ndim == 3:
                gt_heatmap = gt_heatmap[0]
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(gt_heatmap, cmap='viridis')
            axes[0].set_title("Ground Truth Heatmap")
            axes[1].imshow(logits, cmap='viridis')
            axes[1].set_title("Predicted Heatmap")
            for ax in axes:
                ax.axis('off')
            plt.tight_layout()
            vid = os.path.basename(sample['video_file']).split('.')[0]
            frame = sample['center_idx']
            plt.savefig(os.path.join(save_dir, f"{vid}_f{frame}_heatmap.png"), dpi=150)
            plt.close()

def plot_train_curves(train_kl, train_l2, train_total, save_path):
    plt.figure()
    plt.plot(train_kl, label="Train KL Div")
    plt.plot(train_l2, label="Train L2 Error")
    plt.plot(train_total, label="Train Total Loss")
    plt.xlabel("Step")
    plt.ylabel("Metric")
    plt.title("Training Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_val_curves(val_kl, val_l2, val_total, save_path):
    plt.figure()
    plt.plot(val_kl, label="Val KL Div")
    plt.plot(val_l2, label="Val L2 Error")
    plt.plot(val_total, label="Val Total Loss")
    plt.xlabel("Step")
    plt.ylabel("Metric")
    plt.title("Validation Losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_pck(preds, targets, threshold):
    dists = np.linalg.norm(preds - targets, axis=-1)
    return np.mean(dists < threshold)

def plot_single_curve(values, label, save_path):
    plt.figure()
    plt.plot(values, label=label)
    plt.xlabel("Validation Step")
    plt.ylabel(label)
    plt.title(label)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

import matplotlib.pyplot as plt
import cv2
import os

# Define visualization function
def overlay_gt_pred_on_frames(dataset, annotation_data, model, save_dir, num_frames=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(min(num_frames, len(dataset))):
            sample = dataset[i]
            device = next(model.parameters()).device
            input_tensor = sample['images'].unsqueeze(0).to(device)
            logits = model(input_tensor)
            pred_coords = softargmax_2d(logits)[0].cpu().numpy()
            gt_coords = np.array(annotation_data[os.path.basename(sample['video_file'])][sample['center_idx']])
            
            cap = cv2.VideoCapture(sample['video_file'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, sample['center_idx'])
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            
            plt.figure(figsize=(6, 4))
            plt.imshow(frame[..., ::-1])  # BGR to RGB
            plt.scatter(gt_coords[0, 0], gt_coords[0, 1], c='lime', marker='o', label='GT')
            plt.scatter(pred_coords[0, 0], pred_coords[0, 1], c='red', marker='x', label='Pred')
            plt.legend()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"frame_overlay_{i:02d}.png"))
            plt.close()
# ----------------------------
# CONFIG
# ----------------------------
TRAIN_ANNOTATION_PATH = "/home/ubuntu/stats-320-file/calms21_task1_train.npy"
TEST_ANNOTATION_PATH = "/home/ubuntu/stats-320-file/calms21_task1_test.npy"
VIDEO_DIR = "/home/ubuntu/stats-320-file/task1_videos_mp4/train"
VIDEO_DIR_T = "/home/ubuntu/stats-320-file/task1_videos_mp4/test"
SAVE_DIR = "./experiment_a_again_again"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 5
PATIENCE = 10
WINDOW_SIZE = 5
LEARNING_RATE = 1e-4
SIGMA_START = 3.00
SIGMA_END = 1.5

# ---------------------------- DATA ----------------------------
print("🔄 Loading annotations...")
train_annots = load_annotations(TRAIN_ANNOTATION_PATH)
test_annots_full = load_annotations(TEST_ANNOTATION_PATH)

# Set fixed seed for reproducible data split
SEED = 73
rng = random.Random(SEED)

# Split test videos into validation and final test sets (video-wise)
test_videos = list(test_annots_full.keys())
rng.shuffle(test_videos)
split_index = len(test_videos) // 2
val_videos = test_videos[:split_index]
final_test_videos = test_videos[split_index:]

val_annots = {vid: test_annots_full[vid] for vid in val_videos}
test_annots = {vid: test_annots_full[vid] for vid in final_test_videos}

# Limit train and val pairs to 1000 and 200 respectively (with deterministic sort)
train_pairs, _ = split_by_video(train_annots, window_size=WINDOW_SIZE)
train_pairs.sort()
train_pairs = train_pairs[:4000]

val_pairs, _ = split_by_video(val_annots, window_size=WINDOW_SIZE)
val_pairs.sort()
val_pairs = val_pairs[:200]

# Print actual sample counts for debug
print(f"✅ Training on {len(train_pairs)} frames")
print(f"✅ Validating on {len(val_pairs)} frames")

train_dataset = PoseDataset(VIDEO_DIR, train_annots, WINDOW_SIZE, train_pairs, only_keypoint=0)
val_dataset = PoseDataset(VIDEO_DIR_T, val_annots, WINDOW_SIZE, val_pairs, only_keypoint=0)

# Deterministic test split too
test_pairs = []
half_window = WINDOW_SIZE // 2
for vid in sorted(test_annots.keys()):
    kps = test_annots[vid]
    for idx in range(half_window, len(kps) - half_window):
        test_pairs.append((vid, idx))
test_pairs = test_pairs[:200]

test_dataset = PoseDataset(VIDEO_DIR_T, test_annots, WINDOW_SIZE, test_pairs, only_keypoint=0)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)



# ----------------------------
# MODEL
# ----------------------------
model = PoseTransformer(num_keypoints=1).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

import itertools

TOTAL_STEPS = 12000
VAL_INTERVAL = 100
LOG_INTERVAL = 50
SIGMA_START = 3.0
SIGMA_END = 1.5
PATIENCE = 5

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_iter = itertools.cycle(train_loader)

step = 0
best_l2 = float('inf')
best_pck_005 = -1.0

patience_counter = 0
train_kl_history, train_l2_history = [], []
train_total_loss_history = []
train_pck_005_history, train_pck_01_history = [], []
val_kl_history, val_l2_history = [], []
val_total_loss_history = []
val_pck_005_history, val_pck_01_history = [], []

print(f"🚀 Starting step-based training for {TOTAL_STEPS} steps...")

csv_path = os.path.join(SAVE_DIR, "metrics.csv")
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Step', 'Train_KL', 'Train_L2', 'Train_Total_Loss',
                     'Train_PCK@0.05', 'Train_PCK@0.1',
                     'Val_L2', 'Val_KL', 'Val_Total_Loss',
                     'Val_PCK@0.05', 'Val_PCK@0.1'])

    pbar = trange(TOTAL_STEPS, desc="Training Steps")
    while step < TOTAL_STEPS:
        batch = next(train_iter)
        images = batch['images'].to(DEVICE)
        heatmaps = batch['heatmaps'][:, 0:1].to(DEVICE)

        progress = step / TOTAL_STEPS
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * (progress ** 0.5)
        train_dataset.sigma = sigma
        val_dataset.sigma = sigma

        model.train()
        logits = model(images)
        kl_loss = kl_heatmap_loss(logits, heatmaps)
        coord_loss = coordinate_l2_loss(logits, heatmaps)
        loss = 10.0 * kl_loss + 0.01 * coord_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        train_kl_history.append(kl_loss.item())
        train_l2_history.append(coord_loss.item())
        train_total_loss_history.append(loss.item())
        pbar.update(1)

        do_validation = (step % VAL_INTERVAL == 0 and step > 0) or (step == 0)

        if step % LOG_INTERVAL == 0:
            with torch.no_grad():
                overlay_dir = os.path.join(SAVE_DIR, f"train_overlay_step_{step:05d}")
                overlay_gt_pred_on_frames(train_dataset, train_annots, model, overlay_dir, num_frames=5)
                pred_coords = softargmax_2d(logits, orig_w=1024, orig_h=570).cpu().numpy()
                gt_coords   = softargmax_2d(heatmaps, orig_w=1024, orig_h=570).cpu().numpy()
                distances   = np.linalg.norm(pred_coords - gt_coords, axis=-1)  # shape: (B,)

                # Estimate visibility from GT heatmaps
                conf = heatmaps.cpu().numpy().max(axis=(2, 3))  # shape: (B, 1)
                visible = (conf > 0.1).squeeze(-1)              # shape: (B,)

                visible_dists = distances[visible]
                if len(visible_dists) > 0:
                    train_pck_005 = np.mean(visible_dists < 0.05 * np.sqrt(1024**2 + 570**2))
                    train_pck_01  = np.mean(visible_dists < 0.10 * np.sqrt(1024**2 + 570**2))
                else:
                    train_pck_005 = float('nan')
                    train_pck_01  = float('nan')

                print(f"📏 Step {step} | Mean dist: {distances.mean():.2f} | Max: {distances.max():.2f} | Min: {distances.min():.2f} | Visible: {visible.sum()} / {len(visible)}")

                train_pck_005_history.append(train_pck_005)
                train_pck_01_history.append(train_pck_01)

        # Optional validation step
        if do_validation:
            model.eval()
            val_kl_total, val_l2_total, val_loss_total = 0.0, 0.0, 0.0
            val_preds, val_gts = [], []
            skipped = 0

            with torch.no_grad():
                for i in range(len(val_dataset)):
                    sample = val_dataset[i]
                    input_tensor = sample['images'].unsqueeze(0).to(DEVICE)
                    heatmap = sample['heatmaps'].unsqueeze(0).to(DEVICE)

                    logits = model(input_tensor)
                    kl = kl_heatmap_loss(logits, heatmap).item()
                    l2 = coordinate_l2_loss(logits, heatmap).item()
                    total = 10.0 * kl + 0.01 * l2

                    val_kl_total += kl
                    val_l2_total += l2
                    val_loss_total += total

                    # Compute coordinates in original resolution
                    pred_coords = softargmax_2d(logits, orig_w=1024, orig_h=570)[0].cpu().numpy()
                    gt_coords = softargmax_2d(heatmap, orig_w=1024, orig_h=570)[0].cpu().numpy()

                    # Visibility check
                    conf = heatmap.cpu().numpy().max()
                    if conf > 0.1:
                        val_preds.append(pred_coords)
                        val_gts.append(gt_coords)
                    else:
                        skipped += 1

            num_val = len(val_dataset)
            val_kl = val_kl_total / num_val
            val_l2 = val_l2_total / num_val
            val_loss = val_loss_total / num_val

            val_kl_history.append(val_kl)
            val_l2_history.append(val_l2)
            val_total_loss_history.append(val_loss)

            val_preds = np.array(val_preds)
            val_gts = np.array(val_gts)
            if len(val_preds) > 0:
                val_pck_005 = compute_pck(val_preds, val_gts, threshold=0.05 * np.sqrt(1024**2 + 570**2))
                val_pck_01 = compute_pck(val_preds, val_gts, threshold=0.10 * np.sqrt(1024**2 + 570**2))
            else:
                val_pck_005 = float('nan')
                val_pck_01 = float('nan')

            val_pck_005_history.append(val_pck_005)
            val_pck_01_history.append(val_pck_01)

            print(f"🧼 Skipped {skipped} invisible keypoints during validation PCK computation")

        if step % LOG_INTERVAL == 0:
            print_str = f"📉 Step {step:5d} | Train Loss: {loss.item():.4f} | KL: {kl_loss.item():.4f} | L2: {coord_loss.item():.2f} | Sigma: {sigma:.2f}"
            print_str += f" | Train PCK@0.05: {train_pck_005:.3f} | PCK@0.1: {train_pck_01:.3f}"
            if do_validation:
                print_str += f"\n🔍 Val Loss: {val_loss:.4f} | Val KL: {val_kl:.4f} | Val L2: {val_l2:.2f}"
                print_str += f"\n🔍 Val PCK@0.05: {val_pck_005:.3f} | PCK@0.1: {val_pck_01:.3f}"
            print(print_str)

            val_kl_out = val_kl if do_validation else 'NaN'
            val_l2_out = val_l2 if do_validation else 'NaN'
            val_loss_out = val_loss if do_validation else 'NaN'
            val_pck_005_out = val_pck_005 if do_validation else 'NaN'
            val_pck_01_out = val_pck_01 if do_validation else 'NaN'

            writer.writerow([
                step,
                kl_loss.item(),
                coord_loss.item(),
                loss.item(),
                train_pck_005,
                train_pck_01,
                val_l2_out,
                val_kl_out,
                val_loss_out,
                val_pck_005_out,
                val_pck_01_out
            ])

            plot_train_curves(
                train_kl_history, train_l2_history, train_total_loss_history,
                os.path.join(SAVE_DIR, "train_loss_curve.png")
            )

        if do_validation:
            plot_single_curve(val_kl_history, "Val KL", os.path.join(SAVE_DIR, "val_kl_curve.png"))
            plot_single_curve(val_l2_history, "Val L2", os.path.join(SAVE_DIR, "val_l2_curve.png"))
            plot_single_curve(val_total_loss_history, "Val Total Loss", os.path.join(SAVE_DIR, "val_total_loss_curve.png"))
            plot_single_curve(val_pck_005_history, "Val PCK@0.05", os.path.join(SAVE_DIR, "val_pck_005_curve.png"))
            plot_single_curve(val_pck_01_history, "Val PCK@0.1", os.path.join(SAVE_DIR, "val_pck_01_curve.png"))

            viz_dir = os.path.join(SAVE_DIR, f"viz_step_{step:05d}")
            heatmap_dir = os.path.join(viz_dir, "heatmaps")
            visualize_predictions(model, val_dataset, val_annots, viz_dir)
            save_heatmap_visualizations(model, val_dataset, heatmap_dir)

            if step >= 8000:
                if val_l2 < best_l2:
                    best_l2 = val_l2
                    patience_counter = 0
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pose_transformer_best.pt"))
                else:
                    patience_counter += 1
                    if step >= 5000 and patience_counter >= PATIENCE:
                        print(f"⏹️ Early stopping at step {step}. Best Val L2: {best_l2:.2f}")
                        break
            else:
                print(f"🕐 Skipping early stopping check (warmup) at step {step}")



        step += 1


# Final save
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "pose_transformer_final.pt"))
print("✅ Final model saved.")
