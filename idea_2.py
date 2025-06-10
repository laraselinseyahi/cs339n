import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import torchvision.transforms.functional as TF
import numpy as np

# ✅ same as lab: model architecture (PoseNet)
class PoseNet(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, num_keypoints, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

# ✅ same as lab: normalize images
def normalize_image(img):
    return img.float() / 255

# ⚠️ deviation: added resizing transform to enforce fixed input size
class ResizePair:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, sample):
        img, tgt = sample['image'], sample['target']
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
        img = TF.resize(img, self.size, antialias=True)
        tgt = TF.resize(tgt, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        return {'image': img.float(), 'target': tgt.float()}

# ⚠️ deviation: paired frame dataset to support temporal consistency
class PairedFrameDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = [normalize_image(img) for img in images]
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images) - 1

    def __getitem__(self, idx):
        sample_t = {'image': self.images[idx], 'target': self.targets[idx]}
        sample_tp1 = {'image': self.images[idx+1], 'target': self.targets[idx+1]}
        if self.transform:
            sample_t = self.transform(sample_t)
            sample_tp1 = self.transform(sample_tp1)
        return sample_t['image'], sample_t['target'], sample_tp1['image'], sample_tp1['target']

# ✅ same as lab: binary cross entropy + adam optimizer
# ⚠️ deviation: temporal consistency loss via dropout
def generate_dropout_mask(shape, drop_prob=0.3):
    K, H, W = shape
    keep = torch.rand(K) > drop_prob
    return keep[:, None, None].expand(K, H, W).float()

# 🧠 main training function
def train_dropout_temporal_consistency(data_path="lab3_data.pt", epochs=5, drop_prob=0.3):
    data = torch.load(data_path)
    train_images = data["train_images"]
    train_targets = data["train_targets"]
    keypoint_names = data["keypoint_names"]
    num_keypoints = len(keypoint_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ResizePair((256, 256))  # ⚠️ deviation: added resizing to align inputs
    ])

    train_ds = PairedFrameDataset(train_images, train_targets, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)

    model = PoseNet(num_keypoints).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        print(f"\n🚀 starting epoch {epoch+1}...")

        for img_t, tgt_t, img_tp1, tgt_tp1 in train_dl:
            img_t, img_tp1 = img_t.to(device), img_tp1.to(device)
            tgt_t, tgt_tp1 = tgt_t.to(device), tgt_tp1.to(device)

            pred_t = model(img_t)
            pred_tp1 = model(img_tp1)

            # ⚠️ deviation: dropout + temporal consistency
            dropout_mask = generate_dropout_mask(tgt_tp1.shape[1:], drop_prob).to(device)
            masked_tgt_tp1 = tgt_tp1 * dropout_mask

            loss_pred_t = bce_loss(pred_t, tgt_t)
            loss_pred_tp1 = bce_loss(pred_tp1, masked_tgt_tp1)

            consistency_mask = 1.0 - dropout_mask
            heat_t = torch.sigmoid(pred_t.detach())
            heat_tp1 = torch.sigmoid(pred_tp1)
            loss_consistency = mse_loss(heat_tp1 * consistency_mask, heat_t * consistency_mask)

            loss = loss_pred_t + loss_pred_tp1 + loss_consistency

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"📉 epoch {epoch+1} complete. total loss = {total_loss:.4f}")

    torch.save(model.state_dict(), "pose_model_dropout.pt")
    print("✅ model saved to pose_model_dropout.pt")


if __name__ == "__main__":
    train_dropout_temporal_consistency()
