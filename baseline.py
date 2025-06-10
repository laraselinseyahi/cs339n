# this script implements the original lab version of pose estimation using binary dilation
# it mimics the lab setup exactly — including the dilatetargets transform — for fair comparison against gaussian smoothing

# import required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # use headless backend
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from torchvision.transforms import Compose
import numpy as np

# load lab-provided pose tracking dataset
data = torch.load("lab3_data.pt")

# extract dataset components
train_images = data["train_images"]
train_targets = data["train_targets"]
val_images = data["val_images"]
val_targets = data["val_targets"]
keypoint_names = data["keypoint_names"]
num_keypoints = len(keypoint_names)

# select compute device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform to convert image to (C, H, W)
class TransposeAndMove:
    def __call__(self, sample):
        img, tgt = sample['image'], sample['target']
        img = img.permute(2, 0, 1)
        return {'image': img, 'target': tgt}

# transform to dilate binary target masks
class DilateTargets:
    def __init__(self, width=3):
        self.width = width

    def __call__(self, sample):
        img, tgt = sample['image'], sample['target']
        K, H, W = tgt.shape
        dilated = torch.zeros_like(tgt, dtype=torch.float32)
        for k in range(K):
            mask = tgt[k].numpy()
            dilated_mask = binary_dilation(mask, structure=np.ones((self.width, self.width))).astype(np.float32)
            dilated[k] = torch.tensor(dilated_mask)
        return {'image': img, 'target': dilated}

# dataset class matching the lab implementation
class KeypointDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = [img.float() / 255 for img in images]
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = {'image': self.images[idx], 'target': self.targets[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample['image'], sample['target']

# simple CNN for heatmap regression
class PoseNet(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

# dataset with binary dilation (lab baseline)
train_ds = KeypointDataset(
    train_images,
    train_targets,
    transform=Compose([
        TransposeAndMove(),
        DilateTargets(width=3)
    ])
)

val_ds = KeypointDataset(
    val_images,
    val_targets,
    transform=Compose([
        TransposeAndMove(),
        DilateTargets(width=3)
    ])
)

# dataloaders
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=1)

# model, loss, optimizer
model = PoseNet(num_keypoints).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

import matplotlib.pyplot as plt

loss_per_epoch = []

for epoch in range(5):
    model.train()
    total_loss = 0.0
    print(f"\n🚀 starting epoch {epoch+1}...")

    for i, (imgs, tgts) in enumerate(train_dl):
        imgs, tgts = imgs.to(device), tgts.to(device)
        out = model(imgs)
        loss = loss_fn(out, tgts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(train_dl)}] loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_dl)
    loss_per_epoch.append(avg_loss)
    print(f"📈 epoch {epoch+1} complete. avg loss: {avg_loss:.4f}")

# save model for comparison
torch.save(model.state_dict(), "pose_model_dilated.pt")

# plot and save loss curve
plt.plot(loss_per_epoch, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss_dilated.png")

# save model for comparison
torch.save(model.state_dict(), "pose_model_dilated.pt")
