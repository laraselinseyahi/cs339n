# this script implements markerless pose estimation using a cnn that regresses to soft gaussian heatmaps
# it mimics the lab setup exactly — same dataset, model, training loop — but replaces binary dilation with gaussian blur
# the goal is to evaluate whether smooth target heatmaps improve model convergence and localization

# import required libraries
import torch  # pytorch core
import torch.nn as nn  # neural network layers
import torch.nn.functional as F  # activation functions and pooling
from torch.utils.data import Dataset, DataLoader  # custom dataset and batching
import matplotlib
matplotlib.use('Agg')  # set backend to non-gui to avoid runtime errors on headless servers
import matplotlib.pyplot as plt  # for saving visualizations
from scipy.ndimage import gaussian_filter  # for applying gaussian smoothing
from torchvision.transforms import Compose  # for chaining transforms

# load dataset provided in the lab
data = torch.load("lab3_data.pt")  # contains train/val images and binary keypoint targets

# extract training and validation splits
train_images = data["train_images"]
train_targets = data["train_targets"]
val_images = data["val_images"]
val_targets = data["val_targets"]
keypoint_names = data["keypoint_names"]  # list of keypoint names

# calculate number of keypoints from target shape
num_keypoints = len(keypoint_names)

# set device to use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define a transform that converts image from (H, W, C) to (C, H, W)
class TransposeAndMove:
    def __call__(self, sample):
        img, tgt = sample['image'], sample['target']  # unpack image and target
        img = img.permute(2, 0, 1)  # convert to channel-first format
        return {'image': img, 'target': tgt}  # return modified sample

# define a custom transform that applies gaussian blur to each keypoint mask
class GaussianBlurTargets:
    def __init__(self, sigma=1.5):  # sigma controls the spread of the gaussian
        self.sigma = sigma

    def __call__(self, sample):
        img, tgt = sample['image'], sample['target']  # unpack sample
        K, H, W = tgt.shape  # get keypoint map dimensions
        blurred = torch.zeros_like(tgt, dtype=torch.float32)  # create a float32 tensor to store blurred maps
        for k in range(K):  # loop over each keypoint
            mask = tgt[k].float().numpy()  # convert to float and numpy for filtering
            filtered = gaussian_filter(mask, sigma=self.sigma)  # apply gaussian blur
            blurred[k] = torch.tensor(filtered, dtype=torch.float32)  # store result back as float32 tensor
        return {'image': img, 'target': blurred}  # return processed sample

# define dataset class used by the lab
class KeypointDataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = [img.float() / 255 for img in images]  # normalize images to [0, 1]
        self.targets = targets  # store binary keypoint masks
        self.transform = transform  # optional transform to apply

    def __len__(self):
        return len(self.images)  # return total number of samples

    def __getitem__(self, idx):
        sample = {'image': self.images[idx], 'target': self.targets[idx]}  # pack image and target
        if self.transform:
            sample = self.transform(sample)  # apply transform if provided
        return sample['image'], sample['target']  # return processed image and target

# define a simple convolutional neural network for pose estimation
class PoseNet(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)  # conv layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # conv layer 2
        self.conv3 = nn.Conv2d(64, num_keypoints, kernel_size=1)  # final layer for output heatmaps

    def forward(self, x):
        x = F.relu(self.conv1(x))  # apply conv1 and relu
        x = F.relu(self.conv2(x))  # apply conv2 and relu
        return self.conv3(x)  # return raw output (logits)

# build training and validation datasets with transforms
train_ds = KeypointDataset(
    train_images,
    train_targets,
    transform=Compose([
        TransposeAndMove(),  # convert image to channel-first format
        GaussianBlurTargets(sigma=1.5)  # apply gaussian smoothing to targets
    ])
)

val_ds = KeypointDataset(
    val_images,
    val_targets,
    transform=Compose([
        TransposeAndMove(),
        GaussianBlurTargets(sigma=1.5)
    ])
)

# wrap datasets in dataloaders
train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)  # shuffle for training
val_dl = DataLoader(val_ds, batch_size=1)

# initialize model and move to device
model = PoseNet(num_keypoints).to(device)

# use binary cross entropy with logits (sigmoid applied internally)
loss_fn = nn.BCEWithLogitsLoss()

# use Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training loop
for epoch in range(5):  # train for 5 epochs
    model.train()
    total_loss = 0.0
    print(f"\n starting epoch {epoch+1}...")

    for i, (imgs, tgts) in enumerate(train_dl):
        imgs, tgts = imgs.to(device), tgts.to(device)  # move to gpu if available

        out = model(imgs)  # forward pass
        loss = loss_fn(out, tgts)  # compute loss

        optimizer.zero_grad()  # clear previous gradients
        loss.backward()  # compute new gradients
        optimizer.step()  # update model weights

        total_loss += loss.item()  # accumulate loss

        if (i + 1) % 10 == 0:  # print progress every 10 steps
            print(f" [{i + 1}/{len(train_dl)}] loss: {loss.item():.4f}")

    print(f"📈 epoch {epoch+1} complete. total loss: {total_loss:.4f}")  # end of epoch summary

# save the model to file
torch.save(model.state_dict(), "pose_model.pt")
print(" model saved to pose_model.pt")

# visualization helper to inspect model predictions on a sample image
def visualize_prediction(model, dataset, index=0, output_path="output.png"):
    model.eval()  # switch to eval mode
    img, tgt = dataset[index]  # get one sample
    img_tensor = img.to(device).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        out = torch.sigmoid(model(img_tensor)).squeeze(0).cpu()  # get predicted heatmaps

    img_np = img.permute(1, 2, 0).cpu()  # convert image back to HWC for plotting

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(img_np)
    axs[0].set_title("image")

    heatmap = out.max(0)[0]  # max over keypoints to get a unified heatmap
    axs[1].imshow(img_np)
    axs[1].imshow(heatmap, alpha=0.6, cmap='jet')
    axs[1].set_title("predicted keypoints heatmap")

    plt.tight_layout()
    plt.savefig(output_path)
    print(f" saved visualization to {output_path}")
