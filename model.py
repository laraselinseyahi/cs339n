import torch
import torch.nn as nn

IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 256
NUM_KEYPOINTS = 7

class PoseTransformer(nn.Module):
    def _init_(self, patch_size=PATCH_SIZE, embed_dim=EMBED_DIM, num_heads=4, depth=6, num_keypoints=NUM_KEYPOINTS):
        super()._init_()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (IMAGE_SIZE // patch_size) ** 2  # 14x14 = 196
        self.temporal_embed = nn.Parameter(torch.zeros(1, 99 * num_patches, embed_dim))  # assume max T=99

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=depth
        )

        self.decode_conv = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_keypoints, kernel_size=1)
        )

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def forward(self, x):
        # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # flatten temporal dim
        x = self.patch_embed(x)     # [B*T, D, 14, 14]

        Hf, Wf = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B*T, 196, D]

        x = x.reshape(B, T * self.num_patches, self.embed_dim)  # no view issues

        x = x + self.temporal_embed[:, :x.size(1), :]  # [B, T*196, D]
        x = self.encoder(x)  # transformer attention over flattened temporal patches

        x = x.view(B, T, self.num_patches, self.embed_dim)  # [B, T, 196, D]
        x_center = x[:, T // 2]  # [B, 196, D] — center frame
        x_center = x_center.transpose(1, 2).reshape(B, self.embed_dim, Hf, Wf)  # [B, D, 14, 14]
        return self.decode_conv(x_center)  # [B, 7, 28, 28]
