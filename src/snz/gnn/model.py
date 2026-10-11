"""ShapeGNN (texture-blind), RGB control, datasets and training (spec §5.3)."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import ResNet18_Weights, resnet18

from snz.gnn.graph import N_POINTS, neighbor_index, node_features, size_features
from snz.gnn.outlines import OutlineSet

N_CLASSES = 10


def _mlp(i: int, o: int) -> nn.Module:
    return nn.Sequential(nn.Linear(i, o), nn.LeakyReLU(0.2))


class ShapeGNN(nn.Module):
    """EdgeConv over a fixed ring graph of contour points (plain PyTorch, no PyG)."""

    def __init__(self, node_dim: int = 5, size_dim: int = 6, n_classes: int = N_CLASSES,
                 use_size: bool = True):
        super().__init__()
        self.register_buffer("idx", torch.as_tensor(neighbor_index(N_POINTS)), persistent=False)
        self.e1, self.e2, self.e3 = _mlp(2 * node_dim, 64), _mlp(128, 64), _mlp(128, 128)
        self.lin = _mlp(256, 512)
        self.use_size = use_size
        self.size = _mlp(size_dim, 32) if use_size else None
        self.head = nn.Sequential(_mlp(1024 + (32 if use_size else 0), 256), nn.Dropout(0.3),
                                  nn.Linear(256, n_classes))

    def _edge(self, h: Tensor, mlp: nn.Module) -> Tensor:
        nb = h[:, self.idx]                       # [B, N, K, F]
        hi = h.unsqueeze(2).expand_as(nb)
        return mlp(torch.cat([hi, nb - hi], -1)).max(dim=2).values

    def forward(self, nodes: Tensor, size: Tensor) -> Tensor:
        h1 = self._edge(nodes, self.e1)
        h2 = self._edge(h1, self.e2)
        h3 = self._edge(h2, self.e3)
        h = self.lin(torch.cat([h1, h2, h3], -1))
        g = torch.cat([h.max(dim=1).values, h.mean(dim=1)], -1)
        if self.use_size:
            g = torch.cat([g, self.size(size)], -1)
        return self.head(g)


def _augment(contour: np.ndarray, color: np.ndarray, rng: np.random.Generator):
    shift = int(rng.integers(N_POINTS))
    c, col = np.roll(contour, shift, axis=0), np.roll(color, shift, axis=0)
    centroid = c.mean(axis=0)
    if rng.random() < 0.5:  # mirror, then reverse so the order stays clockwise
        c = c.copy()
        c[:, 0] = 2 * centroid[0] - c[:, 0]
        c, col = c[::-1].copy(), col[::-1].copy()
    a = np.deg2rad(rng.uniform(-10, 10))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return ((c - centroid) @ rot.T + centroid).astype(np.float32), col


class GraphDataset(Dataset):
    """variant: "gnn" (geometry + size), "shape" (size zeroed), "color" (+ RGB at points)."""

    def __init__(self, o: OutlineSet, targets: np.ndarray, variant: str, augment: bool,
                 seed: int = 0):
        self.o, self.t, self.variant, self.augment = o, np.asarray(targets), variant, augment
        self.rng = np.random.default_rng(seed)
        if not augment:
            self.cache = [self._features(o.contour[i], o.color[i], i) for i in range(len(o.score))]

    def __len__(self) -> int:
        return len(self.o.score)

    def _features(self, contour, color, i, noise: float = 0.0):
        nodes = node_features(contour)
        if self.variant == "color":
            nodes = np.concatenate([nodes, color.astype(np.float32) / 255.0], axis=1)
        size = size_features(self.o.box[i], contour, tuple(self.o.image_wh[i]))
        if self.variant == "shape":
            size = np.zeros_like(size)
        elif noise:
            size = size + self.rng.normal(0, noise, size.shape).astype(np.float32)
        return torch.from_numpy(nodes.astype(np.float32)), torch.from_numpy(size.astype(np.float32))

    def __getitem__(self, i: int):
        if self.augment:
            c, col = _augment(self.o.contour[i], self.o.color[i], self.rng)
            nodes, size = self._features(c, col, i, noise=0.01)
        else:
            nodes, size = self.cache[i]
        return nodes, size, torch.tensor(self.t[i], dtype=torch.int64)


_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class CropDataset(Dataset):
    """Box crops (+10% per side) at 112x112, ImageNet-normalized; crops cached per image."""

    def __init__(self, o: OutlineSet, targets: np.ndarray, image_dir: Path,
                 file_name: Callable[[str], str] | None = None, size: int = 112,
                 augment: bool = False):
        self.t, self.augment = np.asarray(targets), augment
        self.crops = np.zeros((len(o.score), size, size, 3), np.uint8)
        for i in np.unique(o.image):
            name = o.names[i]
            img = Image.open(Path(image_dir) / (file_name(name) if file_name else name))
            img = img.convert("RGB")
            for r in np.flatnonzero(o.image == i):
                x1, y1, x2, y2 = o.box[r]
                mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
                crop = img.crop((x1 - mx, y1 - my, x2 + mx, y2 + my)).resize((size, size))
                self.crops[r] = np.asarray(crop)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, i: int):
        x = TF.to_tensor(self.crops[i])
        if self.augment and torch.rand(()) < 0.5:
            x = x.flip(-1)
        return TF.normalize(x, _MEAN, _STD), torch.tensor(self.t[i], dtype=torch.int64)


def rgb_model(pretrained: bool = True) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, N_CLASSES)
    return m


def _forward(model: nn.Module, batch, device: str) -> tuple[Tensor, Tensor]:
    *inputs, target = [b.to(device) for b in batch]
    return model(*inputs), target


def train_network(model: nn.Module, dataset: Dataset, *, epochs: int, lr: float,
                  batch_size: int, weight_decay: float, device: str, seed: int = 0) -> nn.Module:
    epochs = int(os.environ.get("SNZ_GNN_EPOCHS", epochs))  # tests only
    torch.manual_seed(seed)
    keep = [i for i in range(len(dataset)) if int(dataset.t[i]) >= 0]
    loader = DataLoader(Subset(dataset, keep), batch_size=batch_size, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(seed),
                        drop_last=len(keep) > batch_size)
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs * len(loader)))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for _ in range(epochs):
        for batch in loader:
            logits, target = _forward(model, batch, device)
            loss = loss_fn(logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


@torch.no_grad()
def predict_probs(model: nn.Module, dataset: Dataset, *, device: str,
                  batch_size: int = 1024) -> np.ndarray:
    model.to(device).eval()
    out = [torch.softmax(_forward(model, b, device)[0].float(), -1).cpu()
           for b in DataLoader(dataset, batch_size=batch_size)]
    return torch.cat(out).numpy() if out else np.zeros((0, N_CLASSES), np.float32)
