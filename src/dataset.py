import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes
from torchvision.transforms import functional as TF

# まず全部を background/void=19 にする
ID_TO_20CLASS = np.full(256, 19, dtype=np.uint8)

mapping = {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}
for k, v in mapping.items():
    ID_TO_20CLASS[k] = v


class Cityscapes20ClassDataset(Dataset):
    def __init__(self, root, split="train", mode="fine", image_size=None):
        self.image_size = image_size
        self.num_classes = 20
        self.ds = Cityscapes(
            root=root,
            split=split,
            mode=mode,
            target_type="semantic",
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, target = self.ds[idx]

        image = TF.pil_to_tensor(image).float() / 255.0
        target_np = np.array(target, dtype=np.uint8)
        mask = torch.from_numpy(ID_TO_20CLASS[target_np]).long()   # (H, W), 0..19

        if self.image_size is not None:
            image = TF.resize(
                image,
                self.image_size,
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resize(
                mask.unsqueeze(0),
                self.image_size,
                interpolation=TF.InterpolationMode.NEAREST,
            ).squeeze(0).long()

        onehot = F.one_hot(mask, num_classes=self.num_classes)      # (H, W, 20)
        onehot = onehot.permute(2, 0, 1).float()                   # (20, H, W)

        return image, onehot, mask