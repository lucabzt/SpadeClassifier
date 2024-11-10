"""
Custom dataset class to convert a dataset in COCO Format to a multi-class detection one.
Input: Image
Output: One-Hot encoded Vector of 53 playing cards
If you want to change the mapping from int to card go to mapping.py
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CocoDetection
from mapping import cards


class PlayingCardDataset(Dataset):
    def __init__(self, images_path, labels_path):
        transform = transforms.Compose([
            transforms.Resize((640, 480)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor()
        ])

        # Load COCO dataset with transform applied
        coco_dataset = CocoDetection(root=images_path,
                                     annFile=labels_path,
                                     transform=transform)
        self.data = coco_dataset
        self.len = len(coco_dataset)
        self.label_size = len(cards)

    def __getitem__(self, index):
        image, lab = self.data.__getitem__(index)
        one_hot = torch.zeros(self.label_size)
        one_hot[lab[0]['category_id'] - 1] = 1
        return image, one_hot

    def __len__(self):
        return self.len