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
    def __init__(self, images_path, labels_path, img_size=(640,480)):
        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
            ])

        # Load COCO dataset
        coco_dataset = CocoDetection(root=images_path,
                                     annFile=labels_path)
        self.data = coco_dataset
        self.len = len(coco_dataset)
        self.label_size = len(cards)
        self.img_size = img_size

    def __getitem__(self, index):
        image, anno = self.data.__getitem__(index)
        image = self.transforms(image)
        label_T = torch.tensor([label['category_id']-1 for label in anno])
        return image, label_T

    def __len__(self):
        return self.len
    
    @staticmethod
    def apply_transformation(images):
        """
        Method for applying more transformations to the training data.
        Supports batched inputs.
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor()
        ])
        
        # Apply transformations to each image in the batch
        transformed_images = torch.stack([transform(image) for image in images])
        return transformed_images