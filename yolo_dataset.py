import os
import torch
from torch.utils.data import Dataset
import cv2


class YoloCustomDataset(Dataset):
    def __init__(self, images_path, labels_path, img_size=(240,240)):
        self.img_size = img_size
        self.pairs: list[tuple[str, str]] = []
        images = os.listdir(images_path)
        labels = os.listdir(labels_path)

        # Load the image and labels paths into a list
        for index in range(len(images)):
            self.pairs.append((os.path.join(images_path, images[index]), os.path.join(labels_path, labels[index])))

    def __getitem__(self, index):
        image_path, label_path = self.pairs[index]

        # Load image and apply transformations
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.img_size)
        image = torch.tensor(image)
        image = image.permute(2, 0, 1).float()

        # Open the labels file
        with open(label_path, 'r') as file:
            # Read lines and split each line into a list of values
            class_ids = [int(line.split()[0]) for line in file]

        # Create one hot tensor
        one_hot = torch.zeros(52)
        indeces = torch.tensor(class_ids)
        one_hot[indeces] = 1
        return image, one_hot

    def __len__(self):
        return len(self.pairs)
