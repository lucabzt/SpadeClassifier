import torch
import matplotlib.pyplot as plt
from yolo_dataset import YoloCustomDataset
from mapping import cards_large

PATH_TO_IMAGES = 'data/playing_cards_large/train/images'
PATH_TO_LABELS = 'data/playing_cards_large/train/labels'

data = YoloCustomDataset(PATH_TO_IMAGES, PATH_TO_LABELS)
image, label = data[20]
plt.imshow(image)
label_int = [int(l) for l in label.nonzero()]
label_str = [cards_large[l] for l in label_int]
print("\n---------------------\n")
print(label_str)
plt.show()