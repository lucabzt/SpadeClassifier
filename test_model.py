import torch
import random
from yolo_dataset import YoloCustomDataset
import matplotlib.pyplot as plt
from SpadeClassifier import SpadeClassifier
from mapping import cards_large

val_set = YoloCustomDataset('./data/playing_cards_large/valid/images', './data/playing_cards_large/valid/labels')
model = SpadeClassifier(53)
model.load_state_dict(torch.load('./pretrained_models/model_142/model.pt', map_location='cpu'))


for i in range(5):
    plt.clf()
    index = random.randint(0, len(val_set) - 1)
    image, labels = val_set[index]
    labels_indices = labels.nonzero()

    # Get logits
    preds = model(image.unsqueeze(0))
    preds_cropped = preds[0, :53]

    # Get topk
    indices = torch.topk(preds_cropped, 3).indices
    preds = [cards_large[i] for i in indices]

    # Print labels
    print(indices)
    print(labels_indices)

    image = (image * 255).int().permute(1, 2, 0).cpu().numpy()
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Preds: {preds}")
    plt.show()

