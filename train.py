"""
Skript for training the SpadeClassifier model on the playing_card_dataset.
"""

# IMPORTS
import torch
from dataset import PlayingCardDataset
from yolo_dataset import YoloCustomDataset
from torch.utils.data import DataLoader
from SpadeClassifier import SpadeClassifier
import matplotlib.pyplot as plt
import os


# PARAMS
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
TRAIN_SET = 'data/playing_cards_large/train'
TEST_SET = 'data/playing_cards_large/test'
VAL_SET = 'data/playing_cards_large/val'
IMG_SIZE = (240,240)
CONFIDENCE_TRESHOLD = 0.7
print(f"MODEL RUNNING ON DEVICE: {device}")


# SAVE GPU FROM SETTING ON FIRE
if device != 'cpu':
    torch.cuda.set_per_process_memory_fraction(0.8, device=0)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# DATASET, train/test split, create dataloaders
train_set: YoloCustomDataset = YoloCustomDataset(os.path.join(TRAIN_SET, 'images'), os.path.join(TRAIN_SET, 'labels'), img_size=IMG_SIZE)
test_set: YoloCustomDataset = YoloCustomDataset(os.path.join(TEST_SET, 'images'), os.path.join(TEST_SET, 'labels'), img_size=IMG_SIZE)
train_load, test_load = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True), DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


# LOAD MODEL
model = SpadeClassifier(53).to(device)
model.load_state_dict(torch.load("pretrained_models/model_142/model.pt", weights_only=True, map_location=device))


# TRAINING PARAMS
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
loss_fn = torch.nn.BCELoss()
train_loss = []
test_loss = []
epochs = 200


# EVALUATE
def compute_pos_neg(labels, preds):
    # Convert predictions to binary values using threshold
    preds_binary = (preds >= CONFIDENCE_TRESHOLD).int()
    labels = labels.int()

    # Calculate True Positives, False Positives, True Negatives, False Negatives
    tp = torch.sum((preds_binary == 1) & (labels == 1)).item()
    fp = torch.sum((preds_binary == 1) & (labels == 0)).item()
    tn = torch.sum((preds_binary == 0) & (labels == 0)).item()
    fn = torch.sum((preds_binary == 0) & (labels == 1)).item()

    return tp, fp, tn, fn


def calculate_metrics(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


# TRAINING CODE
def train_one_epoch() -> None:
    torch.cuda.empty_cache()
    running_loss = 0.0
    total_tp = total_fp = total_tn = total_fn = 0

    model.train()
    for iteration, data in enumerate(train_load):
        # Get data and move to the correct device
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)[:, :52]
        outputs = torch.sigmoid(outputs)

        # Calculate batch metrics and accumulate
        tp, fp, tn, fn = compute_pos_neg(labels, outputs > CONFIDENCE_TRESHOLD)
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        # Loss calculation
        loss = loss_fn(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track running loss
        running_loss += loss.item()

        if iteration % 100 == 0:
            # Compute metrics over the entire training set
            accuracy, precision, recall, f1 = calculate_metrics(total_tp, total_fp, total_tn, total_fn)
            print(f"Metrics - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    train_loss.append(running_loss / len(train_load))


def test_one_epoch() -> None:
    torch.cuda.empty_cache()
    running_loss = 0.0
    total_tp = total_fp = total_tn = total_fn = 0

    model.eval()
    with torch.no_grad():
        for image_batch, label_batch in test_load:
            # Get data and move to the correct device
            images = image_batch.to(device)
            labels = label_batch.to(device)

            # Forward pass
            outputs = model(images)[:, :52]
            outputs = torch.sigmoid(outputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            # Calculate batch metrics and accumulate
            tp, fp, tn, fn = compute_pos_neg(labels, outputs > CONFIDENCE_TRESHOLD)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn

    # Compute metrics over the entire test set
    accuracy, precision, recall, f1 = calculate_metrics(total_tp, total_fp, total_tn, total_fn)
    print("--------------------")
    print(f"Test Metrics - Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")
    print(f"Test Loss: {running_loss / len(test_load):.4f}")

    test_loss.append(running_loss / len(test_load))


# TRAINING LOOP
for epoch in range(epochs):
    print(f"-- Starting Epoch {epoch}: --")
    train_one_epoch()
    test_one_epoch()
    torch.cuda.empty_cache()  # Empty memory cache of GPU

    # Save plot and model to file
    os.makedirs(f'pretrained_models/model_{epoch}', exist_ok=True)
    plt.clf()
    plt.plot(train_loss, label="Training loss")
    plt.plot(test_loss, label="Test Loss")
    plt.legend()
    plt.savefig(f"pretrained_models/model_{epoch}/plot.png")
    torch.save(model.state_dict(), f"pretrained_models/model_{epoch}/model.pt")

torch.save(model.state_dict(), f"pretrained_models/model_{test_loss[-1]:.4f}_.pt")