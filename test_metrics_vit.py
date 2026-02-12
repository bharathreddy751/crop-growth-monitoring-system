import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os

print("="*60)
print("ViT TEST METRICS")
print("="*60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_PATH = os.path.join(BASE_DIR, "sunflowerdataset_split", "val")  # use val as test


# ---------------------------------------
# TRANSFORM
# ---------------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# ---------------------------------------
# DATA
# ---------------------------------------
dataset = datasets.ImageFolder(TEST_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)


# ---------------------------------------
# MODEL
# ---------------------------------------
model = ViTForImageClassification.from_pretrained("vit_sunflower_model")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)

        outputs = model(pixel_values=images)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())


# ---------------------------------------
# METRICS
# ---------------------------------------
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="macro")
recall = recall_score(all_labels, all_preds, average="macro")
f1 = f1_score(all_labels, all_preds, average="macro")

print("\nAccuracy :", round(acc*100,2), "%")
print("Precision:", round(precision*100,2), "%")
print("Recall   :", round(recall*100,2), "%")
print("F1 Score :", round(f1*100,2), "%")

print("\nConfusion Matrix:\n")
print(confusion_matrix(all_labels, all_preds))

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))
