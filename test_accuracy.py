from transformers import ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import os

# --------------------------------------------------
# 1. PATH SETUP
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "sunflowerdataset",
    "Sunflower Stage Original"
)

MODEL_PATH = os.path.join(BASE_DIR, "vit_sunflower_model")

print("=" * 60)
print("TESTING ViT SUNFLOWER CLASSIFIER")
print("=" * 60)

# --------------------------------------------------
# 2. SAME TRANSFORMS AS TRAINING (IMPORTANT)
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# 3. LOAD DATASET
# --------------------------------------------------
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)

print("Detected Classes:", dataset.classes)
print("Total images:", len(dataset))

# --------------------------------------------------
# 4. SPLIT 80-20 (TRAIN / TEST)
# --------------------------------------------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print("Test images:", len(test_dataset))

loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# --------------------------------------------------
# 5. LOAD TRAINED MODEL
# --------------------------------------------------
model = ViTForImageClassification.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

print("Device:", device)

# --------------------------------------------------
# 6. TEST ACCURACY
# --------------------------------------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(pixel_values=images)
        preds = torch.argmax(outputs.logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total

print("\n" + "=" * 60)
print(f"FINAL TEST ACCURACY: {accuracy:.2f}%")
print("=" * 60)
