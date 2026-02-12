# =====================================================
# FAST CNN BASELINE (ResNet18 Transfer Learning)
# For Sunflower Disease Classification
# =====================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

print("="*60)
print("CNN DISEASE CLASSIFIER (FAST RESNET18)")
print("="*60)


# --------------------------------------------------
# PATHS
# --------------------------------------------------
train_path = "disease_split/train"
val_path   = "disease_split/val"


# --------------------------------------------------
# TRANSFORMS (important for CNN stability)
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# --------------------------------------------------
# DATA
# --------------------------------------------------
train_ds = datasets.ImageFolder(train_path, transform=train_transform)
val_ds   = datasets.ImageFolder(val_path, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16)

print("Classes:", train_ds.classes)
print("Train images:", len(train_ds))
print("Val images:", len(val_ds))


# --------------------------------------------------
# LOAD RESNET18 PRETRAINED
# --------------------------------------------------
model = models.resnet18(pretrained=True)

num_classes = len(train_ds.classes)


# --------------------------------------------------
# 🔥 FREEZE BACKBONE (VERY FAST)
# --------------------------------------------------
for param in model.parameters():
    param.requires_grad = False


# Replace last layer only
model.fc = nn.Linear(model.fc.in_features, num_classes)


# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()

# higher LR since only last layer trains
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------
EPOCHS = 5   # fast + sufficient

for epoch in range(EPOCHS):

    # -------- TRAIN --------
    model.train()
    correct,total = 0,0
    total_loss = 0

    for x,y in train_loader:

        x,y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = out.argmax(1)
        correct += (preds==y).sum().item()
        total += y.size(0)

        total_loss += loss.item()

    train_acc = 100*correct/total


    # -------- VALIDATION --------
    model.eval()
    correct,total = 0,0

    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)

            preds = model(x).argmax(1)
            correct += (preds==y).sum().item()
            total += y.size(0)

    val_acc = 100*correct/total


    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")


print("\n✅ CNN training completed successfully!")
