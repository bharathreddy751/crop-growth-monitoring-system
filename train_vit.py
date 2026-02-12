# from transformers import ViTForImageClassification
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch
# import os

# # --------------------------------------------------
# # 1. PRINT BASIC DEBUG INFO
# # --------------------------------------------------
# print("=" * 60)
# print("ViT SUNFLOWER CLASSIFIER TRAINING")
# print("=" * 60)
# print("Current working directory:", os.getcwd())
# print("Files in project root:", os.listdir())

# # --------------------------------------------------
# # 2. BASE DIRECTORY (location of this file)
# # --------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # --------------------------------------------------
# # 3. DATASET PATH
# # --------------------------------------------------
# DATASET_PATH = os.path.join(
#     BASE_DIR,
#     "sunflowerdataset",
#     "Sunflower Stage Original"
# )
# print("\nDataset path:", DATASET_PATH)
# print("Dataset folders:", os.listdir(DATASET_PATH))

# # --------------------------------------------------
# # 4. IMAGE TRANSFORMS
# # --------------------------------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
# ])

# # --------------------------------------------------
# # 5. LOAD DATASET
# # --------------------------------------------------
# dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
# loader = DataLoader(dataset, batch_size=8, shuffle=True)

# print("\nDetected Classes:")
# for idx, class_name in enumerate(dataset.classes):
#     print(f"  {idx}: {class_name}")
# print(f"\nTotal images: {len(dataset)}")

# # --------------------------------------------------
# # 6. LOAD ViT MODEL (TRANSFER LEARNING)
# # --------------------------------------------------
# print("\nLoading pre-trained ViT model...")
# model = ViTForImageClassification.from_pretrained(
#     "google/vit-base-patch16-224",
#     num_labels=len(dataset.classes),
#     ignore_mismatched_sizes=True  # ⬅️ CRITICAL: Ignore classifier head size mismatch
# )

# # --------------------------------------------------
# # 7. SETUP OPTIMIZER AND DEVICE
# # --------------------------------------------------
# optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# print(f"Training device: {device}")
# print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
# print("=" * 60)

# # --------------------------------------------------
# # 8. TRAINING LOOP
# # --------------------------------------------------
# NUM_EPOCHS = 3

# for epoch in range(NUM_EPOCHS):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
    
#     print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
#     print("-" * 60)
    
#     for batch_idx, (images, labels) in enumerate(loader):
#         images, labels = images.to(device), labels.to(device)
        
#         # Forward pass
#         outputs = model(pixel_values=images, labels=labels)
#         loss = outputs.loss
#         logits = outputs.logits
        
#         # Calculate accuracy
#         predictions = torch.argmax(logits, dim=1)
#         correct += (predictions == labels).sum().item()
#         total += labels.size(0)
        
#         # Backward pass
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
        
#         # Print progress every 10 batches
#         if (batch_idx + 1) % 10 == 0:
#             print(f"  Batch {batch_idx + 1}/{len(loader)} - Loss: {loss.item():.4f}")
    
#     # Epoch summary
#     avg_loss = total_loss / len(loader)
#     accuracy = 100 * correct / total
#     print(f"\n✓ Epoch {epoch + 1} Summary:")
#     print(f"  Average Loss: {avg_loss:.4f}")
#     print(f"  Training Accuracy: {accuracy:.2f}%")

# # --------------------------------------------------
# # 9. SAVE MODEL
# # --------------------------------------------------
# print("\n" + "=" * 60)
# print("Saving trained model...")
# model.save_pretrained("vit_sunflower_model")
# print("✅ Model saved to: vit_sunflower_model/")
# print("=" * 60)
# print("TRAINING COMPLETED SUCCESSFULLY!")
# print("=" * 60)


# ==================================================
# ViT Sunflower Stage Classifier - TRAIN + VALIDATION
# ==================================================

from transformers import ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os

print("="*60)
print("ViT SUNFLOWER CLASSIFIER TRAINING (NO DATA LEAKAGE)")
print("="*60)

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "sunflowerdataset_split", "train")
VAL_PATH   = os.path.join(BASE_DIR, "sunflowerdataset_split", "val")


# --------------------------------------------------
# 2. IMAGE TRANSFORMS
# --------------------------------------------------
# Slight augmentation for better generalization

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# 3. LOAD DATASETS
# --------------------------------------------------

train_dataset = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
val_dataset   = datasets.ImageFolder(VAL_PATH, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("\nDetected Classes:")
for i, c in enumerate(train_dataset.classes):
    print(i, c)

print("\nTrain images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------------------------
# 4. LOAD PRETRAINED ViT
# --------------------------------------------------

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(train_dataset.classes),
    ignore_mismatched_sizes=True
)


# --------------------------------------------------
# 5. DEVICE
# --------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

print("\nDevice:", device)
print("="*60)


# --------------------------------------------------
# 6. TRAINING
# --------------------------------------------------

NUM_EPOCHS = 3

for epoch in range(NUM_EPOCHS):

    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print("-"*50)

    # ===== TRAIN =====
    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(pixel_values=images, labels=labels)

        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)


    train_acc = 100 * train_correct / train_total


    # ===== VALIDATION =====
    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(pixel_values=images)
            preds = torch.argmax(outputs.logits, dim=1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total


    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")


# --------------------------------------------------
# 7. SAVE MODEL
# --------------------------------------------------

print("\nSaving model...")
model.save_pretrained("vit_sunflower_model")

print("✅ Model saved successfully!")
print("="*60)
