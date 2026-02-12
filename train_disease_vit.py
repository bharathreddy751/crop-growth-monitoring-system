# from transformers import ViTForImageClassification, ViTImageProcessor
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch
# import os

# # --------------------------------------------------
# # 1. PRINT BASIC DEBUG INFO
# # --------------------------------------------------
# print("=" * 60)
# print("ViT SUNFLOWER DISEASE CLASSIFIER TRAINING")
# print("=" * 60)
# print("Current working directory:", os.getcwd())

# # --------------------------------------------------
# # 2. BASE DIRECTORY
# # --------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# # --------------------------------------------------
# # 3. DATASET PATH - YOUR DISEASE DATASET
# # --------------------------------------------------
# DATASET_PATH = os.path.join(
#     BASE_DIR,
#     "sunflowerdiseasedataset",
#     "Original Image"
# )

# print("\nDataset path:", DATASET_PATH)

# # Check if dataset exists
# if not os.path.exists(DATASET_PATH):
#     print(f"\n❌ ERROR: Dataset not found at {DATASET_PATH}")
#     print("\nTrying to find the correct path...")
    
#     # Try to auto-detect
#     disease_base = os.path.join(BASE_DIR, "sunflowerdiseasedataset")
#     if os.path.exists(disease_base):
#         print(f"Found base folder. Contents: {os.listdir(disease_base)}")
#         for item in os.listdir(disease_base):
#             test_path = os.path.join(disease_base, item)
#             if os.path.isdir(test_path):
#                 print(f"Using: {test_path}")
#                 DATASET_PATH = test_path
#                 break
    
#     if not os.path.exists(DATASET_PATH):
#         exit(1)

# print("Dataset folders:", os.listdir(DATASET_PATH))

# # --------------------------------------------------
# # 4. IMAGE TRANSFORMS
# # --------------------------------------------------
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # --------------------------------------------------
# # 5. LOAD DATASET
# # --------------------------------------------------
# dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
# loader = DataLoader(dataset, batch_size=8, shuffle=True)

# print("\nDetected Disease Classes:")
# for idx, class_name in enumerate(dataset.classes):
#     print(f"  {idx}: {class_name}")
# print(f"\nTotal images: {len(dataset)}")

# # --------------------------------------------------
# # 6. LOAD ViT MODEL AND PROCESSOR
# # --------------------------------------------------
# print("\nLoading pre-trained ViT model...")
# model = ViTForImageClassification.from_pretrained(
#     "google/vit-base-patch16-224",
#     num_labels=len(dataset.classes),
#     ignore_mismatched_sizes=True
# )

# processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

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
# # 9. SAVE MODEL AND PROCESSOR
# # --------------------------------------------------
# print("\n" + "=" * 60)
# print("Saving trained disease model...")
# model.save_pretrained("vit_disease_model")
# processor.save_pretrained("vit_disease_model")
# print("✅ Model and processor saved to: vit_disease_model/")
# print("=" * 60)
# print("DISEASE MODEL TRAINING COMPLETED SUCCESSFULLY!")
# print("=" * 60)
# print("\nDetected classes (in order):")
# for idx, class_name in enumerate(dataset.classes):
#     print(f'  "{idx}": "{class_name}",')
# print("\n💡 Copy the above class order to update config.json if needed!")
# print("=" * 60)

# =====================================================
# FAST ViT Disease Classifier (Transfer Learning)
# Backbone Frozen → Very Fast Training
# =====================================================

import torch
from transformers import ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

print("="*60)
print("SUNFLOWER DISEASE CLASSIFIER (FAST ViT)")
print("="*60)


# --------------------------------------------------
# PATHS
# --------------------------------------------------
train_path = "disease_split/train"
val_path   = "disease_split/val"


# --------------------------------------------------
# LIGHT AUGMENTATION (fast + good accuracy)
# --------------------------------------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
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
# LOAD PRETRAINED ViT
# --------------------------------------------------
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(train_ds.classes),
    ignore_mismatched_sizes=True
)


# --------------------------------------------------
# 🔥 FREEZE BACKBONE (FASTEST PART)
# --------------------------------------------------
for param in model.vit.parameters():
    param.requires_grad = False


# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# higher LR because only classifier trains
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------
NUM_EPOCHS = 4

for epoch in range(NUM_EPOCHS):

    # ---------- TRAIN ----------
    model.train()
    correct,total = 0,0
    total_loss = 0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        out = model(pixel_values=images, labels=labels)
        loss = out.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = out.logits.argmax(1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()

    train_acc = 100*correct/total


    # ---------- VALIDATION ----------
    model.eval()
    correct,total = 0,0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            preds = model(pixel_values=images).logits.argmax(1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)

    val_acc = 100*correct/total


    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    print(f"Train Loss: {total_loss/len(train_loader):.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")


# --------------------------------------------------
# SAVE
# --------------------------------------------------
model.save_pretrained("vit_disease_model_fast")

print("\n✅ Model saved → vit_disease_model_fast/")
print("Training completed successfully!")
