import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import KFold
from transformers import ViTForImageClassification
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("sunflowerdataset/Sunflower Stage Original", transform=transform)

kfold = KFold(n_splits=5, shuffle=True)

fold_acc = []

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f"\nFOLD {fold+1}")

    train_sub = Subset(dataset, train_ids)
    val_sub = Subset(dataset, val_ids)

    train_loader = DataLoader(train_sub, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=16)

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=len(dataset.classes),
        ignore_mismatched_sizes=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # train 1 epoch only for CV
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        out = model(pixel_values=images, labels=labels)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validate
    model.eval()
    correct,total = 0,0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(pixel_values=images).logits.argmax(1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)

    acc = 100*correct/total
    fold_acc.append(acc)

    print("Accuracy:", acc)


print("\nAverage Cross Validation Accuracy:", np.mean(fold_acc))
