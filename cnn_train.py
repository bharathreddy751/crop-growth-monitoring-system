import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# ---------------------------------
# TRANSFORMS
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_path = "sunflowerdataset_split/train"
val_path   = "sunflowerdataset_split/val"

train_ds = datasets.ImageFolder(train_path, transform=transform)
val_ds   = datasets.ImageFolder(val_path, transform=transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)


# ---------------------------------
# SIMPLE CNN MODEL
# ---------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128*28*28,256),
            nn.ReLU(),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        return self.net(x)


device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimpleCNN(len(train_ds.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ---------------------------------
# TRAINING
# ---------------------------------
EPOCHS = 3

for epoch in range(EPOCHS):

    model.train()
    correct,total = 0,0

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

    train_acc = 100*correct/total


    model.eval()
    correct,total = 0,0

    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(1)

            correct += (preds==y).sum().item()
            total += y.size(0)

    val_acc = 100*correct/total

    print(f"Epoch {epoch+1} | Train:{train_acc:.2f}% | Val:{val_acc:.2f}%")
