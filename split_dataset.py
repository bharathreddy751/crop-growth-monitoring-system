import os, shutil, random

BASE = "sunflowerdiseasedataset/Original Image"
OUT = "disease_split"

ratio = 0.8

# remove old split
if os.path.exists(OUT):
    shutil.rmtree(OUT)

for cls in os.listdir(BASE):

    cls_path = os.path.join(BASE, cls)

    if not os.path.isdir(cls_path):
        continue

    images = [f for f in os.listdir(cls_path)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]

    random.shuffle(images)

    split = int(len(images)*ratio)

    for i, img in enumerate(images):

        if i < split:
            dst = os.path.join(OUT, "train", cls)
        else:
            dst = os.path.join(OUT, "val", cls)

        os.makedirs(dst, exist_ok=True)

        shutil.copy(
            os.path.join(cls_path, img),
            os.path.join(dst, img)
        )

print("✅ Disease dataset split SUCCESS")
