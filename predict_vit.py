import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os

# --------------------------------------------------
# 1. BASE DIRECTORY
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------
# 2. LOAD TRAINED MODEL
# --------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "vit_sunflower_model")

print("Loading model from:", MODEL_PATH)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model = ViTForImageClassification.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on: {device}")

# --------------------------------------------------
# 3. PREDICTION FUNCTION
# --------------------------------------------------
def predict_image(image_path):
    """Predict the sunflower growth stage from an image."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class = torch.max(probs, dim=1)

    label = model.config.id2label[predicted_class.item()]
    
    # Get all class probabilities
    all_probs = probs[0].cpu().numpy()
    
    return label, confidence.item(), all_probs

# --------------------------------------------------
# 4. TEST WITH ONE IMAGE
# --------------------------------------------------
if __name__ == "__main__":
    # Example test image path
    TEST_IMAGE = os.path.join(
        BASE_DIR,
        "sunflowerdataset",
        "Sunflower Stage Original",
        "Stage4 (Full_Bloom)",
        "sunflower test.jpg"  # ⚠️ Change to a real image filename
    )

    if not os.path.exists(TEST_IMAGE):
        print("\n❌ Image not found:", TEST_IMAGE)
        print("\nLooking for available images...")
        
        # Try to find any image in the dataset
        dataset_path = os.path.join(BASE_DIR, "sunflowerdataset", "Sunflower Stage Original")
        for stage_folder in os.listdir(dataset_path):
            stage_path = os.path.join(dataset_path, stage_folder)
            if os.path.isdir(stage_path):
                images = [f for f in os.listdir(stage_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if images:
                    TEST_IMAGE = os.path.join(stage_path, images[0])
                    print(f"✓ Found test image: {TEST_IMAGE}")
                    break
    
    if os.path.exists(TEST_IMAGE):
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Image: {os.path.basename(TEST_IMAGE)}")
        
        label, conf, all_probs = predict_image(TEST_IMAGE)
        
        print(f"\n🌻 Predicted Stage: {label}")
        print(f"📊 Confidence: {round(conf * 100, 2)}%")
        
        print("\n📋 All Class Probabilities:")
        for idx, prob in enumerate(all_probs):
            class_name = model.config.id2label[idx]
            print(f"  {class_name}: {round(prob * 100, 2)}%")
        print("=" * 60)
    else:
        print("❌ No test images found in dataset!")