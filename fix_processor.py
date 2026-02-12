from transformers import ViTImageProcessor
import os

# Download the processor from the original pretrained model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# Save it to your model directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vit_sunflower_model")

processor.save_pretrained(MODEL_PATH)

print(f"✅ Processor saved to {MODEL_PATH}")
print("Now you can run predict_vit.py successfully!")