import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ================================
# CREATE RESULTS FOLDER
# ================================
os.makedirs("Results", exist_ok=True)


# ================================
# 1️⃣ MODEL ACCURACY COMPARISON
# ================================
models = ['CNN', 'ViT']
accuracy = [94.86, 99.60]

plt.figure()
plt.bar(models, accuracy, color=['orange','green'])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison for Crop Stage")

for i,v in enumerate(accuracy):
    plt.text(i, v+0.2, str(v))

plt.ylim(90,101)
plt.savefig("Results/accuracy_comparison.png")
plt.close()


models = ['CNN', 'ViT']
accuracy = [80.85, 93.62]

plt.figure()
plt.bar(models, accuracy, color=['orange','green'])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison for Disease")

for i,v in enumerate(accuracy):
    plt.text(i, v+0.2, str(v))

plt.ylim(50,101)
plt.savefig("Results/accuracy_comparison.png")
plt.close()





# ================================
# 4️⃣ CONFUSION MATRIX
# ================================
cm = np.array([
[48,0,0,0,0],
[0,55,0,0,0],
[0,0,44,0,0],
[0,0,0,43,0],
[0,0,0,1,62]
])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("Results/confusion_matrix.png")
plt.close()



# for i,v in enumerate(values):
#     plt.text(i, v+0.1, str(v))

plt.savefig("Results/metrics_bar.png")
plt.close()


print("\n✅ All graphs saved inside 'results/' folder!")
