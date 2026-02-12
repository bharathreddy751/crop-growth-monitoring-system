# 🌻 Automated Crop Growth Monitoring System

An AI-powered web application developed for detecting sunflower crop growth stages and leaf diseases using Vision Transformers (ViT) and Computer Vision techniques.

The system enables farmers to upload crop images and instantly receive predictions along with AI-generated treatment recommendations and weather-based insights for better decision-making.

---

## 🚀 Key Features

- 🌱 Crop Growth Stage Detection  
- 🦠 Sunflower Disease Identification  
- 📷 Image Upload & Real-Time Prediction  
- 🌦 Weather API Integration  
- 🤖 AI-Based Treatment Recommendations  
- 🌐 Django Web Application Interface  

---

## 🧠 Model Architecture

The system uses a **Vision Transformer (ViT)** model:

- Pretrained on ImageNet  
- Fine-tuned for sunflower crop analysis  
- Capable of identifying growth stages and diseases from field images  

Transfer learning is used to improve efficiency and performance.

---

## 🛠 Technologies Used

- Python  
- PyTorch    
- Django  
- HTML / CSS  
- OpenWeatherMap API  
- Groq API  

---

## 📂 Project Structure
crop-growth-monitoring-system/
│
├── classifier/ # ML model related files <br>
│ ├── predict_vit.py <br>
│ ├── train_vit.py<br>
│ ├── train_disease_vit.py<br>
│<br>
├── templates/ # HTML templates<br>
│ ├── index.html<br>
│ ├── result.html<br>
│<br>
├── static/ # CSS, JS, Images<br>
│<br>
├── manage.py # Django project manager<br>
├── requirements.txt # Python dependencies<br>
├── README.md # Project documentation<br>
└── .gitignore # Ignored files<br>
