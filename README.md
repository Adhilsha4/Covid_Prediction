# 🦠 COVID-19 Lung X-Ray Classification Using Deep Learning

## 📌 Introduction
The COVID-19 pandemic, which began in late 2019, was one of the most devastating global health emergencies of the 21st century. While the world has largely returned to normal, new cases continue to surface. COVID-19 primarily affects the lungs, making early detection essential to avoid severe complications like viral pneumonia.

This project presents an AI-powered application that detects COVID-19 from chest X-ray images using deep learning.

---

## 🎯 Project Goal
To classify lung X-ray images into the following three categories:

- ✅ COVID Positive  
- ❌ COVID Negative  
- 🟡 Viral Pneumonia  

The application uses a trained deep learning model to make fast, reliable predictions from X-ray inputs.

---

## 🧠 Model Architecture
The model is a **Convolutional Neural Network (CNN)** built using the **Keras Sequential API**.

### 📐 Layers:
- Input Layer  
- 2 × Convolutional Layers  
- 2 × Pooling Layers  
- Flatten Layer  
- Dense (Fully Connected) Layer  
- Output Layer  

### 🔧 Activation Functions:
- `ReLU` for all hidden layers  
- `Softmax` for the output layer (3-class classification)

✅ The model achieved ~90% accuracy with strong **precision, recall**, and **F1-score**.

---

## ⚙️ How It Works
1. User uploads a chest X-ray image through the web interface.
2. Image is preprocessed and passed to the trained CNN model.
3. The application returns one of the following predictions:
   - **COVID Positive**
   - **COVID Negative**
   - **Viral Pneumonia**

---

## 🚀 Technologies Used
- **Deep Learning**: Keras  
- **Backend**: TensorFlow  
- **Deployment**: Streamlit  
- **Dataset**: Kaggle (reduced from 4000 to 1000 images)  

---

## 🌍 Real-World Impact
This tool assists healthcare professionals by providing:
- Quick diagnostic support  
- Accessibility in remote or under-resourced regions  
- A low-cost, AI-driven screening tool (not a replacement for clinical testing)

---

## 📁 Project Structure
├── app.py # Streamlit app interface
├── model.h5 # Trained CNN model
├── utils.py # Image preprocessing utilities
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── dataset/ # Chest X-ray images (COVID, Normal, Pneumonia)
