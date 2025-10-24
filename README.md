# Multimodal Emotion Recognition System

This project identifies human emotions by combining **visual**, **audio**, and **text** signals. Instead of relying on a single data source, the system **fuses features from all three modalities** to achieve higher accuracy and more stable predictions.

---

## 🚀 Key Features

- **Facial Emotion Recognition (Visual)**
  - Trained and compared multiple CNN architectures: **VGG19, ResNet50, MobileNetV2, and Xception** for facial feature extraction and expression classification.

- **Speech Emotion Detection (Audio)**
  - Extracts **MFCC features** using Librosa and classifies emotional tone with an **LSTM model**.

- **Text Sentiment Analysis (Text)**
  - Uses **word embeddings + LSTM** to capture emotional meaning from conversational or written text.

- **Multimodal Feature Fusion**
  - A **Dense Fusion Layer** integrates outputs from all three modalities to produce the final unified emotion prediction.

---

## 🧠 Models Used

| Modality | Model(s) Used | Purpose |
|---------|----------------|---------|
| Visual (Face) | **VGG19, ResNet50, MobileNetV2, Xception** | Extract facial emotion features and compare performance |
| Audio (Speech) | **MFCC + LSTM Classifier** | Capture vocal tone and emotional cues |
| Text (Language) | **Word Embeddings + LSTM** | Understand sentiment and linguistic context |
| Fusion Layer | **Dense Neural Layer** | Combine multimodal vectors for final emotion prediction |

---

## 📊 Performance

- Best-performing visual model: **Xception** (highest accuracy + efficiency)
- Overall system accuracy: **95.2%**
- Supports near **real-time inference**

---

## 🛠️ Tech Stack

**Python**, **TensorFlow/Keras**, **OpenCV**, **Librosa**, **NLTK**, **Scikit-learn**

---
