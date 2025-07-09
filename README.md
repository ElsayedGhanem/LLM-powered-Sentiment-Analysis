# LLM-powered-Sentiment-Analysis

This project performs sentiment analysis on the IMDB movie reviews dataset using transformer-based models from Hugging Face (BERT and DistilBERT). The goal is to classify reviews as **Positive** or **Negative**. The project also includes an optional **Streamlit web app** for easy interaction and testing.

---
## 🎯 Project Objectives
- Build a complete sentiment analysis pipeline.
- Preprocess text and tokenize using pretrained Hugging Face tokenizers.
- Fine-tune transformer models (BERT, DistilBERT) for binary classification.
- Visualize training loss over epochs.
- Evaluate performance using accuracy and confusion matrix.
- Save and load trained models for reuse.
---
## 🛠️  Tools & Libraries
- Python (Google Colab)
- PyTorch – for model training and inference
- Hugging Face Transformers – for pretrained BERT models and tokenizers
- Datasets – to load IMDB dataset
- scikit-learn – for confusion matrix
- Matplotlib / Seaborn – for visualization
- NLTK – for basic text preprocessing
- Streamlit – optional web interface
---

 ## 📂 Folder Structure 

 ```
 IMDB_Sentiment_Analysis
│
├── sentiment_analysis_notebook.ipynb
├── sentiment_app.py
├── bert_sentiment_model.pth
└── README.md
 ```       
---
## ⚡How to Run This Project
### 1️⃣ Clone the repository
```bash
git clone https://github.com/ElsayedGhanem/LLM-powered-Sentiment-Analysis.git
cd LLM-powered-Sentiment-Analysis
```
### 2️⃣ Open the Notebook
- You can use Jupyter Notebook or Google Colab.
- Upload the Sentiment_Analysis_BERT.ipynb into Google Colab
- Follow the instructions and run each cell in order.
---

## 📦Model Weights Download

Due to file size limits on GitHub, the trained model file is hosted on Google Drive:

👉 [Download the trained model (.pth) file](https://drive.google.com/file/d/1Jyn2gS5622krssoMnYVaixpOH0Gnsl-V/view?usp=drive_link)
### Instructions:
- Download the .pth file from the link.
- Place it in your project folder.
- Update file paths in the notebook or app if needed.

---

## 💻How to Upload the Trained Model to Google Colab

If you want to test your pretrained weights

### 1️⃣ Upload the bert_sentiment_model.pth into Colab
```bash
from google.colab import files
uploaded = files.upload()
```
### 2️⃣ Load the model
```bash
from transformers import BertForSequenceClassification
import torch
model_checkpoint = "bert-base-uncased"

model = BertForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location=device))
model.eval()
```
---

## 🌐 Deploying with Streamlit

This project includes a simple Streamlit app that allows you to classify any text interactively

### 📌 Features
- Load your trained model.
- Input any custom text.
- Get the sentiment prediction (Positive / Negative).

### 📌 How to Run the Streamlit App 

Follow these simple steps to launch the app locally:

### 1️⃣ Install Required Libraries
Open your terminal and install the necessary packages:
```bash
pip install streamlit
pip install transformers torch
streamlit run sentiment_app.py
```
### 2️⃣ Navigate to the Project Directory
```bash
cd C:\Users\eygha\Desktop\Sentiment_Analysis_Project
```
### 3️⃣ Finally, start the Streamlit app with
```bash
streamlit run sentiment_app.py
```
Once the command runs, your default web browser will open with the interactive sentiment analysis app ready to use! 🚀

