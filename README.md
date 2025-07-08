# LLM-powered-Sentiment-Analysis

This project performs sentiment analysis on the IMDB movie reviews dataset using transformer-based models from Hugging Face (BERT and DistilBERT). The goal is to classify reviews as Positive or Negative.

---
## Project Objectives
- Build a complete sentiment analysis pipeline.
- Preprocess text and tokenize using pretrained Hugging Face tokenizers.
- Fine-tune transformer models (BERT, DistilBERT) for binary classification.
- Visualize training loss over epochs.
- Evaluate performance using accuracy and confusion matrix.
- Save and load trained models for reuse.
---
##  Tools & Libraries
- Python (Google Colab)
- PyTorch – for model training and inference
- Hugging Face Transformers – for pretrained BERT models and tokenizers
- Datasets – to load IMDB dataset
- scikit-learn – for confusion matrix
- Matplotlib / Seaborn – for visualization
- NLTK – for basic text preprocessing
---

 ## Folder Structure 

 ```
 IMDB_Sentiment_Analysis
│
├── sentiment_analysis_notebook.ipynb
├── bert_sentiment_model.pth
└── README.md
 ```       
---
##   How to Run This Project
1-  Clone the repository
```bash
git clone https://github.com/ElsayedGhanem/LLM-powered-Sentiment-Analysis.git
cd LLM-powered-Sentiment-Analysis
```
2- Open the Notebook
- You can use Jupyter Notebook or Google Colab.
- Upload the Sentiment_Analysis_BERT.ipynb into Google Colab
- Read the comment and Run each cell.

---

##    Model Weights Download

Due to file size limits on GitHub, the trained model file is hosted on Google Drive:

👉 Download the trained model weights (.pth file) from Google Drive. [Link](https://drive.google.com/file/d/1qBiJxbn9Hdt3TKPBTYKK2dyK1XJefVCy/view?usp=drive_link)

After downloading, place the .pth file in the project folder. Update any file path in the notebook or app as needed to match the location of the downloaded weights.



---
##   How to Upload The trained Model to Google Colab

If you want to test your pretrained weights
```bash
from google.colab import files
uploaded = files.upload()
```
Load the model
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


