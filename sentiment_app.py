import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# uploadind Model and tokenizer
@st.cache_resource
def load_model():
    model_checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("bert_sentiment_model.pth", map_location=device))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# عنوان
st.title("Sentiment Analysis App ❤️")
st.write("Write anything to check if it is Positive or Negative.")

# إدخال المستخدم
user_input = st.text_area("write your sentence here:")

if st.button("Expect"):
    if user_input.strip() == "":
        st.warning("Please write your sentence first!")
    else:
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()

        label = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Expect: {label}")
