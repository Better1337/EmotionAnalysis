import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

with open("tokenizer_config.json", "r") as f:
    tokenizer_config = json.load(f)

emotion_model = BertForSequenceClassification.from_pretrained(".")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", **tokenizer_config)
emotion_model.eval()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

    emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring",
                "confusion", "curiosity", "desire", "disappointment", "disapproval",
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                "joy", "love", "nervousness", "optimism", "pride", "realization",
                "relief", "remorse", "sadness", "surprise", "neutral"]

    emotion_probabilities = {emotions[i]: float(probabilities[i]) for i in range(len(emotions))}
    sorted_emotions = sorted(emotion_probabilities.items(), key=lambda x: x[1], reverse=True)
    return emotion_probabilities, sorted_emotions[:3]
