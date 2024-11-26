import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Load tokenizer configuration from a JSON file
with open("tokenizer_config.json", "r") as f:
    tokenizer_config = json.load(f)

# Load the pre-trained BERT model for emotion classification
emotion_model = BertForSequenceClassification.from_pretrained(".")

#Load the tokenizer, using the BERT base uncased model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", **tokenizer_config)

# Set the model to evaluation mode
emotion_model.eval()

def predict_emotion(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Get predictions from the model
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)[0]

# Define the list of emotions
    emotions = ["admiration", "amusement", "anger", "annoyance", "approval", "caring",
                "confusion", "curiosity", "desire", "disappointment", "disapproval",
                "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
                "joy", "love", "nervousness", "optimism", "pride", "realization",
                "relief", "remorse", "sadness", "surprise", "neutral"]

    # Create a dictionary of emotions and their probabilities
    emotion_probabilities = {emotions[i]: float(probabilities[i]) for i in range(len(emotions))}
    # Sort the emotions by probability in descending order
    sorted_emotions = sorted(emotion_probabilities.items(), key=lambda x: x[1], reverse=True)
    return emotion_probabilities, sorted_emotions[:3]
