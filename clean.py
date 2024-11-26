import pandas as pd
import re

# Function to clean the text by removing special characters, numbers, and extra spaces
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading and trailing whitespace
    return text

# List of emotion labels
emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Create binary columns for each emotion label based on 'emotion_ids' column
for i, label in enumerate(emotion_labels):
    df[label] = df['emotion_ids'].apply(lambda x: int(str(i) in x.split(',')))

# Keep only the text column and the emotion columns
columns_to_keep = ['text'] + emotion_labels
df = df[columns_to_keep]

# Apply the text cleaning function to the 'text' column
df['text'] = df['text'].apply(clean_text)

# Drop rows where 'text' is missing
df = df.dropna(subset=['text'])  

# Remove rows where no emotion labels are assigned (all zeros in emotion columns)
df = df[(df[emotion_labels] != 0).any(axis=1)] 

# Remove duplicate rows
df = df.drop_duplicates()

# Save the cleaned dataset to a CSV file
df.to_csv('cleaned_dataset.csv', index=False)
