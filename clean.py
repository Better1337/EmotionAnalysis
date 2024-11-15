import pandas as pd
import re
import urllib.request

# def download_goemotions():
#     base_url = "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/"
#     files = ["train.tsv", "dev.tsv", "test.tsv"]
#     dataframes = []

#     for file in files:
#         url = base_url + file
#         df = pd.read_csv(url, sep='\t', header=None, names=['text', 'emotion_ids', 'comment_id'])
#         dataframes.append(df)

#     combined_df = pd.concat(dataframes, ignore_index=True)
#     return combined_df

# df = download_goemotions()

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

for i, label in enumerate(emotion_labels):
    df[label] = df['emotion_ids'].apply(lambda x: int(str(i) in x.split(',')))

columns_to_keep = ['text'] + emotion_labels
df = df[columns_to_keep]

df['text'] = df['text'].apply(clean_text)

df = df.dropna(subset=['text'])  
df = df[(df[emotion_labels] != 0).any(axis=1)] 

df = df.drop_duplicates()


df.to_csv('cleaned_dataset.csv', index=False)


