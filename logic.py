from transcription import transcribe_audio
from emotion_model import predict_emotion

def process_audio(file_path):
    # Transcribe audio to text
    text = transcribe_audio(file_path)
    return text

def process_text_analysis(text):
    # Analyze emotions in the given text
    emotion_probabilities, sorted_emotions = predict_emotion(text)
    return emotion_probabilities, sorted_emotions
