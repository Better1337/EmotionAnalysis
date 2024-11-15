import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from transformers import BertTokenizer, BertForSequenceClassification
import whisper
import json

whisper_model = whisper.load_model("base")

with open(r"tokenizer_config.json", "r") as f:
    tokenizer_config = json.load(f)

emotion_model = BertForSequenceClassification.from_pretrained(".")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", **tokenizer_config)

emotion_model.eval()

def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result['text']

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
    return sorted_emotions

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Emotion Analysis from Audio")
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.label = QLabel("Wybierz plik audio (MP3 lub WAV) do analizy emocji.")
        layout.addWidget(self.label)

        self.button_open = QPushButton("Wybierz plik")
        self.button_open.clicked.connect(self.open_file)
        layout.addWidget(self.button_open)

        self.button_analyze = QPushButton("Analizuj emocje")
        self.button_analyze.clicked.connect(self.analyze_emotion)
        layout.addWidget(self.button_analyze)

        self.result_label = QLabel("")
        layout.addWidget(self.result_label)

        self.setLayout(layout)

    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Wybierz plik audio", "", "Audio Files (*.mp3 *.wav)", options=options)
        if file_name:
            self.file_path = file_name
            self.label.setText(f"Wybrano plik: {file_name}")

    def analyze_emotion(self):
        if hasattr(self, 'file_path'):
            self.label.setText("Transkrypcja audio...")
            QApplication.processEvents()

            text = transcribe_audio(self.file_path)
            self.label.setText("Analiza emocji...")
            QApplication.processEvents()

            emotion_results = predict_emotion(text)
            
            results_text = f"Transkrypcja:\n{text}\n\nNajsilniejsze emocje:\n"
            for emotion, score in emotion_results[:3]:  # Wyświetlamy top 3 emocje
                results_text += f"{emotion}: {score:.2f}\n"
            self.result_label.setText(results_text)
        else:
            self.label.setText("Nie wybrano pliku audio.")

app = QApplication(sys.argv)
ex = EmotionApp()
ex.show()
sys.exit(app.exec_())
