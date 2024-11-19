import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QTextEdit, QHBoxLayout
)
from PyQt5.QtCore import Qt
from transformers import BertTokenizer, BertForSequenceClassification
import whisper
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

whisper_model = whisper.load_model("base")

with open(r"tokenizer_config.json", "r") as f:
    tokenizer_config = json.load(f)

emotion_model = BertForSequenceClassification.from_pretrained(".")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", **tokenizer_config)
emotion_model.eval()

def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path, language="en")
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
    return emotion_probabilities, sorted_emotions[:3] 

class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Emotion Analysis from Audio or Text")
        self.setGeometry(100, 100, 600, 600)
        self.setStyleSheet("""
            QLabel {
                font-size: 14px;
            }
            QPushButton {
                font-size: 14px;
                padding: 10px;
                background-color: #0078D7;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005A9E;
            }
            QPushButton:pressed {
                background-color: #003B73;
            }
        """)

        layout = QVBoxLayout()

        
        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("Paste the text for emotion analysis here...")
        layout.addWidget(self.text_input)

        
        self.button_analyze_text = QPushButton("Analyze emotions from text")
        self.button_analyze_text.clicked.connect(self.analyze_text)
        layout.addWidget(self.button_analyze_text)

       
        layout.addWidget(QLabel("Or"))

        
        self.label = QLabel("Select an audio file (MP3 or WAV) for emotion analysis.")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.button_open = QPushButton("Select file")
        self.button_open.clicked.connect(self.open_file)
        layout.addWidget(self.button_open)

        self.button_analyze_file = QPushButton("Analyze emotions from file")
        self.button_analyze_file.clicked.connect(self.analyze_audio_file)
        self.button_analyze_file.setEnabled(False)
        layout.addWidget(self.button_analyze_file)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        layout.addWidget(self.result_label)

        
        self.figure = plt.Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.button_save = QPushButton("Save results")
        self.button_save.clicked.connect(self.save_results)
        self.button_save.setEnabled(False)
        layout.addWidget(self.button_save)

        self.setLayout(layout)

    def open_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select audio file", "", "Audio Files (*.mp3 *.wav)", options=options)
        if file_name:
            if file_name.endswith((".mp3", ".wav")):
                self.file_path = file_name
                self.label.setText(f"Selected file: {file_name}")
                self.button_analyze_file.setEnabled(True)
            else:
                self.label.setText("Invalid file format. Please select MP3 or WAV.")

    def analyze_audio_file(self):
        if hasattr(self, 'file_path'):
            self.progress_bar.setValue(10)
            QApplication.processEvents()

            text = transcribe_audio(self.file_path)
            self.progress_bar.setValue(50)
            QApplication.processEvents()

            self.process_text_analysis(text)

    def analyze_text(self):
        text = self.text_input.toPlainText()
        if text.strip():
            self.process_text_analysis(text)
        else:
            self.result_label.setText("Please enter text for analysis.")

    def process_text_analysis(self, text):
        emotion_probabilities, top_emotions = predict_emotion(text)
        self.progress_bar.setValue(90)
        QApplication.processEvents()

       
        results_text = f"Transcription:\n{text}\n\nStrongest emotions:\n"
        for emotion, score in top_emotions:
            if score > 0: 
                results_text += f"{emotion}: {score:.2f}\n"
        self.result_label.setText(results_text)
        self.progress_bar.setValue(100)

      
        self.results_text = results_text
        self.button_save.setEnabled(True)

        
        self.plot_emotion_pie_chart(emotion_probabilities)

    def plot_emotion_pie_chart(self, emotion_probabilities, threshold=0.05):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        
        filtered_emotions = {k: v for k, v in emotion_probabilities.items() if v >= threshold}
        
        labels = list(filtered_emotions.keys())
        sizes = list(filtered_emotions.values())

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%', startangle=140,
            textprops={'fontsize': 10}, wedgeprops={'edgecolor': 'black'}
        )

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(8)

        ax.axis('equal')  
        self.canvas.draw()

    def save_results(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save results", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            with open(file_name, "w") as file:
                file.write(self.results_text)

app = QApplication(sys.argv)
ex = EmotionApp()
ex.show()
sys.exit(app.exec_())
