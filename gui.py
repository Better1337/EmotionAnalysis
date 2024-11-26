import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QTextEdit
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from transcription import transcribe_audio
from emotion_model import predict_emotion
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


def plot_emotion_pie_chart(figure, canvas, emotions_percentage):
    figure.clear()
    ax = figure.add_subplot(111)
    labels, sizes = zip(*emotions_percentage)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle
    canvas.draw()


class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowIcon(QIcon('logo.ico'))

    def initUI(self):
        self.setWindowTitle("Emotion Analysis from Audio or Text")
        self.setGeometry(100, 100, 600, 600)

        layout = QVBoxLayout()

        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("Paste the text for emotion analysis here...")
        layout.addWidget(self.text_input)

        self.button_analyze_text = QPushButton("Analyze emotions from text")
        self.button_analyze_text.clicked.connect(self.analyze_text)
        layout.addWidget(self.button_analyze_text)

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
            text = transcribe_audio(self.file_path)
            self.progress_bar.setValue(50)
            self.process_text_analysis(text)

    def analyze_text(self):
        text = self.text_input.toPlainText()
        if text.strip():
            self.process_text_analysis(text)
        else:
            self.result_label.setText("Please enter text for analysis.")

    def process_text_analysis(self, text):
        emotion_probabilities, sorted_emotions = predict_emotion(text)
        self.progress_bar.setValue(90)

        top_emotions = sorted_emotions[:3]
        top_score_sum = sum([score for _, score in sorted_emotions])

        results_text = f"Transcription:\n{text}\n\nStrongest emotions:\n"
        top_emotions_percentage = []
        for emotion, score in top_emotions:
            percentage = round((score / top_score_sum) * 100, 1)
            top_emotions_percentage.append((emotion, percentage))
            results_text += f"{emotion}: {percentage:.1f}%\n"

        other_score = sum([score for _, score in sorted_emotions[3:]])
        other_percentage = round((other_score / top_score_sum) * 100, 1) if other_score > 0 else 0

        if other_percentage > 0:
            top_emotions_percentage.append(("Other", other_percentage))

        self.result_label.setText(results_text)
        self.progress_bar.setValue(100)

        plot_emotion_pie_chart(self.figure, self.canvas, top_emotions_percentage)
