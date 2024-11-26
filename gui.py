import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QProgressBar, QTextEdit
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from logic import process_audio, process_text_analysis
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def plot_emotion_pie_chart(figure, canvas, percentages):
    figure.clear()
    ax = figure.add_subplot(111)

    # Extract labels and sizes
    labels = [item[0] for item in percentages]
    sizes = [item[1] for item in percentages]

    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=140,  
        textprops={'fontsize': 10}, 
        wedgeprops={'edgecolor': 'black'}
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(7)

    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    canvas.draw()


class EmotionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  # Initialize the user interface
        self.setWindowIcon(QIcon('logo.ico'))  # Set the application icon

    def initUI(self):
        # Set the window title and dimensions
        self.setWindowTitle("Emotion Analysis from Audio or Text")
        self.setGeometry(100, 100, 600, 600)

        layout = QVBoxLayout()  # Main vertical layout for the UI components

        # Text input area for emotion analysis
        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("Paste the text for emotion analysis here...")
        layout.addWidget(self.text_input)

        # Button for analyzing emotions from text input
        self.button_analyze_text = QPushButton("Analyze emotions from text")
        self.button_analyze_text.clicked.connect(self.analyze_text)
        layout.addWidget(self.button_analyze_text)

        # Label to display instructions for audio file selection
        self.label = QLabel("Select an audio file (MP3 or WAV) for emotion analysis.")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Button to open a file dialog for selecting an audio file
        self.button_open = QPushButton("Select file")
        self.button_open.clicked.connect(self.open_file)
        layout.addWidget(self.button_open)

        # Button to analyze emotions from the selected audio file
        self.button_analyze_file = QPushButton("Analyze emotions from file")
        self.button_analyze_file.clicked.connect(self.analyze_audio_file)
        self.button_analyze_file.setEnabled(False)  # Disabled until a file is selected
        layout.addWidget(self.button_analyze_file)

        # Progress bar to indicate the progress of analysis
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)  # Initialize the progress bar at 0%
        layout.addWidget(self.progress_bar)

        # Label to display results or messages
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)  # Allow wrapping of long text
        layout.addWidget(self.result_label)

        # Matplotlib figure and canvas for displaying charts
        self.figure = plt.Figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)  # Apply the layout to the main window

    def open_file(self):
        # Open a file dialog to select an audio file
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select audio file", "", "Audio Files (*.mp3 *.wav)", options=options)
        if file_name:
            # Check if the selected file has a valid format
            if file_name.endswith((".mp3", ".wav")):
                self.file_path = file_name
                self.label.setText(f"Selected file: {file_name}")  # Display the selected file path
                self.button_analyze_file.setEnabled(True)  # Enable the analyze button
            else:
                self.label.setText("Invalid file format. Please select MP3 or WAV.")

    def analyze_audio_file(self):
        # Analyze emotions from the selected audio file
        if hasattr(self, 'file_path'):
            self.progress_bar.setValue(10)  # Update progress bar
            text = process_audio(self.file_path)  # Convert audio to text
            self.progress_bar.setValue(50)  # Update progress bar
            self.process_text_analysis(text)  # Process the text for emotion analysis

    def analyze_text(self):
        # Analyze emotions from the text input
        text = self.text_input.toPlainText()
        if text.strip():  # Check if the input is not empty
            self.process_text_analysis(text)  # Process the text for emotion analysis
        else:
            self.result_label.setText("Please enter text for analysis.")  # Display an error message

    def process_text_analysis(self, text):
        # Analyze emotions in the given text
        emotion_probabilities, sorted_emotions = process_text_analysis(text)
        self.progress_bar.setValue(90)  # Update the progress bar

        # Get the top three strongest emotions
        top_emotions = sorted_emotions[:3]
        top_score_sum = sum([score for _, score in sorted_emotions])

        # Prepare the results text with the strongest emotions
        results_text = f"Transcription:\n{text}\n\nStrongest emotions:\n"
        top_emotions_percentage = []
        for emotion, score in top_emotions:
            # Calculate the percentage for each emotion
            percentage = round((score / top_score_sum) * 100, 1)
            top_emotions_percentage.append((emotion, percentage))
            results_text += f"{emotion}: {percentage:.1f}%\n"

        # Calculate the percentage for "Other" emotions
        other_score = sum([score for _, score in sorted_emotions[3:]])
        other_percentage = round((other_score / top_score_sum) * 100, 1) if other_score > 0 else 0

        # Add "Other" to the chart data if it more than 0%
        if other_percentage > 0:
            top_emotions_percentage.append(("Other", other_percentage))

        # Display the results in the GUI
        self.result_label.setText(results_text)
        self.progress_bar.setValue(100)

        # Pass the data to create a pie chart
        plot_emotion_pie_chart(self.figure, self.canvas, top_emotions_percentage)
