# Speech-to-Text and Emotion Analysis System using NLP

## Overview

This project is an application for converting speech to text and analyzing emotions using advanced Natural Language Processing (NLP) techniques. It combines state-of-the-art AI models-OpenAI's Whisper for speech transcription and a custom-trained BERT model for multi-label emotion classification. The system features a user-friendly graphical interface (PyQt5) and is designed for high accuracy and ease of use in real-world scenarios such as social media analysis, customer support, education, and healthcare.

---

## Features

- **Automatic Speech Transcription:** Converts audio files (MP3, WAV) to text using the Whisper model.
- **Emotion Detection in Text:** Analyzes user-provided or transcribed text and identifies up to 27 emotions plus neutral using a BERT model trained on the GoEmotions dataset.
- **Intuitive GUI:** Built with PyQt5, allowing users to select files, input text, and view results, including pie chart visualization of emotion probabilities.
- **Flexible Input:** Supports both direct text input and audio file upload.
- **Visualization:** Displays the top detected emotions and their proportions in the analyzed text.

---

## Model Training Required

**Important:**  
Before running the application, you must train the BERT-based emotion analysis model yourself.  
The repository includes scripts for data cleaning (`clean.py`) and model training (`train.py`).  
You need to:

1. **Prepare and clean the dataset** (e.g., GoEmotions) using `clean.py`.
2. **Train the model** using `train.py`.  
   This will save the trained model in the `emotion_analysis_model` directory, which is required for inference.

*Without this step, the emotion analysis functionality will not work.*

---

## Technologies Used

- Python 3.10+
- PyQt5 (GUI)
- [transformers](https://huggingface.co/transformers/) (HuggingFace) – BERT, Whisper
- torch (PyTorch)
- matplotlib (visualization)
- pandas (data processing)

---

## System Architecture

The application uses the Model-View-Controller (MVC) pattern and consists of four main modules:

- **GUI Module:** Handles user interaction and result presentation (PyQt5).
- **Speech Processing Module:** Transcribes audio to text using Whisper.
- **Emotion Analysis Module:** Classifies emotions in text using a fine-tuned BERT model.
- **Application Logic Module:** Coordinates data flow between modules and manages the analysis process.

---

## How It Works

1. **User selects an audio file or enters text.**
2. **If audio:** The system transcribes speech to text using Whisper.
3. **Text is analyzed:** The BERT model predicts probabilities for each emotion label.
4. **Results are displayed:** Top emotions and a pie chart are shown in the GUI.

---

## Example Use Cases

- Social media sentiment and emotion analysis
- Customer support and feedback analysis
- Educational tools for communication skills
- Healthcare applications (e.g., mood monitoring)

---

## Installation

1. **Clone the repository:**
git clone https://github.com/Better1337/EmotionAnalysis.git
2. **Install dependencies:**
pip install -r requirements.txt
3. **Prepare and train the emotion analysis model:**

**a. Clean and prepare the dataset:**  
Use the provided script to preprocess your dataset (for example, GoEmotions):
python clean.py
This will create a `cleaned_dataset.csv` file.

**b. Train the BERT-based emotion analysis model:**  
Run the training script:
python train.py
fter training, the trained model will be saved in the `emotion_analysis_model` directory.  
**Note:** This step is required-without a trained model, the emotion analysis will not work.

4. **Run the application:**
python app.py
---

### Project Structure

- `app.py` — Main entry point and GUI initialization
- `gui.py` — GUI components and event handling
- `logic.py` — Application logic and data flow
- `transcription.py` — Whisper-based audio transcription
- `emotion_model.py` — BERT-based emotion analysis
- `clean.py` — Data cleaning for training
- `train.py` — Model training script
- `requirements.txt` — Dependencies
- `README.md` — Project documentation



## Sample Usage

### Audio Analysis

1. Select an MP3 or WAV file.
2. Click **Analyze emotions from file**.
3. The system will:
    - Transcribe the audio.
    - Analyze the emotions.
    - Display the strongest emotions.

### Text Analysis

1. Paste any text into the input field.
2. Click **Analyze emotions from text**.
3. The system will:
    - Analyze the emotions in the text.
    - Display emotion probabilities.
    - Show a pie chart of the results.

---