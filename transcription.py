import whisper
# Load the base model
whisper_model = whisper.load_model("base")

# Transcribe the audio file
def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path, language="en")
    return result['text']
