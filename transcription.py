import whisper

whisper_model = whisper.load_model("base")

def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path, language="en")
    return result['text']
