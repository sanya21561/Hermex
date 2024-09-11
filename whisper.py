import time
from faster_whisper import WhisperModel

model = WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe_audio(audio):
    start_time = time.time()
    
    segments, _ = model.transcribe(audio, language="en")
    transcription = ' '.join(segment.text for segment in segments)
    
    latency = time.time() - start_time
    
    return transcription, latency