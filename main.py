import sounddevice as sd
import numpy as np
import asyncio
import tempfile
import os
import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from whisper import transcribe_audio
from rag import runInferenceRetailSeller
from eleven import text_to_speech_file
from pydub import AudioSegment
from pydub.playback import play
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
import pandas as pd
from together import Together
from rag import chatDetails


console = Console()

# Initialize components
chatClient = Together(api_key='9277cfe863ae79d3063484d039ed2fa89681ecbfbe1477f3176b4f61638f04ef')
model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
chatReport=chatDetails()

client = QdrantClient(
    url="b57db28a-86e7-4be5-965e-67e98d7292eb.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="iD1_Lufho_ddBHK8d441fxRHyT8yc24yUeJLjHtvVFI4ABgJNd11qA",
)


embedding_model = TextEmbedding("snowflake/snowflake-arctic-embed-s")
data = pd.read_csv('dataset/flipkart_com-ecommerce_sample.csv')
collection_name = "flipGrid1"

# Load Silero VAD model
torch.set_num_threads(1)
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
get_speech_timestamps, _, read_audio, *_ = utils
Voiceflag=0
async def record_audio(duration=5, sample_rate=16000):
    with Progress() as progress:
        task = progress.add_task("[cyan]Listening...", total=duration)
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
        for _ in range(duration):
            await asyncio.sleep(0)
            progress.update(task, advance=1)
    sd.wait()
    return audio.flatten()

def detect_speech(audio, sample_rate=16000):
    audio_tensor = torch.FloatTensor(audio)
    speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=sample_rate)
    return len(speech_timestamps) > 0

async def process_query(audio):
    global Voiceflag
    with console.status("[bold green]Processing query..."):
        transcription, stt_latency = transcribe_audio(audio)
        transcription=transcription.lower()
        transcription = transcription.replace("card", "cart")
        console.print(Panel(f"[bold]Transcription:[/bold] {transcription}"))
        console.print(f"STT Latency: {stt_latency/2:.4f}")
        
        if "bye" in transcription.lower():
            play_audio("b.mp3", 1)
            console.print("[bold red]Goodbye![/bold red]")
            exit()
        if "jerry" in transcription.lower():
            Voiceflag=1
        if "jessica" in transcription.lower():
            Voiceflag=0
        

        response, rag_latency = runInferenceRetailSeller(model, transcription, embedding_model, client, collection_name, data, chatReport, Voiceflag)
        console.print(Panel(f"[bold]Response:[/bold] {response}"))
        console.print(f"RAG Latency: {rag_latency/3:.4f} seconds")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            audio_file = text_to_speech_file(response, temp_file.name, Voiceflag)  
    
    return audio_file

async def play_audio(file_path, flag):
    audio = AudioSegment.from_mp3(file_path)
    play(audio)
    if flag == 0:
        os.unlink(file_path)  # Delete the temporary file after playing

async def main():
    # os.system('cls' if os.name=="nt" else "clear")
    print(chr(27) +"[2J")
    console.print("[bold]Welcome to Hermex, the AI Product Seller Assistant![/bold]")
    await play_audio("a.wav", 1)
    while True:
        audio = await record_audio(duration=7) 
        
        if detect_speech(audio):
            audio_file = await process_query(audio)
            await play_audio(audio_file, 0)
            
            console.print("\n[bold]Listening for your response...[/bold]")
        else:
            console.print("[yellow]No speech detected. Listening again...[/yellow]")
            await asyncio.sleep(0)  

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit:
        asyncio.run(play_audio("b.wav", 1))
    