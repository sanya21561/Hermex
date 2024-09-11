from elevenlabs import Voice, VoiceSettings
from elevenlabs.client import ElevenLabs
import time 

client = ElevenLabs(
    api_key="sk_34c8f4422d2953c0f8c1e147848247b90c7345f404820f64",
)

def text_to_speech_file(text: str, file_path: str, flag) -> str:
    """
    Converts text to speech and saves the output as an MP3 file.

    Args:
        text (str): The text content to convert to speech.
        file_path (str): The file path where the audio file should be saved.

    Returns:
        str: The file path where the audio file has been saved.
    """
    start_time = time.time()
    voiceId1="cgSgspJ2msm6clMCkdW9"  #JESSICA
    voiceId2="iP95p4xoKVk53GoZ742B"     #Jerry
    if flag==0:
        voicer=voiceId1
    else:
        voicer=voiceId2
    response = client.text_to_speech.convert(
        voice_id=voicer,
        optimize_streaming_latency="1",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    end_time = time.time()

    latency = end_time - start_time

    with open(file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"TTS Generation Time: {latency*1000:.4f} seconds")
    return file_path