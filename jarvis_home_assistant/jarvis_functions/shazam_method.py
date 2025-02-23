import sounddevice as sd
import numpy as np
import wave
import tempfile
from shazamio import Shazam
import asyncio


# Function to capture audio from the microphone
def record_audio(duration=5, samplerate=44100):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    return audio


# Function to save the audio as a temporary WAV file
def save_audio_to_wav(audio_data, samplerate=44100):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmpfile:
        with wave.open(tmpfile, 'wb') as wf:
            wf.setnchannels(2)  # Stereo channels
            wf.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wf.setframerate(samplerate)  # Sample rate (e.g., 44100 Hz)
            wf.writeframes(audio_data.tobytes())  # Write the audio data to the WAV file
        return tmpfile.name


# Synchronous function to recognize audio
def recognize_audio():
    shazam = Shazam()

    # Record audio from the microphone (5 seconds)
    audio_data = record_audio(duration=5, samplerate=44100)

    # Save the audio data to a temporary WAV file
    wav_file = save_audio_to_wav(audio_data)

    # Run the async recognition inside a synchronous context using a new event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(shazam.recognize(wav_file))

    if "track" in result:
        title = result["track"].get("title", "Unknown Title")
        artist = result["track"].get("subtitle", "Unknown Artist")
        return title, artist
    else:
        return None, None  # Return None if no song is found
