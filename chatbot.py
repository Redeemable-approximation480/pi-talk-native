# Thread Controls
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

# Load the speech model
print("Initializing AI Speech Engine...")
from faster_whisper import WhisperModel

stt_model = WhisperModel(
    "tiny", 
    device="cpu", 
    compute_type="int8", 
    cpu_threads=1,
    num_workers=1
)
print("✓ Speech Engine Loaded Successfully!")

# Import required libraries
import queue
import time
import requests
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106  
from luma.core.render import canvas
from PIL import ImageFont
import gpiod
from gpiod.line import Direction, Bias, Value

# Hardware settings
BUTTON_PIN = 24
SAMPLE_RATE = 44100
MICROPHONE_DEVICE = 'hw:2,0'
AUDIO_FILE = "input.wav"
OLLAMA_URL = "http://localhost:11434/api/generate"
CHIP_PATH = "/dev/gpiochip4"

# Set up the OLED screen
try:
    serial = i2c(port=1, address=0x3C)
    device = sh1106(serial) 
    device.clear()
except Exception as e:
    print(f"[ERROR] Screen setup failed: {e}")
    device = None

# Load font or use fallback
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
except IOError:
    font = ImageFont.load_default()

# Audio buffer queue
audio_queue = queue.Queue()

# Sounddevice callback function to catch audio blocks
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# Display static status message
def display_status(text):
    print(f"[STATUS] {text}")
    if device is None:
        return
    with canvas(device) as draw:
        draw.text((0, 24), text, font=font, fill="white")

# Split and show text line by line on screen
def animate_text(text):
    print(f"\n[AI RESPONSE]: {text}\n")
    if device is None:
        return
        
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) < 20:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    lines.append(current_line.strip())

    with canvas(device) as draw:
        y = 0
        for line in lines[:5]:
            draw.text((0, y), line, font=font, fill="white")
            y += 12

# Send prompt to local Ollama API
def fetch_qwen_response(prompt):
    display_status("[THINKING...]")
    payload = {"model": "qwen2.5:1.5b", "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=20)
        return response.json().get("response", "...?")
    except Exception as e:
        print(f"Ollama Request Error: {e}")
        return "Local Qwen2.5 offline. Check if 'ollama serve' is running."

# Record audio while button is active
def record_audio(request):
    display_status("[LISTENING]")
    
    while not audio_queue.empty():
        audio_queue.get()

    print(f"[DEBUG] Opening audio stream with device {MICROPHONE_DEVICE}...")
    try:
        stream = sd.InputStream(
            device=MICROPHONE_DEVICE,
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='int16', 
            callback=audio_callback
        )
        
        with stream:
            print("[DEBUG] Stream open. Recording now...")
            while request.get_value(BUTTON_PIN) == Value.INACTIVE:
                time.sleep(0.01)
                
    except Exception as e:
        print(f"[FATAL ERROR] PortAudio could not open stream: {e}")
        return False

    audio_data = []
    while not audio_queue.empty():
        audio_data.append(audio_queue.get())

    if audio_data:
        recording = np.concatenate(audio_data, axis=0)
        write(AUDIO_FILE, SAMPLE_RATE, recording)
        print(f"[DEBUG] Audio file saved ({len(recording)} frames).")
        return True
    return False

# Main program loop
def main():
    display_status("HOLD BUTTON TO TALK")
    print("[DEBUG] System is running and waiting for inputs!")
    
    line_config = {
        BUTTON_PIN: gpiod.LineSettings(
            direction=Direction.INPUT,
            bias=Bias.PULL_UP
        )
    }

    with gpiod.request_lines(CHIP_PATH, consumer="voice-chatbot", config=line_config) as request:
        while True:
            try:
                if request.get_value(BUTTON_PIN) == Value.INACTIVE:
                    time.sleep(0.05)
                    if request.get_value(BUTTON_PIN) == Value.INACTIVE:
                        if record_audio(request):
                            display_status("[PROCESSING...]")
                            segments, _ = stt_model.transcribe(AUDIO_FILE, beam_size=5)
                            user_prompt = "".join([segment.text for segment in segments]).strip()
                            print(f"User Transcribed Text: '{user_prompt}'")
                            
                            if user_prompt:
                                bot_reply = fetch_qwen_response(user_prompt)
                                animate_text(bot_reply)
                                time.sleep(5)
                            else:
                                display_status("[NO AUDIO]")
                                time.sleep(1.5)
                        display_status("HOLD BUTTON TO TALK")
                time.sleep(0.1)
            except KeyboardInterrupt:
                print("\nExiting application...")
                break

if __name__ == "__main__":
    main()
