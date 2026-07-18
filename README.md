# Pi Talk Native: Custom Embedded Edge-AI Voice Companion

A voice assistant powered by a Raspberry Pi 5 single-board computer.

Rather than going for the easy way out, which is using the APIs of either Google Cloud or OpenAI, this whole project is developed to operate **100% locally and totally offline**. The system captures the hardware interrupts to read the arrays from the microphones and uses the speech-to-text module of the chip itself to translate the voice. This is then processed by an offline large language model to reason and display the results on a physical OLED display through an I2C bus.

This project was designed to test the limits of native-edge AI, low overhead Linux kernel, and multi-core thread optimization.

---

## The Setup
[Arcade Button] ──> [USB Mic Audio Capture] ──> [Whisper Speech-to-Text]
│
▼
[OLED Screen]  <───  [Local Ollama Engine]  <───  [Text Processing]

*   **Brain:** Raspberry Pi 5 (8GB RAM variant).
*   **Operating System:** Raspberry Pi OS Lite (64-bit, keeping things lightweight so there's more memory for the AI).
*   **Hardware Pin Control:** Modern `libgpiod v2` library (the new way to handle GPIO pins on the Pi 5 without dealing with slow lag).
*   **Audio Input:** Standard USB plug-and-play mic running through Python's `sounddevice` library.
*   **Speech-to-Text:** `faster-whisper` (Tiny model, compressed down to `int8` precision so the CPU can actually run it quickly).
*   **Local AI Model:** Ollama running `qwen2.5:1.5b`. It's a super fast, lightweight 1.5-billion parameter model that fits perfectly on the Pi.
*   **Display:** 1.3-inch SH1106 monochrome OLED screen wired using I2C lines.

---

## The Challenges I Faced & How I Fixed Them

Wiring up hardware to local AI models means you run into some challenges where the software and hardware collide. Here are the biggest problems I had to solve:

### 1. The Microphone Crash & Core Lockup Loop
*   **The Problem:** The very first time I tried to record my voice, the code would instantly freeze with massive `paInputOverflowed` errors in the terminal.
*   **What was actually happening:** By default, Python tries to do everything on a single thread. Because running an AI model takes massive processing power, the script was totally locking up the main loop while processing. While the script was frozen, the microphone was still flooding the system with audio data. The queue filled up instantly and crashed. On top of that, the mic hated standard 16,000 Hz settings.
*   **How I fixed it:** I forced the background system environment variables to open up all 4 cores of the Pi 5 (`OMP_NUM_THREADS = "4"`). Then, I rewrote the audio recorder to use a non-blocking queue. Now, the mic streams audio smoothly at its native 44,100 Hz on its own separate loop, so the AI processing never chokes the incoming microphone data.

### 2. The Inverted Button Bug (The Pull-Up Resistor Issue)
*   **The Problem:** When I first tested the button, it acted completely backwards. The system would do absolutely nothing while I held the button down, but the second I let go, it would start recording forever until it broke.
*   **What was actually happening:** I set up the GPIO pin to use an internal Pull-Up resistor. In electronics, a pull-up resistor biases the pin's default state to High (1). When you actually push the button down, it shorts the circuit straight to ground, which drops the value to Low (0). My code was looking for a 1 to start recording, meaning it thought my "idle" time was the command to talk.
*   **How I fixed it:** I changed the condition in the script to look for `Value.INACTIVE` (the low state) instead of high. Now the recording state perfectly matches the exact moments my finger is holding down the button.

### 3. The "TV Snow" Screen Mess
*   **The Problem:** The first time I tried to print text to the OLED screen, the display just showed random static, crazy flickering lines, and broken pixel blocks.
*   **What was actually happening:** Almost every tutorial online assumes you're using a common SSD1306 screen controller. My display actually used an SH1106 chip. They use the exact same I2C address (`0x3C`), but their internal memory layouts are totally different. Sending standard commands meant the data was getting shoved into the wrong pixel slots.
*   **How I fixed it:** I swapped out the basic display drivers for the specific `sh1106` framework from the `luma.oled` library. I also added a hard `device.clear()` command right when the script boots up to wipe out any leftover voltage garbage before printing the real UI.

---

## Step-by-Step Installation Guide

If you want to get this exact project running on a Pi 5, follow these steps in order:

### Step 1: Install System Libraries
Update the Pi and download the packages needed for audio drivers and fonts:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-dev libportaudio2 libasound2-dev fonts-dejavu git
Step 2: Set Up Ollama
```

### Step 2: Set Up Ollama
Install the local AI engine and pull down the lightweight Qwen model:
```bash
# Get the official installer script
curl -fsSL [https://ollama.com/install.sh](https://ollama.com/install.sh) | sh

# Pull the model to the board
ollama run qwen2.5:1.5b
```
(Once it finishes downloading, you can exit the model shell by typing /bye).

### Step 3: Set Up the Python Environment
Create a clean virtual environment so you don't mess up your global system packages:

```bash
python3 -m venv --system-site-packages env
source env/bin/activate

# Upgrade pip and install the specific hardware/AI libraries
pip install --upgrade pip
pip install sounddevice numpy scipy requests faster-whisper luma.oled gpiod
```

### The Complete Code (chatbot.py)
Here is the full code:

```python
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
```
### What I Learned From This Project
Building this project taught me a ton of practical computer science skills that go way beyond basic coding:

1. Edge AI Constraints: I figured out how to optimize massive machine learning models by tweaking quantization models (int8) and core limitations so they run on small hardware without overheating the board.
2. Low-Level I/O: I got to drop legacy libraries and learn how the newer Linux libgpiod v2 interface handles physical hardware pins directly through kernel chip paths.
3. Audio Pipelines: I learned how to troubleshoot real-time data queues, audio ring buffers, and sample-rate matching constraints on a live hardware stream.
