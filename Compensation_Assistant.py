from dotenv import load_dotenv
import os
import openai
import json
import sys
import sounddevice as sd
import scipy.io.wavfile as wav
from tempfile import NamedTemporaryFile
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("❌ Error: OPENAI_API_KEY not found in environment variables.")
    exit(1)

client = OpenAI(api_key=api_key)

# Prompt cache for efficiency
prompt_cache = {}
response_cache = {}

def load_prompt(filepath):
    if filepath in prompt_cache:
        return prompt_cache[filepath]
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
            prompt = data["prompt"]
            prompt_cache[filepath] = prompt
            return prompt
    except FileNotFoundError:
        print(f"❌ Error: Prompt file not found: {filepath}")
    except json.JSONDecodeError:
        print(f"❌ Error: Prompt file is not a valid JSON: {filepath}")
    except KeyError:
        print(f"❌ Error: 'prompt' key not found in the JSON file: {filepath}")
    return None

with open("prompts.json", "r") as file:
    prompts = json.load(file)
compensation_prompt = prompts["COMPENSATION"]

# Audio recording and transcription
def record_audio(duration=5, samplerate=44100):
    print(f"🎙️ Recording for {duration} seconds...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    temp_file = NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, samplerate, recording)
    print("✅ Recording saved.")
    return temp_file.name

def transcribe_audio(audio_path):
    print("🧠 Transcribing audio...")
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )
    return transcript.text

# Retry logic for robustness
@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def ask_compensation_assistant(user_question):
    if not compensation_prompt:
        return "System prompt could not be loaded. Please check your JSON file."
    if not user_question.strip():
        return "❗ Please ask a valid question."
    if user_question in response_cache:
        return response_cache[user_question]

    messages = [
        {"role": "system", "content": compensation_prompt},
        {"role": "user", "content": user_question}
    ]

    try:
        response = stream_chat_response(messages)
        response_cache[user_question] = response
        return response
    except openai.OpenAIError as e:
        return f"❌ OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"

def stream_chat_response(conversation, model="gpt-4-turbo", temperature=0.7, max_tokens=800):
    assistant_reply = ""
    try:
        response_stream = client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )
        for chunk in response_stream:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    print(content, end="", flush=True)  # Print as it streams
                    assistant_reply += content
        return assistant_reply
    except Exception as e:
        return f"❌ Error generating response: {e}"

# CLI with text/voice support
def run_chat_loop():
    while True:
        try:
            mode = input("\nType 'text' to type a question or 'voice' to speak (or 'exit' to quit): ").strip().lower()
            if mode == "exit":
                print("👋 Goodbye!")
                break
            elif mode == "voice":
                audio_path = record_audio(duration=5)
                user_input = transcribe_audio(audio_path)
                print("🗣️ You said:", user_input)
            elif mode == "text":
                user_input = input("Type your question: ")
            else:
                print("⚠️ Invalid input. Choose 'text', 'voice', or 'exit'.")
                continue

            response = ask_compensation_assistant(user_input)
            # Remove the extra print to avoid double output
            # print("🤖 Compensation Assistant:", response)

        except KeyboardInterrupt:
            print("\n❗ Interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# Main
if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        answer = ask_compensation_assistant(user_input)
        print("Compensation Assistant:", answer)
    else:
        run_chat_loop()