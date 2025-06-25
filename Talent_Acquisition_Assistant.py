from dotenv import load_dotenv
import os
import openai
import json
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("❌ Error: OPENAI_API_KEY not found in environment variables.")
    exit(1)

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
talent_acquisition_prompt = prompts["TALENT_ACQUISITION"]

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def ask_talent_acquisition_assistant(user_question):
    if not talent_acquisition_prompt:
        return "System prompt could not be loaded. Please check your JSON file."
    if not user_question.strip():
        return "❗ Please ask a valid question."
    # Check response cache first
    if user_question in response_cache:
        return response_cache[user_question]
    messages = [
        {"role": "system", "content": talent_acquisition_prompt},
        {"role": "user", "content": user_question}
    ]
    try:
        response = stream_chat_response(messages, openai.api_key)
        response_cache[user_question] = response  # Save to cache
        return response
    except openai.OpenAIError as e:
        return f"❌ OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}"

def stream_chat_response(conversation, api_key, model="gpt-4-turbo", temperature=0.7, max_tokens=800):
    client = OpenAI(api_key=api_key)
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
                    print(content, end="", flush=True)
                    assistant_reply += content
        print()
        return assistant_reply
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return "Sorry, something went wrong. Please try again later."

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        answer = ask_talent_acquisition_assistant(user_input)
        print("Talent Acquisition Assistant:", answer)
    else:
        while True:
            try:
                question = input("Ask the Talent Acquisition Assistant (or type 'exit' to quit): ")
                if question.lower() == "exit":
                    break
                ask_talent_acquisition_assistant(question)
            except KeyboardInterrupt:
                print("\n❗ Interrupted by user. Exiting...")
                break
            except Exception as e:
                print(f"❌ Error during input handling: {str(e)}")