import os
import json
import logging
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

client = OpenAI(api_key=api_key)

# Prompt and response cache
prompt_cache = {}
response_cache = {}
conversation_history = {}

# Load the system prompt (cached)
def load_prompt(filepath="prompts.json", prompt_key="LEARNING_AND_DEVELOPMENT"):
    cache_key = f"{filepath}:{prompt_key}"
    if cache_key in prompt_cache:
        return prompt_cache[cache_key]
    try:
        with open(filepath, "r") as file:
            data = json.load(file)
            prompt = data[prompt_key]
            prompt_cache[cache_key] = prompt
            return prompt
    except Exception as e:
        logger.error(f"Error loading prompt: {e}")
        raise RuntimeError(f"Error loading prompt: {e}")

# Sanitize input to remove potentially malicious content
def sanitize_input(text: str) -> str:
    """Removes special characters and excessive whitespace from input."""
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Keep alphanumeric, spaces, and basic punctuation
    return ' '.join(text.split())  # Normalize whitespace

# Main backend-callable function
def get_learning_and_development_response(user_input: str, user_id: str = None, prompt_file: str = "prompts.json") -> dict:
    """
    Processes transcribed text from the backend and returns the AI response.
    Caches responses to save API cost. Supports conversation history with user_id.
    Returns a dict for structured backend integration.
    """
    if not user_input or not user_input.strip():
        logger.warning("Empty or invalid input received.")
        return {"status": "error", "message": "Please provide a valid question."}

    # Sanitize input
    question = sanitize_input(user_input)
    if not question:
        logger.warning("Input is empty after sanitization.")
        return {"status": "error", "message": "Please provide a valid question."}

    # Validate input (relaxed to allow short but valid queries)
    if len(question.split()) < 2 and not any(keyword in question.lower() for keyword in ["what", "how", "why", "is", "are"]):
        logger.warning(f"Potentially incomplete transcription: {question}")
        return {"status": "error", "message": "Input appears incomplete. Please clarify your question."}

    # Check cache
    if question in response_cache:
        logger.info(f"Cache hit for input: {question}")
        return {"status": "success", "response": response_cache[question], "cached": True}

    # Load system prompt
    try:
        learning_and_development_prompt = load_prompt(prompt_file, "LEARNING_AND_DEVELOPMENT")
    except RuntimeError:
        return {"status": "error", "message": "Failed to load system prompt."}

    # Prepare messages
    messages = [{"role": "system", "content": learning_and_development_prompt}]
    if user_id and user_id in conversation_history:
        messages.extend(conversation_history[user_id])
        logger.info(f"Appended conversation history for user_id: {user_id}")
    messages.append({"role": "user", "content": question})

    try:
        response = _call_openai(messages)
        response_cache[question] = response

        # Update conversation history
        if user_id:
            if user_id not in conversation_history:
                conversation_history[user_id] = []
            conversation_history[user_id].extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": response}
            ])
            conversation_history[user_id] = conversation_history[user_id][-10:]

        logger.info(f"Generated response for input: {question}")
        return {"status": "success", "response": response, "cached": False}
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {"status": "error", "message": f"Failed to generate response: {str(e)}"}

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=5))
def _call_openai(messages, model="gpt-4-turbo", temperature=0.7, max_tokens=800):
    """Calls OpenAI and returns the full response as a string."""
    assistant_reply = ""
    response_stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )
    for chunk in response_stream:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                assistant_reply += content
    return assistant_reply
