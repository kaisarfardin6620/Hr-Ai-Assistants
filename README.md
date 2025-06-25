# HR AI Assistants

A suite of robust, production-ready Python backend modules for various HR domains (Compensation, Compliance, Talent Acquisition, Organizational Development, and more). Each assistant leverages OpenAI’s GPT models to provide intelligent, context-aware responses for HR professionals and employees.

## Features

- Modular backend files for each HR domain (e.g., Compensation, Compliance, Talent Acquisition, etc.)
- Consistent structure: logging, input sanitization, caching, and error handling
- Conversation history per user for context-aware responses
- Easy integration with web APIs (Flask, Express.js, etc.)
- Supports both typed and transcribed (voice) input
- Ready for production deployment

## Project Structure

```
Compensation_backend.py
Compliance-backend.py
HR_Business_Partner_backend.py
HR_Strategy_backend.py
Learning_And_Development_backend.py
Organizational_Development_backend.py
Talent_Acquisition_backend.py
Total_Rewards_backend.py
prompts.json
Compensation_assistant.py
Compliance-assistant.py
HR_Business_Partner_assistant.py
HR_Strategy_assistant.py
Learning_And_Development_assistant.py
Organizational_Development_assistant.py
Talent_Acquisition_assistant.py
Total_Rewards_assistant.py
...
```

## Setup

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your environment variables:**
   - Create a `.env` file with your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

3. **Run a backend module or API:**
   - Example (Flask API for image/chat assistant):
     ```bash
     python alamin_api.py
     ```

4. **Test with the provided test script:**
   ```bash
   python test_express_api.py
   ```

## Usage

- Import and call any backend function (e.g., `get_compensation_response`) in your API or application.
- Each function expects user input and (optionally) a user ID for conversation history.
- Responses are returned as structured dictionaries for easy integration.

## Example

```python
from Compensation_backend import get_compensation_response

result = get_compensation_response("What is the salary range for a data analyst?", user_id="user123")
print(result)
```

## API Integration

- Use `alamin_api.py` to expose assistants as RESTful endpoints.
- Use `server.js` to bridge Python backends with Node.js/Express applications.

## Extending

- Add new HR domains by following the existing backend file structure.
- Update `prompts.json` with new or refined system prompts.
