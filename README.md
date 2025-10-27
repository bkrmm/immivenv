# Gemini prompt script

This repository contains `agent.py`, a small script that constructs a prompt for Google Gemini
(via the `langchain_google_genai` wrapper) using:

- `user_input`: a JSON file (default `user1.json`).
- `context`: a URL string (default: `https://github.com/bkrmm/immigration-agent-test1/`).
- Local markdown files found in the repository root (they are included in the prompt).

Usage

1. Activate your virtual environment (Windows PowerShell):

```powershell
.\immivenv\Scripts\Activate.ps1
```

2. Set your Google API key in the environment variable `GOOGLE_API_KEY` (replace `YOUR_KEY`):

```powershell
z
```

3. Run the script (example):

```powershell
python .\agent.py --user-input .\user1.json --context-url "https://github.com/bkrmm/immigration-agent-test1/"
```

Behavior

- If `GOOGLE_API_KEY` is not set, the script prints the constructed system prompt and the
  combined human input (dry run) so you can inspect the input that will be sent to Gemini.
- If `GOOGLE_API_KEY` is set, the script will call Gemini via the `ChatGoogleGenerativeAI`
  LangChain wrapper and print the model response plus optional metadata.

Notes

- The script attempts a tolerant JSON parse: if the JSON file has trailing text (e.g., notes),
  it will try to parse the first JSON array or object found in the file.
- The script uses the `langchain_google_genai` package included in the workspace.

If you want me to also add a small unit test or change the output format (JSON, file), tell me which format you prefer.
