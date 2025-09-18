# Medical Specialty Classifier - Setup

This project is a Streamlit app for classifying medical transcriptions. It can optionally transcribe audio using the Groq Whisper API. Keep your API keys secret.

Setup
1. Create a virtual environment and install minimal requirements:

```powershell
# Create venv (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements_minimal.txt
```

2. Configure your Groq API key:

Option A — use a `.env` file (recommended for local development):

 - Copy `.env.example` to `.env` and add your key:

```powershell
copy .env.example .env
# Then open .env and set GROQ_API_KEY=<your_key>
```

Option B — set environment variable in PowerShell for the session:

```powershell
$env:GROQ_API_KEY = "your_groq_key_here"
```

Run the app

```powershell
# From the project root
streamlit run streamlit_medical_classifier.py
```

Security
- `.env` is ignored by `.gitignore`. Do not commit secrets to source control.
