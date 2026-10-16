# 📄 Resume ATS Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) friendliness and tells you exactly how to improve it, powered by Google Gemini Flash.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- **Estimated ATS score (0-100)** with a breakdown: keywords, content & impact, formatting, structure, readability
- **Prioritised improvements** (High / Medium / Low) with example rewrites
- **Keyword gap analysis**, optionally against a pasted job description
- **Rule-based ATS checks**: email, phone, links, standard section headings, length, bullets, quantified results
- **Downloadable Markdown report**

> The score is an AI-generated estimate to guide your edits. It is not the output of a real ATS.

## Project structure

```
resume-ats-checker/
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. Install Python 3.10 or newer.
2. Get a free Gemini API key at <https://aistudio.google.com/apikey>.
3. Install and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

4. Paste your API key in the sidebar, or set it once so you are not asked each time:

   - Create `.streamlit/secrets.toml` in the project folder:
     ```toml
     GEMINI_API_KEY = "your-key-here"
     # optional:
     # GEMINI_MODEL = "gemini-flash-latest"
     ```
   - or set an environment variable: `export GEMINI_API_KEY="your-key-here"`

> ⚠️ Never commit your key. Add `.streamlit/secrets.toml` to `.gitignore`.

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | Streamlit secrets, env var, or sidebar | none (required) |
| `GEMINI_MODEL` | Streamlit secrets, env var, or sidebar | `gemini-flash-latest` |

`gemini-flash-latest` always points to Google's newest Flash model. To pin a specific version, set `GEMINI_MODEL` (for example `gemini-2.5-flash`). If you see a "model not found" error, Google has renamed or retired that model, so change it in the sidebar.

## Deploy on Streamlit Community Cloud

1. Push this folder to a **GitHub repository** (`app.py` and `requirements.txt` at the repo root).
2. Go to <https://share.streamlit.io> and sign in with GitHub.
3. Click **Create app**, choose the repo, branch `main`, and main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Limitations

- Scanned or image-only PDFs have no extractable text, so an ATS could not read them either. Use a text-based PDF or DOCX.
- Uploads are limited to 5 MB, and only the first ~20,000 characters are analysed.
- Resume text is sent to the Gemini API. Do not upload documents you are not comfortable sharing with it.

## Tech stack

[Streamlit](https://streamlit.io) · [Google Gemini](https://ai.google.dev) via `google-genai` · [pypdf](https://pypdf.readthedocs.io) · [python-docx](https://python-docx.readthedocs.io)
