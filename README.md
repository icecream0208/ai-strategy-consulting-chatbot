# AI Strategy Consulting Chatbot

## What this repo contains
A minimal runnable version of a grounded chatbot:
- FastAPI backend (`main.py`)
- Browser frontend (`index.html`)
- Local grounding data in `data/` (PDF files)
- Dependency list (`requirements.txt`)

## Main purpose
Provide a chatbot that reads local PDF and CSV files and returns grounded answers in conversation.

## Key Work
- Python backend implementation
- FastAPI API service (`POST /chat`)
- CrewAI multi-agent orchestration for grounded analysis

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` in repo root:
   ```env
   OPENAI_API_KEY=your_key_here
   ```
4. Start backend:
   ```bash
   python main.py
   ```
5. Open `index.html` in browser.

## API
- `POST /chat`
- Request:
  ```json
  {"message": "Summarize Canada's strengths"}
  ```
- Response:
  ```json
  {"reply": "...grounded answer..."}
  ```

## Grounding files
- `data/Data_Combined.pdf`
- `data/Project_Report.pdf`
- `data/Combined_AI_Target_Scores.csv` (add your local CSV file)

## Notes
- The chatbot is intended for local demo workflows.
- For production use, add authentication, storage-backed chat history, and structured citations.
