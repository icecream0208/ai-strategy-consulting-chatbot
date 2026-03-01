# ================================================================

# main.py - Grounded AI Strategy Chatbot
# Run this file to start the FastAPI server
# Read README.md for setup and running instructions
# ---------------------------------------------------------------
### Test question: Could you give me a bullet point summary of this Canadian AI Strategy project?
### Follow-up: Could you summarize the previous findings in two paragraphs?

# ================================================================

# ---- Setup environment and imports ----
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ---- CrewAI and tool imports ----
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool
from PyPDF2 import PdfReader
import pandas as pd

# ---- FastAPI imports ----
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===============================================================
VERBOSE = True
MAX_HISTORY = 10  # Max number of past turns to keep in history
# ===============================================================
# ---- File paths  ----
NB_PDF_FILE_PATH = "data/Data_Combined.pdf"
CSV_FILE_PATH = "data/Combined_AI_Target_Scores.csv"
REPORT_PDF_FILE_PATH = "data/Project_Report.pdf"

# ===============================================================
# ---- Conversation history (in-memory) ----
# Each item: {"role": "user" or "assistant", "content": "text"}
conversation_history = []

# ===============================================================
# ---- LLM Setup ----
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment or .env file.")

os.environ["OPENAI_API_KEY"] = API_KEY

llm_data = LLM(
    model="gpt-4.1",
    api_key=API_KEY,
    temperature=0.1,
)

llm_report = LLM(
    model="gpt-4.1",
    api_key=API_KEY,
    temperature=0.2,
)

llm_chat = LLM(
    model="gpt-4.1",
    api_key=API_KEY,
    temperature=0.5,
)

# ===============================================================
# ---- Tools ----

@tool("Provide text content from a PDF file given its file path.")
def read_pdf(file_path: str) -> str:
    """
    Extract and return text from a PDF file.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A single string containing the text from all pages,
        separated by blank lines.
    """
    # DEBUG: show when the tool is actually called
    # print(f"[TOOL] read_pdf called with: {file_path}", flush=True)

    reader = PdfReader(file_path)
    pages_text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text.strip())

    return "\n\n".join(pages_text)


@tool("Provide a text preview of a CSV file given its file path.")
def read_csv(file_path: str, n_rows: int = 10) -> str:
    """
    Load a CSV file and return a small text preview for the LLM.

    Args:
        file_path: Path to the CSV file.
        n_rows: Number of rows to include in the preview (default: 10).

    Returns:
        A string containing the column names and the first `n_rows` rows
        as CSV-formatted text.
    """
    # DEBUG: show when the tool is actually called
    # print(f"[TOOL] read_csv called with: {file_path}, n_rows={n_rows}", flush=True)

    df = pd.read_csv(file_path)
    preview = df.head(n_rows)
    return preview.to_csv(index=False)

# ===============================================================
# ---- Agents ----

data_analyst = Agent(
    role="AI Strategy Data Analyst",
    goal=(
        "Process, interpret, and summarize insights from structured CSV data "
        "and the accompanying methodology contained in the PDF (exported Jupyter Notebook). "
        "Produce accurate, evidence-based analytical summaries and "
        "pass them to the chat agent in a clean, structured format."
    ),
    backstory=(
        "You are a meticulous data analyst specializing in policy analytics and AI strategy evaluation. "
        "You have extensive experience interpreting multi-source datasets, including CSV metrics, "
        "evaluation tables, and full analytical workflows documented in Jupyter Notebooks. "
        "You understand how data is generated, transformed, cleaned, and scored, and you can reconstruct "
        "the end-to-end pipeline from raw files and documentation.\n"
        "Your job is to deeply understand both the numerical results (CSV files) and the methodology "
        "(PDF notebook), identify patterns, compare countries including Canada, and extract key insights. "
        "You document everything clearly, logically, and in structured bullet points so "
        "the chat agent can reliably answer user questions.\n"
        "You never hallucinate - all insights must be grounded in either the CSVs or the notebook PDF."
    ),
    llm=llm_data,
    tools=[read_pdf, read_csv],
    verbose=VERBOSE,
)

report_analyst = Agent(
    role="AI Strategy Report Analyst",
    goal=(
        "Carefully read the group's final report (PDF) on Canada's AI strategy, "
        "extract the main arguments, findings, comparisons, and recommendations, "
        "and summarize them in a clear, structured way so that the chat agent can "
        "use them to answer questions."
    ),
    backstory=(
        "You are an expert in reading and interpreting policy and strategy reports, "
        "especially those related to national AI strategies. "
        "You excel at distilling long, academic or policy-style documents into concise, "
        "well-organized insights. You pay attention to: the overall structure of the report, "
        "the key findings about Canada, any comparisons with other countries, methodology sections, "
        "and the final recommendations or policy implications. "
        "You always keep track of where each insight comes from in the report (e.g., sections or headings) "
        "so that another agent can refer back to them if needed.\n"
        "You never invent content: everything you say must be grounded in the actual PDF report."
    ),
    llm=llm_report,
    tools=[read_pdf],
    verbose=VERBOSE,
)

chat_agent = Agent(
    role="AI Strategy Consultant",
    goal=(
        "Have a meaningful, business-oriented conversation with users about Canada's AI "
        "strategy by using the team's analysis and recommendations from your crew agents. "
        "Explain insights clearly, compare Canada with other countries, and "
        "translate technical results into actionable policy and PR messages."
    ),
    backstory=(
        "You are a consulting-style AI advisor specializing in national AI strategies, "
        "with a focus on Canada. You have access to the team's quantitative results "
        "such as scores, rankings, indicators, comparisons and their storytelling and recommendations. "
        "You are used to speaking with government officials, PR managers, and business leaders, so "
        "you avoid heavy jargon and emphasize clear takeaways, risks, and practical steps.\n\n"
        "In conversations, you:\n"
        "- Use the data analyst's outputs for numbers, rankings, and factor importance.\n"
        "- Use the report analyst's outputs for narratives, PR angles, visual story ideas, "
        "and projected improvements in Canada's AI competitiveness.\n"
        "- Help users understand how proposed policies and programs could change "
        "Canada's AI score over time.\n"
        "- Always stay grounded in the provided project materials (CSV files, notebook, "
        "final report, slides, and prepared summaries). If information is not in those "
        "sources, you clearly say that it is outside the current analysis instead of guessing."
    ),
    llm=llm_chat,
    verbose=VERBOSE,
)

# ===============================================================
# ---- Helper functions to build tasks with CURRENT history ----

LOCAL = ("You must answer ONLY using the content of the local files "
        "that you access through your tools (read_pdf, read_csv, etc.).\n\n"
        "Rules:\n"
        "1. Before answering, ALWAYS call the appropriate tool to read the file.\n"
        "2. If the answer is not clearly stated in the files, say: "
        "'I cannot find this information in the provided documents.'\n"
        "3. In every answer, include an 'Evidence' section that quotes the exact "
        "text or table from the file you used.\n"
        "4. Never rely on your own outside knowledge.")

def build_history_text() -> str:
    """Return the last MAX_HISTORY turns as a text block for prompts."""
    messages = []
    count = 0

    for msg in reversed(conversation_history):
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()

        messages.append(f"{role.upper()}: {content}")
        count += 1

        if count == MAX_HISTORY:
            break
    
    messages.reverse()

    return "\n".join(messages)

def build_data_summary_task() -> Task:
    return Task(
        description=(
            "Your job in this task is to READ the project data from disk and summarize it.\n\n"
            "You MUST do the following steps in order:\n"
            f"1. Call `read_pdf` on the Jupyter Notebook PDF at: '{NB_PDF_FILE_PATH}'.\n"
            f"2. Call `read_csv` on the provided CSV file at: '{CSV_FILE_PATH}'\n"
            "   Do NOT guess what is inside; always use the tool.\n"
            "3. After reading, produce a detailed summary that can answer the user's question, including:\n"
            "- Key metrics and their meanings\n"
            "- Patterns and trends across countries, especially Canada\n"
            "- Methodology used to generate the data\n"
            "- Any anomalies or important observations\n\n"
            "Summarize in clear bullet points, grounded ONLY in the data and notebook.\n\n"
            "If you cannot access a file or it is missing, explicitly say so.\n\n"
            "Here is the conversation so far:\n{history_text}\n\n"
            "Here is the latest user question:\n{user_message}\n\n"
            f"{LOCAL}\n"
        ),
        expected_output="A structured summary of insights from the CSV data and notebook PDF.",
        agent=data_analyst,
        verbose=VERBOSE,
    )


def build_report_summary_task() -> Task:
    return Task(
        description=(
            "Your job in this task is to READ the final report PDF from disk and summarize it.\n\n"
            "You MUST:\n"
            f"1. Call `read_pdf` on the report file at: '{REPORT_PDF_FILE_PATH}'.\n"
            "2. Based ONLY on the PDF content, produce a clear, structured summary including:\n"
            "- Main arguments and findings about Canada's AI strategy\n"
            "- Comparisons with other countries\n"
            "- Key recommendations and policy implications\n\n"
            "Summarize in your own words. Do NOT provide long verbatim text.\n"
            "Include an 'Evidence' section with short quotes or references to sections/headings.\n"
            "If you cannot read the file, explicitly say that.\n\n"
            "Here is the conversation so far:\n{history_text}\n\n"
            "Here is the latest user question:\n{user_message}\n\n"
            f"{LOCAL}\n"
        ),
        expected_output="A structured summary of insights from the final report PDF.",
        agent=report_analyst,
        verbose=VERBOSE,
    )


def build_final_chat_task(data_task: Task, report_task: Task) -> Task:
    return Task(
        description=(
            "Engage in a meaningful conversation with the user about Canada's AI strategy, "
            "using the summaries provided by the data analyst and report analyst agents. "
            "Answer the user's questions clearly, providing actionable insights and comparisons "
            "where relevant. If the information is not available in the provided summaries, "
            "clearly state that it is outside the current analysis instead of guessing.\n\n"
            "Here is the conversation so far:\n{history_text}\n\n"
            "Here is the latest user question:\n{user_message}\n\n"
        ),
        expected_output=(
            "A clear, informative response to the user's question about Canada's AI strategy."
        ),
        agent=chat_agent,
        context=[data_task, report_task],
        verbose=VERBOSE,
    )

# ===============================================================
# ---- FastAPI setup ----
app = FastAPI()

# Allow calls from a static HTML file opened in browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str  # markdown string

# ===============================================================
# ---- Chat endpoint ----

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Store user message
    user_message = req.message
    conversation_history.append({"role": "user", "content": user_message})

    # Build current conversation history text
    history_text = build_history_text()

    # Build fresh tasks with up-to-date history
    data_task = build_data_summary_task()
    report_task = build_report_summary_task()
    final_chat_task = build_final_chat_task(data_task, report_task)

    # Create crew for this request
    crew = Crew(
        agents=[data_analyst, report_analyst, chat_agent],
        tasks=[data_task, report_task, final_chat_task],
        verbose=VERBOSE,
        process=Process.sequential,
    )

    # Run crew
    result = crew.kickoff(
        inputs={
            "history_text": history_text,
            "user_message": user_message,
        }
    )

    # Extract final reply for markdown text
    try:
        if hasattr(result, "tasks_output") and result.tasks_output:
            # Final task output from chat agent
            final_reply = getattr(result.tasks_output[-1], "raw", str(result))
        elif hasattr(result, "raw"):
            final_reply = result.raw
        else:
            final_reply = str(result)
    except Exception:
        final_reply = str(result)

    # Save reply to history
    conversation_history.append({"role": "assistant", "content": final_reply})

    # Return markdown to frontend
    return ChatResponse(reply=final_reply)

# ===============================================================
# ---- Main entrypoint ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

