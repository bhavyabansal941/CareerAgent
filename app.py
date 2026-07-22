import chainlit as cl
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import base64
import requests
import io
import json
from pypdf import PdfReader
from docx import Document
import openpyxl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from tavily import TavilyClient

load_dotenv()

@cl.data_layer
def get_data_layer():
    db_url = os.getenv("DATABASE_URL")
    # Convert standard postgres:// URL to the async driver format SQLAlchemy needs
    conninfo = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    # asyncpg doesn't accept sslmode in the URL itself — strip it and pass SSL separately
    conninfo = conninfo.split("?")[0]
    return SQLAlchemyDataLayer(conninfo=conninfo, connect_args={"ssl": "require"})


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict,
    default_user: cl.User,
) -> cl.User:
    default_user.metadata["name"] = raw_user_data.get("name") or raw_user_data.get("given_name") or "there"
    return default_user

# ---------------- LLMs ----------------
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
vision_llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.2-11b-vision-preview")

# ---------------- Web search grounding (Tavily) ----------------
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None

@tool
def web_search(query: str) -> str:
    """Search the web for current, up-to-date, or time-sensitive information: exam patterns,
    syllabi, cutoffs, application deadlines, recent policy or pattern changes, current events, or
    anything that may have changed since your training data. Always use this before stating facts
    about things that change over time (exam formats, eligibility criteria, dates, current rankings,
    recent news) rather than relying on memory. Returns a summary plus source URLs to cite.
    For official/government topics, include the official body's name in the query (e.g. "SSC CGL
    exam pattern ssc.gov.in") to help surface authoritative sources over forums or blogs."""
    if not tavily_client:
        return "Web search is not configured (missing TAVILY_API_KEY)."
    try:
        results = tavily_client.search(query, max_results=5, include_answer=True)
        output = ""
        if results.get("answer"):
            output += f"Summary: {results['answer']}\n\n"
        output += "Sources:\n"
        for r in results.get("results", []):
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            snippet = (r.get("content") or "")[:250]
            output += f"- {title} ({url}): {snippet}\n"
        return output or "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"

llm_with_tools = llm.bind_tools([web_search])

# ---------------- Base system prompt ----------------
BASE_PROMPT = """You are CareerAgent, a helpful AI assistant with special expertise in careers —
resumes, job search, interview prep, and skill development. You are NOT restricted to career topics;
answer any question on any subject to the best of your ability.

Match your response length and structure to the question:
- Casual questions, greetings, or small talk ("what are you doing", "how are you", "hi") get a short,
  natural, conversational reply — one or two sentences, no headers, no bullet lists.
- Substantive questions (career advice, technical explanations, how-to requests) get structured,
  detailed answers with headers/bullets where helpful.
Don't pad short questions with unnecessary capability lists or headers just to seem thorough.

You have a web_search tool. Use it whenever a question involves something that can change over time —
exam patterns, syllabi, eligibility criteria, cutoffs, application deadlines, current job openings,
recent policy changes, or any fact you're not certain is still current. Do not answer time-sensitive
questions from memory alone.
When searching for exam patterns, government schemes, official eligibility criteria, or similar official
information, prefer official sources: government/commission websites (.gov, .nic.in, .gov.in), the
official body's own site (e.g. ssc.gov.in for SSC exams, upsc.gov.in for UPSC), or well-established news
outlets. Avoid relying solely on forums, Quora, YouTube, or unverified blogs for facts like this — if the
first search results are mostly low-quality sources, run a follow-up search using different keywords
(such as the official body's name) to try to surface the authoritative source.
When you use web_search, cite the source URLs it returns so the user can verify the information themselves.
If web_search is unavailable or returns nothing useful, say so plainly and tell the user to check the
official source — never state an unverified time-sensitive fact as if it were certain.
CRITICAL: before stating specific facts (dates, eligibility criteria, percentages, processes) about a named
exam, program, or scheme, confirm that a search result actually names and describes that specific thing —
not just a related topic. If the user asks about something and your search results only cover general or
adjacent topics without confirming that specific name exists, say you couldn't find evidence it exists and
ask the user to double-check the name/spelling, rather than inventing plausible-sounding details. Do not
attach source links to a claim unless those sources actually support that specific claim.
Never promise to "remember this for next time" or "update your knowledge going forward" — each
conversation starts fresh with no memory of this correction, so don't imply otherwise."""

# ---------------- Mode-specific prompts ----------------
MODE_PROMPTS = {
    "ats_score": """You are giving an ESTIMATED keyword-alignment check, not a real ATS score — actual ATS
software uses proprietary parsing you don't have access to, and you must be upfront about that.
CRITICAL: You need both a resume AND a specific job description. If either is missing, DO NOT invent one —
ask the user to provide what's missing and stop there.
Once you have both, give:
1. "Estimated Keyword Match: ~X%" (frame as a rough estimate, not a definitive score)
2. Keywords from the JD missing in the resume
3. 3-5 specific, resume-grounded fixes (reference actual resume content, not generic advice)
Always end with: "Note: this is an estimate based on keyword overlap, not an official ATS score.\"""",

    "skill_gap": """Compare the user's ACTUAL current skills (from their resume/background provided below)
against the target role. CRITICAL: if no resume or explicit skill list is available, DO NOT assume a
generic background — ask the user to upload their resume or list their skills, and stop there.
Once you have real information, ground your analysis in their ACTUAL projects/skills (name them specifically),
not generic role descriptions. List: skills they already have, skills they're missing, and a prioritized
learning order.""",

    "roadmap": """Create a structured, realistic learning roadmap for the user's stated career goal.
If a resume is available below, use it to skip steps they've already completed and focus the roadmap
on their actual gaps, rather than starting from zero generically.
Break it into phases with rough time estimates. Be honest about typical timelines — don't oversell speed.""",

    "learning_recs": """Recommend specific, real, well-known learning resources (courses, docs, free platforms)
for the user's stated skill or role. Prefer free/affordable options. Be specific about what each resource covers.
If a resume is available, tailor recommendations to their actual current level rather than assuming beginner.""",

    "mock_interview": """You are conducting a mock interview for the user's target role.
If a resume is available below, tailor questions to their actual projects/experience where relevant.
Ask ONE interview question at a time. After they answer, give brief constructive feedback (strengths + one
improvement), then ask the next question. Keep it realistic and role-appropriate.""",

    "resume_analysis": """Review the resume content provided below. Give feedback on: clarity, structure,
impact of bullet points (are they quantified?), and any red flags. Be specific — quote or reference actual
lines from their resume, not generic resume advice."""
}

STICKY_MODES = ["ats_score", "skill_gap", "resume_analysis", "mock_interview", "roadmap", "learning_recs"]

# ---------------- Chainlit starters (sidebar-style quick actions) ----------------
@cl.set_starters
async def starters():
    return [
        cl.Starter(label="Review my resume", message="Review my resume"),
        cl.Starter(label="ATS score", message="Check ATS score for my resume"),
        cl.Starter(label="Skill gap for a role", message="Skill gap for "),
        cl.Starter(label="Mock interview", message="Mock interview for "),
    ]

# ---------------- Keyword-based fast routing ----------------
def keyword_intent(text: str):
    t = text.lower()
    if any(k in t for k in ["never mind", "new topic", "just chat", "forget that", "general question"]):
        return "general_reset"
    if any(k in t for k in ["ats score", "ats check", "match score", "compatibility score", "keyword match"]):
        return "ats_score"
    if any(k in t for k in ["skill gap", "missing skills", "skills am i missing", "what skills do i need"]):
        return "skill_gap"
    if any(k in t for k in ["roadmap", "learning path", "how do i become", "plan to learn", "path to become"]):
        return "roadmap"
    if any(k in t for k in ["recommend course", "learning resource", "where to learn", "best course"]):
        return "learning_recs"
    if any(k in t for k in ["mock interview", "practice interview", "interview me", "interview prep"]):
        return "mock_interview"
    if any(k in t for k in ["review my resume", "how's my resume", "feedback on resume", "check my resume", "review this file", "review this resume"]):
        return "resume_analysis"
    if any(k in t for k in ["job search", "find jobs", "job openings", "job listings", "jobs for"]):
        return "job_search"
    return None

# ---------------- LLM fallback classifier (for ambiguous/natural phrasing) ----------------
def llm_classify(text: str) -> str:
    prompt = f"""Classify this user message into exactly ONE category. Reply with ONLY the category word, nothing else.

Categories:
- ats_score (checking resume against a job description)
- skill_gap (what skills are missing for a role)
- roadmap (learning path/plan to reach a role)
- learning_recs (asking for courses/resources)
- mock_interview (wants interview practice)
- resume_analysis (wants resume reviewed/critiqued)
- job_search (wants real job listings)
- general (anything else, including general knowledge questions)

Message: "{text}"

Category:"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        result = resp.content.strip().lower()
        valid = set(STICKY_MODES + ["job_search", "general"])
        for v in valid:
            if v in result:
                return v
    except Exception:
        pass
    return "general"

def classify_intent(user_text: str, active_mode: str) -> str:
    kw = keyword_intent(user_text)
    if kw == "general_reset":
        return "general_reset"
    if kw:
        return kw
    # No clear keyword — if short/ambiguous, ask the LLM to classify instead of blindly falling back
    if active_mode in STICKY_MODES and len(user_text.split()) < 12:
        return active_mode  # likely a follow-up in the current flow
    return llm_classify(user_text)

# ---------------- Free job search via Adzuna ----------------
def extract_job_query(text: str):
    """Pull a clean job title/keywords and location out of a natural-language request,
    since Adzuna matches keywords literally and a full sentence rarely matches anything."""
    prompt = f"""Extract the job title/keywords and location from this job search request.
Reply with ONLY a JSON object, no other text, no markdown, in this exact format:
{{"what": "job title or keywords only", "where": "city/location or empty string if none mentioned"}}

Message: "{text}"
JSON:"""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)
        what = data.get("what", "").strip() or text
        where = data.get("where", "").strip()
        return what, where
    except Exception:
        return text, ""

def search_jobs(query: str, where: str = "", location: str = "in"):
    app_id = os.getenv("ADZUNA_APP_ID")
    app_key = os.getenv("ADZUNA_APP_KEY")
    if not app_id or not app_key:
        return None
    url = f"https://api.adzuna.com/v1/api/jobs/{location}/search/1"
    params = {"app_id": app_id, "app_key": app_key, "results_per_page": 10, "what": query, "content-type": "application/json"}
    if where:
        params["where"] = where
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return None

# ---------------- Generic file text extraction ----------------
def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    return "".join(page.extract_text() or "" for page in reader.pages).strip()

def extract_file_text(file_path: str, mime: str, filename: str) -> str:
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        if "pdf" in (mime or "") or filename.lower().endswith(".pdf"):
            return extract_pdf_text(file_bytes)
        if "wordprocessingml" in (mime or "") or "msword" in (mime or "") or filename.lower().endswith((".docx", ".doc")):
            doc = Document(io.BytesIO(file_bytes))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if "spreadsheetml" in (mime or "") or filename.lower().endswith((".xlsx", ".xls")):
            wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
            text = ""
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    text += " ".join(str(c) for c in row if c is not None) + "\n"
            return text.strip()
        if (mime and "text" in mime) or filename.lower().endswith((".txt", ".csv", ".md", ".json")):
            return file_bytes.decode("utf-8", errors="ignore")
        if mime and "image" in mime:
            return None
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"[Could not read file: {str(e)}]"

# ---------------- Chat lifecycle ----------------
@cl.on_chat_start
async def start():
    cl.user_session.set("history", [SystemMessage(content=BASE_PROMPT)])
    cl.user_session.set("resume_text", None)
    cl.user_session.set("active_mode", None)
    # No message sent here — this keeps the starter buttons visible on the welcome screen

@cl.on_chat_resume
async def resume(thread: dict):
    # Rebuild our internal history from the saved thread messages
    history = [SystemMessage(content=BASE_PROMPT)]
    resume_text = None
    active_mode = None

    for step in thread.get("steps", []):
        if step.get("type") == "user_message":
            history.append(HumanMessage(content=step.get("output", "")))
        elif step.get("type") == "assistant_message":
            history.append(AIMessage(content=step.get("output", "")))

    cl.user_session.set("history", history)
    cl.user_session.set("resume_text", resume_text)
    cl.user_session.set("active_mode", active_mode)

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    resume_text = cl.user_session.get("resume_text")

    if message.elements:
        for element in message.elements:
            filename = getattr(element, "name", "file")
            mime = element.mime or ""
            if "image" in mime:
                with open(element.path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                vision_msg = HumanMessage(content=[
                    {"type": "text", "text": "Extract all readable text from this image, verbatim, no commentary."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ])
                resp = vision_llm.invoke([vision_msg])
                resume_text = resp.content
                cl.user_session.set("resume_text", resume_text)
                await cl.Message(content=f"✅ **{filename}** (image) received and read.").send()
            else:
                extracted = extract_file_text(element.path, mime, filename)
                if extracted and not extracted.startswith("[Could not read"):
                    resume_text = extracted
                    cl.user_session.set("resume_text", resume_text)
                    await cl.Message(content=f"✅ **{filename}** received and read.").send()
                else:
                    await cl.Message(content=f"⚠️ Couldn't read **{filename}** — unsupported or corrupted file.").send()

    user_text = message.content or ""
    active_mode = cl.user_session.get("active_mode")
    intent = classify_intent(user_text, active_mode)

    if intent == "general_reset":
        intent = "general"
        cl.user_session.set("active_mode", None)
    elif intent in STICKY_MODES:
        cl.user_session.set("active_mode", intent)
    elif intent == "job_search":
        cl.user_session.set("active_mode", None)

    if intent == "job_search":
        job_title, job_location = extract_job_query(user_text)
        results = search_jobs(job_title, where=job_location)
        if results is None:
            await cl.Message(content="⚠️ Job search needs Adzuna API keys in `.env` — add `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` to enable this.").send()
            return
        if not results and job_location:
            # Retry without the location filter in case it was too narrow/misspelled
            results = search_jobs(job_title)
        if not results:
            await cl.Message(content=f"No live listings found for \"{job_title}\"{' in ' + job_location if job_location else ''}. Try different or broader keywords.").send()
            return
        reply = "**Live job listings:**\n\n"
        for job in results:
            reply += f"- **{job.get('title')}** at {job.get('company', {}).get('display_name', 'N/A')} — {job.get('location', {}).get('display_name', 'N/A')}\n"
        await cl.Message(content=reply).send()
        return

    system_content = BASE_PROMPT
    if intent in MODE_PROMPTS:
        system_content += "\n\n" + MODE_PROMPTS[intent]
    if resume_text:
        system_content += f"\n\nUser's resume/background content:\n{resume_text[:3000]}"

    conversation = [SystemMessage(content=system_content)] + history[1:] + [HumanMessage(content=user_text)]

    try:
        response = llm_with_tools.invoke(conversation)

        # Handle web_search tool calls (search grounding for time-sensitive facts)
        max_tool_rounds = 3
        rounds = 0
        while getattr(response, "tool_calls", None) and rounds < max_tool_rounds:
            conversation.append(response)
            for tool_call in response.tool_calls:
                if tool_call["name"] == "web_search":
                    result = web_search.invoke(tool_call["args"])
                else:
                    result = f"Unknown tool: {tool_call['name']}"
                conversation.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            response = llm_with_tools.invoke(conversation)
            rounds += 1
    except Exception:
        # Groq occasionally rejects a malformed function call (tool_use_failed) — fall back to a
        # plain, non-tool response rather than crashing the request.
        try:
            response = llm.invoke(conversation)
        except Exception:
            response = AIMessage(content="Sorry, I ran into an error generating a response. Please try again.")

    history.append(HumanMessage(content=user_text))
    history.append(AIMessage(content=response.content))
    cl.user_session.set("history", history)

    await cl.Message(content=response.content).send()