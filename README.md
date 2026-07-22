# 💼 CareerAgent

An AI-powered career assistant that reviews resumes, checks ATS keyword alignment, identifies skill gaps, runs mock interviews, and searches live job listings — all through a conversational chat interface.

**🔗 Live demo:** [careeragent-n127.onrender.com](https://careeragent-n127.onrender.com)

> Note: hosted on Render's free tier, so the app may take 30–50 seconds to wake up if it's been idle.

## Features

- **Resume review** — upload a PDF, DOCX, XLSX, or image of your resume and get specific, grounded feedback (not generic advice)
- **ATS keyword check** — estimated keyword-alignment score against a job description, with a clear disclaimer that it's an estimate, not an official ATS score
- **Skill gap analysis** — compares your actual background against a target role
- **Mock interviews** — one question at a time, with feedback after each answer
- **Learning roadmaps & course recommendations** — tailored to your current level if a resume is on file
- **Live job search** — real listings pulled from the Adzuna API
- **Google sign-in** with persistent chat history, so past conversations are saved across sessions
- **Multi-format file support** — PDF, DOCX, XLSX, plain text, and images (read via a vision model)
- **General-purpose chat** — not restricted to career topics; handles any question

## How it works

CareerAgent classifies each message's intent (keyword matching first, LLM classification as a fallback for ambiguous phrasing) and routes it to a specialized prompt for that task — resume analysis, ATS scoring, skill gaps, roadmaps, learning recommendations, or mock interviews. If you've uploaded a resume, its content is injected into context so responses stay grounded in your actual background rather than generic advice.

## Tech stack

| Layer | Tech |
|---|---|
| Chat UI | [Chainlit](https://chainlit.io) |
| LLM | Groq (`llama-3.3-70b-versatile`), via LangChain |
| Vision (resume images) | Groq (`llama-3.2-11b-vision-preview`) |
| Auth | Google OAuth |
| Persistence | PostgreSQL (via Chainlit's SQLAlchemy data layer) |
| Job search | Adzuna API |
| File parsing | pypdf, python-docx, openpyxl |
| Hosting | Render |

## Running locally

1. Clone the repo and install dependencies:
   ```bash
   git clone https://github.com/bhavyabansal941/CareerAgent.git
   cd CareerAgent
   pip install -r requirements.txt
   ```

2. Create a `.env` file with:
   ```
   GROQ_API_KEY=your_groq_key
   OAUTH_GOOGLE_CLIENT_ID=your_google_client_id
   OAUTH_GOOGLE_CLIENT_SECRET=your_google_client_secret
   CHAINLIT_AUTH_SECRET=your_chainlit_secret
   CHAINLIT_URL=http://localhost:8000
   DATABASE_URL=your_postgres_url
   ADZUNA_APP_ID=your_adzuna_app_id       # optional — enables live job search
   ADZUNA_APP_KEY=your_adzuna_app_key     # optional — enables live job search
   ```

3. Run the app:
   ```bash
   chainlit run app.py
   ```

## Screenshot
<img width="1901" height="931" alt="image" src="https://github.com/user-attachments/assets/a29693ef-de19-4fa4-ab02-8b228a33d704" />



## Author

**Bhavya Bansal**
[GitHub](https://github.com/bhavyabansal941) · [LinkedIn](https://linkedin.com/in/bhavya-bansal-aa70a3301)
