import io
import re
import time
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CareerGap AI", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# ================= CUSTOM THEME =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.cg-header { display:flex; align-items:center; gap:16px; padding:18px 22px; border-radius:16px;
    background: linear-gradient(135deg,#0f2027 0%,#203a43 55%,#2c5364 100%); margin-bottom:14px;
    border:1px solid #2c5364; }
.cg-logo { font-size:42px; }
.cg-header h1 { color:#fff; font-weight:800; font-size:30px; margin:0; letter-spacing:-0.5px; }
.cg-header h1 span { background: linear-gradient(90deg,#ffd54f,#ffb300); -webkit-background-clip:text;
    -webkit-text-fill-color:transparent; }
.cg-header p { color:#9fb8c4; margin:2px 0 0 0; font-size:13px; }
.cg-badge { display:inline-block; background:#ffd54f; color:#0f2027; font-size:11px; font-weight:700;
    padding:2px 10px; border-radius:20px; margin-left:8px; vertical-align:middle; }
.cg-trace { background:#16222e; border:1px solid #24394a; border-radius:12px; padding:10px 14px; margin:6px 0; }
.cg-trace b { color:#7ee2a8; font-size:13px; } .cg-trace small { color:#8aa5b5; }
div[data-testid="stMetric"] { background:#16222e; border:1px solid #24394a; border-radius:12px; padding:10px; }
div[data-testid="stMetric"] label { color:#9fb8c4 !important; }
.stButton button[kind="primary"] { background: linear-gradient(90deg,#ff5252,#ff8a65); border:none; font-weight:700; }
#MainMenu, footer { visibility:hidden; }
div[data-testid="stStatusWidget"] { background:#16222e; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cg-header"><div class="cg-logo">🚀</div><div>
<h1>CareerGap <span>AI</span><span class="cg-badge">TRACK C · C4</span></h1>
<p>Skill-Gap-to-Job Matching Agent — Profile → Match → Gap → Train → Report. Multi-agent pipeline with embedding retrieval.</p>
</div></div>
""", unsafe_allow_html=True)

# ============ FREE LLM LAYER (optional — app works fully without it) ============
def get_gemini_key():
    """Key priority: Streamlit secrets -> sidebar text input."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass
    return st.session_state.get("gemini_key_input", "").strip()


def get_llm_backend():
    """Return (backend_name, sdk_module) or (None, None)."""
    try:
        from google import genai
        return "google-genai", genai
    except ImportError:
        pass
    try:
        import google.generativeai as genai
        return "google-generativeai", genai
    except ImportError:
        return None, None


GEMINI_MODELS = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.0-flash"]

def _gemini_generate(sdk, backend, key, prompt):
    """Try Gemini models newest->oldest; return (text, model_used)."""
    last_err = None
    for model in GEMINI_MODELS:
        try:
            if backend == "google-genai":
                client = sdk.Client(api_key=key)
                resp = client.models.generate_content(model=model, contents=prompt)
                text = (resp.text or "").strip()
            else:
                sdk.configure(api_key=key)
                m = sdk.GenerativeModel(model)
                resp = m.generate_content(prompt)
                text = (resp.text or "").strip()
            if text:
                return text, model
        except Exception as e:
            last_err = e
    raise last_err if last_err else RuntimeError("No Gemini model responded.")


def llm_career_summary(skills, top, gap, roadmap_lines):
    """3-sentence AI career summary. Returns (text, used_llm, status). Falls back to template with visible reason."""
    fallback = (f"You are closest to job-readiness for {top['title']} at {gap['readiness']}% readiness. "
                f"Closing {len(gap['critical'])} critical gap(s) — {', '.join(gap['critical']) or 'none'} — is the "
                f"highest-leverage next step, and the roadmap below sequences exactly that.")
    key = get_gemini_key()
    if not key:
        return fallback, False, "No API key set — paste a free key in the sidebar."
    backend, sdk = get_llm_backend()
    if backend is None:
        return fallback, False, "Gemini SDK not installed — add `google-genai` to requirements.txt."
    prompt = (f"You are a career advisor. In 3 short, concrete, encouraging sentences (no fluff), advise a job seeker.\n"
              f"Their skills: {skills}.\nTop job match: {top['title']} at {top['company']} ({top['match_score']}% match, "
              f"{gap['readiness']}% readiness).\nCritical gaps: {gap['critical']}. Important gaps: {gap['important']}.\n"
              f"Recommended roadmap: {roadmap_lines}\nMention the single most valuable skill to learn first.")
    try:
        text, model_used = _gemini_generate(sdk, backend, key, prompt)
        return text, True, f"connected via {backend} ({model_used})"
    except Exception as e:
        return fallback, False, f"Gemini API call failed: {type(e).__name__}: {e}"


def llm_gap_explanation(skills, job, gap):
    """Grounded LLM explanation of WHY each critical gap matters. Returns (text, used_llm, status)."""
    fallback = (f"For **{job['title']}**, you already cover {', '.join(gap['matched']) or 'none of the required skills'}. "
                f"The critical missing skill(s) — {', '.join(gap['critical']) or 'none'} — are required in the job posting, "
                f"so each one directly blocks applications until closed. Start with the highest-demand gap in the roadmap below.")
    key = get_gemini_key()
    if not key:
        return fallback, False, "No API key set."
    backend, sdk = get_llm_backend()
    if backend is None:
        return fallback, False, "Gemini SDK not installed."
    prompt = (f"Explain in 2-3 short sentences, based ONLY on these facts, why the missing skills matter for this job and "
              f"which single skill to learn first:\nJob: {job['title']} at {job['company']}.\nRequired skills: {job['required_skills']}.\n"
              f"Candidate has: {gap['matched']}.\nCritical gaps: {gap['critical']}.\nImportant gaps: {gap['important']}.")
    try:
        text, model_used = _gemini_generate(sdk, backend, key, prompt)
        return text, True, f"Gemini ({model_used})"
    except Exception as e:
        return fallback, False, f"LLM fallback used: {type(e).__name__}: {e}"


def test_llm_connection():
    """Live ping to Gemini. Returns (ok, message)."""
    key = get_gemini_key()
    if not key:
        return False, "No API key set."
    backend, sdk = get_llm_backend()
    if backend is None:
        return False, "Gemini SDK not installed."
    try:
        text, model_used = _gemini_generate(sdk, backend, key, 'Reply with exactly: OK')
        return True, f"✓ Connected ({backend}, {model_used}) — Gemini replied: {text[:50]}"
    except Exception as e:
        return False, f"✗ {type(e).__name__}: {e}"

# ================= EMBEDDING LAYER (sentence-transformers, cached, graceful fallback) =================
@st.cache_resource(show_spinner=False)
def get_embedder():
    """Load all-MiniLM-L6-v2 once. Returns None if unavailable -> keyword fallback everywhere."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None

# ================= DATA (self-contained) =================
SKILL_ONTOLOGY = {
    "VLSI": ["Digital Electronics","Verilog","SystemVerilog","UVM","SVA","AMBA","RTL Design","ASIC","FPGA","VHDL"],
    "Embedded": ["C","Embedded C","Arduino","STM32","UART","SPI","I2C","RTOS","Microcontrollers","ARM"],
    "Software": ["Python","Java","JavaScript","SQL","Git","REST API","Django","React","Node.js","Docker"],
    "Data/AI": ["Python","Pandas","NumPy","Machine Learning","Deep Learning","SQL","Statistics","TensorFlow","PyTorch","NLP"],
    "Networking": ["TCP/IP","Networking Basics","CCNA","Linux","Routing","Switching","Network Security","Wireshark"],
    "IoT": ["Arduino","Raspberry Pi","MQTT","Embedded C","Sensors","IoT Protocols","Cloud IoT","Python"],
}
ALL_SKILLS = sorted({s for v in SKILL_ONTOLOGY.values() for s in v})

SKILL_SYNONYMS = {
    "system verilog":"SystemVerilog","sv":"SystemVerilog","systemverilog":"SystemVerilog",
    "verilog hdl":"Verilog","verilog":"Verilog","vhdl":"VHDL",
    "uvm":"UVM","universal verification methodology":"UVM","sva":"SVA","assertions":"SVA",
    "amba":"AMBA","axi":"AMBA","digital electronics":"Digital Electronics","digital logic":"Digital Electronics",
    "rtl":"RTL Design","rtl design":"RTL Design","asic":"ASIC","fpga":"FPGA",
    "embedded c":"Embedded C","c programming":"C","c language":"C","c":"C",
    "arduino":"Arduino","stm32":"STM32","arm cortex":"ARM","arm":"ARM",
    "uart":"UART","spi":"SPI","i2c":"I2C","rtos":"RTOS",
    "microcontroller":"Microcontrollers","microcontrollers":"Microcontrollers",
    "python":"Python","java":"Java","javascript":"JavaScript","js":"JavaScript",
    "sql":"SQL","mysql":"SQL","postgresql":"SQL","git":"Git","github":"Git",
    "rest api":"REST API","rest":"REST API","api development":"REST API",
    "django":"Django","react":"React","reactjs":"React","react.js":"React",
    "node":"Node.js","nodejs":"Node.js","node.js":"Node.js","docker":"Docker","containers":"Docker",
    "pandas":"Pandas","numpy":"NumPy","machine learning":"Machine Learning","ml":"Machine Learning",
    "deep learning":"Deep Learning","dl":"Deep Learning","statistics":"Statistics","stats":"Statistics",
    "tensorflow":"TensorFlow","pytorch":"PyTorch","nlp":"NLP","natural language processing":"NLP",
    "tcp/ip":"TCP/IP","tcp ip":"TCP/IP","networking":"Networking Basics","networking basics":"Networking Basics",
    "ccna":"CCNA","linux":"Linux","routing":"Routing","switching":"Switching",
    "network security":"Network Security","wireshark":"Wireshark","raspberry pi":"Raspberry Pi","mqtt":"MQTT",
    "sensors":"Sensors","iot protocols":"IoT Protocols","iot":"IoT Protocols",
    "cloud iot":"Cloud IoT","aws iot":"Cloud IoT","matlab":"MATLAB",
}

JOBS = [
    dict(job_id="J01",company="TechChip",title="RTL Design Intern",location="Bangalore",domain="VLSI",education="B.Tech ECE",required_skills=["Verilog","Digital Electronics","SystemVerilog"],preferred_skills=["Python","Linux"],experience="0",salary=25000),
    dict(job_id="J02",company="VerifyTech",title="Verification Engineer",location="Hyderabad",domain="VLSI",education="B.Tech ECE",required_skills=["Verilog","SystemVerilog","UVM","SVA"],preferred_skills=["AMBA","Python"],experience="0-1",salary=45000),
    dict(job_id="J03",company="SiliconWorks",title="ASIC Design Engineer",location="Bangalore",domain="VLSI",education="B.Tech ECE",required_skills=["Verilog","RTL Design","Digital Electronics","VHDL"],preferred_skills=["FPGA","ASIC"],experience="1-2",salary=60000),
    dict(job_id="J04",company="ChipCraft",title="FPGA Developer",location="Pune",domain="VLSI",education="B.Tech ECE/EEE",required_skills=["Verilog","FPGA","VHDL"],preferred_skills=["Embedded C","C"],experience="0-1",salary=40000),
    dict(job_id="J05",company="NanoFab",title="Physical Design Engineer",location="Bangalore",domain="VLSI",education="B.Tech ECE",required_skills=["Digital Electronics","ASIC","RTL Design"],preferred_skills=["Verilog","Linux"],experience="1-3",salary=70000),
    dict(job_id="J06",company="TestSilicon",title="DFT Engineer",location="Hyderabad",domain="VLSI",education="B.Tech ECE",required_skills=["Digital Electronics","Verilog","ASIC"],preferred_skills=["SystemVerilog"],experience="0-2",salary=55000),
    dict(job_id="J07",company="AnalogWorks",title="Analog Layout Engineer",location="Bangalore",domain="VLSI",education="B.Tech ECE",required_skills=["Digital Electronics","VHDL"],preferred_skills=["ASIC","RTL Design"],experience="1",salary=50000),
    dict(job_id="J08",company="VerifyTech",title="Verification Intern",location="Hyderabad",domain="VLSI",education="B.Tech ECE",required_skills=["Verilog","SystemVerilog"],preferred_skills=["UVM","Python"],experience="0",salary=22000),
    dict(job_id="J09",company="SiliconEdge",title="RTL Design Engineer",location="Chennai",domain="VLSI",education="B.Tech ECE",required_skills=["Verilog","Digital Electronics","RTL Design","SystemVerilog"],preferred_skills=["UVM"],experience="1-2",salary=65000),
    dict(job_id="J10",company="VerifyTech",title="AMBA Protocol Verification Engineer",location="Hyderabad",domain="VLSI",education="B.Tech ECE",required_skills=["SystemVerilog","UVM","AMBA","SVA"],preferred_skills=["Verilog"],experience="1-3",salary=72000),
    dict(job_id="J11",company="EmbeddedWorks",title="Embedded Engineer",location="Chennai",domain="Embedded",education="B.Tech ECE/EEE",required_skills=["C","Embedded C","Microcontrollers","UART","SPI"],preferred_skills=["RTOS","Python"],experience="0",salary=28000),
    dict(job_id="J12",company="MicroSys",title="Firmware Developer",location="Pune",domain="Embedded",education="B.Tech",required_skills=["C","Embedded C","STM32","ARM"],preferred_skills=["RTOS"],experience="0-1",salary=35000),
    dict(job_id="J13",company="IoTCraft",title="IoT Firmware Engineer",location="Bangalore",domain="Embedded",education="B.Tech",required_skills=["Embedded C","Arduino","I2C","SPI"],preferred_skills=["MQTT","Python"],experience="0",salary=30000),
    dict(job_id="J14",company="RealTimeSys",title="RTOS Developer",location="Hyderabad",domain="Embedded",education="B.Tech",required_skills=["C","RTOS","ARM","Microcontrollers"],preferred_skills=["STM32"],experience="1-2",salary=50000),
    dict(job_id="J15",company="MicroSys",title="Embedded Systems Intern",location="Pune",domain="Embedded",education="B.Tech",required_skills=["C","Embedded C"],preferred_skills=["Arduino","UART"],experience="0",salary=15000),
    dict(job_id="J16",company="AutoElectron",title="Automotive Embedded Engineer",location="Chennai",domain="Embedded",education="B.Tech",required_skills=["C","Embedded C","Microcontrollers","ARM"],preferred_skills=["UART","SPI","RTOS"],experience="1-3",salary=55000),
    dict(job_id="J17",company="ChipCraft",title="Hardware-Firmware Engineer",location="Pune",domain="Embedded",education="B.Tech",required_skills=["Embedded C","STM32","SPI","I2C"],preferred_skills=["C"],experience="0-1",salary=38000),
    dict(job_id="J18",company="TestSilicon",title="Embedded Test Engineer",location="Hyderabad",domain="Embedded",education="B.Tech",required_skills=["C","Embedded C","UART"],preferred_skills=["Python","Linux"],experience="0-2",salary=32000),
    dict(job_id="J19",company="IoTCraft",title="IoT Developer",location="Bangalore",domain="IoT",education="B.Tech Any",required_skills=["Arduino","Raspberry Pi","MQTT","Sensors"],preferred_skills=["Python","Cloud IoT"],experience="0-1",salary=32000),
    dict(job_id="J20",company="SmartGrid",title="IoT Solutions Engineer",location="Delhi NCR",domain="IoT",education="B.Tech",required_skills=["IoT Protocols","MQTT","Cloud IoT","Python"],preferred_skills=["Sensors"],experience="1-2",salary=48000),
    dict(job_id="J21",company="HomeSense",title="Smart Home Systems Engineer",location="Bangalore",domain="IoT",education="B.Tech",required_skills=["Arduino","Sensors","IoT Protocols"],preferred_skills=["Raspberry Pi","MQTT"],experience="0",salary=28000),
    dict(job_id="J22",company="IoTCraft",title="IoT Intern",location="Bangalore",domain="IoT",education="B.Tech",required_skills=["Arduino","Raspberry Pi"],preferred_skills=["Python","Sensors"],experience="0",salary=15000),
    dict(job_id="J23",company="FactoryNet",title="Industrial IoT Engineer",location="Pune",domain="IoT",education="B.Tech",required_skills=["IoT Protocols","MQTT","Sensors","Cloud IoT"],preferred_skills=["Python","Networking Basics"],experience="1-3",salary=55000),
    dict(job_id="J24",company="SmartGrid",title="Connected Devices Engineer",location="Delhi NCR",domain="IoT",education="B.Tech",required_skills=["Raspberry Pi","Python","MQTT"],preferred_skills=["Cloud IoT","Arduino"],experience="0-1",salary=34000),
    dict(job_id="J25",company="CodeNest",title="Python Developer",location="Remote",domain="Software",education="B.Tech Any / BCA",required_skills=["Python","SQL","Git","REST API"],preferred_skills=["Django","Docker"],experience="0-1",salary=40000),
    dict(job_id="J26",company="WebWorks",title="Full Stack Developer",location="Bangalore",domain="Software",education="B.Tech / BCA",required_skills=["JavaScript","React","Node.js","SQL"],preferred_skills=["Git","Docker"],experience="0-2",salary=55000),
    dict(job_id="J27",company="ServerSide",title="Backend Developer",location="Hyderabad",domain="Software",education="B.Tech / BCA",required_skills=["Python","Django","SQL","REST API"],preferred_skills=["Docker","Git"],experience="1-2",salary=60000),
    dict(job_id="J28",company="CodeNest",title="Software Engineer Intern",location="Remote",domain="Software",education="B.Tech / BCA",required_skills=["Python","Git"],preferred_skills=["SQL","REST API"],experience="0",salary=18000),
    dict(job_id="J29",company="WebWorks",title="Frontend Developer",location="Pune",domain="Software",education="B.Tech / BCA",required_skills=["JavaScript","React","Git"],preferred_skills=["Node.js"],experience="0-1",salary=42000),
    dict(job_id="J30",company="ServerSide",title="DevOps-leaning Software Engineer",location="Hyderabad",domain="Software",education="B.Tech",required_skills=["Python","Docker","Git","SQL"],preferred_skills=["REST API"],experience="1-3",salary=65000),
    dict(job_id="J31",company="DataSense",title="Data Analyst",location="Bangalore",domain="Data/AI",education="B.Tech / BSc Any",required_skills=["Python","Pandas","SQL","Statistics"],preferred_skills=["NumPy"],experience="0-1",salary=38000),
    dict(job_id="J32",company="AIWorks",title="Machine Learning Engineer Intern",location="Hyderabad",domain="Data/AI",education="B.Tech",required_skills=["Python","Machine Learning","NumPy","Pandas"],preferred_skills=["TensorFlow"],experience="0",salary=25000),
    dict(job_id="J33",company="DataSense",title="Data Scientist",location="Bangalore",domain="Data/AI",education="B.Tech / MSc",required_skills=["Python","Machine Learning","Statistics","Pandas"],preferred_skills=["Deep Learning","SQL"],experience="1-3",salary=75000),
    dict(job_id="J34",company="AIWorks",title="NLP Engineer",location="Hyderabad",domain="Data/AI",education="B.Tech",required_skills=["Python","NLP","Machine Learning","Deep Learning"],preferred_skills=["PyTorch"],experience="1-2",salary=80000),
    dict(job_id="J35",company="AIWorks",title="Deep Learning Researcher",location="Remote",domain="Data/AI",education="M.Tech / MSc",required_skills=["Python","Deep Learning","PyTorch","TensorFlow"],preferred_skills=["NLP"],experience="2-4",salary=95000),
    dict(job_id="J36",company="NetCore",title="Network Engineer",location="Chennai",domain="Networking",education="B.Tech / Diploma",required_skills=["TCP/IP","Networking Basics","Routing","Switching"],preferred_skills=["CCNA"],experience="0-1",salary=30000),
    dict(job_id="J37",company="SecureNet",title="Network Security Analyst",location="Delhi NCR",domain="Networking",education="B.Tech",required_skills=["Network Security","TCP/IP","Linux"],preferred_skills=["Wireshark"],experience="1-2",salary=50000),
    dict(job_id="J38",company="CorpIT",title="Systems & Network Administrator",location="Mumbai",domain="Networking",education="B.Tech / Diploma",required_skills=["Linux","Networking Basics","TCP/IP"],preferred_skills=["Routing","Switching"],experience="0-2",salary=35000),
    dict(job_id="J39",company="NetCore",title="CCNA Network Intern",location="Chennai",domain="Networking",education="B.Tech / Diploma",required_skills=["Networking Basics","TCP/IP"],preferred_skills=["CCNA","Routing"],experience="0",salary=16000),
    dict(job_id="J40",company="SecureNet",title="Network Operations Engineer",location="Delhi NCR",domain="Networking",education="B.Tech",required_skills=["TCP/IP","Routing","Switching","Linux"],preferred_skills=["Network Security","Wireshark"],experience="1-3",salary=52000),
]

COURSES = [
    dict(course_id="C01",course_name="SystemVerilog Fundamentals",provider="NPTEL",skills=["SystemVerilog"],duration_weeks=6,cost=0,level="Beginner"),
    dict(course_id="C02",course_name="UVM Verification Mastery",provider="Udemy",skills=["UVM","SystemVerilog"],duration_weeks=8,cost=799,level="Intermediate"),
    dict(course_id="C03",course_name="Digital Electronics Foundations",provider="NPTEL",skills=["Digital Electronics"],duration_weeks=8,cost=0,level="Beginner"),
    dict(course_id="C04",course_name="Verilog HDL for Beginners",provider="Coursera",skills=["Verilog"],duration_weeks=5,cost=0,level="Beginner"),
    dict(course_id="C05",course_name="Advanced RTL Design",provider="NPTEL",skills=["RTL Design","Verilog"],duration_weeks=8,cost=0,level="Intermediate"),
    dict(course_id="C06",course_name="AMBA Bus Protocols",provider="Udemy",skills=["AMBA"],duration_weeks=3,cost=999,level="Intermediate"),
    dict(course_id="C07",course_name="SystemVerilog Assertions (SVA)",provider="Udemy",skills=["SVA","SystemVerilog"],duration_weeks=4,cost=599,level="Intermediate"),
    dict(course_id="C08",course_name="ASIC Design Flow",provider="NPTEL",skills=["ASIC","RTL Design"],duration_weeks=10,cost=0,level="Advanced"),
    dict(course_id="C09",course_name="FPGA Design with Verilog",provider="Coursera",skills=["FPGA","Verilog"],duration_weeks=6,cost=1499,level="Intermediate"),
    dict(course_id="C10",course_name="VHDL Programming",provider="NPTEL",skills=["VHDL"],duration_weeks=6,cost=0,level="Beginner"),
    dict(course_id="C11",course_name="Embedded C Programming",provider="Skill India",skills=["Embedded C","C"],duration_weeks=6,cost=0,level="Beginner"),
    dict(course_id="C12",course_name="C Programming Essentials",provider="NPTEL",skills=["C"],duration_weeks=4,cost=0,level="Beginner"),
    dict(course_id="C13",course_name="Arduino for Beginners",provider="Udemy",skills=["Arduino"],duration_weeks=3,cost=499,level="Beginner"),
    dict(course_id="C14",course_name="STM32 Microcontroller Programming",provider="Udemy",skills=["STM32","ARM"],duration_weeks=6,cost=899,level="Intermediate"),
    dict(course_id="C15",course_name="Real-Time Operating Systems (RTOS)",provider="Coursera",skills=["RTOS"],duration_weeks=6,cost=1299,level="Intermediate"),
    dict(course_id="C16",course_name="Microcontrollers & Embedded Systems",provider="NPTEL",skills=["Microcontrollers","C"],duration_weeks=8,cost=0,level="Beginner"),
    dict(course_id="C17",course_name="UART/SPI/I2C Protocols Deep Dive",provider="Udemy",skills=["UART","SPI","I2C"],duration_weeks=4,cost=699,level="Intermediate"),
    dict(course_id="C18",course_name="ARM Cortex-M Programming",provider="Coursera",skills=["ARM","Embedded C"],duration_weeks=6,cost=1199,level="Intermediate"),
    dict(course_id="C19",course_name="Raspberry Pi Projects",provider="Udemy",skills=["Raspberry Pi"],duration_weeks=3,cost=399,level="Beginner"),
    dict(course_id="C20",course_name="MQTT & IoT Messaging Protocols",provider="Coursera",skills=["MQTT","IoT Protocols"],duration_weeks=3,cost=599,level="Beginner"),
    dict(course_id="C21",course_name="IoT Sensors & Actuators",provider="Skill India",skills=["Sensors"],duration_weeks=4,cost=0,level="Beginner"),
    dict(course_id="C22",course_name="Cloud for IoT (AWS IoT)",provider="Coursera",skills=["Cloud IoT"],duration_weeks=5,cost=1499,level="Intermediate"),
    dict(course_id="C23",course_name="IoT Protocols & Architecture",provider="NPTEL",skills=["IoT Protocols"],duration_weeks=6,cost=0,level="Beginner"),
    dict(course_id="C24",course_name="Python for Everybody",provider="Coursera",skills=["Python"],duration_weeks=6,cost=0,level="Beginner"),
    dict(course_id="C25",course_name="Python for Data Science",provider="edX",skills=["Python","Pandas","NumPy"],duration_weeks=6,cost=0,level="Beginner"),
    dict(course_id="C26",course_name="Java Programming Fundamentals",provider="NPTEL",skills=["Java"],duration_weeks=8,cost=0,level="Beginner"),
    dict(course_id="C27",course_name="Modern JavaScript",provider="Udemy",skills=["JavaScript"],duration_weeks=5,cost=599,level="Beginner"),
    dict(course_id="C28",course_name="SQL for Data Analysis",provider="Coursera",skills=["SQL"],duration_weeks=4,cost=0,level="Beginner"),
    dict(course_id="C29",course_name="Git & Version Control",provider="Udemy",skills=["Git"],duration_weeks=2,cost=0,level="Beginner"),
    dict(course_id="C30",course_name="Building REST APIs",provider="Udemy",skills=["REST API"],duration_weeks=4,cost=699,level="Intermediate"),
    dict(course_id="C31",course_name="Django Web Framework",provider="Udemy",skills=["Django","Python","REST API"],duration_weeks=6,cost=999,level="Intermediate"),
    dict(course_id="C32",course_name="React.js Complete Guide",provider="Udemy",skills=["React","JavaScript"],duration_weeks=7,cost=1299,level="Intermediate"),
    dict(course_id="C33",course_name="Node.js Backend Development",provider="Coursera",skills=["Node.js","JavaScript"],duration_weeks=6,cost=999,level="Intermediate"),
    dict(course_id="C34",course_name="Docker & Containers",provider="Udemy",skills=["Docker"],duration_weeks=3,cost=599,level="Intermediate"),
    dict(course_id="C35",course_name="Machine Learning Specialization",provider="Coursera",skills=["Machine Learning","Python","Statistics"],duration_weeks=10,cost=0,level="Intermediate"),
    dict(course_id="C36",course_name="Deep Learning Specialization",provider="Coursera",skills=["Deep Learning","TensorFlow"],duration_weeks=12,cost=1999,level="Advanced"),
    dict(course_id="C37",course_name="Statistics for Data Science",provider="NPTEL",skills=["Statistics"],duration_weeks=6,cost=0,level="Beginner"),
    dict(course_id="C38",course_name="TensorFlow for AI",provider="Coursera",skills=["TensorFlow","Deep Learning"],duration_weeks=6,cost=999,level="Intermediate"),
    dict(course_id="C39",course_name="PyTorch Fundamentals",provider="Udemy",skills=["PyTorch","Deep Learning"],duration_weeks=6,cost=899,level="Intermediate"),
    dict(course_id="C40",course_name="Natural Language Processing",provider="Coursera",skills=["NLP","Deep Learning"],duration_weeks=8,cost=1499,level="Advanced"),
    dict(course_id="C41",course_name="Networking Basics (CCNA prep)",provider="NPTEL",skills=["Networking Basics","TCP/IP"],duration_weeks=8,cost=0,level="Beginner"),
    dict(course_id="C42",course_name="CCNA Certification Prep",provider="Skill India",skills=["CCNA","Routing","Switching"],duration_weeks=10,cost=0,level="Intermediate"),
    dict(course_id="C43",course_name="Linux System Administration",provider="Coursera",skills=["Linux"],duration_weeks=6,cost=0,level="Beginner"),
    dict(course_id="C44",course_name="Network Security Fundamentals",provider="Udemy",skills=["Network Security"],duration_weeks=6,cost=999,level="Intermediate"),
    dict(course_id="C45",course_name="Wireshark Packet Analysis",provider="Udemy",skills=["Wireshark"],duration_weeks=3,cost=499,level="Beginner"),
]

# ================= EMBEDDING RETRIEVAL (vector layer with keyword fallback) =================
def _job_texts():
    return [f"{j['title']} {j['domain']} {' '.join(j['required_skills'])} {' '.join(j['preferred_skills'])}" for j in JOBS]

def _course_texts():
    return [f"{c['course_name']} by {c['provider']} teaches {', '.join(c['skills'])} ({c['level']})" for c in COURSES]

@st.cache_data(show_spinner=False)
def job_embeddings():
    """Embeddings for all job postings. None if the embedder is unavailable."""
    emb = get_embedder()
    if emb is None:
        return None
    try:
        return emb.encode(_job_texts(), normalize_embeddings=True)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def course_embeddings():
    """Embeddings for the course catalog. None if the embedder is unavailable."""
    emb = get_embedder()
    if emb is None:
        return None
    try:
        return emb.encode(_course_texts(), normalize_embeddings=True)
    except Exception:
        return None

# ================= TOOL REGISTRY (decorated, documented, trace-logged) =================
TOOL_REGISTRY = {}
TOOL_TRACE_KEY = "tool_trace"

def tool(fn):
    """Decorator: register fn as an agent tool (name + one-line doc) and log every call with runtime to the tool trace."""
    TOOL_REGISTRY[fn.__name__] = (fn.__doc__ or "").strip().split("\n")[0]
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        try:
            summary = f"({', '.join(str(a)[:40] for a in args[:2])})" if args else "()"
            st.session_state.setdefault(TOOL_TRACE_KEY, []).append(
                {"tool": fn.__name__, "desc": TOOL_REGISTRY[fn.__name__], "args": summary, "seconds": round(elapsed, 3)})
        except Exception:
            pass
        return result
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper

# ================= AGENT 0 — PDF DOCUMENT INTAKE =================
@st.cache_data(show_spinner=False)
def extract_pdf_text(file_bytes: bytes):
    """Multi-backend PDF text extraction. Returns (text, backend) or (None, error_details)."""
    attempts = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        if text.strip():
            return text.strip(), "pypdf"
        attempts.append("pypdf: 0 words (scanned/image PDF?)")
    except ImportError:
        attempts.append("pypdf not installed")
    except Exception as e:
        attempts.append(f"pypdf: {type(e).__name__}: {e}")
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        if text.strip():
            return text.strip(), "PyPDF2"
        attempts.append("PyPDF2: 0 words (scanned/image PDF?)")
    except ImportError:
        attempts.append("PyPDF2 not installed")
    except Exception as e:
        attempts.append(f"PyPDF2: {type(e).__name__}: {e}")
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        if text.strip():
            return text.strip(), "pdfplumber"
        attempts.append("pdfplumber: 0 words (scanned/image PDF?)")
    except ImportError:
        attempts.append("pdfplumber not installed")
    except Exception as e:
        attempts.append(f"pdfplumber: {type(e).__name__}: {e}")
    return None, " | ".join(attempts)

# ================= PROFILE EXTRACTION =================
def extract_skills(text):
    lower = text.lower()
    found = set()
    cands = list(SKILL_SYNONYMS.keys()) + [s.lower() for s in ALL_SKILLS]
    for cand in sorted(set(cands), key=len, reverse=True):
        if re.search(r"(?<![a-z0-9])" + re.escape(cand) + r"(?![a-z0-9])", lower):
            canon = SKILL_SYNONYMS.get(cand)
            if canon:
                found.add(canon)
    return sorted(found)

def extract_education(text):
    lower = text.lower()
    if re.search(r"b\.?tech.*ece|electronics.*communication", lower): return "B.Tech ECE"
    if re.search(r"b\.?tech.*eee|electrical.*electronics", lower): return "B.Tech EEE"
    if re.search(r"b\.?tech.*cse|computer science", lower): return "B.Tech CSE"
    if re.search(r"\bbca\b", lower): return "BCA"
    if re.search(r"m\.?tech", lower): return "M.Tech"
    if re.search(r"\bmsc\b", lower): return "MSc"
    if "diploma" in lower: return "Diploma"
    return "Not specified"

INTEREST_KEYWORDS = {"VLSI":["vlsi","chip design","rtl","asic","fpga","semiconductor"],
    "Embedded":["embedded","firmware","microcontroller"],"IoT":["iot","internet of things","smart devices"],
    "Software":["web development","full stack","software engineering","app development"],
    "Data/AI":["data science","machine learning","artificial intelligence","ai","ml"],
    "Networking":["networking","network security","ccna"]}

def extract_interests(text):
    lower = text.lower()
    hits = sorted({d for d,kws in INTEREST_KEYWORDS.items() if any(k in lower for k in kws)})
    return hits or ["Any"]

# ================= AGENT TOOLS =================
@tool
def t_extract_profile(resume_text: str, extra_skills: list):
    """Extract structured profile (skills, education, interests) from free-text resume."""
    skills = sorted(set(extract_skills(resume_text)) | set(extra_skills))
    return {"raw_text": resume_text, "skills": skills, "education": extract_education(resume_text),
            "interests": extract_interests(resume_text), "simulated": list(extra_skills)}

@tool
def t_match_jobs(profile: dict, location: str):
    """Score all job postings for the profile using keyword overlap + semantic embedding similarity."""
    skills, education, interests = profile["skills"], profile["education"], profile["interests"]
    emb = job_embeddings()
    prof_emb = None
    if emb is not None:
        try:
            prof_emb = get_embedder().encode([profile["raw_text"]], normalize_embeddings=True)[0]
        except Exception:
            prof_emb = None
    results = []
    for i, job in enumerate(JOBS):
        req, pref = job["required_skills"], job["preferred_skills"]
        m_req = set(skills) & set(req)
        m_pref = set(skills) & set(pref)
        kw_score = min(100.0, len(m_req)/len(req)*100 + (len(m_pref)/len(pref)*15 if pref else 0))
        sem = float(emb[i] @ prof_emb) if (emb is not None and prof_emb is not None) else None
        if sem is not None:
            skill_score = 0.6*kw_score + 0.4*(100.0*sem)
        else:
            skill_score = kw_score
        edu_score = 100.0 if (education != "Not specified" and education.split()[-1].upper() in job["education"].upper()) else \
                    (60.0 if education.split(".")[0].upper() in job["education"].upper() else (50.0 if education == "Not specified" else 20.0))
        interest_score = 60.0 if "Any" in interests else (100.0 if job["domain"] in interests else 20.0)
        loc_score = 80.0 if job["location"] == "Remote" or location == "Not specified" else (100.0 if location == job["location"] else 40.0)
        final = skill_score*0.60 + edu_score*0.15 + interest_score*0.15 + loc_score*0.10
        results.append({**job, "match_score": round(final,1), "keyword_score": round(kw_score,1),
                        "semantic_score": round(sem,3) if sem is not None else None,
                        "matched_skills": sorted(m_req), "missing_skills": sorted(set(req)-set(skills))})
    results = sorted(results, key=lambda r: r["match_score"], reverse=True)
    for r in results[:10]:
        ttr = time_to_ready(skills, r, st.session_state.get("free_only_flag", False))
        r["weeks_to_ready"] = ttr["weeks"]
        r["cost_to_ready"] = ttr["cost"]
        r["n_courses"] = len(ttr["courses"])
    return results

@tool
def t_analyze_gap(skills: list, job: dict):
    """Identify matched/critical/important gaps and readiness for one job, with skill demand across the job market."""
    matched = sorted(set(skills) & set(job["required_skills"]))
    critical = sorted(set(job["required_skills"]) - set(skills))
    important = sorted(set(job["preferred_skills"]) - set(skills))
    demand = {s: sum(1 for j in JOBS if s in j["required_skills"] or s in j["preferred_skills"]) for s in critical+important}
    readiness = round(len(matched)/len(job["required_skills"])*100,1) if job["required_skills"] else 100.0
    return {"matched": matched, "critical": critical, "important": important, "demand": demand, "readiness": readiness}

@tool
def t_retrieve_courses(skill: str, free_only: bool = False):
    """Retrieve the best course for a skill via embedding similarity over the catalog, with keyword fallback. Returns (course, match_score)."""
    embs = course_embeddings()
    if embs is not None:
        try:
            s_emb = get_embedder().encode([skill], normalize_embeddings=True)[0]
            sims = embs @ s_emb
            order = np.argsort(-sims)
            for idx in order:
                c = COURSES[idx]
                if free_only and c["cost"] > 0:
                    continue
                if sims[idx] >= 0.20:
                    return c, float(sims[idx])
        except Exception:
            pass
    cands = [c for c in COURSES if skill in c["skills"] and (not free_only or c["cost"] == 0)]
    if cands:
        return sorted(cands, key=lambda c: (c["cost"] > 0, c["duration_weeks"]))[0], None
    return None, None

@tool
def t_build_roadmap(gap: dict, free_only: bool = False):
    """Build a week-by-week learning roadmap: gaps ranked by market demand, mapped to retrieved courses."""
    critical, important, demand = gap["critical"], gap["important"], gap["demand"]
    ordered = sorted(critical, key=lambda s: -demand.get(s,0)) + sorted(important, key=lambda s: -demand.get(s,0))
    roadmap, seen, week = [], set(), 0
    for skill in ordered:
        c, sim = t_retrieve_courses(skill, free_only)
        if not c or c["course_id"] in seen:
            continue
        seen.add(c["course_id"])
        start = week + 1
        week += c["duration_weeks"]
        roadmap.append({"Skill": skill, "Priority": "🔴 Critical" if skill in critical else "🟡 Important",
                        "Course": c["course_name"], "Provider": c["provider"],
                        "Schedule": f"Week {start}-{week}", "Duration (wks)": c["duration_weeks"],
                        "Cost": "Free" if c["cost"]==0 else f"₹{c['cost']}",
                        "Retrieval": f"{round(sim*100)}% semantic match" if sim is not None else "keyword match",
                        "Unlocks (jobs)": demand.get(skill,0)})
    return roadmap, week

@tool
def t_time_to_ready(skills: list, job: dict, free_only: bool = False):
    """Compulsory add-on: total weeks and cost to close all gaps for ANY target job, honoring the free-only filter."""
    gap = t_analyze_gap(skills, job)
    ordered = sorted(gap["critical"], key=lambda s: -gap["demand"].get(s,0)) + sorted(gap["important"], key=lambda s: -gap["demand"].get(s,0))
    seen, weeks, cost, names = set(), 0, 0, []
    for skill in ordered:
        c, _ = t_retrieve_courses(skill, free_only)
        if not c or c["course_id"] in seen:
            continue
        seen.add(c["course_id"])
        weeks += c["duration_weeks"]
        cost += c["cost"]
        names.append(c["course_name"])
    return {"weeks": weeks, "cost": cost, "courses": names, "gap": gap}

def time_to_ready(skills, job, free_only=False):
    """Internal (unlogged) variant used inside other tools to avoid double trace entries."""
    gap = {"matched": sorted(set(skills) & set(job["required_skills"])),
           "critical": sorted(set(job["required_skills"]) - set(skills)),
           "important": sorted(set(job["preferred_skills"]) - set(skills))}
    demand = {s: sum(1 for j in JOBS if s in j["required_skills"] or s in j["preferred_skills"]) for s in gap["critical"]+gap["important"]}
    ordered = sorted(gap["critical"], key=lambda s: -demand.get(s,0)) + sorted(gap["important"], key=lambda s: -demand.get(s,0))
    seen, weeks, cost, names = set(), 0, 0, []
    for skill in ordered:
        c, _ = t_retrieve_courses(skill, free_only)
        if not c or c["course_id"] in seen:
            continue
        seen.add(c["course_id"])
        weeks += c["duration_weeks"]
        cost += c["cost"]
        names.append(c["course_name"])
    return {"weeks": weeks, "cost": cost, "courses": names}

# ================= SIDEBAR =================
with st.sidebar:
    st.header("Your Profile")
    pdf_file = st.file_uploader("📄 Upload resume PDF (optional)", type=["pdf"])
    resume_text = st.text_area("...or paste your resume / profile text", height=150,
        value="B.Tech ECE, final year student, fresher, based in Bangalore. Skills: Python, C, Verilog, Digital Electronics, basic MATLAB. Interested in VLSI and chip design roles.")
    location = st.selectbox("Preferred location", ["Not specified","Bangalore","Hyderabad","Chennai","Pune","Delhi NCR","Mumbai","Remote"])
    extra_skills = st.multiselect("💡 \"What if I learn this?\" — simulate adding skills", ALL_SKILLS)
    free_only = st.checkbox("Show FREE courses only")
    st.session_state["free_only_flag"] = free_only
    with st.expander("🔑 Free AI narration (optional)"):
        st.caption("Free Gemini key — no credit card: aistudio.google.com/apikey")
        st.text_input("Gemini API key", type="password", key="gemini_key_input")
        if st.button("🔌 Test API connection", use_container_width=True):
            ok, msg = test_llm_connection()
            (st.success if ok else st.error)(msg)
    go = st.button("🚀 Analyze", type="primary", use_container_width=True)

# --- PDF intake with loading state ---
if pdf_file is not None:
    with st.spinner("📄 Parsing PDF…"):
        pdf_text, pdf_info = extract_pdf_text(pdf_file.read())
    if pdf_text:
        if pdf_text not in resume_text:
            resume_text = (resume_text + "\n" + pdf_text).strip()
        st.sidebar.success(f"✓ PDF parsed with {pdf_info} — {len(pdf_text.split())} words added")
    else:
        st.sidebar.error(f"PDF parse failed: {pdf_info} → add `pypdf` to requirements.txt")

# ================= MAIN PIPELINE =================
if go:
    if not resume_text.strip():
        st.warning("Please paste your resume text or upload a PDF first.")
        st.stop()

    st.session_state[TOOL_TRACE_KEY] = []
    backend_note = "embeddings ON (all-MiniLM-L6-v2)" if get_embedder() is not None else "embeddings unavailable — keyword fallback"

    with st.status("🤖 Agent pipeline running…", expanded=True) as status:
        st.write("👤 **Profile Agent** — extracting skills, education, interests…")
        profile = t_extract_profile(resume_text, extra_skills)
        st.write(f"🎯 **Job Match Agent** — scoring {len(JOBS)} postings ({backend_note})…")
        matches = t_match_jobs(profile, location)

        top = matches[0]
        if top["match_score"] < 50.0:
            in_interest = [m for m in matches if "Any" not in profile["interests"] and m["domain"] in profile["interests"]]
            if in_interest:
                top = in_interest[0]
            st.write(f"🧭 **Orchestrator** — top match only {matches[0]['match_score']}% (<50%). Branched to best fit in interest domains → **{top['title']}** ({top['match_score']}%)")

        st.write(f"🔍 **Gap Agent** — analyzing '{top['title']}'…")
        gap = t_analyze_gap(profile["skills"], top)
        st.write("📚 **Training Agent** — retrieving courses & ranking by demand…")
        roadmap, total_weeks = t_build_roadmap(gap, free_only)
        st.write("📄 **Report Agent** — compiling report…")
        status.update(label="✅ Pipeline complete — scroll down for results", state="complete", expanded=False)

    # --- Agent trace cards ---
    st.subheader("🤖 Agent Activity Trace")
    c1, c2, c3 = st.columns(3)
    c1.success(f"✓ Profile Agent — {len(profile['skills'])} skills (edu: {profile['education']})")
    c2.success(f"✓ Job Match Agent — {len(matches)} jobs evaluated")
    c3.success(f"✓ Gap Agent — {len(gap['critical'])} critical gap(s) for '{top['title']}'")

    total_cost = sum(int(r["Cost"].replace("₹","")) for r in roadmap if r["Cost"] != "Free")
    base_profile = {"raw_text": resume_text, "skills": extract_skills(resume_text), "education": profile["education"],
                    "interests": profile["interests"], "simulated": []}
    base_matches = t_match_jobs(base_profile, location)
    before_unlocked = sum(1 for m in base_matches if m["match_score"] >= 50)
    after_unlocked = sum(1 for m in matches if m["match_score"] >= 50)
    st.info(f"✓ Training Agent — {len(roadmap)} course(s), {total_weeks} weeks, ₹{total_cost}  |  "
            f"✓ Report Agent — compiled  |  Jobs ≥50%: {before_unlocked} → {after_unlocked} (+{after_unlocked-before_unlocked} unlocked by simulation)")

    with st.expander("🔧 Tool call trace (decorated tools, documented + timed)"):
        for i, t in enumerate(st.session_state[TOOL_TRACE_KEY], 1):
            st.markdown(f'<div class="cg-trace"><b>{i}. {t["tool"]}</b> <small>{t["desc"]}</small><br>'
                        f'<small>args: {t["args"]} · ⏱ {t["seconds"]}s</small></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Top Job Matches", "🎯 Skill Gap", "🗺️ Learning Roadmap", "📄 Career Report"])

    with tab1:
        st.caption("Time-to-ready (Key Feature 7): total weeks & cost to close each job's gaps — recomputed with your free-only filter and simulated skills.")
        cols = ["job_id","title","company","location","domain","match_score","keyword_score","weeks_to_ready","cost_to_ready","matched_skills","missing_skills"]
        df = pd.DataFrame(matches[:10])[cols].rename(columns={"match_score":"Match %","keyword_score":"Keyword %",
                                                              "weeks_to_ready":"Weeks to ready","cost_to_ready":"Cost to ready (₹)"})
        st.dataframe(df, use_container_width=True, hide_index=True)

    with tab2:
        cm1, cm2 = st.columns([1,3])
        cm1.metric(f"Readiness for {top['title']}", f"{gap['readiness']}%")
        cm2.progress(int(gap["readiness"])/100)
        st.write(f"**✅ Matched:** {', '.join(gap['matched']) or 'none'}")
        no_gaps_msg = "none — you are fully qualified for this role!"
        st.error("**🔴 Critical gaps (required):** " + (", ".join(gap["critical"]) or no_gaps_msg))
        st.warning(f"**🟡 Important gaps (preferred):** {', '.join(gap['important']) or 'none'}")
        explain, used_llm, llm_status = llm_gap_explanation(profile["skills"], top, gap)
        st.markdown("**🧠 Why these gaps matter**")
        if used_llm:
            st.caption(f"✨ Grounded explanation via {llm_status}")
            st.info(explain)
        else:
            st.caption(f"Template explanation in use — {llm_status}")
            st.info(explain)

    with tab3:
        st.write(f"**Total time to ready:** {total_weeks} weeks   |   **Total cost:** ₹{total_cost}")
        if roadmap:
            st.dataframe(pd.DataFrame(roadmap), use_container_width=True, hide_index=True)
            st.caption("Retrieval column shows how each course was matched to the gap (semantic similarity % or keyword fallback) — traceable RAG.")
        else:
            st.balloons()
            st.success("No gaps — you are already job-ready for this role!")

    with tab4:
        roadmap_lines = [f"{r['Schedule']}: {r['Skill']} → {r['Course']} ({r['Provider']}, {r['Cost']})" for r in roadmap]
        summary, used_llm, llm_status = llm_career_summary(profile["skills"], top, gap, roadmap_lines)
        if used_llm:
            st.caption(f"✨ Summary generated with Gemini ({llm_status})")
        else:
            st.warning(f"Template summary in use. Reason: {llm_status}")
        st.info(summary)

        report_md = [
            "# CareerGap AI — Career Readiness Report",
            f"**Education:** {profile['education']}   **Experience:** Fresher/Student",
            f"**Skills ({len(profile['skills'])}):** {', '.join(profile['skills'])}",
            f"**Retrieval backend:** {backend_note}",
            "",
            f"## Target Role: {top['title']} @ {top['company']} ({top['location']})",
            f"**Match: {top['match_score']}%   Readiness: {gap['readiness']}%**",
            "",
            "### Top 5 Matches — with time-to-ready",
            "| Job | Match % | Weeks to ready | Cost (₹) |",
            "|---|---|---|---|",
        ]
        for m in matches[:5]:
            report_md.append(f"| {m['title']} @ {m['company']} | {m['match_score']}% | {m.get('weeks_to_ready','—')} | {m.get('cost_to_ready','—')} |")
        report_md += ["", "### Skill Gaps",
                      f"- 🔴 Critical: {', '.join(gap['critical']) or 'none'}",
                      f"- 🟡 Important: {', '.join(gap['important']) or 'none'}",
                      "", f"## Recommended Roadmap — {total_weeks} weeks, ₹{total_cost}"]
        for r in roadmap:
            report_md.append(f"- **{r['Schedule']}** — {r['Skill']} → {r['Course']} ({r['Provider']}, {r['Cost']}) [{r['Retrieval']}] — unlocks {r['Unlocks (jobs)']} jobs")
        report_md += ["", "## Advisor Summary", summary]
        report_text = "\n".join(report_md)
        st.markdown(report_text)
        st.download_button("⬇️ Download Report (.md)", report_text,
                           file_name="career_readiness_report.md", mime="text/markdown")
else:
    st.info("Fill in your profile on the left and click **Analyze**. Try the 💡 simulator: add a skill and watch match scores, time-to-ready, and unlocked-job counts update live.")
