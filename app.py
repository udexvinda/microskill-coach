# app.py — MicroSkill Coach (Streamlit Cloud edition)
# - Upload LinkedIn PDF → analyze → compute gaps → HTML report
# - Optional Deep AI (OpenAI) for summary/axes/courses
# - Uses openai==1.51.0 (version-agnostic call style)
from __future__ import annotations
import base64, io, os, re, json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pypdf import PdfReader

# ---------------- Page & session ----------------
st.set_page_config(page_title="MicroSkill Coach", page_icon="🎯", layout="wide")
ss = st.session_state
ss.setdefault("analyzed", False)
ss.setdefault("computed", False)
ss.setdefault("profile", None)   # {raw, sections, detected, name}
ss.setdefault("gpt", None)       # {summary, role_label, axis_scores, top_gaps, courses}
ss.setdefault("report_html", None)

# ------------- Role targets and axes -------------
TARGET_ROLES: Dict[str, Dict[str, float]] = {
    "Data Analyst": {
        "SQL":0.9,"Python":0.9,"Data Visualization":0.8,"Statistics":0.8,"ETL":0.6,
        "Dashboarding":0.7,"Business Communication":0.6,"Domain Knowledge":0.5
    },
    "Ops Excellence Lead": {
        "Lean Six Sigma":0.9,"BPMN":0.8,"Process Mining":0.8,"Automation/RPA":0.7,
        "Change Management":0.7,"Stakeholder Management":0.7,"Data Analysis":0.7,"TRIZ":0.5
    },
    "AI Product Manager": {
        "Product Strategy":0.9,"User Research":0.7,"AI/ML Fundamentals":0.8,"Prompt Engineering":0.6,
        "Data Analysis":0.7,"Experimentation":0.7,"Roadmapping":0.7,"Go-To-Market":0.6
    },
}
ROLE_AXES: Dict[str, Dict[str, List[str]]] = {
    "Data Analyst": {
        "Data": ["ETL","SQL"],
        "Programming": ["Python"],
        "Visualization": ["Data Visualization","Dashboarding"],
        "Statistics": ["Statistics"],
        "Business": ["Business Communication","Domain Knowledge"],
        "Ops": ["Data Analysis"],
    },
    "Ops Excellence Lead": {
        "Lean/CI": ["Lean Six Sigma"],
        "Process": ["BPMN","Process Mining"],
        "Automation": ["Automation/RPA"],
        "Change": ["Change Management","Stakeholder Management"],
        "Analytics": ["Data Analysis"],
        "Innovation": ["TRIZ"],
    },
    "AI Product Manager": {
        "Strategy": ["Product Strategy","Roadmapping"],
        "Research": ["User Research"],
        "AI/ML": ["AI/ML Fundamentals","Prompt Engineering"],
        "Analytics": ["Data Analysis","Experimentation"],
        "GTM": ["Go-To-Market"],
        "Communication": ["Business Communication"],
    },
}

# ------------- Keyword heuristics -------------
SKILL_KEYWORDS: Dict[str, List[str]] = {
    "SQL":["sql","postgres","mysql","t-sql","mssql","sqlite"],
    "Python":["python","pandas","numpy","scikit-learn","matplotlib"],
    "Data Visualization":["tableau","power bi","plotly","visualization","data viz","lookerstudio","looker"],
    "Statistics":["hypothesis test","anova","regression","statistic","bayes"],
    "ETL":["etl","extract transform","data pipeline","airflow","dlt"],
    "Dashboarding":["dashboard","tableau","power bi","metabase"],
    "Business Communication":["stakeholder","presentation","communication","storytelling"],
    "Domain Knowledge":["insurance","banking","manufacturing","energy","shared services","healthcare"],
    "Lean Six Sigma":["lean","six sigma","dmaic","kaizen","value stream","black belt","green belt"],
    "BPMN":["bpmn","process map","swimlane","camunda","bpmn.io"],
    "Process Mining":["process mining","event log","conformance","celonis","apromore","pm4py"],
    "Automation/RPA":["rpa","uipath","automation anywhere","blue prism","automation"],
    "Change Management":["change management","adkar","prosci","change adoption"],
    "Stakeholder Management":["stakeholder","steering committee","executive","buy-in"],
    "Data Analysis":["analysis","analytics","kpi","metrics","reporting"],
    "TRIZ":["triz","contradiction","inventive principles","ariz"],
    "Product Strategy":["product strategy","north star","kpi","product vision"],
    "User Research":["interview","usability","persona","survey"],
    "AI/ML Fundamentals":["machine learning","ml","classification","regression","model","embedding","llm"],
    "Prompt Engineering":["prompt","rag","few-shot","system prompt"],
    "Experimentation":["ab test","experiment","controlled","cohort"],
    "Roadmapping":["roadmap","quarter","milestone"],
    "Go-To-Market":["gtm","launch","pricing","positioning"],
}

COURSES = [
    {"title":"Intro to SQL","provider":"Coursera","url":"https://example.com/sql","skills":["SQL"],"hours":8,"level":"Beginner"},
    {"title":"Practical Python for Data","provider":"edX","url":"https://example.com/python","skills":["Python","Data Analysis"],"hours":12,"level":"Beginner"},
    {"title":"Tableau for Data Viz","provider":"Udacity","url":"https://example.com/tableau","skills":["Data Visualization","Dashboarding"],"hours":10,"level":"Beginner"},
    {"title":"Lean Six Sigma Yellow Belt","provider":"Coursera","url":"https://example.com/lss","skills":["Lean Six Sigma"],"hours":14,"level":"Beginner"},
    {"title":"Process Mining Fundamentals","provider":"edX","url":"https://example.com/pm","skills":["Process Mining"],"hours":16,"level":"Intermediate"},
    {"title":"Prompt Engineering Basics","provider":"YouTube","url":"https://example.com/prompt","skills":["Prompt Engineering"],"hours":4,"level":"Beginner"},
    {"title":"Statistics with Python","provider":"Coursera","url":"https://example.com/stats","skills":["Statistics","Python"],"hours":20,"level":"Intermediate"},
    {"title":"Power BI Dashboarding","provider":"Udemy","url":"https://example.com/pbi","skills":["Dashboarding","Data Visualization"],"hours":9,"level":"Beginner"},
]

# ------------- Utils -------------
def read_pdf_text(file) -> str:
    reader = PdfReader(file)
    pages = [(p.extract_text() or "") for p in reader.pages]
    return re.sub(r"\s+", " ", "\n".join(pages)).strip()

def infer_profile_sections(text: str) -> Dict[str, str]:
    lower = text.lower()
    def grab(start: str, nexts: List[str]) -> str:
        s = lower.find(start)
        if s == -1: return ""
        e = len(text)
        for nxt in nexts:
            i = lower.find(nxt)
            if i != -1 and i > s: e = min(e, i)
        return text[s:e].strip()
    return {
        "about": grab("about", ["experience","education","skills","certifications"]),
        "experience": grab("experience", ["education","skills","certifications"]),
        "education": grab("education", ["skills","certifications"]),
        "skills": grab("skills", ["certifications","licenses","accomplishments"]),
        "certifications": grab("certifications", ["skills","projects","honors"]),
    }

def detect_skills(text: str) -> Dict[str, float]:
    t = text.lower(); out={}
    for skill, kws in SKILL_KEYWORDS.items():
        hits = sum(1 for kw in kws if re.search(rf"\b{re.escape(kw)}\b", t))
        out[skill] = (min(1.0, 0.5 + 0.1*(hits-1))) if hits else 0.0
    return out

def role_gap_scores(role_skills: Dict[str,float], detected: Dict[str,float], axes_map: Dict[str,List[str]]):
    rows=[]
    for skill, w in role_skills.items():
        cur=float(detected.get(skill,0)); gap=max(0.0, 1.0-cur)
        rows.append({"Skill":skill,"Weight":w,"Current":round(cur,2),"Target":1.0,"Gap":round(gap,2),"WeightedGap":round(gap*w,2)})
    df = pd.DataFrame(rows).sort_values(["WeightedGap","Gap"], ascending=False, ignore_index=True)
    axis_scores = {axis: (float(np.mean([detected.get(s,0.0) for s in skills])) if skills else 0.0)
                   for axis, skills in axes_map.items()}
    return df, axis_scores

def radar_chart(axis_scores: Dict[str,float], title: str):
    labels = list(axis_scores.keys())
    values = list(axis_scores.values())
    N = max(1, len(labels))
    if not labels: labels, values = ["No axes"], [0.0]
    angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
    vals = values + values[:1]
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], labels)
    ax.set_rlabel_position(0); ax.set_ylim(0,1)
    ax.plot(angles, vals); ax.fill(angles, vals, alpha=0.1)
    plt.title(title)
    return fig

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

def make_html_report(name, role_label, gaps_df, recs, axes, radar_b64, ai_summary: str|None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    gaps_html = gaps_df.head(10).to_html(index=False)
    axes_html = "".join([f"<li><b>{k}</b>: {v:.2f}</li>" for k, v in axes.items()])
    courses_li = "\n".join([
        f"<li><a href='{c.get('url','')}' target='_blank'>{c.get('title','Course')}</a> — "
        f"{c.get('provider','')} • {c.get('hours','')}h • {c.get('level','')} • "
        f"Skills: {', '.join(c.get('skills', []))}</li>"
        for c in recs
    ])
    summary_html = f"<p>{ai_summary}</p>" if ai_summary else "<p class='muted'>No AI summary (offline mode).</p>"
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>MicroSkill Coach Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#222}}
h1,h2{{margin-bottom:8px}} .muted{{color:#666}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:8px}} th{{background:#f6f6f6}}
.radar{{text-align:center;margin:16px 0}}
</style></head><body>
<h1>MicroSkill Coach Report</h1>
<p class='muted'>Generated: {now}</p>
<p><b>Profile:</b> {name or 'N/A'}<br><b>Target Role:</b> {role_label}</p>
<h2>AI Summary</h2>{summary_html}
<h2>Competency Radar</h2><div class='radar'><img src='{radar_b64}' alt='Radar'></div>
<h2>Top Skill Gaps</h2>{gaps_html}
<h2>Axis Scores</h2><ul>{axes_html}</ul>
<h2>Recommended Courses</h2><ul>{courses_li}</ul>
<hr><p class='muted'>Privacy: Generated from your uploaded PDF only. AI is optional and uses your secret key.</p>
</body></html>"""

# ------------- Sidebar: Deep AI toggle & key -------------
with st.sidebar:
    st.markdown("### Modes")
    deep_ai = st.toggle("Enable Deep AI (OpenAI)", value=True,
                        help="If ON and key is set in Secrets, AI adds summary/axes/courses.")
    OPENAI_API_KEY = (st.secrets.get("openai", {}).get("api_key") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")
    if deep_ai and not OPENAI_API_KEY:
        st.caption("Deep AI is ON but no key found in Secrets. Falling back to offline heuristics.")

# ------------- Main UI -------------
st.title("🎯 MicroSkill Coach — Begin Your LSS Journey")
st.caption("Upload → Analyze → Compute → Download report (HTML). AI optional.")

uploaded = st.file_uploader("Upload your LinkedIn PDF", type=["pdf"])
role_choice = st.selectbox("Target role", list(TARGET_ROLES.keys()), index=0)

# Horizontal action bar
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1: analyze_btn  = st.button("Analyze & Suggest Roles", type="primary", disabled=not uploaded)
with c2: compute_btn  = st.button("Compute Gaps & Plan", disabled=not ss.get("analyzed", False))
with c3: download_btn = st.button("Download Report (HTML)", disabled=not ss.get("computed", False))
with c4: reset_btn    = st.button("Reset")

# ---------- 1) Analyze ----------
if analyze_btn and uploaded:
    try:
        raw = read_pdf_text(uploaded)
        if not raw:
            st.error("Could not extract text. Ensure the PDF is text-based (not scanned).")
            st.stop()
        sections = infer_profile_sections(raw)
        detected = detect_skills(raw)
        first_line = raw.split("\n")[0].strip()
        name_guess = first_line if 2 <= len(first_line.split()) <= 5 else ""

        ss.profile = {"raw": raw, "sections": sections, "detected": detected, "name": name_guess}
        ss.analyzed, ss.computed, ss.report_html, ss.gpt = True, False, None, None

        # Deep AI (OpenAI) — summary/axes/courses as JSON
        if deep_ai and OPENAI_API_KEY:
            try:
                import openai
                openai.api_key = OPENAI_API_KEY
                prompt = (
                    "You are a career coach. Read the PROFILE TEXT and return JSON with keys:\n"
                    "summary (<=120 words), role_label, axis_scores (object of 4-8 axes with 0..1), "
                    "top_gaps (array of {skill,reason}), courses (array of {title,provider,url,hours,level,skills}).\n"
                    "Courses must align to the gaps and be realistic.\nPROFILE TEXT:\n" + raw[:5000]
                )
                r = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.2,
                    messages=[{"role":"system","content":"Return valid JSON only."},
                              {"role":"user","content":prompt}],
                )
                data = json.loads(r.choices[0].message.content.strip())
                ss.gpt = {
                    "summary": str(data.get("summary","")),
                    "role_label": (str(data.get("role_label","")) or None),
                    "axis_scores": data.get("axis_scores", {}),
                    "top_gaps": data.get("top_gaps", []),
                    "courses": data.get("courses", []),
                }
                if ss.gpt.get("role_label") in TARGET_ROLES:
                    role_choice = ss.gpt["role_label"]
            except Exception as e:
                st.info(f"AI skipped (using heuristics). Detail: {e}")

        st.success("Profile parsed. Now click **Compute Gaps & Plan**.")
    except Exception as e:
        st.exception(e)

# ---------- 2) Compute ----------
if compute_btn and ss.get("analyzed") and ss.get("profile"):
    try:
        detected = ss.profile["detected"]
        role_skills = TARGET_ROLES[role_choice]
        axes_map = ROLE_AXES[role_choice]
        gaps_df, axis_scores = role_gap_scores(role_skills, detected, axes_map)

        # Blend AI axis suggestions if present (display only)
        if ss.gpt and isinstance(ss.gpt.get("axis_scores"), dict):
            for ax, val in ss.gpt["axis_scores"].items():
                if ax in axis_scores:
                    axis_scores[ax] = float(np.clip(0.5*axis_scores[ax] + 0.5*float(val), 0.0, 1.0))

        # Courses: prefer AI list; fallback to catalog
        if ss.gpt and ss.gpt.get("courses"):
            recs = [{
                "title": c.get("title","Course"), "provider": c.get("provider",""),
                "url": c.get("url",""), "skills": c.get("skills", []),
                "hours": c.get("hours",""), "level": c.get("level",""),
            } for c in ss.gpt["courses"][:8]]
        else:
            wanted = gaps_df.sort_values("WeightedGap", ascending=False)["Skill"].tolist()
            recs = [c for c in COURSES if set(c["skills"]) & set(wanted)][:6]

        # Tabs
        st.success("Plan computed. Review tabs and download the report.")
        t1, t2, t3, t4, t5 = st.tabs(["Summary","Radar","Gaps","Courses","Raw Text"])

        with t1:
            st.subheader("Profile Summary")
            if ss.gpt and ss.gpt.get("summary"): st.markdown(ss.gpt["summary"])
            else: st.caption("No AI summary (offline mode).")
            det_df = pd.DataFrame([{"Skill":k,"Score":round(v,2)} for k,v in detected.items()])
            st.dataframe(det_df.sort_values("Score", ascending=False), use_container_width=True)

        with t2:
            st.subheader("Competency Radar")
            fig = radar_chart(axis_scores, f"{role_choice} — Current Profile")
            st.pyplot(fig, use_container_width=True)

        with t3:
            st.subheader("Top Skill Gaps (weighted)")
            st.dataframe(gaps_df, use_container_width=True)

        with t4:
            st.subheader("Recommended Courses")
            cdf = pd.DataFrame(recs)
            if not cdf.empty:
                st.dataframe(cdf[["title","provider","hours","level","url","skills"]], use_container_width=True)
            else:
                st.info("No course recommendations available.")

        with t5:
            st.subheader("Raw Extracted Text (first 3,000 chars)")
            st.code(ss.profile["raw"][:3000] + ("..." if len(ss.profile["raw"]) > 3000 else ""))

        # HTML report
        radar_b64 = fig_to_b64(fig)
        ai_summary = ss.gpt["summary"] if ss.gpt and ss.gpt.get("summary") else None
        ss.report_html = make_html_report(ss.profile["name"], role_choice, gaps_df, recs, axis_scores, radar_b64, ai_summary)
        ss.computed = True
    except Exception as e:
        st.exception(e)

# ---------- 3) Download ----------
if download_btn and ss.get("computed") and ss.get("report_html"):
    st.download_button(
        "Download Report (HTML)",
        data=ss.report_html.encode("utf-8"),
        file_name=f"microskill_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
        mime="text/html",
    )

# ---------- 4) Reset ----------
if reset_btn:
    for k in ["analyzed","computed","profile","gpt","report_html"]:
        if k in ss: del ss[k]
    st.rerun()