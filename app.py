"""
CareerCompass - Main Web Application
Resume Intelligence and Job Matching Platform
Built with Streamlit & Python
"""

import streamlit as st
import os
import sqlite3
import pandas as pd
import numpy as np
import altair as alt

from src.parser import ResumeParser
from src.matcher import MatchEngine
from src.retrieval import KnowledgeRetriever
from src.assistant import CareerAssistant
from src.extensions import CareerExtensions

st.set_page_config(
    page_title="CareerCompass | Resume Intelligence Platform",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.15);
    }
    .metric-card h4 {
        color: #94A3B8 !important;
        font-size: 1.05rem !important;
        margin-bottom: 0.3rem !important;
    }
    .metric-card h2 {
        color: #38BDF8 !important;
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        margin: 0.2rem 0 !important;
    }
    .metric-card p {
        color: #CBD5E1 !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }
    .score-badge-high {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
    }
    .score-badge-med {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
    }
    .score-badge-low {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
    }
    .tool-badge {
        display: inline-block;
        background: #1E3A8A;
        color: #93C5FD;
        border: 1px solid #3B82F6;
        border-radius: 6px;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        font-family: monospace;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "careercompass.db")

@st.cache_resource
def load_engines():
    parser = ResumeParser()
    matcher = MatchEngine()
    retriever = KnowledgeRetriever()
    assistant = CareerAssistant()
    extensions = CareerExtensions(matcher)
    return parser, matcher, retriever, assistant, extensions

parser, matcher, retriever, assistant, extensions = load_engines()

@st.cache_data
def load_db_data():
    conn = sqlite3.connect(DB_PATH)
    jobs_df = pd.read_sql_query("""
        SELECT j.job_id, j.title, c.name as company, j.role_category,
               j.experience_min, j.experience_max, j.salary_min, j.salary_max, j.salary_avg,
               j.location, j.is_remote, j.posted_date, j.description
        FROM jobs j
        JOIN companies c ON j.company_id = c.company_id
    """, conn)
    
    resumes_df = pd.read_sql_query("""
        SELECT resume_id, candidate_name, email, phone, target_role,
               years_experience, experience_level, education_level, degree, raw_text
        FROM resumes
    """, conn)
    conn.close()
    return jobs_df, resumes_df

jobs_df, resumes_df = load_db_data()

# -------------------------------------------------------------
# App Header
# -------------------------------------------------------------
st.markdown("""
<div style="display: flex; align-items: center; gap: 14px; margin-bottom: 0.3rem;">
    <svg width="44" height="44" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink: 0;">
        <defs>
            <linearGradient id="compassGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#38BDF8"/>
                <stop offset="100%" stop-color="#818CF8"/>
            </linearGradient>
            <linearGradient id="needleNorth" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#38BDF8"/>
                <stop offset="100%" stop-color="#2563EB"/>
            </linearGradient>
            <linearGradient id="needleSouth" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stop-color="#64748B"/>
                <stop offset="100%" stop-color="#475569"/>
            </linearGradient>
        </defs>
        <circle cx="24" cy="24" r="22" stroke="url(#compassGrad)" stroke-width="1.8" stroke-dasharray="3 3" opacity="0.65"/>
        <circle cx="24" cy="24" r="18" stroke="url(#compassGrad)" stroke-width="2.2" fill="#0F172A"/>
        <!-- Cardinal ticks -->
        <line x1="24" y1="8" x2="24" y2="11" stroke="#38BDF8" stroke-width="2" stroke-linecap="round"/>
        <line x1="24" y1="37" x2="24" y2="40" stroke="#38BDF8" stroke-width="2" stroke-linecap="round"/>
        <line x1="8" y1="24" x2="11" y2="24" stroke="#38BDF8" stroke-width="2" stroke-linecap="round"/>
        <line x1="37" y1="24" x2="40" y2="24" stroke="#38BDF8" stroke-width="2" stroke-linecap="round"/>
        <!-- Compass Needles -->
        <polygon points="24,9 28,24 24,22 20,24" fill="url(#needleNorth)"/>
        <polygon points="24,39 28,24 24,26 20,24" fill="url(#needleSouth)"/>
        <!-- Pivot Core -->
        <circle cx="24" cy="24" r="3" fill="#FFFFFF"/>
        <circle cx="24" cy="24" r="1.5" fill="#0F172A"/>
    </svg>
    <div class="main-title" style="margin-bottom: 0;">CareerCompass</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="sub-title">Resume Intelligence & Job Matching Platform — Powered by Machine Learning & Grounded RAG</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🎯 Resume Match & Remediation",
    "📊 Job Market Dashboard",
    "🤖 Conversational Assistant (Tool Calling)",
    "👥 Batch Candidate Screening"
])

# -------------------------------------------------------------
# TAB 1: RESUME MATCH & REMEDIATION
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("1. Candidate Resume Input")
    col_input_type, col_sample = st.columns([2, 2])
    
    with col_input_type:
        input_mode = st.radio("Choose Input Method:", ["Upload File (PDF / DOCX / TXT)", "Select Sample Candidate Profile", "Paste Raw Resume Text"], horizontal=True)
        
    resume_data = None
    
    if input_mode == "Upload File (PDF / DOCX / TXT)":
        uploaded_file = st.file_uploader("Upload Resume Document", type=["pdf", "docx", "txt"])
        if uploaded_file:
            with st.spinner("Parsing document structure, contact details, and skill entities..."):
                resume_data = parser.parse_file(uploaded_file, filename=uploaded_file.name)
            st.success(f"Successfully parsed resume! Extracted **{len(resume_data['skills'])} skills** and **{resume_data['years_experience']} years of experience**.")
    elif input_mode == "Select Sample Candidate Profile":
        sample_roles = resumes_df["target_role"].unique().tolist()
        sel_role = st.selectbox("Filter Sample Profiles by Discipline:", sample_roles)
        role_candidates = resumes_df[resumes_df["target_role"] == sel_role]
        cand_choice = st.selectbox("Select Candidate:", role_candidates["candidate_name"].tolist())
        sel_cand_row = role_candidates[role_candidates["candidate_name"] == cand_choice].iloc[0]
        resume_data = parser.parse_text(sel_cand_row["raw_text"])
        st.info(f"Loaded Profile: **{cand_choice}** ({sel_cand_row['target_role']}, {sel_cand_row['years_experience']} yrs exp, {sel_cand_row['education_level']})")
    else:
        raw_input = st.text_area(
            "Paste Resume Content",
            height=160,
            value="""Jordan Miller | Senior DevOps Engineer
            jordan.miller@example.org | 6 years of experience
            Skills: Docker, Kubernetes, Terraform, AWS, Linux, CI/CD, Python
            Education: B.S. in Computer Science, Stanford University"""
        )
        if raw_input:
            resume_data = parser.parse_text(raw_input)

    st.divider()
    
    # Target Job Description Selection
    st.subheader("2. Target Job Description")
    jd_mode = st.radio("Choose Job Description Source:", ["Select from 10,600+ Verified Postings", "Enter Custom Job Description"], horizontal=True)
    
    target_job = None
    if jd_mode == "Select from 10,600+ Verified Postings":
        col_j1, col_j2 = st.columns(2)
        with col_j1:
            filter_role = st.selectbox("Role Discipline:", ["All"] + list(jobs_df["role_category"].unique()))
        with col_j2:
            filtered_jobs = jobs_df if filter_role == "All" else jobs_df[jobs_df["role_category"] == filter_role]
            selected_job_title = st.selectbox("Select Job Opening:", filtered_jobs["title"] + " @ " + filtered_jobs["company"] + " [" + filtered_jobs["job_id"] + "]")
            
        sel_id = selected_job_title.split("[")[-1].replace("]", "").strip()
        job_row = jobs_df[jobs_df["job_id"] == sel_id].iloc[0]
        
        # Get required skills from database
        conn = sqlite3.connect(DB_PATH)
        job_skills_query = """
            SELECT s.name FROM job_skills js
            JOIN skills s ON js.skill_id = s.skill_id
            WHERE js.job_id = ?
        """
        j_skills = [r[0] for r in conn.cursor().execute(job_skills_query, (sel_id,)).fetchall()]
        conn.close()
        
        target_job = {
            "job_id": job_row["job_id"],
            "title": job_row["title"],
            "company": job_row["company"],
            "role_category": job_row["role_category"],
            "experience_min": job_row["experience_min"],
            "experience_max": job_row["experience_max"],
            "salary_range": f"${int(job_row['salary_min']):,} - ${int(job_row['salary_max']):,}",
            "location": job_row["location"],
            "skills": j_skills,
            "description": job_row["description"]
        }
        
        st.markdown(f"**Selected Position**: `{target_job['title']}` at **{target_job['company']}** | **Required Skills**: {', '.join(j_skills)} | **Exp**: {target_job['experience_min']} - {target_job['experience_max']} yrs")
    else:
        custom_title = st.text_input("Job Title", value="Senior Cloud Infrastructure Engineer")
        custom_skills = st.text_input("Required Skills (comma-separated)", value="Docker, Kubernetes, AWS, Terraform, CI/CD, Python, Prometheus")
        custom_exp = st.slider("Minimum Years Experience Required", 0, 12, 5)
        custom_desc = st.text_area("Full Job Description", value="Looking for an experienced engineer to lead our cloud infrastructure, manage Kubernetes clusters, and automate Terraform CI/CD pipelines.")
        target_job = {
            "title": custom_title,
            "company": "Custom Company",
            "role_category": "Cloud & DevOps Engineer",
            "experience_min": custom_exp,
            "experience_max": custom_exp + 3,
            "skills": [s.strip() for s in custom_skills.split(",") if s.strip()],
            "description": custom_desc
        }

    st.divider()
    
    # Run Match Engine
    if st.button("🚀 Calculate Match Score & Generate Improvement Plan", type="primary", use_container_width=True):
        if not resume_data:
            st.error("Please provide a resume to analyze.")
        else:
            with st.spinner("Analyzing candidate profile, running predictive classifier, and retrieving remediation roadmap..."):
                match_result = matcher.calculate_match(resume_data, target_job)
                
            score = match_result["overall_match_score"]
            bd = match_result["breakdown"]
            verdict = match_result["classifier_prediction"]
            prob = match_result["strong_match_probability"]
            
            st.markdown("### Match Intelligence Assessment")
            
            col_score, col_verdict = st.columns([1.5, 3])
            with col_score:
                badge_class = "score-badge-high" if score >= 75 else "score-badge-med" if score >= 50 else "score-badge-low"
                st.markdown(f'<div class="{badge_class}">{score} / 100</div>', unsafe_allow_html=True)
                st.caption("Overall Match Score")
            with col_verdict:
                st.markdown(f"#### Classifier Verdict: **{verdict}**")
                st.markdown(f"**Strong Match Model Confidence**: `{prob}%`")
                st.info("Features derived from skill overlap, experience delta, education tier, and dense semantic embedding.")

            st.markdown("#### Transparent Calculation Breakdown")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🛠️ Skill Match</h4>
                    <h2>{bd['skill_match']['score']} <span style="font-size:1.1rem;color:#94A3B8;font-weight:400;">/ 40</span></h2>
                    <p>{len(bd['skill_match']['matched_skills'])} matched, {len(bd['skill_match']['missing_skills'])} missing</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⏳ Experience Fit</h4>
                    <h2>{bd['experience_fit']['score']} <span style="font-size:1.1rem;color:#94A3B8;font-weight:400;">/ 25</span></h2>
                    <p>Candidate: {bd['experience_fit']['candidate_years']} yrs (Req: {bd['experience_fit']['required_years']})</p>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎓 Education</h4>
                    <h2>{bd['education_alignment']['score']} <span style="font-size:1.1rem;color:#94A3B8;font-weight:400;">/ 15</span></h2>
                    <p>Level: {bd['education_alignment']['candidate_level']}</p>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🧠 Semantic Fit</h4>
                    <h2>{bd['semantic_relevance']['score']} <span style="font-size:1.1rem;color:#94A3B8;font-weight:400;">/ 20</span></h2>
                    <p>Similarity: {round(bd['semantic_relevance']['raw_similarity']*100, 1)}%</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            
            # Skills Analysis & Ranked Gaps
            col_matched, col_gaps = st.columns(2)
            with col_matched:
                st.markdown("#### ✅ Matched Core Competencies")
                if bd['skill_match']['matched_skills']:
                    chips = "".join([f"<span style='background:rgba(16,185,129,0.2);color:#34D399;border:1px solid rgba(16,185,129,0.4);padding:4px 12px;margin:4px;border-radius:15px;display:inline-block;font-weight:600;'>✓ {s}</span>" for s in bd['skill_match']['matched_skills']])
                    st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.write("No direct skills matched.")
                    
            with col_gaps:
                st.markdown("#### ⚠️ Missing Skills (Ranked by Score Impact)")
                ranked = match_result["ranked_missing_skills"]
                if ranked:
                    for item in ranked:
                        crit_color = "#DC2626" if item["importance"] == "Critical" else "#D97706"
                        st.markdown(
                            f"- **{item['skill']}** &nbsp; "
                            f"<span style='color:{crit_color};font-weight:700;'>+{item['score_impact_points']} pts potential gain</span> &nbsp; "
                            f"*(Priority: {item['importance']})*",
                            unsafe_allow_html=True
                        )
                else:
                    st.success("Candidate possesses all required skills!")

            st.markdown("---")
            
            # Phased Improvement Plan citing Knowledge Base
            st.markdown("### 📚 Grounded Improvement & Remediation Plan")
            st.caption("Generated by RAG retrieval engine referencing curated taxonomies, interview question banks, and learning materials.")
            
            missing_skill_names = [item["skill"] for item in match_result["ranked_missing_skills"]]
            plan = retriever.generate_improvement_plan(missing_skill_names, target_role=target_job.get("role_category", "Engineering"))
            
            for phase in plan["phases"]:
                with st.expander(f"📌 {phase['phase']} — Goal: {phase['goals']}", expanded=True):
                    for task in phase["tasks"]:
                        st.markdown(f"- {task}")
                        
            st.markdown("##### 🔗 Explicit Knowledge Base Citations:")
            for src in plan["cited_sources"]:
                st.markdown(f"1. `{src}`")
                
            # Extension: Resume Rewriting Suggestions
            with st.expander("✨ Tailored Resume Rewriting Suggestions (STAR Method)", expanded=False):
                rewrites = extensions.generate_rewrite_suggestions(resume_data, target_job)
                for rw in rewrites:
                    st.markdown(f"**Section**: `{rw['section']}`")
                    st.markdown(f"❌ *Previous*: {rw['current_generic_pattern']}")
                    st.markdown(f"✅ *Optimized*: {rw['recommended_rewrite']}")
                    st.caption(f"💡 *Rationale*: {rw['rationale']}")
                    st.divider()

# -------------------------------------------------------------
# TAB 2: JOB MARKET INTELLIGENCE DASHBOARD
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("Job Market Intelligence & Skills Analytics")
    st.caption("Analytical exploration of 10,600+ job postings across 8 engineering disciplines.")
    
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dash_role = st.multiselect("Filter Disciplines:", jobs_df["role_category"].unique().tolist(), default=jobs_df["role_category"].unique().tolist()[:4])
    with col_f2:
        dash_exp = st.selectbox("Experience Level:", ["All Levels", "Junior (0-2 yrs)", "Mid-Level (3-5 yrs)", "Senior (6-8 yrs)", "Lead (9+ yrs)"])
    with col_f3:
        dash_remote = st.selectbox("Work Arrangement:", ["All Arrangements", "Remote Only", "On-site / Hybrid Only"])

    # Filter data
    df_filtered = jobs_df.copy()
    if dash_role:
        df_filtered = df_filtered[df_filtered["role_category"].isin(dash_role)]
    if dash_exp == "Junior (0-2 yrs)":
        df_filtered = df_filtered[df_filtered["experience_min"] <= 2]
    elif dash_exp == "Mid-Level (3-5 yrs)":
        df_filtered = df_filtered[(df_filtered["experience_min"] > 2) & (df_filtered["experience_min"] <= 5)]
    elif dash_exp == "Senior (6-8 yrs)":
        df_filtered = df_filtered[(df_filtered["experience_min"] > 5) & (df_filtered["experience_min"] <= 8)]
    elif dash_exp == "Lead (9+ yrs)":
        df_filtered = df_filtered[df_filtered["experience_min"] > 8]
        
    if dash_remote == "Remote Only":
        df_filtered = df_filtered[df_filtered["is_remote"] == 1]
    elif dash_remote == "On-site / Hybrid Only":
        df_filtered = df_filtered[df_filtered["is_remote"] == 0]

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active Postings", f"{len(df_filtered):,}")
    k2.metric("Average Salary", f"${int(df_filtered['salary_avg'].mean()):,}" if len(df_filtered) > 0 else "$0")
    remote_ratio = round((df_filtered['is_remote'].sum() / len(df_filtered) * 100), 1) if len(df_filtered) > 0 else 0
    k3.metric("Remote Share", f"{remote_ratio}%")
    top_comp = df_filtered["company"].mode()[0] if len(df_filtered) > 0 else "N/A"
    k4.metric("Top Hiring Org", top_comp)

    st.markdown("---")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### 📈 Top In-Demand Technical Skills")
        conn = sqlite3.connect(DB_PATH)
        sel_jobs_list = "('" + "','".join(df_filtered["job_id"].tolist()[:2000]) + "')"
        if len(df_filtered) > 0:
            skills_query = f"""
                SELECT s.name as skill_name, COUNT(js.job_id) as demand_count
                FROM job_skills js
                JOIN skills s ON js.skill_id = s.skill_id
                WHERE js.job_id IN {sel_jobs_list}
                GROUP BY s.name
                ORDER BY demand_count DESC LIMIT 12
            """
            top_s_df = pd.read_sql_query(skills_query, conn)
            top_s_df["pct"] = round((top_s_df["demand_count"] / len(df_filtered)) * 100, 1)
            
            chart_skills = alt.Chart(top_s_df).mark_bar(color="#3B82F6").encode(
                x=alt.X("demand_count:Q", title="Number of Postings"),
                y=alt.Y("skill_name:N", sort="-x", title="Skill"),
                tooltip=["skill_name", "demand_count", "pct"]
            ).properties(height=320)
            st.altair_chart(chart_skills, use_container_width=True)
        conn.close()

    with col_c2:
        st.markdown("#### 💰 Salary Distribution by Role Category")
        chart_salary = alt.Chart(df_filtered).mark_boxplot(extent="min-max", color="#10B981").encode(
            x=alt.X("role_category:N", title="Role Category", axis=alt.Axis(labelAngle=-25)),
            y=alt.Y("salary_avg:Q", title="Average Annual Salary ($USD)"),
            tooltip=["role_category", "salary_avg"]
        ).properties(height=320)
        st.altair_chart(chart_salary, use_container_width=True)

    col_c3, col_c4 = st.columns(2)
    with col_c3:
        st.markdown("#### 📍 Top Hiring Metro Locations")
        loc_df = df_filtered["location"].value_counts().head(8).reset_index()
        loc_df.columns = ["location", "count"]
        chart_loc = alt.Chart(loc_df).mark_bar(color="#8B5CF6").encode(
            x=alt.X("count:Q", title="Job Postings"),
            y=alt.Y("location:N", sort="-x", title="Location"),
            tooltip=["location", "count"]
        ).properties(height=280)
        st.altair_chart(chart_loc, use_container_width=True)
        
    with col_c4:
        st.markdown("#### 📅 Monthly Hiring Velocity Over Time")
        df_filtered["month"] = pd.to_datetime(df_filtered["posted_date"]).dt.to_period("M").astype(str)
        month_df = df_filtered.groupby("month").size().reset_index(name="postings")
        chart_trend = alt.Chart(month_df).mark_line(point=True, color="#EC4899").encode(
            x=alt.X("month:N", title="Posting Month"),
            y=alt.Y("postings:Q", title="Volume of Postings"),
            tooltip=["month", "postings"]
        ).properties(height=280)
        st.altair_chart(chart_trend, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: CONVERSATIONAL ASSISTANT WITH TOOL CALLING
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("Conversational Career Assistant")
    st.caption("Demonstrating autonomous tool calling grounded in the relational SQLite database and knowledge base.")
    
    st.markdown("""
    **Try one of these queries to trigger tool execution:**
    - `What is the demand and average salary for Kubernetes?`
    - `Find high paying senior backend jobs in New York`
    - `Give me technical interview questions for System Design`
    - `What are the learning resources for Docker?`
    """)
    
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "Hello! I am your CareerCompass assistant. I can query our live database of 10,600+ job postings and retrieve interview resources using function tool calling. What would you like to explore?"}
        ]
        
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_prompt = st.chat_input("Ask about jobs, salaries, skill demand, or interview prep...")
    if user_prompt:
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Executing tool calling on database and knowledge base..."):
                reply, tool_data = assistant.answer_query(user_prompt)
            st.markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})

# -------------------------------------------------------------
# TAB 4: BATCH CANDIDATE SCREENING (EXTENSION)
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("Batch Resume Screening & Candidate Ranking")
    st.caption("Screen multiple candidate resumes simultaneously against a single target job opening.")
    
    col_b_job, col_b_cand = st.columns([1.5, 1])
    with col_b_job:
        batch_job_role = st.selectbox("Select Target Job Role:", jobs_df["role_category"].unique(), key="batch_role")
        batch_jobs = jobs_df[jobs_df["role_category"] == batch_job_role]
        batch_job_pick = st.selectbox("Select Job Position:", batch_jobs["title"] + " @ " + batch_jobs["company"] + " [" + batch_jobs["job_id"] + "]")
        b_id = batch_job_pick.split("[")[-1].replace("]", "").strip()
        b_job_row = jobs_df[jobs_df["job_id"] == b_id].iloc[0]
        
        conn = sqlite3.connect(DB_PATH)
        b_skills = [r[0] for r in conn.cursor().execute("SELECT s.name FROM job_skills js JOIN skills s ON js.skill_id = s.skill_id WHERE js.job_id = ?", (b_id,)).fetchall()]
        conn.close()
        
        batch_target_job = {
            "title": b_job_row["title"],
            "company": b_job_row["company"],
            "role_category": b_job_row["role_category"],
            "experience_min": b_job_row["experience_min"],
            "experience_max": b_job_row["experience_max"],
            "skills": b_skills,
            "description": b_job_row["description"]
        }
    with col_b_cand:
        cohort_size = st.slider("Candidate Cohort Size to Screen:", 5, 25, 10)
        
    if st.button("⚡ Run Batch Screening & Rank Candidates", type="primary"):
        with st.spinner(f"Evaluating and ranking {cohort_size} candidate profiles..."):
            sample_candidates = resumes_df.sample(n=cohort_size, random_state=42).to_dict("records")
            # Parse skills for each
            parsed_list = []
            for c in sample_candidates:
                p = parser.parse_text(c["raw_text"])
                p["candidate_name"] = c["candidate_name"]
                p["email"] = c["email"]
                parsed_list.append(p)
                
            leaderboard = extensions.batch_score_resumes(parsed_list, batch_target_job)
            
        st.markdown(f"#### 🏆 Candidate Leaderboard for `{batch_target_job['title']}`")
        
        board_rows = []
        for item in leaderboard:
            board_rows.append({
                "Rank": f"#{item['rank']}",
                "Candidate": item["candidate_name"],
                "Email": item["email"],
                "Match Score": f"{item['overall_score']} / 100",
                "Skill Points": f"{item['skill_score']} / 40",
                "Experience Points": f"{item['experience_score']} / 25",
                "Missing Skills": item["missing_skills_count"],
                "Verdict": item["prediction"]
            })
            
        st.dataframe(pd.DataFrame(board_rows), use_container_width=True)
        top_cand = leaderboard[0]
        st.success(f"🥇 Top Recommended Candidate: **{top_cand['candidate_name']}** with a score of **{top_cand['overall_score']} / 100** ({top_cand['prediction']}).")
