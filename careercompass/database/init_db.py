"""
CareerCompass - Database Initialization and Ingestion
Loads schema.sql, populates all tables from processed CSVs, and executes analytical queries.
"""

import sqlite3
import pandas as pd
import json
import os

BASE_DIR = r"C:\Users\priya\.gemini\antigravity\scratch\careercompass"
DB_PATH = os.path.join(BASE_DIR, "database", "careercompass.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "database", "schema.sql")
QUERIES_PATH = os.path.join(BASE_DIR, "database", "queries.sql")
JOBS_CSV = os.path.join(BASE_DIR, "data", "processed", "jobs_cleaned.csv")
RESUMES_CSV = os.path.join(BASE_DIR, "data", "processed", "resumes_cleaned.csv")
KB_PATH = os.path.join(BASE_DIR, "knowledge_base", "skills_taxonomy.json")

def init_database():
    print(f"Connecting to database at {DB_PATH}...")
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Execute Schema
    print(f"Executing schema from {SCHEMA_PATH}...")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    cursor.executescript(schema_sql)
    print("Schema created successfully.")
    
    # 2. Insert Skills from Taxonomy
    with open(KB_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
        
    skill_to_id = {}
    for cat_name, cat_skills in taxonomy.items():
        for skill_name, meta in cat_skills.items():
            cursor.execute(
                "INSERT INTO skills (name, category, tier) VALUES (?, ?, ?)",
                (skill_name, meta.get("category", cat_name), meta.get("tier", "High"))
            )
            skill_to_id[skill_name.lower()] = cursor.lastrowid
            
    # Also ensure common generic skills exist
    common_extras = [("Git", "Tools"), ("Agile", "Methodology"), ("Linux", "DevOps"), ("REST APIs", "Backend"), ("Unit Testing", "Testing"), ("Jira", "Tools"), ("Code Review", "Best Practices")]
    for s_name, s_cat in common_extras:
        if s_name.lower() not in skill_to_id:
            cursor.execute("INSERT INTO skills (name, category, tier) VALUES (?, ?, ?)", (s_name, s_cat, "Medium"))
            skill_to_id[s_name.lower()] = cursor.lastrowid
            
    conn.commit()
    print(f"Loaded {len(skill_to_id)} skills into master skills table.")
    
    # 3. Insert Companies and Jobs
    jobs_df = pd.read_csv(JOBS_CSV)
    print(f"Ingesting {len(jobs_df)} jobs and companies...")
    
    company_to_id = {}
    unique_companies = jobs_df[["company", "company_industry", "company_size"]].drop_duplicates()
    for _, comp in unique_companies.iterrows():
        cursor.execute(
            "INSERT INTO companies (name, industry, size_band) VALUES (?, ?, ?)",
            (comp["company"], comp["company_industry"], comp["company_size"])
        )
        company_to_id[comp["company"]] = cursor.lastrowid
        
    job_skills_to_insert = []
    
    for _, row in jobs_df.iterrows():
        c_id = company_to_id[row["company"]]
        cursor.execute(
            """INSERT INTO jobs (
                job_id, company_id, title, role_category, experience_min, experience_max,
                salary_min, salary_max, salary_avg, location, is_remote, salary_imputed,
                posted_date, description
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["job_id"], c_id, row["title"], row["role_category"],
                int(row["experience_min"]), int(row["experience_max"]),
                float(row["salary_min"]), float(row["salary_max"]), float(row["salary_avg"]),
                row["normalized_location"], 1 if row["is_remote"] else 0,
                1 if row["salary_imputed"] else 0, row["posted_date"], row["description"]
            )
        )
        
        # Link skills
        skills_str = str(row.get("skills_normalized", ""))
        for s in [x.strip() for x in skills_str.split(",") if x.strip()]:
            s_lower = s.lower()
            if s_lower in skill_to_id:
                job_skills_to_insert.append((row["job_id"], skill_to_id[s_lower], 1))
            else:
                # Insert ad-hoc skill
                cursor.execute("INSERT INTO skills (name, category, tier) VALUES (?, ?, ?)", (s, "General", "Medium"))
                new_id = cursor.lastrowid
                skill_to_id[s_lower] = new_id
                job_skills_to_insert.append((row["job_id"], new_id, 1))
                
    cursor.executemany("INSERT OR IGNORE INTO job_skills (job_id, skill_id, is_required) VALUES (?, ?, ?)", job_skills_to_insert)
    conn.commit()
    print(f"Ingested {len(jobs_df)} jobs and {len(job_skills_to_insert)} job-skill associations.")
    
    # 4. Insert Resumes
    resumes_df = pd.read_csv(RESUMES_CSV)
    print(f"Ingesting {len(resumes_df)} resumes...")
    
    resume_skills_to_insert = []
    for _, row in resumes_df.iterrows():
        cursor.execute(
            """INSERT INTO resumes (
                resume_id, candidate_name, email, phone, target_role,
                years_experience, experience_level, education_level, degree, university, raw_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["resume_id"], row["candidate_name"], row["email"], row["phone"],
                row["target_role"], int(row["years_experience"]), row["experience_level"],
                row["education_level"], row["degree"], row["university"], row["raw_text"]
            )
        )
        
        skills_str = str(row.get("skills_normalized", ""))
        for s in [x.strip() for x in skills_str.split(",") if x.strip()]:
            s_lower = s.lower()
            if s_lower in skill_to_id:
                resume_skills_to_insert.append((row["resume_id"], skill_to_id[s_lower], "Competent"))
            else:
                cursor.execute("INSERT INTO skills (name, category, tier) VALUES (?, ?, ?)", (s, "General", "Medium"))
                new_id = cursor.lastrowid
                skill_to_id[s_lower] = new_id
                resume_skills_to_insert.append((row["resume_id"], new_id, "Competent"))
                
    cursor.executemany("INSERT OR IGNORE INTO resume_skills (resume_id, skill_id, proficiency_level) VALUES (?, ?, ?)", resume_skills_to_insert)
    conn.commit()
    print(f"Ingested {len(resumes_df)} resumes and {len(resume_skills_to_insert)} resume-skill associations.")
    
    # 5. Validate Analytical Queries
    print("Testing analytical queries execution...")
    # Test Query 1 sample
    top_skills_test = cursor.execute("""
        SELECT j.role_category, s.name, COUNT(js.job_id) as freq
        FROM jobs j
        JOIN job_skills js ON j.job_id = js.job_id
        JOIN skills s ON js.skill_id = s.skill_id
        GROUP BY j.role_category, s.name
        ORDER BY freq DESC LIMIT 5
    """).fetchall()
    print(f"Top 5 demanded skills test: {top_skills_test}")
    
    # Test Query 2 sample
    top_hiring_test = cursor.execute("""
        SELECT c.name, COUNT(j.job_id) as total_jobs, ROUND(AVG(j.salary_avg), 0) as avg_salary
        FROM companies c JOIN jobs j ON c.company_id = j.company_id
        GROUP BY c.name ORDER BY total_jobs DESC LIMIT 3
    """).fetchall()
    print(f"Top 3 hiring companies test: {top_hiring_test}")
    
    conn.close()
    print("Database initialization and validation completed successfully.")

if __name__ == "__main__":
    init_database()
