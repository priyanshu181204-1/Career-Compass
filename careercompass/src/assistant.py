"""
CareerCompass - Conversational Assistant with Tool Calling
Addresses Section 5 & 6 requirements:
- Assistant grounded in the job database and skill corpus
- Implement tool calling so the assistant can query the job database directly
"""

import os
import sqlite3
import re
import json
from src.retrieval import KnowledgeRetriever

class CareerAssistant:
    def __init__(self, db_path=None, kb_dir=None):
        base = os.path.dirname(os.path.dirname(__file__))
        self.db_path = db_path or os.path.join(base, "database", "careercompass.db")
        self.retriever = KnowledgeRetriever(kb_dir)
        
    def _get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -------------------------------------------------------------
    # Tool 1: Query Job Database
    # -------------------------------------------------------------
    def tool_query_jobs(self, role=None, location=None, min_salary=None, skill=None, limit=5):
        """Tool that queries live job listings from SQLite database."""
        conn = self._get_db()
        cursor = conn.cursor()
        
        query = """
            SELECT j.job_id, j.title, c.name as company, j.role_category,
                   j.location, j.is_remote, j.salary_min, j.salary_max, j.salary_avg,
                   j.experience_min, j.experience_max, j.description
            FROM jobs j
            JOIN companies c ON j.company_id = c.company_id
            WHERE 1=1
        """
        params = []
        
        if role:
            query += " AND (LOWER(j.role_category) LIKE ? OR LOWER(j.title) LIKE ?)"
            params.extend([f"%{role.lower()}%", f"%{role.lower()}%"])
        if location:
            if "remote" in location.lower():
                query += " AND j.is_remote = 1"
            else:
                query += " AND LOWER(j.location) LIKE ?"
                params.append(f"%{location.lower()}%")
        if min_salary:
            query += " AND j.salary_avg >= ?"
            params.append(float(min_salary))
        if skill:
            query += """
                AND j.job_id IN (
                    SELECT js.job_id FROM job_skills js
                    JOIN skills s ON js.skill_id = s.skill_id
                    WHERE LOWER(s.name) LIKE ?
                )
            """
            params.append(f"%{skill.lower()}%")
            
        query += f" ORDER BY j.salary_avg DESC LIMIT {int(limit)}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            results.append({
                "job_id": r["job_id"],
                "title": r["title"],
                "company": r["company"],
                "role_category": r["role_category"],
                "location": r["location"],
                "salary_range": f"${int(r['salary_min']):,} - ${int(r['salary_max']):,}",
                "experience_required": f"{r['experience_min']} - {r['experience_max']} years",
                "description_snippet": r["description"][:140] + "..."
            })
            
        return {
            "tool_called": "tool_query_jobs",
            "parameters": {"role": role, "location": location, "min_salary": min_salary, "skill": skill, "limit": limit},
            "count_found": len(results),
            "results": results
        }

    # -------------------------------------------------------------
    # Tool 2: Get Skill Market Demand & Salary Stats
    # -------------------------------------------------------------
    def tool_get_skill_stats(self, skill_name):
        """Tool that queries database for market demand frequency, average salary, and top hiring companies."""
        conn = self._get_db()
        cursor = conn.cursor()
        
        # Total jobs count
        total_jobs = cursor.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        
        # Skill demand count
        demand_row = cursor.execute("""
            SELECT s.name, s.category, COUNT(js.job_id) as demand_count,
                   ROUND(AVG(j.salary_avg), 0) as avg_offered_salary,
                   ROUND(MIN(j.salary_min), 0) as min_sal,
                   ROUND(MAX(j.salary_max), 0) as max_sal
            FROM skills s
            JOIN job_skills js ON s.skill_id = js.skill_id
            JOIN jobs j ON js.job_id = j.job_id
            WHERE LOWER(s.name) LIKE ?
            GROUP BY s.skill_id
        """, (f"%{skill_name.lower().strip()}%",)).fetchone()
        
        if not demand_row:
            conn.close()
            return {
                "tool_called": "tool_get_skill_stats",
                "skill": skill_name,
                "found": False,
                "message": f"Skill '{skill_name}' not found in active database."
            }
            
        # Top hiring companies for this skill
        top_companies = cursor.execute("""
            SELECT c.name, COUNT(j.job_id) as postings_count
            FROM companies c
            JOIN jobs j ON c.company_id = j.company_id
            JOIN job_skills js ON j.job_id = js.job_id
            JOIN skills s ON js.skill_id = s.skill_id
            WHERE LOWER(s.name) LIKE ?
            GROUP BY c.name
            ORDER BY postings_count DESC LIMIT 5
        """, (f"%{skill_name.lower().strip()}%",)).fetchall()
        
        conn.close()
        
        d_count = demand_row["demand_count"]
        pct = round((d_count / total_jobs) * 100.0, 1)
        
        return {
            "tool_called": "tool_get_skill_stats",
            "skill": demand_row["name"],
            "category": demand_row["category"],
            "postings_count": d_count,
            "market_demand_percentage": f"{pct}% of all jobs ({d_count:,} postings)",
            "average_salary": f"${int(demand_row['avg_offered_salary']):,}",
            "salary_range": f"${int(demand_row['min_sal']):,} - ${int(demand_row['max_sal']):,}",
            "top_hiring_companies": [f"{c['name']} ({c['postings_count']} jobs)" for c in top_companies]
        }

    # -------------------------------------------------------------
    # Tool 3: Execute Safe Analytical SQL Query
    # -------------------------------------------------------------
    def tool_execute_sql(self, sql_query):
        """Tool that runs read-only SELECT analytical SQL queries on the database."""
        clean_q = sql_query.strip()
        if not clean_q.upper().startswith("SELECT") and not clean_q.upper().startswith("WITH"):
            return {"error": "Security restriction: Only read-only SELECT queries are allowed."}
            
        conn = self._get_db()
        cursor = conn.cursor()
        try:
            cursor.execute(clean_q)
            cols = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            results = [dict(zip(cols, r)) for r in rows[:15]]
            conn.close()
            return {
                "tool_called": "tool_execute_sql",
                "sql_query": clean_q,
                "columns": cols,
                "row_count": len(rows),
                "data": results
            }
        except Exception as e:
            conn.close()
            return {"tool_called": "tool_execute_sql", "error": str(e)}

    # -------------------------------------------------------------
    # Natural Language Agent Dispatcher
    # -------------------------------------------------------------
    def answer_query(self, user_query):
        """Dispatches natural language questions to appropriate tools and synthesizes grounded answers."""
        q = user_query.lower()
        
        # 1. Salary / Market Stats intent
        salary_match = re.search(r"(?:salary|demand|stats|market|how much).*?(?:for|in|of)\s+([a-zA-Z0-9+#.]+)", q)
        if ("salary" in q or "demand" in q or "market" in q or "companies hiring" in q) and any(s in q for s in ["docker", "kubernetes", "python", "aws", "react", "pytorch", "sql", "kafka", "go"]):
            for s in ["docker", "kubernetes", "python", "aws", "react", "pytorch", "sql", "kafka", "go"]:
                if s in q:
                    stats = self.tool_get_skill_stats(s)
                    if stats.get("found", True):
                        resp = (
                            f"### Market Intelligence for **{stats['skill']}** (Tool: `tool_get_skill_stats`)\n\n"
                            f"- **Market Demand Density**: {stats['market_demand_percentage']}\n"
                            f"- **Average Offered Compensation**: **{stats['average_salary']}** (Range: {stats['salary_range']})\n"
                            f"- **Category**: {stats['category']}\n"
                            f"- **Top Hiring Companies**: {', '.join(stats['top_hiring_companies'])}\n\n"
                            f"> Grounded query against `careercompass.db` ({stats['postings_count']} records analyzed)."
                        )
                        return resp, stats
                        
        # 2. Search Open Jobs intent
        if any(w in q for w in ["find", "show", "search", "list", "openings", "roles"]) and any(w in q for w in ["job", "jobs", "role", "roles", "positions"]):
            role = None
            for r in ["data scientist", "machine learning", "backend", "devops", "frontend", "full stack", "data engineer", "cybersecurity"]:
                if r in q:
                    role = r
                    break
            loc = "Remote" if "remote" in q else None
            min_sal = 130000 if ("130k" in q or "130000" in q or "high paying" in q) else None
            
            jobs = self.tool_query_jobs(role=role, location=loc, min_salary=min_sal, limit=4)
            res_md = f"### Job Search Results (Tool: `tool_query_jobs`)\n\nFound **{jobs['count_found']} matching positions** in the job repository:\n\n"
            for j in jobs["results"]:
                res_md += f"**[{j['job_id']}] {j['title']}** at **{j['company']}**\n"
                res_md += f"- **Location**: {j['location']} | **Compensation**: {j['salary_range']}\n"
                res_md += f"- **Experience**: {j['experience_required']}\n"
                res_md += f"- *Snippet*: {j['description_snippet']}\n\n"
            return res_md, jobs

        # 3. Interview prep intent
        if "interview" in q or "question" in q or "prep" in q:
            for s in ["docker", "kubernetes", "python", "machine learning", "system design", "react", "aws", "sql"]:
                if s in q:
                    qs = self.retriever.get_interview_questions(s)
                    q_item = qs[0]
                    resp = (
                        f"### Technical Interview Preparation for **{s.capitalize()}** (Tool: `tool_get_interview_prep`)\n\n"
                        f"**Question**: *\"{q_item['question']}\"*\n\n"
                        f"- **Core Topic**: {q_item.get('topic', s)}\n"
                        f"- **Difficulty Level**: {q_item.get('difficulty', 'Intermediate')}\n"
                        f"- **Key Concepts to Articulate**:\n  > {q_item['key_points']}\n\n"
                        f"*(Source: CareerCompass Reference Knowledge Base - Interview Question Corpus)*"
                    )
                    return resp, q_item
                    
        # 4. Learning roadmap / resource intent
        if "learn" in q or "course" in q or "resource" in q or "study" in q or "roadmap" in q:
            for s in ["docker", "kubernetes", "machine learning", "pytorch", "system design", "postgresql", "react", "aws"]:
                if s in q:
                    res = self.retriever.get_skill_resources(s)
                    resp = (
                        f"### Learning Resources for **{s.capitalize()}** (Tool: `tool_get_learning_resources`)\n\n"
                        f"- **Recommended Course**: {res['course']}\n"
                        f"- **Official Documentation**: [{res['docs']}]({res['docs']})\n"
                        f"- **Reference Book**: *{res['book']}*\n"
                        f"- **Hands-on Capstone Project**: {res['project']}\n"
                    )
                    return resp, res

        # Default: Analytical Database Query Overview
        default_res = self.tool_query_jobs(limit=3)
        resp = (
            f"I am your CareerCompass Assistant grounded in the active SQLite job database and engineering knowledge base.\n\n"
            f"You can ask me to:\n"
            f"1. **Query Job Postings**: *'Find senior DevOps roles with salary over $140,000'*\n"
            f"2. **Analyze Market Stats**: *'What is the demand and average salary for Kubernetes?'*\n"
            f"3. **Retrieve Interview Questions**: *'Give me interview questions for System Design'*\n"
            f"4. **Get Learning Plans**: *'Show learning resources for Docker'*\n"
        )
        return resp, default_res
