"""
CareerCompass - Automated Test Suite
Verifies all deliverables and required components.
"""

import os
import unittest
import sqlite3
import pandas as pd
import json

from src.parser import ResumeParser
from src.matcher import MatchEngine
from src.semantic import SemanticMatcher
from src.retrieval import KnowledgeRetriever
from src.assistant import CareerAssistant
from src.extensions import CareerExtensions

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestCareerCompass(unittest.TestCase):
    def setUp(self):
        self.parser = ResumeParser()
        self.matcher = MatchEngine()
        self.semantic = SemanticMatcher()
        self.retriever = KnowledgeRetriever()
        self.assistant = CareerAssistant()
        self.extensions = CareerExtensions(self.matcher)
        self.db_path = os.path.join(BASE_DIR, "database", "careercompass.db")

    def test_01_dataset_counts(self):
        """Verify minimum 10,000 jobs and 500 resumes as required by Section 4."""
        jobs_path = os.path.join(BASE_DIR, "data", "processed", "jobs_cleaned.csv")
        resumes_path = os.path.join(BASE_DIR, "data", "processed", "resumes_cleaned.csv")
        
        self.assertTrue(os.path.exists(jobs_path), "Clean jobs CSV missing")
        self.assertTrue(os.path.exists(resumes_path), "Clean resumes CSV missing")
        
        jobs_df = pd.read_csv(jobs_path)
        resumes_df = pd.read_csv(resumes_path)
        
        self.assertGreaterEqual(len(jobs_df), 10000, f"Jobs count {len(jobs_df)} < 10,000")
        self.assertGreaterEqual(len(resumes_df), 500, f"Resumes count {len(resumes_df)} < 500")
        self.assertEqual(jobs_df["salary_min"].isna().sum(), 0, "Unresolved missing salary bands")

    def test_02_database_integrity(self):
        """Verify SQLite database schema and foreign key relations."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check tables
        tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        for t in ["companies", "jobs", "skills", "job_skills", "resumes", "resume_skills"]:
            self.assertIn(t, tables, f"Table {t} missing in database")
            
        # Check row counts
        job_count = cursor.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        self.assertGreaterEqual(job_count, 10000)
        
        # Test analytical queries execution
        queries_file = os.path.join(BASE_DIR, "database", "queries.sql")
        self.assertTrue(os.path.exists(queries_file))
        conn.close()

    def test_03_resume_parser(self):
        """Verify parsing of skills, experience, and education."""
        sample_text = """
        Jordan Miller
        jordan.m@synthetic-talent.org | (555) 123-4567
        Senior DevOps Engineer with 5 years of experience building Kubernetes clusters and AWS infrastructure.
        Skills: Docker, Kubernetes, AWS, Terraform, CI/CD, Linux, Python
        Education: B.S. in Computer Science, Stanford University (2021)
        """
        parsed = self.parser.parse_text(sample_text)
        self.assertIn("Docker", parsed["skills"])
        self.assertIn("Kubernetes", parsed["skills"])
        self.assertIn("AWS", parsed["skills"])
        self.assertEqual(parsed["years_experience"], 5)
        self.assertEqual(parsed["education_level"], "Bachelor")

    def test_04_semantic_comparison(self):
        """Verify TF-IDF vs dense embeddings on synonym mismatch."""
        resume_text = "Experience with K8s, PSQL, Golang, and Amazon Web Services."
        job_text = "Required skills: Kubernetes, PostgreSQL, Go, and AWS."
        
        comp = self.semantic.compare_methods(resume_text, job_text)
        self.assertIn("tfidf_similarity", comp)
        self.assertIn("dense_similarity", comp)
        self.assertGreater(comp["dense_similarity"], comp["tfidf_similarity"])
        self.assertGreater(comp["embedding_gain"], 0.2)

    def test_05_predictive_matching_and_breakdown(self):
        """Verify 0-100 score, 4-part visible breakdown, and ranked missing skills."""
        resume_data = {
            "skills": ["Python", "Docker", "Git"],
            "years_experience": 4,
            "education_level": "Bachelor",
            "target_role": "Backend Engineer",
            "raw_text": "Backend engineer with 4 years Python and Docker experience."
        }
        job_data = {
            "title": "Senior Backend Engineer",
            "role_category": "Backend Engineer",
            "skills": "Python, Docker, Kubernetes, AWS, PostgreSQL",
            "experience_min": 4,
            "experience_max": 7,
            "description": "Seeking Senior Backend Engineer proficient in Python, Docker, Kubernetes, AWS, and PostgreSQL."
        }
        
        res = self.matcher.calculate_match(resume_data, job_data)
        self.assertIn("overall_match_score", res)
        self.assertTrue(0.0 <= res["overall_match_score"] <= 100.0)
        
        bd = res["breakdown"]
        self.assertIn("skill_match", bd)
        self.assertIn("experience_fit", bd)
        self.assertIn("education_alignment", bd)
        self.assertIn("semantic_relevance", bd)
        
        # Check ranked missing skills
        self.assertGreater(len(res["ranked_missing_skills"]), 0)
        top_missing = res["ranked_missing_skills"][0]
        self.assertIn("skill", top_missing)
        self.assertIn("score_impact_points", top_missing)

    def test_06_rag_retrieval_and_improvement_plan(self):
        """Verify improvement plan citing specific resources."""
        plan = self.retriever.generate_improvement_plan(["Kubernetes", "AWS"], target_role="Cloud & DevOps Engineer")
        self.assertGreaterEqual(len(plan["phases"]), 4)
        self.assertGreater(len(plan["cited_sources"]), 0)
        self.assertTrue(any("kubernetes.io" in s.lower() or "cke" in s.lower() or "oreilly" in s.lower() for s in plan["cited_sources"]))

    def test_07_assistant_tool_calling(self):
        """Verify assistant executes tools against live database."""
        # Skill stats tool
        stats = self.assistant.tool_get_skill_stats("Docker")
        self.assertIn("postings_count", stats)
        self.assertGreater(stats["postings_count"], 1000)
        
        # Query jobs tool
        jobs = self.assistant.tool_query_jobs(role="DevOps", limit=3)
        self.assertEqual(jobs["count_found"], 3)

    def test_08_extensions(self):
        """Verify batch scoring, rewrites, and salary estimation."""
        cand_list = [
            {"candidate_name": "Alice", "skills": ["Python", "Docker"], "years_experience": 3, "education_level": "Bachelor"},
            {"candidate_name": "Bob", "skills": ["Python", "Docker", "AWS", "Kubernetes"], "years_experience": 6, "education_level": "Master"}
        ]
        target_job = {"title": "DevOps Engineer", "skills": ["Python", "Docker", "AWS", "Kubernetes"], "experience_min": 4, "experience_max": 7}
        
        leaderboard = self.extensions.batch_score_resumes(cand_list, target_job)
        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(leaderboard[0]["candidate_name"], "Bob")  # Bob has higher match
        
        sal = self.extensions.predict_expected_salary(cand_list[1], target_role="Cloud & DevOps Engineer")
        self.assertIn("estimated_median_salary", sal)

if __name__ == "__main__":
    unittest.main()
