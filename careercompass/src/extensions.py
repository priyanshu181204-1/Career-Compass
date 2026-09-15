"""
CareerCompass - Optional Extensions Module
Addresses Section 9:
1. Batch scoring of multiple resumes against one job posting
2. Structured resume rewriting suggestions (STAR method & ATS optimization)
3. Salary prediction integrated into candidate evaluation
"""

import os
import pandas as pd
from src.matcher import MatchEngine

class CareerExtensions:
    def __init__(self, match_engine=None):
        self.matcher = match_engine or MatchEngine()

    def batch_score_resumes(self, resumes_list, job_data):
        """Scores and ranks multiple candidate profiles against a single job posting."""
        leaderboard = []
        for idx, r in enumerate(resumes_list):
            res = self.matcher.calculate_match(r, job_data)
            leaderboard.append({
                "rank": 0,
                "candidate_name": r.get("candidate_name", f"Candidate #{idx+1}"),
                "email": r.get("email", "N/A"),
                "overall_score": res["overall_match_score"],
                "skill_score": res["breakdown"]["skill_match"]["score"],
                "experience_score": res["breakdown"]["experience_fit"]["score"],
                "prediction": res["classifier_prediction"],
                "missing_skills_count": len(res["breakdown"]["skill_match"]["missing_skills"]),
                "details": res
            })
            
        leaderboard.sort(key=lambda x: x["overall_score"], reverse=True)
        for i, item in enumerate(leaderboard):
            item["rank"] = i + 1
        return leaderboard

    def generate_rewrite_suggestions(self, resume_data, job_data):
        """
        Generates structured bullet-point rewrite suggestions incorporating
        missing keywords and quantifiable STAR impact metrics.
        """
        match_res = self.matcher.calculate_match(resume_data, job_data)
        missing_skills = match_res["breakdown"]["skill_match"]["missing_skills"]
        target_title = job_data.get("title", "Target Role")
        
        suggestions = []
        
        # Suggestion 1: Impact Quantification
        suggestions.append({
            "section": "Professional Experience (Metrics & Scale)",
            "current_generic_pattern": "Responsible for managing and maintaining backend services.",
            "recommended_rewrite": (
                f"Architected and maintained mission-critical services using {resume_data.get('skills', ['Python'])[0]}, "
                f"supporting 50,000+ daily active users and improving system throughput by 32%."
            ),
            "rationale": "Replaces passive duty description with active verb, quantified user scale, and metric improvement."
        })
        
        # Suggestion 2: Keyword integration for top missing skill
        if missing_skills:
            top_gap = missing_skills[0]
            suggestions.append({
                "section": f"Keyword Integration ({top_gap})",
                "current_generic_pattern": "Deployed code to cloud environment and automated build steps.",
                "recommended_rewrite": (
                    f"Automated containerized deployment pipelines using **{top_gap}** and CI/CD workflows, "
                    f"reducing release cycle duration from 45 minutes to under 8 minutes."
                ),
                "rationale": f"Explicitly incorporates '{top_gap}' from the job requirements within a high-impact engineering context."
            })
            
        # Suggestion 3: Actionable Summary
        suggestions.append({
            "section": "Executive Summary Tailoring",
            "current_generic_pattern": f"Software professional looking for challenging opportunities in tech.",
            "recommended_rewrite": (
                f"Results-driven {target_title} with {resume_data.get('years_experience', 3)} years of experience "
                f"specializing in {', '.join(resume_data.get('skills', ['Core Tech'])[:3])}. Proven track record "
                f"driving scalable systems and engineering reliability."
            ),
            "rationale": "Directly anchors candidate identity to the job title and top required competencies."
        })
        
        return suggestions

    def predict_expected_salary(self, resume_data, target_role="Software Engineer"):
        """Predicts estimated market compensation based on role, years of experience, and skills."""
        base_salaries = {
            "Data Scientist": (100000, 155000),
            "Machine Learning Engineer": (120000, 185000),
            "Backend Engineer": (105000, 165000),
            "Cloud & DevOps Engineer": (115000, 175000),
            "Frontend Engineer": (95000, 150000),
            "Full Stack Engineer": (105000, 160000),
            "Data Engineer": (110000, 170000),
            "Cybersecurity Engineer": (110000, 175000)
        }
        
        role_base = base_salaries.get(target_role, (100000, 160000))
        yrs = float(resume_data.get("years_experience", 2))
        
        # Experience multiplier
        exp_factor = 1.0 + min(1.2, yrs * 0.08)
        
        # Skill bonus
        skills = resume_data.get("skills", [])
        premium_skills = ["Kubernetes", "PyTorch", "AWS", "Kafka", "Large Language Models", "System Design"]
        bonus_count = sum(1 for s in skills if s in premium_skills)
        skill_factor = 1.0 + (bonus_count * 0.03)
        
        est_min = round(role_base[0] * exp_factor * skill_factor * 0.85, -3)
        est_max = round(role_base[1] * exp_factor * skill_factor * 0.95, -3)
        est_mid = round((est_min + est_max) / 2, -3)
        
        return {
            "target_role": target_role,
            "years_experience": yrs,
            "estimated_median_salary": f"${int(est_mid):,}",
            "estimated_salary_range": f"${int(est_min):,} - ${int(est_max):,}",
            "premium_skills_detected": [s for s in skills if s in premium_skills]
        }
