"""
CareerCompass - Match Engine and Scoring Module
Addresses Section 5 & 6 requirements:
- Match score out of 100 with visible breakdown of its calculation
- Sub-scores: Skills (40%), Experience (25%), Education (15%), Semantic Alignment (20%)
- Missing skills ranked by their effect on the score
- Classifier prediction for strong match
"""

import os
import joblib
import numpy as np
from src.semantic import SemanticMatcher

EDU_RANK = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}

class MatchEngine:
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(base_dir, "models", "match_classifier.joblib")
            
        self.model_data = joblib.load(model_path)
        self.classifier = self.model_data["model"]
        self.feature_names = self.model_data["feature_names"]
        self.semantic_matcher = SemanticMatcher()
        
    def calculate_match(self, resume_data, job_data):
        """
        Calculates match score out of 100, full breakdown, classifier prediction,
        and ranked missing skills.
        """
        # 1. Skill Extraction & Overlap
        r_skills = set([s.strip().lower() for s in resume_data.get("skills", [])])
        
        # Handle job skills input (can be string or list)
        j_skills_raw = job_data.get("skills", "")
        if isinstance(j_skills_raw, list):
            j_skills_list = [s.strip() for s in j_skills_raw if s.strip()]
        else:
            j_skills_list = [s.strip() for s in str(j_skills_raw).split(",") if s.strip()]
            
        j_skills_lower = set([s.lower() for s in j_skills_list])
        
        matched_skills = []
        missing_skills = []
        for s in j_skills_list:
            if s.lower() in r_skills:
                matched_skills.append(s)
            else:
                missing_skills.append(s)
                
        if j_skills_lower:
            skill_overlap_ratio = len(matched_skills) / len(j_skills_lower)
        else:
            skill_overlap_ratio = 0.5
            
        # Skill sub-score (0 to 40 pts)
        skill_score = round(skill_overlap_ratio * 40.0, 1)
        
        # 2. Experience Fit
        r_exp = float(resume_data.get("years_experience", 2))
        j_min_exp = float(job_data.get("experience_min", 2))
        j_max_exp = float(job_data.get("experience_max", max(j_min_exp + 2, 5)))
        exp_delta = r_exp - j_min_exp
        
        if r_exp < j_min_exp:
            exp_ratio = max(0.2, r_exp / max(1.0, j_min_exp))
            exp_fit_score = exp_ratio * 0.85
        elif r_exp <= j_max_exp + 3:
            exp_fit_score = 1.0
        else:
            exp_fit_score = max(0.75, 1.0 - 0.02 * (r_exp - (j_max_exp + 3)))
            
        # Experience sub-score (0 to 25 pts)
        experience_score = round(exp_fit_score * 25.0, 1)
        
        # 3. Education Fit
        r_edu = EDU_RANK.get(resume_data.get("education_level", "Bachelor"), 2)
        req_edu = 3 if j_min_exp >= 7 else 2
        edu_ratio = min(1.0, max(0.5, (r_edu + 1) / (req_edu + 1)))
        
        # Education sub-score (0 to 15 pts)
        education_score = round(edu_ratio * 15.0, 1)
        
        # 4. Semantic Text Similarity
        r_text = resume_data.get("raw_text", "")
        j_desc = job_data.get("description", "")
        if r_text and j_desc:
            semantic_sim = self.semantic_matcher.compute_dense_similarity(r_text, j_desc)
        else:
            semantic_sim = skill_overlap_ratio * 0.7 + 0.2
        semantic_sim = max(0.1, min(1.0, semantic_sim))
        
        # Semantic sub-score (0 to 20 pts)
        semantic_score = round(semantic_sim * 20.0, 1)
        
        # Total Match Score out of 100
        total_match_score = round(skill_score + experience_score + education_score + semantic_score, 1)
        total_match_score = max(5.0, min(100.0, total_match_score))
        
        # 5. Classifier Model Prediction
        r_role = str(resume_data.get("target_role", "")).lower()
        j_role = str(job_data.get("role_category", "")).lower()
        role_align = 1.0 if (r_role and j_role and r_role == j_role) else 0.5
        
        feat_vector = np.array([[
            skill_overlap_ratio,
            exp_delta,
            exp_fit_score,
            edu_ratio,
            semantic_sim,
            role_align,
            len(missing_skills)
        ]])
        
        is_strong_prob = float(self.classifier.predict_proba(feat_vector)[0][1])
        classifier_prediction = "Strong Match" if is_strong_prob >= 0.50 else "Not a Strong Match"
        
        # 6. Rank Missing Skills by Impact on Score
        ranked_missing = []
        base_missing_count = len(missing_skills)
        for s in missing_skills:
            # Marginal gain on skill score if candidate had this skill
            simulated_matched = len(matched_skills) + 1
            simulated_overlap = simulated_matched / max(1, len(j_skills_list))
            simulated_skill_score = simulated_overlap * 40.0
            point_gain = round(simulated_skill_score - skill_score, 1)
            ranked_missing.append({
                "skill": s,
                "score_impact_points": max(1.0, point_gain),
                "importance": "Critical" if point_gain >= 5.0 else "High" if point_gain >= 3.0 else "Medium"
            })
            
        ranked_missing.sort(key=lambda x: x["score_impact_points"], reverse=True)
        
        return {
            "overall_match_score": total_match_score,
            "classifier_prediction": classifier_prediction,
            "strong_match_probability": round(is_strong_prob * 100, 1),
            "breakdown": {
                "skill_match": {
                    "score": skill_score,
                    "max_points": 40,
                    "percentage": round((skill_score / 40.0) * 100, 1),
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills
                },
                "experience_fit": {
                    "score": experience_score,
                    "max_points": 25,
                    "percentage": round((experience_score / 25.0) * 100, 1),
                    "candidate_years": r_exp,
                    "required_years": f"{int(j_min_exp)} - {int(j_max_exp)} years"
                },
                "education_alignment": {
                    "score": education_score,
                    "max_points": 15,
                    "percentage": round((education_score / 15.0) * 100, 1),
                    "candidate_level": resume_data.get("education_level", "Bachelor")
                },
                "semantic_relevance": {
                    "score": semantic_score,
                    "max_points": 20,
                    "percentage": round((semantic_score / 20.0) * 100, 1),
                    "raw_similarity": round(semantic_sim, 4)
                }
            },
            "ranked_missing_skills": ranked_missing
        }
