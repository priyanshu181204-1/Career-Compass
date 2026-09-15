"""
CareerCompass - Data Preparation and Cleaning Pipeline
Addresses Section 5 requirements:
1. Deduplication of job listings (removes duplicate postings based on exact content hash)
2. Inconsistent location string normalization (maps 20+ variations to standard city, state, remote flags)
3. Missing salary band imputation using conditional medians by role & seniority
4. Skill normalization and parsing against taxonomy
"""

import os
import json
import pandas as pd
import numpy as np
import hashlib

def load_taxonomy(kb_path):
    with open(kb_path, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    synonym_map = {}
    for cat, skills in taxonomy.items():
        for canon, meta in skills.items():
            synonym_map[canon.lower()] = canon
            for syn in meta.get("synonyms", []):
                synonym_map[syn.lower()] = canon
    return synonym_map, taxonomy

LOCATION_MAP = {
    "nyc": ("New York", "NY", False),
    "new york, ny": ("New York", "NY", False),
    "new york city, new york": ("New York", "NY", False),
    "manhattan, ny": ("New York", "NY", False),
    "san francisco, ca": ("San Francisco", "CA", False),
    "sf, california": ("San Francisco", "CA", False),
    "san francisco bay area": ("San Francisco", "CA", False),
    "silicon valley, ca": ("San Jose", "CA", False),
    "seattle, wa": ("Seattle", "WA", False),
    "seattle, washington": ("Seattle", "WA", False),
    "austin, tx": ("Austin", "TX", False),
    "austin, texas": ("Austin", "TX", False),
    "atx": ("Austin", "TX", False),
    "boston, ma": ("Boston", "MA", False),
    "boston, massachusetts": ("Boston", "MA", False),
    "chicago, il": ("Chicago", "IL", False),
    "chicago, illinois": ("Chicago", "IL", False),
    "remote": ("Remote", "US", True),
    "remote - us": ("Remote", "US", True),
    "fully remote": ("Remote", "US", True),
    "remote (united states)": ("Remote", "US", True),
    "us remote / anywhere": ("Remote", "US", True),
    "los angeles, ca": ("Los Angeles", "CA", False),
    "la, california": ("Los Angeles", "CA", False),
    "denver, co": ("Denver", "CO", False),
    "denver, colorado": ("Denver", "CO", False),
    "atlanta, ga": ("Atlanta", "GA", False),
    "atlanta, georgia": ("Atlanta", "GA", False)
}

def normalize_location(raw_loc):
    if not isinstance(raw_loc, str):
        return "Unknown", "US", False
    norm_key = raw_loc.strip().lower()
    if norm_key in LOCATION_MAP:
        return LOCATION_MAP[norm_key]
    
    if "remote" in norm_key:
        return "Remote", "US", True
    return raw_loc.strip(), "US", False

def clean_jobs(raw_csv_path, output_csv_path, kb_path):
    print(f"Loading raw jobs from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    initial_count = len(df)
    print(f"Initial raw jobs count: {initial_count}")
    
    # 1. Deduplication using full content hash
    df["content_hash"] = df.apply(
        lambda r: hashlib.md5(f"{str(r['company']).strip().lower()}_{str(r['title']).strip().lower()}_{str(r['description']).strip()}_{str(r['skills']).strip()}".encode()).hexdigest(),
        axis=1
    )
    dup_count = df.duplicated(subset=["content_hash"]).sum()
    df = df.drop_duplicates(subset=["content_hash"]).reset_index(drop=True)
    print(f"Deduplication complete: Identified and removed {dup_count} duplicate listings ({len(df)} remaining unique listings).")
    
    # 2. Location Normalization
    loc_tuples = [normalize_location(l) for l in df["location_raw"]]
    df["location_city"] = [t[0] for t in loc_tuples]
    df["location_state"] = [t[1] for t in loc_tuples]
    df["is_remote"] = [t[2] for t in loc_tuples]
    df["normalized_location"] = df.apply(
        lambda r: "Remote - US" if r["is_remote"] else f"{r['location_city']}, {r['location_state']}",
        axis=1
    )
    print("Location normalization complete.")
    
    # 3. Missing Salary Bands Imputation
    missing_sal_count = df["salary_min"].isna().sum()
    df["salary_imputed"] = df["salary_min"].isna()
    
    df["exp_tier"] = pd.cut(
        df["experience_min"],
        bins=[-1, 2, 5, 8, 100],
        labels=["Junior", "Mid", "Senior", "Lead"]
    )
    
    grouped_medians = df.groupby(["role_category", "exp_tier"], observed=False)[["salary_min", "salary_max"]].median()
    
    def impute_salary(row, col):
        if pd.notna(row[col]):
            return float(row[col])
        rc = row["role_category"]
        et = row["exp_tier"]
        try:
            val = grouped_medians.loc[(rc, et), col]
            if pd.isna(val):
                val = df[col].median()
        except KeyError:
            val = df[col].median()
        return round(float(val), -3)
        
    df["salary_min"] = df.apply(lambda r: impute_salary(r, "salary_min"), axis=1)
    df["salary_max"] = df.apply(lambda r: impute_salary(r, "salary_max"), axis=1)
    df["salary_avg"] = (df["salary_min"] + df["salary_max"]) / 2.0
    print(f"Salary imputation complete: Imputed {missing_sal_count} missing salary records using conditional medians.")
    
    # 4. Skill Standardization
    synonym_map, _ = load_taxonomy(kb_path)
    
    def standardize_skills(skills_str):
        if not isinstance(skills_str, str):
            return ""
        items = [s.strip() for s in skills_str.split(",") if s.strip()]
        std = []
        for s in items:
            canon = synonym_map.get(s.lower(), s)
            if canon not in std:
                std.append(canon)
        return ", ".join(std)
        
    df["skills_normalized"] = df["skills"].apply(standardize_skills)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"Cleaned job listings ({len(df)} rows) successfully saved to {output_csv_path}")
    return df

def clean_resumes(raw_csv_path, output_csv_path, kb_path):
    print(f"Loading raw resumes from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    synonym_map, _ = load_taxonomy(kb_path)
    
    def standardize_skills(skills_str):
        if not isinstance(skills_str, str):
            return ""
        items = [s.strip() for s in skills_str.split(",") if s.strip()]
        std = []
        for s in items:
            canon = synonym_map.get(s.lower(), s)
            if canon not in std:
                std.append(canon)
        return ", ".join(std)
        
    df["skills_normalized"] = df["skills"].apply(standardize_skills)
    df.to_csv(output_csv_path, index=False)
    print(f"Cleaned resumes ({len(df)} rows) saved to {output_csv_path}")
    return df

if __name__ == "__main__":
    base = r"C:\Users\priya\.gemini\antigravity\scratch\careercompass"
    raw_jobs = os.path.join(base, "data", "raw", "jobs_raw.csv")
    clean_jobs_out = os.path.join(base, "data", "processed", "jobs_cleaned.csv")
    raw_res = os.path.join(base, "data", "raw", "resumes.csv")
    clean_res_out = os.path.join(base, "data", "processed", "resumes_cleaned.csv")
    kb = os.path.join(base, "knowledge_base", "skills_taxonomy.json")
    
    clean_jobs(raw_jobs, clean_jobs_out, kb)
    clean_resumes(raw_res, clean_res_out, kb)
