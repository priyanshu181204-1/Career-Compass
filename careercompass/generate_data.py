import json
import random
import os
import csv
from datetime import datetime, timedelta

random.seed(42)

BASE_DIR = r"C:\Users\priya\.gemini\antigravity\scratch\careercompass"
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
KB_DIR = os.path.join(BASE_DIR, "knowledge_base")
SAMPLE_RESUME_DIR = os.path.join(BASE_DIR, "data", "resumes_sample")

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(SAMPLE_RESUME_DIR, exist_ok=True)

# Load existing KB
with open(os.path.join(KB_DIR, "role_definitions.json"), "r") as f:
    ROLE_DEFINITIONS = json.load(f)

COMPANIES = [
    ("Google", "Technology", "Enterprise", "Mountain View, CA"),
    ("Microsoft", "Technology", "Enterprise", "Redmond, WA"),
    ("Amazon", "E-commerce & Cloud", "Enterprise", "Seattle, WA"),
    ("Meta", "Social Media & AI", "Enterprise", "Menlo Park, CA"),
    ("Apple", "Consumer Electronics", "Enterprise", "Cupertino, CA"),
    ("Netflix", "Entertainment & Streaming", "Enterprise", "Los Gatos, CA"),
    ("Uber", "Mobility & Logistics", "Growth", "San Francisco, CA"),
    ("Airbnb", "Travel & Hospitality", "Growth", "San Francisco, CA"),
    ("Stripe", "Fintech & Payments", "Growth", "South San Francisco, CA"),
    ("Databricks", "Data & AI Platforms", "Growth", "San Francisco, CA"),
    ("Snowflake", "Cloud Data Warehouse", "Enterprise", "Bozeman, MT"),
    ("Palantir", "Big Data Analytics", "Enterprise", "Denver, CO"),
    ("Twilio", "Communications API", "Growth", "San Francisco, CA"),
    ("Shopify", "E-commerce Platform", "Enterprise", "Ottawa, Canada"),
    ("Salesforce", "Enterprise CRM", "Enterprise", "San Francisco, CA"),
    ("Atlassian", "Developer Tools", "Enterprise", "Sydney, Australia"),
    ("Cloudflare", "Edge Security & CDN", "Growth", "San Francisco, CA"),
    ("Coinbase", "Cryptocurrency & Web3", "Growth", "Remote - US"),
    ("DoorDash", "On-Demand Delivery", "Growth", "San Francisco, CA"),
    ("Spotify", "Audio Streaming", "Enterprise", "New York, NY"),
    ("Adobe", "Creative Software", "Enterprise", "San Jose, CA"),
    ("Oracle", "Cloud Infrastructure & DB", "Enterprise", "Austin, TX"),
    ("Cisco", "Networking & Hardware", "Enterprise", "San Jose, CA"),
    ("Intel", "Semiconductors", "Enterprise", "Santa Clara, CA"),
    ("NVIDIA", "GPU & Accelerated Computing", "Enterprise", "Santa Clara, CA"),
    ("CrowdStrike", "Cybersecurity", "Growth", "Austin, TX"),
    ("Zscaler", "Zero Trust Cloud Security", "Growth", "San Jose, CA"),
    ("Palo Alto Networks", "Cybersecurity", "Enterprise", "Santa Clara, CA"),
    ("Pinterest", "Social Discovery", "Growth", "San Francisco, CA"),
    ("Reddit", "Community Platform", "Growth", "San Francisco, CA"),
    ("Robinhood", "Fintech & Trading", "Growth", "Menlo Park, CA"),
    ("Instacart", "Grocery Delivery", "Growth", "San Francisco, CA"),
    ("Wayfair", "E-commerce Furniture", "Enterprise", "Boston, MA"),
    ("HubSpot", "Inbound Marketing CRM", "Growth", "Cambridge, MA"),
    ("Square / Block", "Payments & Financial Services", "Enterprise", "Oakland, CA"),
    ("Lyft", "Ridesharing", "Growth", "San Francisco, CA"),
    ("ServiceNow", "Enterprise Workflow Automation", "Enterprise", "Santa Clara, CA"),
    ("Workday", "Enterprise Cloud HR/Finance", "Enterprise", "Pleasanton, CA"),
    ("Splunk", "Data Observability", "Enterprise", "San Francisco, CA"),
    ("MongoDB Inc.", "Database Systems", "Growth", "New York, NY")
]

RAW_LOCATIONS = [
    "New York, NY", "NYC", "New York City, New York", "Manhattan, NY",
    "San Francisco, CA", "SF, California", "San Francisco Bay Area", "Silicon Valley, CA",
    "Seattle, WA", "Seattle, Washington",
    "Austin, TX", "Austin, Texas", "ATX",
    "Boston, MA", "Boston, Massachusetts",
    "Chicago, IL", "Chicago, Illinois",
    "Remote", "Remote - US", "Fully Remote", "Remote (United States)", "US Remote / Anywhere",
    "Los Angeles, CA", "LA, California",
    "Denver, CO", "Denver, Colorado",
    "Atlanta, GA", "Atlanta, Georgia"
]

TITLES_BY_ROLE = {
    "Data Scientist": ["Data Scientist", "Associate Data Scientist", "Senior Data Scientist", "Staff Data Scientist", "Principal Data Scientist", "Lead Data Scientist", "Quantitative Researcher - Data Science"],
    "Machine Learning Engineer": ["Machine Learning Engineer", "Junior ML Engineer", "Senior Machine Learning Engineer", "Staff MLOps Engineer", "AI Research Engineer", "Deep Learning Engineer", "Lead ML Systems Engineer"],
    "Backend Engineer": ["Backend Software Engineer", "Software Engineer - Backend", "Senior Backend Engineer", "Staff Software Engineer - Infrastructure", "Distributed Systems Engineer", "API Platform Engineer", "Lead Backend Architect"],
    "Cloud & DevOps Engineer": ["DevOps Engineer", "Cloud Infrastructure Engineer", "Senior Site Reliability Engineer (SRE)", "Platform Engineer", "Staff DevOps Architect", "Cloud Security Engineer", "Lead Infrastructure Engineer"],
    "Frontend Engineer": ["Frontend Engineer", "UI/UX Software Engineer", "Senior Frontend Engineer", "Staff Web Engineer", "Lead React Developer", "Client Applications Engineer"],
    "Full Stack Engineer": ["Full Stack Software Engineer", "Senior Full Stack Engineer", "Staff Software Engineer", "Product Engineer", "Lead Full Stack Developer", "Founding Software Engineer"],
    "Data Engineer": ["Data Engineer", "Senior Data Engineer", "Staff Data Platform Engineer", "Big Data Engineer", "Analytics Engineer", "Lead ETL Pipeline Engineer"],
    "Cybersecurity Engineer": ["Information Security Engineer", "Security Operations Engineer", "Senior Cloud Security Engineer", "Application Security Specialist", "Threat Intelligence Analyst", "Lead Cyber Defense Engineer"]
}

TEAMS = ["Infrastructure Team", "Growth Platform", "Core AI Org", "Identity & Auth", "Checkout Systems", "Search & Discovery", "Data Platform", "Security Operations", "Customer Experience", "Platform Reliability", "Payments Engine", "Developer Experience"]

def generate_job_postings(num_unique=10600, num_duplicates=400):
    jobs = []
    base_date = datetime(2025, 9, 1)
    
    for i in range(1, num_unique + 1):
        job_id = f"JOB-{i:06d}"
        role_cat = random.choice(list(ROLE_DEFINITIONS.keys()))
        role_info = ROLE_DEFINITIONS[role_cat]
        title = random.choice(TITLES_BY_ROLE[role_cat])
        company_info = random.choice(COMPANIES)
        company_name = company_info[0]
        team_name = random.choice(TEAMS)
        
        if any(w in title for w in ["Junior", "Associate"]):
            exp_min = random.randint(0, 1)
            exp_max = exp_min + random.randint(1, 2)
            sal_mult = 0.8
        elif any(w in title for w in ["Senior"]):
            exp_min = random.randint(5, 7)
            exp_max = exp_min + random.randint(2, 4)
            sal_mult = 1.3
        elif any(w in title for w in ["Staff", "Principal", "Lead", "Architect"]):
            exp_min = random.randint(8, 11)
            exp_max = exp_min + random.randint(3, 5)
            sal_mult = 1.65
        else:
            exp_min = random.randint(2, 4)
            exp_max = exp_min + random.randint(2, 3)
            sal_mult = 1.05

        core = role_info["core_skills"]
        secondary = role_info["secondary_skills"]
        num_core = random.randint(3, len(core))
        num_sec = random.randint(1, min(3, len(secondary)))
        chosen_skills = list(set(random.sample(core, num_core) + random.sample(secondary, num_sec)))
        random.shuffle(chosen_skills)
        skills_str = ", ".join(chosen_skills)
        
        # ~7% missing salary
        if random.random() < 0.07:
            salary_min = None
            salary_max = None
        else:
            base_min = int(role_info["salary_range"]["min"] * sal_mult * random.uniform(0.92, 1.08))
            base_max = int(role_info["salary_range"]["max"] * sal_mult * random.uniform(0.92, 1.08))
            salary_min = round(base_min, -3)
            salary_max = round(base_max, -3)
        
        raw_location = random.choice(RAW_LOCATIONS)
        description = (
            f"Job Req #{job_id}: {company_name} is seeking a skilled {title} to join our {team_name}. "
            f"Key responsibilities include designing scalable systems, participating in technical reviews, and driving product impact. "
            f"Required technical competencies: {skills_str}. Experience required: {exp_min} to {exp_max} years. "
            f"We offer competitive compensation, comprehensive health benefits, and flexible work options."
        )
        post_date = base_date + timedelta(days=random.randint(0, 365))
        
        jobs.append({
            "job_id": job_id,
            "title": title,
            "company": company_name,
            "company_industry": company_info[1],
            "company_size": company_info[2],
            "location_raw": raw_location,
            "role_category": role_cat,
            "experience_min": exp_min,
            "experience_max": exp_max,
            "salary_min": salary_min,
            "salary_max": salary_max,
            "skills": skills_str,
            "posted_date": post_date.strftime("%Y-%m-%d"),
            "description": description
        })
    
    # Deliberate duplicate listings (exact copy with same job content)
    for d in range(num_duplicates):
        orig = random.choice(jobs[:3000])
        dup = orig.copy()
        dup["job_id"] = f"JOB-DUP-{d+1:04d}"
        jobs.append(dup)
        
    return jobs

jobs_data = generate_job_postings(10600, 400)
csv_path = os.path.join(DATA_RAW_DIR, "jobs_raw.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=jobs_data[0].keys())
    writer.writeheader()
    writer.writerows(jobs_data)

print(f"Generated {len(jobs_data)} job postings in {csv_path} (including 400 intentional duplicates).")
