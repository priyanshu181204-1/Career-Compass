-- CareerCompass: Relational Database Schema
-- Compatible with SQLite3 and PostgreSQL

DROP TABLE IF EXISTS resume_skills;
DROP TABLE IF EXISTS job_skills;
DROP TABLE IF EXISTS resumes;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS skills;
DROP TABLE IF EXISTS companies;

-- Companies Table
CREATE TABLE companies (
    company_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    industry TEXT,
    size_band TEXT,
    headquarters TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Skills Master Table
CREATE TABLE skills (
    skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    tier TEXT DEFAULT 'High'
);

-- Jobs Table
CREATE TABLE jobs (
    job_id TEXT PRIMARY KEY,
    company_id INTEGER NOT NULL,
    title TEXT NOT NULL,
    role_category TEXT NOT NULL,
    experience_min INTEGER NOT NULL,
    experience_max INTEGER NOT NULL,
    salary_min REAL NOT NULL,
    salary_max REAL NOT NULL,
    salary_avg REAL NOT NULL,
    location TEXT NOT NULL,
    is_remote BOOLEAN NOT NULL DEFAULT 0,
    salary_imputed BOOLEAN NOT NULL DEFAULT 0,
    posted_date DATE NOT NULL,
    description TEXT NOT NULL,
    FOREIGN KEY (company_id) REFERENCES companies (company_id) ON DELETE CASCADE
);

-- Job-Skill Relationship (Many-to-Many)
CREATE TABLE job_skills (
    job_id TEXT NOT NULL,
    skill_id INTEGER NOT NULL,
    is_required BOOLEAN DEFAULT 1,
    PRIMARY KEY (job_id, skill_id),
    FOREIGN KEY (job_id) REFERENCES jobs (job_id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills (skill_id) ON DELETE CASCADE
);

-- Resumes Table
CREATE TABLE resumes (
    resume_id TEXT PRIMARY KEY,
    candidate_name TEXT NOT NULL,
    email TEXT,
    phone TEXT,
    target_role TEXT NOT NULL,
    years_experience INTEGER NOT NULL,
    experience_level TEXT NOT NULL,
    education_level TEXT NOT NULL,
    degree TEXT,
    university TEXT,
    raw_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Resume-Skill Relationship (Many-to-Many)
CREATE TABLE resume_skills (
    resume_id TEXT NOT NULL,
    skill_id INTEGER NOT NULL,
    proficiency_level TEXT DEFAULT 'Competent',
    PRIMARY KEY (resume_id, skill_id),
    FOREIGN KEY (resume_id) REFERENCES resumes (resume_id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills (skill_id) ON DELETE CASCADE
);

-- Analytical Indexes
CREATE INDEX idx_jobs_role ON jobs (role_category);
CREATE INDEX idx_jobs_company ON jobs (company_id);
CREATE INDEX idx_jobs_salary ON jobs (salary_avg);
CREATE INDEX idx_jobs_remote ON jobs (is_remote);
CREATE INDEX idx_jobs_date ON jobs (posted_date);
CREATE INDEX idx_job_skills_skill ON job_skills (skill_id);
CREATE INDEX idx_resumes_role ON resumes (target_role);
