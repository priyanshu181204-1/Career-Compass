# CareerCompass: Resume Intelligence & Job Matching Platform
**Presentation Slide Deck (10 Slides)**
*Cohort: CSE VII, Batch 1 | Domain: Recruitment Technology and HR Analytics | Date: September 2026*

---

## Slide 1: Title & Overview
- **Project**: CP-01 CareerCompass — Resume Intelligence and Job Matching Platform
- **Institution**: Be Practical Tech Solutions & Sharda University, Greater Noida
- **Team**: Batch 1, Room 204, Block 3 (CSE VII)
- **Core Mission**: Consolidate machine learning on 10,600+ structured hiring records with an unstructured text retrieval layer (RAG) to provide transparent candidate match scoring, explainable gap ranking, and actionable career remediation.

---

## Slide 2: Problem Statement & Motivation
- **The Recruitment Asymmetry**:
  - Recruiters screen applications in seconds via opaque ATS parsers. Over 75% of resumes are discarded without explanation.
  - Candidates receive generic rejections with no guidance on how to close the qualification gap.
  - Lexical keyword mismatches (e.g. "K8s" vs "Kubernetes") create unnecessary false rejections.
- **CareerCompass Solution**:
  - Transparent 0–100 match score with a visible 4-part breakdown (Skills, Experience, Education, Semantics).
  - Missing skills ranked by their marginal contribution to the overall score.
  - Actionable 4-week remediation plan citing specific documentation, books, and courses.

---

## Slide 3: End-to-End System Architecture & Data Engineering
- **Primary Data**: 10,600+ clean job postings across 8 major engineering disciplines.
- **Secondary Data**: 550 synthetic resumes with structured metadata and raw text.
- **Knowledge Base**: Curated skill taxonomy (52 skills), role definition rubrics, technical interview Q&A, and reference learning resources.
- **Data Preparation Pipeline**:
  - *Deduplication*: Identified and removed 400 duplicate listings via MD5 content hashing.
  - *Location Normalization*: Standardized 25 messy variations into canonical City, State, and Remote indicators.
  - *Missing Salary Imputation*: Resolved 721 missing salary bands using conditional medians grouped by role category and seniority.

---

## Slide 4: Relational Database Schema & Analytical Queries
- **3NF Relational Structure**: SQLite / PostgreSQL compatible schema:
  - `companies` (company_id, name, industry, size_band, headquarters)
  - `jobs` (job_id, company_id, title, role_category, experience_min/max, salary_min/max/avg, location, is_remote)
  - `skills` (skill_id, name, category, tier)
  - `job_skills` (job_id, skill_id, is_required)
  - `resumes` & `resume_skills` (candidate records, education, experience, skills)
- **Analytical Queries (`queries.sql`)**:
  1. Top 10 demanded skills per role category with market penetration percentages.
  2. Hiring volume, remote share, and average offered salary by company and industry.
  3. Monthly hiring velocity and posting trends over time.
  4. Salary progression by role category across 4 experience tiers.
  5. Cross-skill co-occurrence matrix (e.g. Python + Docker pairings).

---

## Slide 5: Exploratory Data Analysis — Five Key Market Findings
1. **Power-Law Skill Concentration**: Top 6 skills (Docker 45.2%, Python 45.0%, AWS 37.8%, Kubernetes 28.9%, CI/CD 28.6%, SQL 26.6%) appear in over 78% of postings, whereas niche frameworks appear in under 6%.
2. **Experience Threshold Divergence**: Cloud/DevOps (median 5.0 yrs) and Cybersecurity (5.0 yrs) mandate significantly higher minimum tenure than Frontend (3.0 yrs) and Full Stack (3.0 yrs).
3. **Kubernetes & Cloud Salary Premium**: Postings requiring Kubernetes average **$182,229** vs $148,810 without (+22.5% statistically significant premium, p < 0.001).
4. **Remote Compensation Parity**: Remote roles (19.8% share) average **$158,824**, achieving wage parity with Tier-1 hubs (San Francisco $162k, New York $160k).
5. **The Experience-Skill Elasticity Tradeoff**: Candidates demonstrating multi-disciplinary skill breadth (6+ core skills) earn 18% higher compensation, compensating for lower raw tenure.

---

## Slide 6: Predictive Matching Model & Class Imbalance
- **Classification Task**: Predict whether a candidate-job pair is a `Strong Match` (1) vs `Weak/Non-Match` (0).
- **Engineered Features**:
  - `skill_overlap_ratio` (Jaccard index)
  - `experience_delta` (candidate exp - required min exp)
  - `experience_fit_score` (non-linear penalty for under/over qualification)
  - `education_fit_score` (ordinal degree alignment)
  - `semantic_similarity` (dense embedding cosine similarity)
  - `role_discipline_match` (binary indicator)
- **Class Imbalance & Results**:
  - Positive match skew: 19.5% strong matches vs 80.5% non-matches.
  - Handled via `class_weight='balanced'`.
  - Random Forest Classifier achieved **1.0000 F1-Score** and **1.0000 PR-AUC** (outperforming Logistic Regression baseline F1: 0.8187).
  - Feature importances: Skill Overlap (26.2%) and Semantic Similarity (24.2%) dominate decision boundaries.
- **Metric Defense**:
  - Recruiters prioritize **Precision** (minimizing false positive candidate screenings); candidates require **Recall** (avoiding false negative rejections).
  - For CareerCompass, **Balanced F1-Score** (harmonic mean of Precision & Recall) is the single most vital metric because it prevents both misleading endorsements and unfair exclusions.

---

## Slide 7: NLP Text Processing: TF-IDF vs. Dense Semantic Embeddings
- **The Vocabulary Mismatch Problem**:
  - Resume uses: *"Experienced with K8s, PSQL, Golang, Py, and Amazon Web Services."*
  - Job Description requires: *"Proficient in Kubernetes, PostgreSQL, Go, Python, and AWS."*
- **Empirical Experiment**:
  - **TF-IDF Baseline**: 0.0000 cosine similarity (complete false negative due to zero lexical token overlap).
  - **Dense Semantic Embeddings (LSA / SVD)**: 0.3816 cosine similarity (+0.3816 gain).
  - Captures conceptual synonyms and domain context where lexical matchers fail.

---

## Slide 8: Grounded RAG Retrieval & Autonomous Tool Calling
- **RAG Remediation Pipeline**:
  - Queries skill taxonomies, role definitions, and interview question corpus.
  - Synthesizes a structured 4-week roadmap:
    - *Week 1*: Core fundamentals and official docs citations.
    - *Week 2*: Guided coursework (CKA, Coursera, Udemy).
    - *Week 3*: Production portfolio capstone projects.
    - *Week 4*: Mock technical interview defense with key articulate points.
- **Autonomous Tool Calling**:
  - `tool_query_jobs(role, location, min_salary, skill)`
  - `tool_get_skill_stats(skill_name)`
  - `tool_get_interview_prep(skill_name)`
  - `tool_execute_sql(sql_query)`

---

## Slide 9: Live Application Demonstration
- **Streamlit Web Application (`app.py`)**:
  - *Tab 1*: Resume upload (PDF/DOCX/text), 0–100 radial score badge, 4-card breakdown, ranked missing skills, and STAR resume rewriting suggestions.
  - *Tab 2*: Job market dashboard with dynamic multi-select filters, boxplots, metro rankings, and velocity trends.
  - *Tab 3*: Conversational career assistant executing live tools against `careercompass.db`.
  - *Tab 4*: Batch candidate screening with sorted leaderboard and top candidate selection.

---

## Slide 10: Limitations & Future Enhancements
- **Limitations**:
  - Static skill taxonomy requires periodic ingestion of newly emerging tools (e.g. newly released LLM libraries).
  - Experience extraction relies on date ranges; non-traditional career paths require enhanced NLP.
- **Future Roadmap**:
  - Fine-tuned domain-specific LLM sentence embeddings.
  - Voice-driven mock interview simulator with real-time feedback.
  - Multi-tenant enterprise dashboard for corporate HR departments.
