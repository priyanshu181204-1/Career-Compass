-- CareerCompass: Analytical SQL Queries

-- 1. Top 10 Most Demanded Skills per Role Category
-- Calculates frequency and percentage of total job postings in each role
WITH RoleTotals AS (
    SELECT role_category, COUNT(*) AS total_jobs
    FROM jobs
    GROUP BY role_category
),
SkillCounts AS (
    SELECT 
        j.role_category,
        s.name AS skill_name,
        s.category AS skill_category,
        COUNT(js.job_id) AS demand_count,
        ROUND(COUNT(js.job_id) * 100.0 / rt.total_jobs, 1) AS demand_pct,
        DENSE_RANK() OVER (PARTITION BY j.role_category ORDER BY COUNT(js.job_id) DESC) as rank_in_role
    FROM jobs j
    JOIN job_skills js ON j.job_id = js.job_id
    JOIN skills s ON js.skill_id = s.skill_id
    JOIN RoleTotals rt ON j.role_category = rt.role_category
    GROUP BY j.role_category, s.name, s.category, rt.total_jobs
)
SELECT role_category, rank_in_role, skill_name, skill_category, demand_count, demand_pct
FROM SkillCounts
WHERE rank_in_role <= 10
ORDER BY role_category, rank_in_role;

-- 2. Hiring Volume by Company and Industry
-- Identifies the top hiring organizations, average compensation, and remote ratio
SELECT 
    c.name AS company_name,
    c.industry,
    c.size_band,
    COUNT(j.job_id) AS total_postings,
    ROUND(AVG(j.salary_avg), 0) AS avg_offered_salary,
    ROUND(SUM(CASE WHEN j.is_remote = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(j.job_id), 1) AS remote_pct
FROM companies c
JOIN jobs j ON c.company_id = j.company_id
GROUP BY c.company_id, c.name, c.industry, c.size_band
ORDER BY total_postings DESC;

-- 3. Job Posting Trends and Hiring Velocity Over Time
-- Aggregates monthly posting volume, distinct active roles, and average salary trends
SELECT 
    strftime('%Y-%m', posted_date) AS posting_month,
    COUNT(job_id) AS monthly_postings,
    COUNT(DISTINCT role_category) AS distinct_roles_active,
    ROUND(AVG(salary_avg), 0) AS avg_salary_trend,
    ROUND(SUM(CASE WHEN is_remote = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(job_id), 1) AS remote_share_pct
FROM jobs
GROUP BY posting_month
ORDER BY posting_month ASC;

-- 4. Salary Bands by Role Category and Experience Level
-- Shows compensation progression from Junior to Lead roles
SELECT 
    role_category,
    CASE 
        WHEN experience_min <= 2 THEN 'Junior (0-2 yrs)'
        WHEN experience_min <= 5 THEN 'Mid-Level (3-5 yrs)'
        WHEN experience_min <= 8 THEN 'Senior (6-8 yrs)'
        ELSE 'Lead / Staff (9+ yrs)'
    END AS experience_bracket,
    COUNT(job_id) AS sample_size,
    ROUND(MIN(salary_min), 0) AS min_starting_salary,
    ROUND(AVG(salary_avg), 0) AS avg_salary,
    ROUND(MAX(salary_max), 0) AS max_ceiling_salary
FROM jobs
GROUP BY role_category, experience_bracket
ORDER BY role_category, AVG(salary_avg) ASC;

-- 5. Cross-Skill Co-occurrence: Top Paired Skills
-- Computes the most frequent technology pairings in job descriptions
SELECT 
    s1.name AS skill_a,
    s2.name AS skill_b,
    COUNT(*) AS co_occurrence_count
FROM job_skills js1
JOIN job_skills js2 ON js1.job_id = js2.job_id AND js1.skill_id < js2.skill_id
JOIN skills s1 ON js1.skill_id = s1.skill_id
JOIN skills s2 ON js2.skill_id = s2.skill_id
GROUP BY s1.name, s2.name
ORDER BY co_occurrence_count DESC
LIMIT 20;
