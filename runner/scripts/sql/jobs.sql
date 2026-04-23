CREATE TABLE IF NOT EXISTS jobs (
  id BIGSERIAL PRIMARY KEY,
  company TEXT NOT NULL,
  role TEXT NOT NULL,
  job_description TEXT NOT NULL,
  required_skills TEXT[] NOT NULL DEFAULT '{}',
  required_experience_level TEXT,
  responsibilities TEXT[] NOT NULL DEFAULT '{}',
  required_education TEXT,
  required_certifications TEXT[] NOT NULL DEFAULT '{}',
  required_languages TEXT[] NOT NULL DEFAULT '{}',
  application_status TEXT NOT NULL DEFAULT 'saved' CHECK (
    application_status IN ('saved', 'applied', 'screening', 'interview', 'offer', 'rejected', 'withdrawn')
  ),
  applied_at TIMESTAMPTZ,
  expected_salary_min NUMERIC(12, 2),
  expected_salary_max NUMERIC(12, 2),
  process_steps TEXT[] NOT NULL DEFAULT '{}',
  submitted_resume_version TEXT,
  submitted_cover_letter TEXT,
  submitted_answers JSONB NOT NULL DEFAULT '{}'::jsonb,
  candidate_notes TEXT,
  last_status_update_at TIMESTAMPTZ,
  source_url TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CHECK (
    expected_salary_min IS NULL
    OR expected_salary_max IS NULL
    OR expected_salary_min <= expected_salary_max
  )
);