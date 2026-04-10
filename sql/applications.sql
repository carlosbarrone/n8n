CREATE TABLE applications (
  id SERIAL PRIMARY KEY,
  company TEXT,
  role TEXT,
  job_description TEXT,
  requirements TEXT,
  expected_salary TEXT,
  status TEXT,
  cv TEXT,
  cover_letter TEXT,
  answers JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);