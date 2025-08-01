import random
import pandas as pd
import numpy as np


random.seed(42)
np.random.seed(42)

skill_pool = ['Python', 'SQL', 'Java', 'C++', 'Excel', 'Tableau', 'AWS', 'Linux', 'Docker', 'Spark']
education_levels = ['High School', "Associate's", "Bachelor's", "Master's", "PhD"]

n_samples = 500

data = []

for _ in range(n_samples):
    candidate_skills = random.sample(skill_pool, k=random.randint(2, 6))
    candidate_exp = np.random.randint(0, 15)  # years experience
    candidate_edu = random.choices(education_levels, weights=[5, 10, 30, 40, 15])[0]

    job_skills = random.sample(skill_pool, k=random.randint(2, 5))
    job_min_exp = np.random.randint(0, 10)
    job_req_edu = random.choices(education_levels, weights=[10, 15, 35, 30, 10])[0]

    skill_overlap = len(set(candidate_skills).intersection(set(job_skills)))

    edu_match = int(education_levels.index(candidate_edu) >= education_levels.index(job_req_edu))

    exp_gap = candidate_exp - job_min_exp

    label = int((skill_overlap >= 2) and (exp_gap >= 0) and (edu_match == 1))

    data.append({
        'candidate_skills': candidate_skills,
        'candidate_exp': candidate_exp,
        'candidate_edu': candidate_edu,
        'job_skills': job_skills,
        'job_min_exp': job_min_exp,
        'job_req_edu': job_req_edu,
        'skill_overlap': skill_overlap,
        'edu_match': edu_match,
        'exp_gap': exp_gap,
        'fit_label': label
    })

df = pd.DataFrame(data)

df.to_csv('dummy_job_fit_data.csv', index=False)

print(df.head())
