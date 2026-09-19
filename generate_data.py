"""
Synthetic dataset generator for VU-style distance learning dropout risk.
Feature choices are grounded in known distance-education dropout research:
- LMS engagement (logins, video watch %) -> strongest predictor in literature
- Assignment/quiz timeliness and scores -> academic engagement signal
- GPA trend (declining vs improving) -> strong early warning signal
- Forum/discussion participation -> social integration proxy
- Session attendance -> commitment signal
- Part-time job / semester load -> known external stress factors
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 3000

def clip(x, lo, hi):
    return np.clip(x, lo, hi)

# Base "engagement latent factor" per student - drives most other features
engagement = np.random.beta(2, 2, N)  # 0 (disengaged) to 1 (highly engaged)

weekly_lms_logins = clip(np.random.normal(2 + 10 * engagement, 2.5, N), 0, 21)
video_watch_pct = clip(np.random.normal(20 + 70 * engagement, 15, N), 0, 100)
avg_quiz_score = clip(np.random.normal(35 + 55 * engagement, 12, N), 0, 100)
assignment_completion_rate = clip(np.random.normal(30 + 65 * engagement, 15, N), 0, 100)
avg_submission_delay_days = clip(np.random.normal(5 - 4.5 * engagement, 2, N), 0, 10)
forum_posts_per_month = clip(np.random.poisson(1 + 6 * engagement, N), 0, 30)
session_attendance_pct = clip(np.random.normal(25 + 65 * engagement, 18, N), 0, 100)

# GPA trend: declining engagement -> declining GPA
gpa_trend = np.random.normal(-1.2 + 2.4 * engagement, 0.6, N)  # negative = declining
current_gpa = clip(np.random.normal(2.0 + 1.8 * engagement, 0.5, N), 0, 4)

part_time_job = np.random.binomial(1, 0.4, N)
semester = np.random.randint(1, 9, N)

# Dropout risk probability - logistic combination of factors (ground truth signal)
z = (
    2.0
    - 0.09 * weekly_lms_logins
    - 0.012 * video_watch_pct
    - 0.012 * avg_quiz_score
    - 0.012 * assignment_completion_rate
    + 0.15 * avg_submission_delay_days
    - 0.05 * forum_posts_per_month
    - 0.009 * session_attendance_pct
    - 0.55 * gpa_trend
    - 0.25 * current_gpa
    + 0.35 * part_time_job
    + np.random.normal(0, 0.6, N)
)
prob_dropout = 1 / (1 + np.exp(-z))
dropout = np.random.binomial(1, prob_dropout)

df = pd.DataFrame({
    "weekly_lms_logins": weekly_lms_logins.round(1),
    "video_watch_pct": video_watch_pct.round(1),
    "avg_quiz_score": avg_quiz_score.round(1),
    "assignment_completion_rate": assignment_completion_rate.round(1),
    "avg_submission_delay_days": avg_submission_delay_days.round(1),
    "forum_posts_per_month": forum_posts_per_month,
    "session_attendance_pct": session_attendance_pct.round(1),
    "gpa_trend": gpa_trend.round(2),
    "current_gpa": current_gpa.round(2),
    "part_time_job": part_time_job,
    "semester": semester,
    "dropout_risk": dropout,
})

df.to_csv("/home/claude/vu-dropout/vu_dropout_dataset.csv", index=False)
print(df["dropout_risk"].value_counts(normalize=True))
print(df.head())
print(f"\nTotal rows: {len(df)}")
