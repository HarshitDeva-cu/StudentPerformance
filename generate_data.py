import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic student data
n_students = 100

# Generate data
data = {
    'student_id': [f'STU{i:03d}' for i in range(1, n_students + 1)],
    'gender': np.random.choice(['Male', 'Female'], n_students),
    'age': np.random.randint(15, 22, n_students),
    'parent_education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_students),
    'parent_income': np.random.choice(['Low', 'Medium', 'High'], n_students),
    'study_hours': np.random.uniform(0, 8, n_students).round(1),
    'attendance_percentage': np.random.uniform(40, 100, n_students).round(1),
    'internal_marks': np.random.randint(0, 100, n_students),
    'final_marks': np.random.randint(0, 100, n_students),
    'school_type': np.random.choice(['Public', 'Private'], n_students),
    'extra_classes': np.random.choice(['Yes', 'No'], n_students),
    'test_preparation_course': np.random.choice(['Completed', 'Not Completed'], n_students),
    'peer_group_study': np.random.choice(['Yes', 'No'], n_students),
    'daily_screen_time': np.random.uniform(1, 10, n_students).round(1),
    'library_usage': np.random.choice(['Never', 'Rarely', 'Sometimes', 'Often', 'Daily'], n_students),
    'sports_participation': np.random.choice(['Yes', 'No'], n_students),
    'discipline_score': np.random.randint(1, 11, n_students),
    'monthly_absent_days': np.random.randint(0, 15, n_students),
    'sleep_hours': np.random.uniform(4, 10, n_students).round(1),
    'mental_stress_level': np.random.choice(['Low', 'Medium', 'High'], n_students),
    'learning_disability': np.random.choice(['Yes', 'No'], n_students, p=[0.1, 0.9]),
    'family_support': np.random.choice(['Low', 'Medium', 'High'], n_students),
}

# Create DataFrame
df = pd.DataFrame(data)


# Generate status based on various factors (realistic logic)
def determine_status(row):
    score = 0

    # Academic factors
    if row['attendance_percentage'] >= 75:
        score += 2
    if row['study_hours'] >= 4:
        score += 2
    if row['internal_marks'] >= 60:
        score += 2
    if row['final_marks'] >= 50:
        score += 3

    # Support factors
    if row['extra_classes'] == 'Yes':
        score += 1
    if row['test_preparation_course'] == 'Completed':
        score += 1
    if row['peer_group_study'] == 'Yes':
        score += 1
    if row['family_support'] == 'High':
        score += 1

    # Negative factors
    if row['daily_screen_time'] > 6:
        score -= 1
    if row['monthly_absent_days'] > 5:
        score -= 2
    if row['sleep_hours'] < 6:
        score -= 1
    if row['mental_stress_level'] == 'High':
        score -= 1
    if row['learning_disability'] == 'Yes':
        score -= 1

    # Library usage bonus
    library_score = {'Never': -1, 'Rarely': 0, 'Sometimes': 1, 'Often': 2, 'Daily': 3}
    score += library_score[row['library_usage']]

    # Discipline bonus
    if row['discipline_score'] >= 7:
        score += 1

    # Determine pass/fail with some randomness
    threshold = 6 + np.random.normal(0, 1)
    return 'pass' if score >= threshold else 'fail'


# Apply the function to determine status
df['status'] = df.apply(determine_status, axis=1)

# Save to CSV
df.to_csv('student_data.csv', index=False)
print(f"Generated {len(df)} student records")
print(f"Pass rate: {(df['status'] == 'pass').sum() / len(df) * 100:.1f}%")
print(f"Fail rate: {(df['status'] == 'fail').sum() / len(df) * 100:.1f}%")
print("\nFirst 5 records:")
print(df.head())