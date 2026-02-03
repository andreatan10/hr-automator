import pandas as pd
import sys
sys.path.append('models')
from role_classifier import RoleClassifier

# Load the extracted resume data
df = pd.read_excel('extracted_resume_data.xlsx')

# Filter only rejected candidates
rejected_df = df[df['Decision'] == 'reject'].copy()
print(f"Total rejected candidates: {len(rejected_df)}")

# Filter 1: Invalid names
invalid_names = rejected_df[rejected_df['Name'].isin(['invalid', 'Contact Information'])]
invalid_names.to_excel('invalid_candidates.xlsx', index=False)
print(f"Invalid names filtered: {len(invalid_names)}")

# Continue with valid candidates
valid_df = rejected_df[~rejected_df['Name'].isin(['invalid', 'Contact Information'])]

# Load the role classifier model
classifier = RoleClassifier()
classifier.load_model('models/role_classifier')
print("Role classifier loaded successfully")

# Run predictions on valid rejected candidates
filtered_out = []
for idx, row in valid_df.iterrows():
    education = str(row.get('Education', '')) if pd.notna(row.get('Education', '')) else ''
    experience = str(row.get('Experience', '')) if pd.notna(row.get('Experience', '')) else ''
    skills = str(row.get('Skills', '')) if pd.notna(row.get('Skills', '')) else ''
    
    text = f"Education: {education}. Experience: {experience}. Skills: {skills}"
    predictions = classifier.predict(text, top_k=1)
    
    original_role = str(row.get('Role', '')).lower()
    predicted_role = predictions[0]['role'].lower()
    confidence = predictions[0]['confidence']
    
    # Filter 2 & 3: Role mismatch or low confidence
    if original_role != predicted_role or confidence < 0.6:
        filtered_out.append({
            'Name': row.get('Name', ''),
            'Original_Role': row.get('Role', ''),
            'Predicted_Role': predictions[0]['role'],
            'Confidence': confidence,
            'Filter_Reason': 'Role mismatch' if original_role != predicted_role else 'Low confidence',
            'Skills': skills,
            'Education': education,
            'Experience': experience
        })

# Save filtered out candidates
filtered_df = pd.DataFrame(filtered_out)
filtered_df.to_excel('filtered_candidates.xlsx', index=False)

print(f"Candidates filtered out: {len(filtered_df)}")
print(f"- Role mismatches: {len(filtered_df[filtered_df['Filter_Reason'] == 'Role mismatch'])}")
print(f"- Low confidence: {len(filtered_df[filtered_df['Filter_Reason'] == 'Low confidence'])}")