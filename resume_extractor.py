import pandas as pd
import re
from typing import Dict, List, Optional

class ResumeExtractor:
    def __init__(self, excel_file: str):
        self.df = pd.read_excel(excel_file)
    
    def extract_name(self, resume_text: str) -> Optional[str]:
        """Extract candidate name from resume text"""
        patterns = [
            r'candidate profile:\s*([^*\n]+)',
            r'interview for.*role[*\s]*([^*\n]+)',
            r'^([A-Z][a-z]+ [A-Z][a-z]+)',
            r'Here\'s a sample resume for ([^:]+):',
            r'Here is a sample resume for ([^:]+):',
            r'^([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE | re.MULTILINE)
            if match:
                name = match.group(1).strip()
                
                # Check if name is "Here Is" (fuzzy matching fallback)
                if name.lower() == "here is":
                    return self.fuzzy_name_match(resume_text)
                
                return name
        return None
    
    def fuzzy_name_match(self, resume_text: str) -> Optional[str]:
        """Extract name using fuzzy matching from email/linkedin"""
        contact_info = self.extract_contact_info(resume_text)
        
        # Check if both email and linkedin are blank
        if not contact_info['email'] and not contact_info['linkedin']:
            return "invalid"
        
        # Extract potential names from email and linkedin
        potential_names = []
        
        if contact_info['email']:
            email_name = contact_info['email'].split('@')[0]
            # Convert email format to name (e.g., john.doe -> John Doe)
            name_parts = re.split(r'[._-]', email_name)
            if len(name_parts) >= 2:
                potential_names.append(' '.join([part.capitalize() for part in name_parts[:2]]))
        
        if contact_info['linkedin']:
            linkedin_name = contact_info['linkedin'].split('/')[-1]
            # Convert linkedin format to name
            name_parts = re.split(r'[._-]', linkedin_name)
            if len(name_parts) >= 2:
                potential_names.append(' '.join([part.capitalize() for part in name_parts[:2]]))
        
        # Find best match in resume text
        for potential_name in potential_names:
            if potential_name.lower() in resume_text.lower():
                return potential_name
        
        return potential_names[0] if potential_names else "invalid"
    
    def extract_contact_info(self, resume_text: str) -> Dict[str, Optional[str]]:
        """Extract contact information"""
        contact = {'phone': None, 'email': None, 'linkedin': None, 'github': None}
        
        # Try flexible contact section patterns
        contact_patterns = [
            r'Contact Information[:\s]*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)',
            r'Contact[:\s]*(.*?)(?=\n\n|\n[A-Z][a-z]+:|$)',
            r'\*\s*Email:[^\n]*\n[^\n]*\n[^\n]*\n[^\n]*'  # Bullet point format
        ]
        
        contact_text = resume_text
        for pattern in contact_patterns:
            match = re.search(pattern, resume_text, re.DOTALL | re.IGNORECASE)
            if match:
                contact_text = match.group(1) if match.lastindex else match.group(0)
                break
        
        # Email patterns
        email_patterns = [
            r'Email:\s*\[([^\]]+)\]\([^\)]+\)',  # [email](mailto:email)
            r'Email:\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',  # Email: email
            r'\*\s*Email:\s*\[([^\]]+)\]',  # * Email: [email]
            r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'  # Any email
        ]
        
        for pattern in email_patterns:
            match = re.search(pattern, contact_text)
            if match:
                contact['email'] = match.group(1).strip()
                break
        
        # Phone patterns
        phone_patterns = [
            r'Phone:\s*([\(\)\d\s-]+)',
            r'\*\s*Phone:\s*([\(\)\d\s-]+)',
            r'(\([0-9]{3}\)\s*[0-9]{3}-[0-9]{4})'
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, contact_text)
            if match:
                contact['phone'] = match.group(1).strip()
                break
        
        # LinkedIn patterns
        linkedin_patterns = [
            r'LinkedIn:\s*(linkedin\.com/in/[^\s\n]+)',
            r'\*\s*LinkedIn:\s*(linkedin\.com/in/[^\s\n]+)',
            r'(linkedin\.com/in/[^\s\n]+)'
        ]
        
        for pattern in linkedin_patterns:
            match = re.search(pattern, contact_text)
            if match:
                contact['linkedin'] = match.group(1).strip()
                break
        
        # GitHub patterns
        github_patterns = [
            r'GitHub:\s*(github\.com/[^\s\n]+)',
            r'\*\s*GitHub:\s*(github\.com/[^\s\n]+)',
            r'(github\.com/[^\s\n]+)'
        ]
        
        for pattern in github_patterns:
            match = re.search(pattern, contact_text)
            if match:
                contact['github'] = match.group(1).strip()
                break
        
        return contact
    
    def extract_education(self, resume_text: str) -> List[str]:
        """Extract education information"""
        education = []
        
        # Look for education section
        education_section = re.search(r'Education:(.*?)(?=\n\n|\n[A-Z]|$)', resume_text, re.DOTALL | re.IGNORECASE)
        if education_section:
            edu_text = education_section.group(1)
            # Extract degree patterns
            degrees = re.findall(r'([A-Z][^,\n]*(?:degree|science|arts|engineering)[^,\n]*)', edu_text, re.IGNORECASE)
            education.extend([deg.strip() for deg in degrees])
        
        # Look for degree patterns
        degree_patterns = [r'(Bachelor|Master|PhD|B\.S\.|M\.S\.|Ph\.D\.).*?(?:in|of)\s+([^\n,]+)']
        for pattern in degree_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            education.extend([f"{m[0]} in {m[1].strip()}" for m in matches])
        
        # Look for school/institute names
        school_patterns = [
            r'([A-Z][^\n,]*(?:University|College|Institute|School)[^\n,]*)',
            r'(\b[A-Z][a-z]+\s+(?:University|College|Institute|School)\b)',
            r'(\b(?:University|College|Institute|School)\s+of\s+[A-Z][^\n,]*)',
        ]
        
        for pattern in school_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            education.extend([match.strip() for match in matches])
        
        return list(set(education))
    
    def extract_experience(self, resume_text: str) -> List[str]:
        """Extract work experience"""
        experience = []
        
        # Look for experience section
        exp_patterns = [
            r'Professional Experience:(.*?)(?=\n\n[A-Z]|Education:|$)',
            r'Work Experience:(.*?)(?=\n\n[A-Z]|Education:|$)',
            r'Experience:(.*?)(?=\n\n[A-Z]|Education:|$)'
        ]
        
        for pattern in exp_patterns:
            exp_section = re.search(pattern, resume_text, re.DOTALL | re.IGNORECASE)
            if exp_section:
                exp_text = exp_section.group(1)
                # Extract job titles and companies
                jobs = re.findall(r'([A-Z][^,\n]*(?:Engineer|Scientist|Analyst|Manager|Developer)[^,\n]*)', exp_text)
                experience.extend([job.strip() for job in jobs])
                break
        
        # Extract job titles with common patterns
        job_patterns = [r'(\b(?:Senior|Junior|Lead|Principal)?\s*(?:Data Scientist|Engineer|Analyst|Manager|Developer|Consultant)\b)']
        for pattern in job_patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            experience.extend(matches)
        
        return list(set(experience))
    
    def extract_skills(self, resume_text: str) -> List[str]:
        """Extract technical skills"""
        skills = []
        
        # Look for skills section
        skills_section = re.search(r'(?:Technical )?Skills:(.*?)(?=\n\n[A-Z]|Professional|Education|$)', resume_text, re.DOTALL | re.IGNORECASE)
        if skills_section:
            skills_text = skills_section.group(1)
            # Extract programming languages, tools, etc.
            skill_patterns = [
                r'Programming languages?:\s*([^\n]+)',
                r'Data Engineering tools?:\s*([^\n]+)',
                r'Cloud Platforms?:\s*([^\n]+)',
                r'([A-Za-z]+(?:\s+[A-Za-z]+)*(?:,\s*)?)'
            ]
            
            for pattern in skill_patterns:
                matches = re.findall(pattern, skills_text, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, str) and len(match.strip()) > 2:
                        skills.extend([s.strip() for s in match.split(',')])
        
        # Also look for skills mentioned in the text
        common_skills = ['Python', 'R', 'SQL', 'Java', 'Spark', 'Airflow', 'AWS', 'GCP', 'Azure', 
                        'Machine Learning', 'TensorFlow', 'scikit-learn', 'Tableau', 'Power BI']
        
        for skill in common_skills:
            if skill.lower() in resume_text.lower() and skill not in skills:
                skills.append(skill)
        
        return list(set(skills))  # Remove duplicates
    
    def extract_all_resumes(self) -> List[Dict]:
        """Extract information from all resumes in the dataset"""
        extracted_data = []
        
        for idx, row in self.df.iterrows():
            resume_text = str(row['Resume'])
            
            extracted_info = {
                'original_index': idx,
                'role': row['Role'],
                'name': self.extract_name(resume_text),
                'contact_info': self.extract_contact_info(resume_text),
                'education': self.extract_education(resume_text),
                'experience': self.extract_experience(resume_text),
                'skills': self.extract_skills(resume_text),
                'decision': row['Decision'],
                'reason': row['Reason_for_decision']
            }
            
            extracted_data.append(extracted_info)
        
        return extracted_data
    
    def save_extracted_data(self, output_file: str = 'extracted_resume_data.xlsx'):
        """Save extracted data to Excel file"""
        extracted_data = self.extract_all_resumes()
        
        # Flatten the data for Excel export
        flattened_data = []
        for item in extracted_data:
            # Check if both email and linkedin are blank, set name to "invalid"
            name = item['name']
            if not item['contact_info']['email'] and not item['contact_info']['linkedin']:
                name = "invalid"
            
            flat_item = {
                'Original_Index': item['original_index'],
                'Role': item['role'],
                'Name': name,
                'Phone': item['contact_info']['phone'],
                'Email': item['contact_info']['email'],
                'LinkedIn': item['contact_info']['linkedin'],
                'GitHub': item['contact_info']['github'],
                'Education': '; '.join(item['education']) if item['education'] else '',
                'Experience': '; '.join(item['experience']) if item['experience'] else '',
                'Skills': '; '.join(item['skills']) if item['skills'] else '',
                'Decision': item['decision'],
                'Reason': item['reason']
            }
            flattened_data.append(flat_item)
        
        df_output = pd.DataFrame(flattened_data)
        df_output.to_excel(output_file, index=False)
        print(f"Extracted data saved to {output_file}")
        return df_output

# Usage example
if __name__ == "__main__":
    extractor = ResumeExtractor('Resumes (For Applicants).xlsx')
    result_df = extractor.save_extracted_data()
    print(f"Processed {len(result_df)} resumes")
    print("\nSample extracted data:")
    print(result_df.head())