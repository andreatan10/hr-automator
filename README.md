# HR Automator

AI-powered resume classification system using DistilBERT for automated role prediction and candidate assessment.

## Features

- **Resume Data Extraction**: Extract structured data from resume datasets
- **Role Classification**: DistilBERT-based model for predicting job roles from resume content
- **Web Interface**: User-friendly interface for HR officers to upload and analyse resumes
- **Candidate Filtering**: Automated filtering based on confidence scores and role matching

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract Resume Data

Run the resume extractor to create the structured dataset:

```bash
python resume_extractor.py
```

This creates `extracted_resume_data.xlsx` containing structured resume information.

### 3. Train the Role Classifier Model

Navigate to the models folder and train the DistilBERT classifier:

```bash
python models/role_classifier.py
```

This will:
- Train a DistilBERT model on the extracted resume data
- Save the trained model to `models/role_classifier/`
- Generate test predictions and performance metrics

## Usage for HR Officers

### Web Interface

1. **Start the Application**:
   ```bash
   python app.py
   ```

2. **Access the Interface**:
   - Open your browser and go to `http://localhost:5000`

3. **Upload Resume Data**:
   - Click "Select Files" to upload an Excel file containing resume data
   - For testing, use the `test_data.xlsx` file

4. **View Results**:
   - The system will automatically classify candidates and display:
     - Predicted job roles
     - Confidence scores
     - Key skills

### Candidate Filtering (Testing)

For testing the filtering functionality without the web interface:

```bash
python candidate_filter.py
```

This will:
- Process rejected candidates from the extracted resume data
- Generate `invalid_candidates.xlsx` with candidates having invalid names
- Generate `filtered_candidates.xlsx` with candidates flagged for role mismatch or low confidence

### Features

- **Confidence-based Filtering**: Candidates with low confidence (<60%) are flagged for review
- **Invalid Resume Detection**: Automatically flags resumes with invalid names
- **Visual Indicators**: Color-coded confidence levels (green/yellow/red)

## File Structure

```
├── app.py                      # Flask web application
├── resume_extractor.py         # Resume data extraction
├── models/
│   └── role_classifier.py      # DistilBERT model training
├── webpage.html               # Frontend interface
├── requirements.txt           # Python dependencies
└── Resumes (For Applicants).xlsx # Raw resume dataset
```

## Model Performance

The DistilBERT classifier achieves:
- High accuracy on role classification:
   Accuracy: 0.9760,
   ROC-AUC: 0.9997,
   PR-AUC: 0.9897
- Robust confidence scoring
- Support for multiple job categories

## Requirements

- Python 3.8+
- All packages listed in `requirements.txt`