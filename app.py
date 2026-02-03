from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import sys
import os
sys.path.append('models')
from role_classifier import RoleClassifier

app = Flask(__name__)
CORS(app)

# Load model once at startup
classifier = RoleClassifier()
classifier.load_model('models/role_classifier')

@app.route('/')
def index():
    with open('webpage.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Update the HTML to include file upload functionality
    updated_html = html_content.replace(
        '<button class="flex items-center gap-2 cursor-pointer overflow-hidden rounded-lg h-10 px-6 bg-primary text-white text-sm font-bold">',
        '<input type="file" id="fileInput" accept=".xlsx,.xls" style="display: none;" onchange="uploadFile()">'
        '<button onclick="document.getElementById(\'fileInput\').click()" class="flex items-center gap-2 cursor-pointer overflow-hidden rounded-lg h-10 px-6 bg-primary text-white text-sm font-bold">'
    )
    
    # Add JavaScript for file upload
    js_script = """
    <script>
    async function uploadFile() {
        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];
        
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const results = await response.json();
            updateTable(results);
        } catch (error) {
            console.error('Error:', error);
        }
    }
    
    function updateTable(results) {
        const tbody = document.querySelector('tbody');
        tbody.innerHTML = '';
        
        results.forEach(result => {
            const row = `
                <tr class="hover:bg-slate-50 dark:hover:bg-white/5 transition-colors">
                    <td class="px-6 py-4">
                        <div class="flex items-center gap-3">
                            <div class="size-8 rounded bg-primary/20 flex items-center justify-center text-primary">
                                <span class="material-symbols-outlined text-lg font-bold">description</span>
                            </div>
                            <div>
                                <p class="font-bold text-slate-900 dark:text-white">${result.name}</p>
                            </div>
                        </div>
                    </td>
                    <td class="px-6 py-4 font-semibold text-slate-700 dark:text-slate-300">${result.predicted_role}</td>
                    <td class="px-6 py-4">
                        <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold ${getConfidenceClass(result.confidence)}">
                            ${(result.confidence * 100).toFixed(1)}%
                        </span>
                    </td>
                    <td class="px-6 py-4">
                        <div class="flex flex-wrap gap-1">
                            ${result.skills.split(';').slice(0, 3).map(skill => 
                                `<span class="px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-[10px] font-bold text-slate-600 dark:text-slate-400 uppercase">${skill.trim()}</span>`
                            ).join('')}
                        </div>
                    </td>
                    <td class="px-6 py-4 text-right">
                        <div class="flex justify-end gap-2">
                            ${result.confidence < 0.6 ? 
                                '<button class="p-1.5 hover:bg-red-100 dark:hover:bg-red-900/40 text-red-600 dark:text-red-400 rounded transition-colors"><span class="material-symbols-outlined">close</span></button>' :
                                '<button class="p-1.5 hover:bg-green-100 dark:hover:bg-green-900/40 text-green-600 dark:text-green-400 rounded transition-colors"><span class="material-symbols-outlined">check</span></button>'
                            }
                        </div>
                    </td>
                </tr>
            `;
            tbody.innerHTML += row;
        });
    }
    
    function getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400';
        if (confidence >= 0.6) return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400';
        return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400';
    }
    </script>
    """
    
    updated_html = updated_html.replace('</body>', js_script + '</body>')
    return updated_html

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read Excel file
        df = pd.read_excel(file)
        
        results = []
        for idx, row in df.iterrows():
            name = str(row.get('Name', 'Unknown'))
            
            # Flag invalid resumes
            if name.lower() == 'invalid':
                results.append({
                    'name': name,
                    'predicted_role': 'Invalid Resume',
                    'confidence': 0.0,
                    'skills': 'N/A',
                    'education': 'N/A',
                    'experience': 'N/A'
                })
                continue
            
            education = str(row.get('Education', '')) if pd.notna(row.get('Education', '')) else ''
            experience = str(row.get('Experience', '')) if pd.notna(row.get('Experience', '')) else ''
            skills = str(row.get('Skills', '')) if pd.notna(row.get('Skills', '')) else ''
            
            text = f"Education: {education}. Experience: {experience}. Skills: {skills}"
            predictions = classifier.predict(text, top_k=1)
            
            results.append({
                'name': name,
                'predicted_role': predictions[0]['role'],
                'confidence': float(predictions[0]['confidence']),
                'skills': skills,
                'education': education,
                'experience': experience
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)