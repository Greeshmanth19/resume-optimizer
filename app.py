# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import PyPDF2
from docx import Document
import logging
from werkzeug.utils import secure_filename
from openai import OpenAI
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx'}
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "enter your openai API key here"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_resume(file_path):
    """Extract text from resume file"""
    try:
        text = ""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
        return ""

def process_markdown(text):
    """Convert markdown formatting to HTML"""
    # Replace markdown bold with HTML bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Replace markdown bullet points with HTML list items
    text = re.sub(r'- (.*?)(?:\n|$)', r'<li>\1</li>', text)
    
    # Wrap lists in <ul> tags
    text = re.sub(r'(<li>.*?</li>)+', r'<ul>\g<0></ul>', text, flags=re.DOTALL)
    
    # Replace markdown headings
    text = re.sub(r'# (.*?)(?:\n|$)', r'<h1>\1</h1>', text)
    text = re.sub(r'## (.*?)(?:\n|$)', r'<h2>\1</h2>', text)
    text = re.sub(r'### (.*?)(?:\n|$)', r'<h3>\1</h3>', text)
    
    # Add paragraph tags to plain text blocks
    paragraphs = text.split('\n\n')
    for i, paragraph in enumerate(paragraphs):
        if not paragraph.startswith('<'):
            paragraphs[i] = f'<p>{paragraph}</p>'
    
    text = '\n'.join(paragraphs)
    
    return text

def summarize_resume(resume_text):
    """Use GPT to summarize resume"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert resume analyzer. Extract key skills, experiences, and qualifications from the resume."},
                {"role": "user", "content": f"Summarize this resume and extract key skills, experiences, and qualifications in detail:\n\n{resume_text}"}
            ]
        )
        summary = response.choices[0].message.content
        return process_markdown(summary)  # Process markdown formatting
    except Exception as e:
        logger.error(f"Error summarizing resume: {e}")
        return "Failed to generate resume summary."

def extract_keywords(text):
    """Extract keywords from text using GPT"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract the most important keywords from the text as a comma-separated list."},
                {"role": "user", "content": f"Extract 10-15 most important skills, technologies, and qualifications from this resume as a comma-separated list:\n\n{text}"}
            ]
        )
        keywords = response.choices[0].message.content.split(',')
        return [keyword.strip() for keyword in keywords]
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def extract_job_keywords(job_description):
    """Extract keywords from job description using GPT"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Extract the most important keywords from the job description that would be critical for a resume to include."},
                {"role": "user", "content": f"Extract 10-15 most important skills, technologies, and qualifications from this job description as a comma-separated list:\n\n{job_description}"}
            ]
        )
        keywords = response.choices[0].message.content.split(',')
        return [keyword.strip() for keyword in keywords]
    except Exception as e:
        logger.error(f"Error extracting job keywords: {e}")
        return []

def analyze_job_fit(resume_keywords, resume_summary, job_description):
    """Analyze how well a resume fits a job description"""
    try:
        # Extract keywords from job description
        job_keywords = extract_job_keywords(job_description)
        
        # Find missing keywords
        # Convert all to lowercase for case-insensitive comparison
        resume_keywords_lower = [keyword.lower() for keyword in resume_keywords]
        job_keywords_lower = [keyword.lower() for keyword in job_keywords]
        
        missing_keywords = []
        for job_keyword in job_keywords:
            # Check if any resume keyword contains this job keyword
            if not any(job_keyword.lower() in resume_keyword.lower() for resume_keyword in resume_keywords_lower):
                missing_keywords.append(job_keyword)
        
        # Calculate match percentage
        if len(job_keywords) > 0:
            match_percentage = int(100 * (1 - len(missing_keywords) / len(job_keywords)))
        else:
            match_percentage = 100
        
        # Generate improvement suggestions
        improvement_suggestions = generate_improvement_suggestions(resume_summary, job_description, missing_keywords)
        
        return {
            "match_percentage": match_percentage,
            "job_keywords": job_keywords,
            "missing_keywords": missing_keywords,
            "improvement_suggestions": improvement_suggestions
        }
    except Exception as e:
        logger.error(f"Error analyzing job fit: {e}")
        return {
            "match_percentage": 0,
            "job_keywords": [],
            "missing_keywords": [],
            "improvement_suggestions": ["Error analyzing job fit. Please try again."]
        }

def generate_improvement_suggestions(resume_summary, job_description, missing_keywords):
    """Generate specific suggestions to improve resume for job fit"""
    try:
        if not missing_keywords:
            return ["Your resume contains all the important keywords for this job!"]
        
        # Join missing keywords for prompt
        missing_keywords_str = ", ".join(missing_keywords)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert resume writer helping job seekers optimize their resumes for specific job descriptions."},
                {"role": "user", "content": f"""
                Here's a summary of the candidate's resume:
                {resume_summary}
                
                Here's the job description they're applying for:
                {job_description}
                
                The candidate's resume is missing these important keywords: {missing_keywords_str}
                
                Provide 3-5 specific, actionable suggestions for how they could update their resume to better match this job description. 
                Focus on adding the missing keywords naturally, removing irrelevant content, and emphasizing relevant experience.
                Format each suggestion as a separate paragraph with a clear action item.
                """}
            ]
        )
        
        # Process the markdown in the response
        suggestions_text = response.choices[0].message.content.strip()
        processed_text = process_markdown(suggestions_text)
        
        # Split by HTML paragraph tags
        suggestions = re.findall(r'<p>(.*?)</p>', processed_text, re.DOTALL)
        if not suggestions:
            # Fallback to splitting by newlines if no paragraphs found
            suggestions = [s for s in re.split(r'\n+', suggestions_text) if s.strip()]
        
        return suggestions
    except Exception as e:
        logger.error(f"Error generating improvement suggestions: {e}")
        return ["Failed to generate improvement suggestions. Please try again."]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload-resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PDF or DOCX file.'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process resume
        resume_text = parse_resume(file_path)
        if not resume_text:
            return jsonify({'error': 'Could not extract text from resume'}), 400
            
        # Generate summary and keywords
        summary = summarize_resume(resume_text)
        keywords = extract_keywords(resume_text)
        
        # Clean up
        os.remove(file_path)
        
        return jsonify({
            'summary': summary,
            'keywords': keywords
        })
        
    except Exception as e:
        logger.error(f"Error processing resume: {e}")
        return jsonify({'error': 'An error occurred while processing the resume'}), 500

@app.route('/api/analyze-job-fit', methods=['POST'])
def analyze_job_fit_endpoint():
    data = request.json
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    resume_keywords = data.get('resume_keywords', [])
    resume_summary = data.get('resume_summary', '')
    job_description = data.get('job_description', '')
    
    if not resume_keywords or not resume_summary or not job_description:
        return jsonify({'error': 'Missing required data (resume_keywords, resume_summary, or job_description)'}), 400
    
    try:
        analysis_results = analyze_job_fit(resume_keywords, resume_summary, job_description)
        return jsonify(analysis_results)
    except Exception as e:
        logger.error(f"Error analyzing job fit: {e}")
        return jsonify({'error': 'An error occurred while analyzing job fit'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
