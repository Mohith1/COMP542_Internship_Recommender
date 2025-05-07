from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from io import BytesIO
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder


app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
model_dir = "model/job_recommendation_model"
tokenizer_dir = "model/job_recommendation_tokenizer"

tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

labels = [
    'Data Science', 'Human Resources', 'Advocate', 'Arts', 'Web Designing', 'Mechanical Engineer',
    'Sales', 'Health and fitness', 'Civil Engineer', 'Java Developer', 'Business Analyst',
    'SAP Developer', 'Automation Testing', 'Electrical Engineering', 'Operations Manager',
    'Python Developer', 'DevOps Engineer', 'Network Security Engineer', 'PMO', 'Database',
    'Hadoop', 'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing'
]

SKILL_KEYWORDS = [
    # Software Engineering
    'python', 'java', 'c++', 'c#', 'javascript', 'typescript', 'html', 'css', 'react', 'angular', 'vue.js',
    'node.js', 'express.js', 'next.js', 'nuxt.js', 'spring', 'spring boot', 'django', 'flask',
    'ruby on rails', 'php', 'laravel', 'dotnet', '.net core', 'asp.net', 'golang', 'swift', 'kotlin',
    'flutter', 'react native', 'xamarin', 'android', 'ios', 'unity', 'unreal engine',

    # DevOps / Cloud / Infrastructure
    'git', 'github', 'gitlab', 'bitbucket', 'docker', 'kubernetes', 'helm', 'jenkins', 'ansible',
    'terraform', 'vagrant', 'aws', 'azure', 'gcp', 'cloudformation', 'linux', 'windows server', 'bash',
    'powershell', 'zsh', 'nginx', 'apache', 'devops', 'ci/cd', 'monitoring', 'prometheus', 'grafana',
    'splunk', 'datadog', 'new relic', 'elk stack',

    # Databases & Storage
    'sql', 'mysql', 'postgresql', 'sqlite', 'oracle', 'mssql', 'mongodb', 'redis', 'cassandra',
    'dynamodb', 'firebase', 'elasticsearch', 'neo4j',

    # APIs & Testing
    'rest api', 'graphql', 'soap', 'postman', 'swagger', 'openapi', 'unit testing', 'integration testing',
    'system testing', 'jest', 'mocha', 'chai', 'pytest', 'junit', 'selenium', 'cypress', 'playwright',

    # Data Science / ML / AI
    'machine learning', 'deep learning', 'data science', 'nlp', 'natural language processing',
    'computer vision', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'openai', 'huggingface',
    'data visualization', 'matplotlib', 'seaborn', 'plotly', 'pandas', 'numpy', 'scipy', 'excel',
    'tableau', 'power bi', 'hadoop', 'spark', 'databricks', 'mlflow', 'airflow', 'bigquery', 'snowflake',
    'data engineering', 'etl', 'data lake', 'data warehouse',

    # UI/UX & Design
    'figma', 'sketch', 'adobe xd', 'photoshop', 'illustrator', 'invision', 'wireframing', 'prototyping',
    'user research', 'usability testing', 'responsive design', 'accessibility', 'material design',
    'tailwindcss', 'bootstrap', 'sass', 'less',

    # Product & Project Management
    'product management', 'product strategy', 'roadmapping', 'agile', 'scrum', 'kanban',
    'jira', 'trello', 'asana', 'monday.com', 'notion', 'confluence', 'okrs', 'kpis',

    # Marketing & Growth
    'seo', 'sem', 'ppc', 'google ads', 'adwords', 'facebook ads', 'instagram ads', 'linkedin ads',
    'email marketing', 'mailchimp', 'constant contact', 'hubspot', 'drip', 'klaviyo',
    'content marketing', 'copywriting', 'social media', 'google analytics', 'hotjar', 'ahrefs', 'moz',
    'a/b testing', 'conversion optimization', 'crm', 'customer segmentation', 'lead generation',
    'marketing automation', 'campaign management',

    # Sales & Customer Support
    'salesforce', 'hubspot crm', 'zoho crm', 'cold calling', 'sales strategy', 'account management',
    'customer success', 'live chat', 'zendesk', 'intercom', 'freshdesk', 'ticketing systems',

    # Finance & Business
    'excel', 'quickbooks', 'xero', 'erp', 'financial modeling', 'forecasting', 'budgeting', 'bookkeeping',
    'data analysis', 'sql for finance', 'market analysis', 'valuation', 'kpis',

    # Cybersecurity
    'cybersecurity', 'network security', 'ethical hacking', 'penetration testing', 'owasp',
    'vulnerability assessment', 'firewalls', 'siem', 'wireshark', 'metasploit', 'nmap', 'burpsuite',

    # Human Resources
    'recruitment', 'talent acquisition', 'human Resource software', 'bamboohr', 'workday', 'greenhouse', 'ats',
    'onboarding', 'employee engagement', 'human Resource analytics', 'payroll', 'performance management',

    # Soft Skills (bonus)
    'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
    'adaptability', 'time management', 'creativity', 'collaboration',

    # Languages (bonus)
    'english', 'spanish', 'french', 'german', 'chinese', 'japanese', 'hindi', 'arabic'
]


# Extract text using PyMuPDF
def extract_text_with_fitz(file_bytes: bytes):
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        text = " ".join([page.get_text() for page in doc])
    return text

# Clean and preprocess the text
def clean_text(text: str) -> str:
    text = re.sub(r'[^a-zA-Z0-9\.\+\#\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# Extract matched skills
def match_skills(text: str):
    found_skills = set()
    for skill in SKILL_KEYWORDS:
        if skill.lower() in text:
            found_skills.add(skill)
    return sorted(found_skills)

# 2. Create a prediction function
def predict(texts):
    # If only one sentence, make it a list
    if isinstance(texts, str):
        texts = [texts]

    encodings = tokenizer(
        texts,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

    return preds.cpu().numpy()

# Load and fit the LabelEncoder
df = pd.read_csv('data/UpdatedResumeDataSet.csv')
label_encoder = LabelEncoder()
label_encoder.fit(df['Category'])


# API endpoint
@app.post("/predict-pdf")
async def predict_from_pdf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Read file content
    contents = await file.read()

    # Extract and clean text
    extracted_text = extract_text_with_fitz(contents)
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    cleaned_text = clean_text(extracted_text)

    # Match skills
    matched_skills = match_skills(cleaned_text)
    for skill in matched_skills:
        print(f"- {skill}")
    global output_parsed_text
    output_parsed_text = " ".join(matched_skills)

    # Predict category
    # predicted_category = output_parsed_text

    predictions = predict(output_parsed_text)
    decoded_predictions = label_encoder.inverse_transform(predictions)
    predicted_category = decoded_predictions[0]

    return {
        "predicted_category": predicted_category,
        "matched_skills": output_parsed_text
    }
