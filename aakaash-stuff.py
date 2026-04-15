import os
import glob
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# 1. Your Financial Metrics Dictionary
FINANCIAL_METRICS = {
    'demand': ['demand', 'consumption', 'spending', 'purchasing', 'buying', 'market demand', 'consumer demand', 'order volume'],
    'hiring': ['hiring', 'employment', 'jobs', 'labor', 'workforce', 'recruitment', 'staffing', 'unemployment', 'headcount'],
    'pricing': ['pricing', 'price', 'inflation', 'costs', 'rates', 'fees', 'tariffs', 'margin pressure'],
    'capex': ['capex', 'capital expenditure', 'investment', 'infrastructure', 'construction', 'equipment', 'facilities'],
    'AI': ['artificial intelligence', 'ai', 'machine learning', 'automation', 'technology', 'innovation', 'generative ai'],
    'GDP': ['gdp', 'gross domestic product', 'economic growth', 'economy', 'output', 'macro environment'],
    'housing': ['housing', 'real estate', 'property', 'mortgage', 'home prices', 'construction', 'residential'],
    'trade': ['trade', 'imports', 'exports', 'tariffs', 'commerce', 'supply chain', 'freight'],
    'monetary_policy': ['monetary policy', 'interest rates', 'federal reserve', 'central bank', 'rate hikes', 'rate cuts'],
    'fiscal_policy': ['fiscal policy', 'budget', 'deficit', 'taxes', 'spending', 'stimulus', 'subsidies']
}

# 2. Financial Sentiment Lexicons
POSITIVE_FIN_WORDS = set([
    'achieve', 'advantage', 'better', 'boom', 'breakthrough', 'confident', 'exceed', 
    'excellent', 'expand', 'gain', 'grow', 'growth', 'highest', 'improve', 'improvement', 
    'increase', 'innovative', 'momentum', 'opportunity', 'outperform', 'profit', 
    'profitable', 'progress', 'record', 'solid', 'strength', 'strong', 'success', 'surge'
])

NEGATIVE_FIN_WORDS = set([
    'adverse', 'against', 'bad', 'challenge', 'challenging', 'crisis', 'damage', 
    'decline', 'decrease', 'default', 'deficit', 'depress', 'deteriorate', 'difficult', 
    'disappoint', 'disappointing', 'down', 'drop', 'fail', 'failure', 'fall', 'hardship', 
    'loss', 'negative', 'penalty', 'recession', 'risk', 'shortfall', 'sluggish', 
    'struggle', 'suffer', 'threat', 'volatile', 'weak', 'weaken', 'worse'
])

def get_lexicon_sentiment(text):
    words = [word.lower() for word in word_tokenize(text) if word.isalpha()]
    
    if not words:
        return 0.0
        
    pos_count = sum(1 for word in words if word in POSITIVE_FIN_WORDS)
    neg_count = sum(1 for word in words if word in NEGATIVE_FIN_WORDS)
    
    if pos_count == 0 and neg_count == 0:
        return 0.0
        
    score = (pos_count - neg_count) / (pos_count + neg_count)
    return score

def analyze_transcripts(folder_path):
    company_data = {}
    
    search_pattern = os.path.join(folder_path, "**", "*.txt")
    txt_files = glob.glob(search_pattern, recursive=True)
    
    print(f"--> SYSTEM FOUND {len(txt_files)} TEXT FILES <--")
    
    for file_path in txt_files:
        # Find the relative path from GardDat to the file
        # E.g., "abbott\2025\Q2\transcript.txt"
        relative_path = os.path.relpath(file_path, folder_path)
        
        # Split the path by the OS folder separator (\ on Windows, / on Mac)
        path_parts = relative_path.split(os.sep)
        
        if len(path_parts) > 1:
            # If it's inside a subfolder, the VERY FIRST folder is the company name (e.g., 'abbott')
            company_name = path_parts[0]
        else:
            # Fallback: If a file is just sitting loose in GardDat, use its filename
            company_name = os.path.basename(file_path).replace('.txt', '')
            
        if company_name not in company_data:
            company_data[company_name] = {metric: [] for metric in FINANCIAL_METRICS.keys()}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
            
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            for metric, keywords in FINANCIAL_METRICS.items():
                if any(keyword in sentence_lower for keyword in keywords):
                    sentiment_score = get_lexicon_sentiment(sentence)
                    company_data[company_name][metric].append(sentiment_score)
                    
    return company_data

def generate_absolute_scores(company_data):
    aggregated_data = []
    
    for company, metrics in company_data.items():
        row = {'Company': company}
        for metric, scores in metrics.items():
            if scores: 
                row[metric] = sum(scores) / len(scores)
            else:
                row[metric] = None
        aggregated_data.append(row)
        
    df = pd.DataFrame(aggregated_data)
    df.set_index('Company', inplace=True)
    
    return df

# ==========================================
# Execution
# ==========================================
folder_dir = r"C:\Users\rajaa\OneDrive\Desktop\GardDat"

print(f"Scanning folder: {folder_dir}")
raw_sentiment_data = analyze_transcripts(folder_dir)

if not raw_sentiment_data:
    print("\n❌ CRASH PREVENTED: The script found 0 files. ")
else:
    results_df = generate_absolute_scores(raw_sentiment_data)
    results_df = results_df.round(2)
    
    print("\n✅ Success! Aggregated Sentiment Scores (-1.0 to +1.0):")
    print(results_df)