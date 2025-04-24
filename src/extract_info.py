import json
import time
import os
import pandas as pd
from tqdm import tqdm
import requests
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inequality_analysis.log"),
        logging.StreamHandler()
    ]
)

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))   

def analyze_abstract_with_llm(title, abstract, model="gpt-4.1-nano"): # gpt-4o-mini
    """Use OpenAI's API to analyze the abstract with our refined prompt"""
    prompt = f"""
    Analyze the following academic paper title and abstract:
    
    Title: {title}
    Abstract: {abstract}
    
    First, determine if this paper genuinely addresses social inequality (socioeconomic, racial, gender, health, educational inequalities, etc.).
    
    Guidelines for classification:
    1. Consider as a "social inequality" paper if it:
       - Addresses social discrimination or inequality related to race, gender, class, education, health, information, geographical location, age, disability, etc.
       - Discusses social bias or fairness issues in algorithms or AI systems
       - Analyzes unequal impacts of technology across different social groups
    
    Return ONLY a JSON object with the following structure:
    
    {{
        "is_social_inequality": true/false, (whether this paper genuinely addresses social inequality)
        "reason": "brief explanation of determination", (within 3 sentences)
        
        // Only complete the fields below if is_social_inequality is true
        "inequality_type": ["type1", "type2", ...], (select from: "economic", "income", "wealth", "socioeconomic", "class", "gender", "racial", "ethnic", "nationality", "religion", "linguistic", "health", "educational", "informational", "digital", "geographic", "urban-rural", "environmental", "climate", "disability", "age", "other")
        "other_detail": "describe only if 'other' selected above", (10 words or less)
        "affected_populations": ["group1", "group2", ...], (groups affected by the inequality - keep each entry concise, under 5 words)
        "methodology": ["method1", "method2", ...], (select from: "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Quantitative Analysis", "Qualitative Study", "Literature Review", "Algorithm Auditing", "Statistical Analysis", "Ethics Analysis", "Survey", "Experiment", "Case Study", "Dataset Creation", "Model Development", "System Design", "Theoretical Analysis", "Other")
        "methodology_detail": "additional details only if needed", (10 words or less)
        "geographic_focus": ["region1", "region2", ...], (specific regions or countries studied; null if global or unspecified)
        "ai_relationship": "relationship type", (select ONE: "AI as cause/amplifier", "AI as solution", "AI as measurement tool", "AI as subject of regulation", "Not AI-related", "Unclear")
        "confidence": 0.X (confidence in this analysis, value between 0-1. e.g., 0.7 for 70% confident)
    }}
    
    BE CONCISE. Return ONLY valid JSON with NO additional explanation. Be precise and base your analysis strictly on the information provided.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research assistant analyzing academic papers on inequality and AI."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        # Parse the response to ensure it's valid JSON
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        logging.error(f"Error analyzing abstract: {e}")
        return None

def process_paper(paper):
    """Process a single paper - for parallel processing"""
    try:
        # Extract the paper's title and abstract
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        
        # Skip papers with empty abstracts
        if not abstract:
            return None
        
        # Analyze with LLM
        analysis = analyze_abstract_with_llm(title, abstract)
        
        if analysis:
            # Add original paper metadata and analysis to results
            result = {
                'id': paper['id'],
                'title': title,
                'year': paper.get('year'),
                'authors': paper.get('authors', []),
                'categories': paper.get('categories', []),
                'is_ai_related_original': paper.get('is_ai_related', False),
                'analysis': analysis
            }
            return result
        
    except Exception as e:
        logging.error(f"Error processing paper {paper.get('id', 'unknown')}: {e}")
    
    return None

def process_papers_parallel(papers, output_file, max_workers=5, max_papers=None, batch_size=50):
    """Process papers in parallel with checkpoint saving and filtering by social inequality relevance"""
    results = []
    filtered_results = []  # New: to store only social inequality papers
    
    # Check if output file exists and load any previous results
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            filtered_results = json.load(f)
            logging.info(f"Loaded {len(filtered_results)} previously analyzed social inequality papers")
    
    # Get list of already processed paper IDs
    processed_ids = {paper['id'] for paper in filtered_results if 'id' in paper}
    
    # Filter papers that haven't been processed yet
    papers_to_process = [p for p in papers if p['id'] not in processed_ids]
    
    # Apply max_papers limit if specified
    if max_papers and len(papers_to_process) > max_papers:
        papers_to_process = papers_to_process[:max_papers]
        
    logging.info(f"Found {len(papers_to_process)} papers to analyze")
    
    # Variables to track statistics
    total_analyzed = 0
    total_social_inequality = 0
    
    # Process in batches, but parallelize within each batch
    for i in range(0, len(papers_to_process), batch_size):
        batch = papers_to_process[i:i + batch_size]
        logging.info(f"Processing batch {i//batch_size + 1}/{(len(papers_to_process) + batch_size - 1)//batch_size}")
        
        batch_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all papers in the batch for processing
            future_to_paper = {executor.submit(process_paper, paper): paper for paper in batch}
            
            # Process as they complete
            for future in tqdm(future_to_paper, total=len(batch), desc="Processing papers"):
                result = future.result()
                if result:
                    total_analyzed += 1
                    # Only include social inequality papers in final results
                    if result.get('analysis', {}).get('is_social_inequality', False):
                        batch_results.append(result)
                        total_social_inequality += 1
        
        # Add batch results to overall results
        filtered_results.extend(batch_results)
        
        # Save checkpoint after each batch
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_results, f, ensure_ascii=False, indent=2)
        
        # Log batch statistics
        batch_analyzed = len(batch)
        batch_social_inequality = len(batch_results)
        logging.info(f"Batch results: {batch_social_inequality}/{batch_analyzed} ({batch_social_inequality/batch_analyzed*100:.1f}%) are social inequality papers")
        logging.info(f"Saved checkpoint with {len(filtered_results)} social inequality papers")
        
        # Brief pause between batches
        time.sleep(2)
    
    # Log final statistics
    if total_analyzed > 0:
        logging.info(f"Analysis complete! Total papers analyzed: {total_analyzed}")
        logging.info(f"Social inequality papers identified: {total_social_inequality} ({total_social_inequality/total_analyzed*100:.1f}%)")
    else:
        logging.info(f"Analysis complete! No papers were analyzed.")
    
    return filtered_results

def load_and_filter_papers(input_file, years_range=None, max_papers=None):
    """Load papers from file with optional filtering by year"""
    # Load papers
    with open(input_file, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    logging.info(f"Loaded {len(papers)} papers from {input_file}")
    
    # Filter by year if specified
    if years_range:
        start_year, end_year = years_range
        papers = [p for p in papers if p.get('year') and start_year <= p.get('year') <= end_year]
        logging.info(f"Filtered to {len(papers)} papers from years {start_year}-{end_year}")
    
    # Apply max papers if specified
    if max_papers and len(papers) > max_papers:
        # Sort by year (descending) to get most recent papers
        papers = sorted(papers, key=lambda p: p.get('year', 0), reverse=True)
        papers = papers[:max_papers]
        logging.info(f"Limited to {len(papers)} papers")
    
    return papers

def log_analysis_statistics(results):
    """Log detailed statistics about the analysis results"""
    if not results:
        logging.info("No results to analyze statistics.")
        return
    
    # Basic counts
    total_papers = len(results)
    
    # Inequality type distribution
    inequality_types = {}
    for paper in results:
        for itype in paper.get('analysis', {}).get('inequality_type', []):
            inequality_types[itype] = inequality_types.get(itype, 0) + 1
    
    # AI relationship distribution
    ai_relationships = {}
    for paper in results:
        rel = paper.get('analysis', {}).get('ai_relationship')
        if rel:
            ai_relationships[rel] = ai_relationships.get(rel, 0) + 1
    
    # Methodology distribution
    methodologies = {}
    for paper in results:
        for method in paper.get('analysis', {}).get('methodology', []):
            methodologies[method] = methodologies.get(method, 0) + 1
    
    # Log statistics
    logging.info("===== Analysis Results Statistics =====")
    logging.info(f"Total social inequality papers: {total_papers}")
    
    logging.info("\nInequality Types Distribution:")
    for itype, count in sorted(inequality_types.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  - {itype}: {count} papers ({count/total_papers*100:.1f}%)")
    
    logging.info("\nAI Relationship Distribution:")
    for rel, count in sorted(ai_relationships.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"  - {rel}: {count} papers ({count/total_papers*100:.1f}%)")
    
    logging.info("\nMethodology Distribution:")
    for method, count in sorted(methodologies.items(), key=lambda x: x[1], reverse=True)[:10]:
        logging.info(f"  - {method}: {count} papers ({count/total_papers*100:.1f}%)")

def main(input_folder, output_folder):
    """
    Process all JSON files in the input folder and save results to the output folder.
    
    Args:
        input_folder: Directory containing JSON files to process
        output_folder: Directory to save analyzed results
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Statistics tracking across all files
    total_papers_processed = 0
    total_social_inequality_papers = 0
    
    # Get all JSON files in the input folder
    json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    logging.info(f"Found {len(json_files)} JSON files in {input_folder}")
    
    # Process each file
    for json_file in json_files:
        input_file = os.path.join(input_folder, json_file)
        output_file = os.path.join(output_folder, json_file)
        
        logging.info(f"Processing file: {json_file}")
        
        # Load papers
        papers = load_and_filter_papers(input_file)
        papers_count = len(papers)
        total_papers_processed += papers_count
        logging.info(f"Loaded {papers_count} papers from {input_file}")
                
        # Process papers
        results = process_papers_parallel(
            papers,
            output_file,
            max_workers=4,
            batch_size=50
        )
        
        total_social_inequality_papers += len(results)
        
        # Log statistics for this file
        log_analysis_statistics(results)
        
        logging.info(f"Completed processing {json_file}")
    
    # Log overall statistics
    if total_papers_processed > 0:
        logging.info("\n===== OVERALL PROCESSING STATISTICS =====")
        logging.info(f"Total papers processed: {total_papers_processed}")
        logging.info(f"Total social inequality papers: {total_social_inequality_papers}")
        logging.info(f"Overall filtering ratio: {total_social_inequality_papers/total_papers_processed*100:.1f}%")
    
    logging.info("All files have been processed!")


if __name__ == "__main__":
    import argparse

    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Process ArXiv papers with LLM analysis")
    parser.add_argument("--input_folder", required=True, help="Folder containing JSON files to process")
    parser.add_argument("--output_folder", required=True, help="Folder to save analysis results")
    
    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder)
