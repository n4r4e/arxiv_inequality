# collect_papers.py
import arxiv
import pandas as pd
from datetime import datetime
import logging
import json
from tqdm import tqdm
import re
import os
import time

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Module-level constants
API_BATCH_SIZE = 1000  # Batch size for API requests
API_DELAY_SECONDS = 3.0  # Delay between API requests
API_NUM_RETRIES = 5  # Number of retries for API requests
API_BATCH_WAIT_TIME = 10  # Seconds to wait between batch requests
RETRY_BACKOFF_FACTOR = 2  # Exponential backoff factor for retries
MAX_RESULTS_PER_QUERY = 30000  # Maximum results per query (API limit)

# Social inequality related keywords
SOCIAL_KEYWORDS = {
    'inequality', 'discrimination', 'disparity', 'fairness', 'ethics', 'bias', 
    'social', 'societal', 'economic', 'political', 'racial', 'ethnic', 'gender', 'health', 'educational', 'income', 'wealth', 'digital', 
    'geographic', 'urban', 'rural', 'environmental', 'climate', 'linguistic', 'cultural', 'religious', 'religion', 'intersectional', 
    'disability', 'ethnicity', 'age', 'ethical', 'race', 'gendered', 'sexual', 'class', 'socioeconomic', 
    'demographic', 'opportunity', 'divide', 'justice', 'gap', 'inclusion', 'exclusion', 'barrier',  
    'marginalized', 'marginal', 'vulnerable', 'group', 'community', 'population', 'segregation',
    'underrepresented', 'minority', 'disadvantaged', 'privileged', 'advantaged', 'prejudice', 'stereotype' 
}

# Social categories
SOCIAL_CATEGORIES = {
    'cs.CY'  # Computer Science and Society
}

# AI-related keywords
AI_TERMS = {
    'artificial intelligence', 'ai', 'machine learning', 'deep learning',
    'neural network', 'cnn', 'rnn', 'gnn', 
    'nlp', 'natural language processing', 'language model', 'llm', 'embedding',
    'reinforcement learning', 'supervised learning', 'unsupervised learning', 'self-supervised learning', 
    'few-shot learning', 'zero-shot learning', 'in-context learning', 'federated learning', 
    'generative ai', 'agi', 'foundation model', 'autoencoder', 'auto-encoder', 'vae', 'gan',
    'transformer', 'bert', 'gpt', 'chatgpt', 'prompt engineering',
    'human-in-the-loop', 'instruction tuning', 'fine-tuning', 'transfer learning',
    'computer vision', 'image recognition', 'facial recognition', 'object detection', 
    'diffusion model', 'stable diffusion', 
    'speech recognition', 'multi modal', 'multi-modal'
}

# Mathematical inequality related terms
MATH_INEQUALITY_TERMS = {
    'eigenvalue', 'lipschitz', 'banach', 'hilbert', 'sobolev', 'fourier',
    'cauchy', 'hölder', 'holder', 'triangle inequality', 'jensen inequality',
    'poincaré inequality', 'markov inequality', 'chebyshev inequality',
    'cauchy-schwarz', 'young inequality', 'weighted inequality', 'frobenius',
    'laplacian', 'hermitian', 'norm', 'sobolev inequality', 'brunn-minkowski',
    'talagrand inequality', 'majorization', 'symmetrization', 'rearrangement',
    'isoperimetric', 'functional inequality', 'integral inequality',
    'logarithmic inequality', 'hardy inequality', 'nash inequality',
    'operator norm', 'asymptotic bound', 'convex function', 'concentration inequality',
    'infimum', 'supremum', 'bounded by', 'upper bound', 'lower bound',
    'differential equation', 'optimization problem', 'theorem', 'lemma', 'corollary',
    'mathematical model', 'convergence rate', 'algebraic', 'topological'
}

# Compile once as global variables
MATH_INEQUALITY_PATTERN_COMPILED = [
    re.compile(r'[<>≤≥]\s*[0-9]'), 
    re.compile(r'[0-9]\s*[<>≤≥]'),
    re.compile(r'[<>≤≥]\s*C\b'), 
    re.compile(r'\bC\s*[<>≤≥]'),
    re.compile(r'≤.*≤'), 
    re.compile(r'≥.*≥')
]

# AI-related categories
AI_CATEGORIES = {'cs.AI', 'cs.LG', 'cs.CV', 'cs.CL', 'cs.NE'}

# Inequality query terms
INEQUALITY_QUERY_TERMS = '(ti:inequality OR abs:inequality OR ' \
                         'ti:discrimination OR abs:discrimination OR ' \
                         'ti:fairness OR abs:fairness OR ' \
                         'ti:"ethic*" OR abs:"ethic*" OR ' \
                         'ti:bias OR abs:bias)'

# Query restricted to CS categories
CS_ONLY_QUERY = ' AND (cat:cs.* OR cat:cs)'


def create_word_pattern(term):
    """Generate a regular expression pattern that includes plural forms of the word"""
    if term.endswith('y') and not term[-2] in 'aeiou':
        # Words ending with a consonant + 'y' (e.g., disparity -> disparities)
        return r'\b' + re.escape(term[:-1]) + r'(y|ies)\b'
    elif term.endswith(('s', 'x', 'z', 'ch', 'sh')):
        # Words ending with s, x, z, ch, or sh (e.g., bias -> biases)
        return r'\b' + re.escape(term) + r'(es)?\b'
    else:
        # General case (e.g., inequality -> inequalities)
        return r'\b' + re.escape(term) + r's?\b'
    
MATH_TERM_PATTERNS = [re.compile(create_word_pattern(term)) for term in MATH_INEQUALITY_TERMS]
SOCIAL_KEYWORD_PATTERNS = [re.compile(create_word_pattern(kw)) for kw in SOCIAL_KEYWORDS]

def create_arxiv_client(page_size=API_BATCH_SIZE, delay_seconds=API_DELAY_SECONDS, num_retries=API_NUM_RETRIES):
    """Create an arXiv API client with specified parameters
    
    Args:
        page_size: Number of results per page
        delay_seconds: Delay between API requests
        num_retries: Number of retries for API requests
        
    Returns:
        arxiv.Client: Configured arXiv client
    """
    return arxiv.Client(page_size=page_size, delay_seconds=delay_seconds, num_retries=num_retries)

def build_date_query(base_query, start_date, end_date):
    """Build a query with date range
    
    Args:
        base_query: Base search query
        start_date: Start date (YYYYMMDD format)
        end_date: End date (YYYYMMDD format)
        
    Returns:
        str: Query with date range
    """
    return f"{base_query} AND submittedDate:[{start_date} TO {end_date}]"

def fetch_batch_with_retry(client, search, max_retries=API_NUM_RETRIES):
    """Fetch a batch of results with retry mechanism
    
    Args:
        client: arXiv API client
        search: Search query
        max_retries: Maximum number of retries
        
    Returns:
        list: List of results
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            batch = list(client.results(search))
            return batch
        except Exception as e:
            retry_count += 1
            logger.warning(f"Batch processing error (attempt {retry_count}/{max_retries}): {e}")
            
            if retry_count < max_retries:
                # Exponential backoff
                wait_time = API_DELAY_SECONDS * (RETRY_BACKOFF_FACTOR ** retry_count)
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Maximum retry count exceeded: {e}")
                return []
    return []

def search_arxiv_by_date_range(query, start_date, end_date, max_results=MAX_RESULTS_PER_QUERY):
    """Search for arXiv papers within a specific date range - improved version considering API limits
    
    Args:
        query: Search query
        start_date: Start date (YYYYMMDD format)
        end_date: End date (YYYYMMDD format)
        max_results: Maximum number of results (API limit: 30000)
        
    Returns:
        tuple: (list of results, total number of results)
    """
    # Create date query
    date_query = build_date_query(query, start_date, end_date)
    logger.info(f"Date range search query: {date_query}")
    
    # Initialize variables
    all_results = []
    total_available_results = 0  # Total number of results reported by API
    first_page_total = 0  # Store total number of results reported on first page
    
    try:
        # Create client
        client = create_arxiv_client()
        
        # First batch search
        search = arxiv.Search(
            query=date_query,
            max_results=API_BATCH_SIZE,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        # Get first batch results
        first_batch = fetch_batch_with_retry(client, search)
        
        if not first_batch:
            logger.info(f"No search results found.")
            return [], 0
            
        all_results.extend(first_batch)
        
        # Check total number of results (from API response)
        if hasattr(search, 'total_results') and search.total_results:
            total_available_results = search.total_results
            first_page_total = search.total_results  # Store total number of results reported on first page
            logger.info(f"Initial filtered results for this query: {total_available_results}")
            
            # Calculate actual number of results to fetch (smaller of API limit or user limit)
            total_to_fetch = min(total_available_results, max_results)
            logger.info(f"Results to collect: {total_to_fetch} (maximum limit: {max_results})")
        else:
            # If total number of results is unknown, assume maximum
            total_to_fetch = max_results
            logger.info(f"Unable to verify total results. Retrieving up to {max_results}.")
        
        # Check number of first batch results already fetched
        fetched_so_far = len(first_batch)
        
        # Use date-based paging for safe API usage
        current_date = None
        
        # Get additional batches if there are more results and maximum limit has not been reached
        while fetched_so_far < total_to_fetch:
            # Calculate next batch size
            next_batch_size = min(API_BATCH_SIZE, total_to_fetch - fetched_so_far)
            
            if next_batch_size <= 0:
                break
                
            logger.info(f"Fetching additional batch: {fetched_so_far+1}~{fetched_so_far+next_batch_size} (out of {total_to_fetch} total)")
            
            # Check date of last fetched paper
            if len(all_results) > 0 and hasattr(all_results[-1], 'published'):
                last_date = all_results[-1].published
                
                # Check to prevent infinite loop if same date repeats
                if current_date == last_date.strftime('%Y%m%d'):
                    logger.warning(f"No further progress at date {current_date}. Ending pagination.")
                    break
                
                # Update current date
                current_date = last_date.strftime('%Y%m%d')
                
                # Create modified query for papers older than the last date
                modified_query = build_date_query(query, start_date, f"{current_date}000000")
                logger.info(f"Date-based paging query: date <= {current_date}")
            else:
                # End if no date information is found
                logger.warning("Date information not found. Ending pagination.")
                break
            
            # Delay to avoid API limits
            logger.info(f"Waiting {API_BATCH_WAIT_TIME} seconds to comply with API limits...")
            time.sleep(API_BATCH_WAIT_TIME)
            
            # Create new search with modified query
            new_search = arxiv.Search(
                query=modified_query,
                max_results=next_batch_size,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            
            # Fetch new batch with retry
            new_batch = fetch_batch_with_retry(client, new_search)
            
            if not new_batch:
                logger.info(f"No more results available. Returning {len(all_results)} total results.")
                break
            
            # Deduplication logic (based on id)
            existing_ids = {paper.entry_id for paper in all_results}
            unique_new_papers = [p for p in new_batch if p.entry_id not in existing_ids]
            
            if not unique_new_papers:
                logger.info(f"No new unique results. Returning {len(all_results)} total results.")
                break
            
            all_results.extend(unique_new_papers)
            logger.info(f"Batch complete: {len(unique_new_papers)} new results retrieved (total {len(all_results)} so far)")
            
            fetched_so_far = len(all_results)
                
        # Check if there were more results than the maximum
        if total_available_results > fetched_so_far:
            logger.warning(f"Note: Not all results were collected due to API limits or set maximum value.")
            logger.warning(f"  - Total available results: {total_available_results}")
            logger.warning(f"  - Actually collected results: {fetched_so_far}")
        
    except Exception as e:
        logger.error(f"Error during search: {e}")
        # Return results collected so far along with original total results count when error occurs
        return all_results, first_page_total
    
    logger.info(f"Total {len(all_results)} papers found (initial filtered results: {first_page_total})")
    
    # Return results along with total count
    return all_results, first_page_total

def parse_date(date_str):
    """Parse date string to datetime object with error handling
    
    Args:
        date_str: Date string
        
    Returns:
        tuple: (datetime object or None, year or None)
    """
    try:
        # Extract only date part (supports various formats)
        if 'T' in date_str:
            date_part = date_str.split('T')[0]
        elif ' ' in date_str:
            date_part = date_str.split(' ')[0]
        else:
            date_part = date_str
            
        # Date conversion
        published_date = datetime.strptime(date_part, '%Y-%m-%d')
        year = published_date.year
        return published_date, year
    except Exception as e:
        # Try to extract only year using regex
        try:
            year_match = re.search(r'(\d{4})', date_str)
            if year_match:
                year = int(year_match.group(1))
                # Temporarily set date to January 1 of that year
                return datetime(year, 1, 1), year
        except Exception:
            pass
            
        return None, None

def extract_paper_metadata(paper):
    """Extract metadata from arXiv paper object - improved date handling
    
    Args:
        paper: arXiv paper object
        
    Returns:
        dict: Paper metadata
    """
    # Extract author data
    authors = [author.name for author in paper.authors]
    
    # Process publication date - improved method
    published_date = None
    year = None
    
    if hasattr(paper, 'published'):
        published_str = str(paper.published)
        published_date, year = parse_date(published_str)
        
        if published_date is None:
            logger.warning(f"Date conversion error for ID: {paper.entry_id}")
    
    # Extract categories
    categories = paper.categories if hasattr(paper, 'categories') else []
    
    # Extract author IDs (if available)
    author_ids = []
    if hasattr(paper, 'authors'):
        for author in paper.authors:
            if hasattr(author, 'id') and author.id:
                author_ids.append(author.id)
    
    # Create metadata dictionary
    metadata = {
        'id': paper.entry_id,
        'title': paper.title,
        'authors': authors,
        'author_ids': author_ids,
        'abstract': paper.summary,
        'published_date': published_date,
        'year': year,
        'categories': categories,
    }
    
    # Include PDF URL
    if hasattr(paper, 'pdf_url') and paper.pdf_url:
        metadata['pdf_url'] = paper.pdf_url
    
    return metadata

def is_math_inequality_paper(paper_metadata):
    """Check if paper is related to mathematical inequality
    
    Args:
        paper_metadata: Paper metadata dictionary
        
    Returns:
        bool: True if paper is related to mathematical inequality
    """    
    # Check for mathematical terms in title and abstract
    title = paper_metadata.get('title', '').lower()
    abstract = paper_metadata.get('abstract', '').lower()
    content = title + ' ' + abstract
    
    # Use precompiled global patterns
    # Check if mathematical terms appear in the title
    title_match = any(pattern.search(title) for pattern in MATH_TERM_PATTERNS)
    
    # Check if multiple mathematical terms appear in the abstract
    abstract_match_count = sum(1 for pattern in MATH_TERM_PATTERNS if pattern.search(abstract))
    abstract_match = abstract_match_count >= 2  # Considered a math-related paper if 2 or more terms are found
    
    # Check for mathematical inequality expressions (e.g., "f(x) ≤ g(x)")
    inequality_match = any(pattern.search(content) for pattern in MATH_INEQUALITY_PATTERN_COMPILED)
    
    # Final decision: If any of the above conditions are met, the paper is considered math-related
    return (title_match or abstract_match or inequality_match)

def is_social_inequality_paper(paper_metadata):
    """Determine if a given paper is related to social inequality.
    
    Args:
        paper_metadata: Paper metadata dictionary
        
    Returns:
        bool: True if paper is related to social inequality
        
    Criteria:
    - Contains at least 1 social inequality keyword in the title, OR
    - Contains at least 2 social inequality keywords in the abstract
    - Papers with mathematical inequality terms (1+ in title or 2+ in abstract) are excluded
    """
    title = paper_metadata.get('title', '').lower()
    abstract = paper_metadata.get('abstract', '').lower()
    categories = paper_metadata.get('categories', [])

    text = title + " " + abstract

    # Exclude papers that include mathematical inequality terms
    math_terms_in_title = sum(1 for pattern in MATH_TERM_PATTERNS if pattern.search(title))
    math_terms_in_abstract = sum(1 for pattern in MATH_TERM_PATTERNS if pattern.search(abstract))

    if math_terms_in_title >= 1 or math_terms_in_abstract >= 2:
        return False
    
    # Count social inequality keywords
    social_keywords_in_title = sum(1 for pattern in SOCIAL_KEYWORD_PATTERNS if pattern.search(title))
    social_keywords_in_abstract = sum(1 for pattern in SOCIAL_KEYWORD_PATTERNS if pattern.search(abstract))

    # Only pass if there's at least 1 keyword in the title or 2 or more in the abstract
    return social_keywords_in_title >= 1 or social_keywords_in_abstract >= 2

def is_ai_paper(paper_metadata):
    """Check if paper is AI-related
    
    Args:
        paper_metadata: Paper metadata dictionary
        
    Returns:
        bool: True if paper is AI-related
    """
    title = paper_metadata.get('title', '').lower()
    abstract = paper_metadata.get('abstract', '').lower()
    content = title + ' ' + abstract
    categories = paper_metadata.get('categories', [])
    
    # Precompile patterns for better performance
    ai_term_patterns = [re.compile(r'\b' + re.escape(term) + r'\b') for term in AI_TERMS]
    
    # Consider AI-related if content contains AI terms or category is AI-related
    return any(pattern.search(content) for pattern in ai_term_patterns) or any(cat in AI_CATEGORIES for cat in categories)

def collect_papers_by_year(year, max_results=MAX_RESULTS_PER_QUERY):
    """Collect inequality related papers for a specific year - process entire year at once
    
    Args:
        year: Year to search
        max_results: Maximum number of results (API limit: 30000)
        
    Returns:
        tuple: (list of filtered papers, number of initial filtered results)
    """
    # Set query for social inequality related papers
    query = f"{INEQUALITY_QUERY_TERMS}{CS_ONLY_QUERY}"

    # Set date range for entire year
    start_date = f"{year}0101"  # January 1
    end_date = f"{year}1231"    # December 31
    
    # Date range search - improved function to retrieve large results (up to 30,000)
    logger.info(f"===== Starting paper collection for year {year} =====")
    papers, total_available = search_arxiv_by_date_range(query, start_date, end_date, max_results)
    
    if not papers:
        logger.warning(f"No search results for year {year}.")
        return [], total_available
    
    # Extract metadata
    all_papers_metadata = []
    for paper in tqdm(papers, desc=f"Processing papers for year {year}"):
        try:
            metadata = extract_paper_metadata(paper)
            all_papers_metadata.append(metadata)
        except Exception as e:
            logger.error(f"Error processing paper: {e}")
    
    # Filter mathematical inequality papers and select only social inequality papers
    filtered_papers = []
    for paper in all_papers_metadata:
        if not is_math_inequality_paper(paper) and is_social_inequality_paper(paper):
            # Check if filtered paper is AI-related (but don't filter)
            paper['is_ai_related'] = is_ai_paper(paper)
            filtered_papers.append(paper)
    
    logger.info(f"Year {year} total search results: {len(all_papers_metadata)} (initial filtered results: {total_available})")
    logger.info(f"Year {year} social inequality related papers: {len(filtered_papers)}")
    
    # Calculate number of AI-related papers (for reference)
    ai_papers = [p for p in filtered_papers if p.get('is_ai_related', False)]
    logger.info(f"Year {year} AI-related papers among social inequality papers: {len(ai_papers)}")
    
    # Check if collected papers reached API limit
    if total_available > max_results:
        logger.warning(f"Note: Initial filtered results for year {year} ({total_available}) exceeded maximum limit ({max_results}), some papers may have been missed.")
    
    return filtered_papers, total_available

def analyze_author_network(papers):
    """Analyze author network
    
    Args:
        papers: List of paper metadata dictionaries
        
    Returns:
        dict: Author network analysis results
    """
    author_papers = {}  # Papers per author
    author_collaborations = {}  # Collaborations between authors
    
    for paper in papers:
        authors = paper.get('authors', [])
        
        # Count papers per author
        for author in authors:
            if author not in author_papers:
                author_papers[author] = 0
            author_papers[author] += 1
        
        # Build author collaboration relationships
        for i, author1 in enumerate(authors):
            if author1 not in author_collaborations:
                author_collaborations[author1] = {}
            
            for j, author2 in enumerate(authors):
                if i != j:  # Exclude self-relationships
                    if author2 not in author_collaborations[author1]:
                        author_collaborations[author1][author2] = 0
                    author_collaborations[author1][author2] += 1
    
    # Most active authors (by paper count)
    top_authors = sorted(author_papers.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Author pairs with most collaborations
    top_collaborations = []
    for author1, collabs in author_collaborations.items():
        for author2, count in collabs.items():
            if author1 < author2:  # Prevent duplicates
                top_collaborations.append((author1, author2, count))
    
    top_collaborations = sorted(top_collaborations, key=lambda x: x[2], reverse=True)[:20]
    
    return {
        'top_authors': top_authors,
        'top_collaborations': top_collaborations,
        'total_authors': len(author_papers)
    }

def analyze_top_authors(papers, top_n=50):
    """Analyze top authors
    
    Args:
        papers: List of paper metadata dictionaries
        top_n: Number of top authors to analyze
        
    Returns:
        dict: Top authors analysis results
    """
    # Count papers per author
    author_papers = {}
    for paper in papers:
        for author in paper.get('authors', []):
            if author not in author_papers:
                author_papers[author] = 0
            author_papers[author] += 1
    
    # Extract top authors
    top_authors = sorted(author_papers.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Count AI-related papers by top authors
    top_author_ai_papers = {}
    for author, _ in top_authors:
        ai_papers = 0
        for paper in papers:
            if author in paper.get('authors', []) and paper.get('is_ai_related', False):
                ai_papers += 1
        top_author_ai_papers[author] = ai_papers
    
    # Analyze top authors' activity by year
    top_author_years = {}
    for author, _ in top_authors:
        years = []
        for paper in papers:
            if author in paper.get('authors', []) and paper.get('year'):
                years.append(paper.get('year'))
        if years:
            top_author_years[author] = {
                'min_year': min(years),
                'max_year': max(years),
                'active_years': len(set(years))
            }
    
    return {
        'top_authors': top_authors,
        'top_author_ai_papers': top_author_ai_papers,
        'top_author_years': top_author_years
    }

def save_papers_to_file(papers, year, output_dir):
    """Save collected paper data to files
    
    Args:
        papers: List of paper metadata dictionaries
        year: Year of papers
        output_dir: Output directory
    """
    if not papers:
        logger.warning(f"No papers to save for year {year}.")
        return
    
    # Set filename by year
    filename_base = os.path.join(output_dir, f"arxiv_social_inequality_{year}")
    
    # Save as CSV file
    df = pd.DataFrame(papers)
    csv_filename = f"{filename_base}.csv"
    df.to_csv(csv_filename, index=False)
    
    # Save as JSON file
    json_filename = f"{filename_base}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        papers_list = df.to_dict(orient='records')
        # Handle datetime objects
        for paper in papers_list:
            if 'published_date' in paper and isinstance(paper['published_date'], datetime):
                paper['published_date'] = paper['published_date'].isoformat()
        json.dump(papers_list, f, ensure_ascii=False, indent=2)
    
    logger.info(f"{year} - Saved {len(papers)} papers ({csv_filename}, {json_filename})")
    
    # Save author network analysis results
    author_network = analyze_author_network(papers)
    network_filename = f"{filename_base}_author_network.json"
    with open(network_filename, 'w', encoding='utf-8') as f:
        json.dump(author_network, f, ensure_ascii=False, indent=2)
    
    logger.info(f"{year} - Saved author network analysis results ({network_filename})")

def analyze_trends(all_papers):
    """Analyze trends by year
    
    Args:
        all_papers: List of all paper metadata dictionaries
        
    Returns:
        dict: Trends analysis results
    """
    # Papers per year
    years = [p.get('year') for p in all_papers if p.get('year') is not None]
    year_counts = {}
    for y in years:
        year_counts[y] = year_counts.get(y, 0) + 1
    
    # AI-related statistics by year
    ai_papers = [p for p in all_papers if p.get('is_ai_related', False)]
    ai_years = [p.get('year') for p in ai_papers if p.get('year') is not None]
    ai_year_counts = {}
    for y in ai_years:
        ai_year_counts[y] = ai_year_counts.get(y, 0) + 1
    
    # Category analysis
    categories = {}
    for paper in all_papers:
        for cat in paper.get('categories', []):
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
    
    # Most common categories
    top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:20]
    
    return {
        'year_counts': year_counts,
        'ai_year_counts': ai_year_counts,
        'top_categories': top_categories
    }

def collect_all_papers(start_year=2015, end_year=2025, max_results_per_year=MAX_RESULTS_PER_QUERY, output_dir="arxiv_papers"):
    """Collect inequality related papers for entire year range - process by year
    
    Args:
        start_year: Start year
        end_year: End year
        max_results_per_year: Maximum papers to collect per year (limited to 30,000)
        output_dir: Output directory for results
        
    Returns:
        list: All collected papers
    """
    # Create results directory
    output_path = os.path.abspath(output_dir)
    os.makedirs(output_path, exist_ok=True)

    os.makedirs(output_path, exist_ok=True)
    logger.info(f"Results directory: {output_path}")

    # List to store results for all years
    all_years_results = []
    
    # Store filtering statistics (initial filtered results and collected results by year)
    filtering_stats = {}
    
    # Set current year as end_year (don't search future years)
    current_year = datetime.now().year
    if end_year > current_year:
        end_year = current_year
    
    # Collect by year
    for year in range(end_year, start_year - 1, -1):  # From recent to past years
        # Collect papers by year
        year_papers, total_available = collect_papers_by_year(
            year, max_results_per_year
        )
        
        # Update filtering statistics
        filtering_stats[year] = {
            'total_available': total_available,
            'collected_papers': len(year_papers),
            'ai_papers_count': len([p for p in year_papers if p.get('is_ai_related', False)])
        }
        
        # Save results
        save_papers_to_file(year_papers, year, output_dir)
        
        # Add to overall results
        all_years_results.extend(year_papers)
        
        # Delay considering API limits
        if year > start_year:
            logger.info(f"Year {year} collection complete. Waiting 15 seconds before next year...")
            time.sleep(15)  # Longer wait time (15 seconds)
    
    # Save overall results
    all_results_filename = os.path.join(output_dir, "arxiv_inequality_all_years.json")
    with open(all_results_filename, 'w', encoding='utf-8') as f:
        # Handle datetime objects
        all_years_json = []
        for paper in all_years_results:
            paper_copy = paper.copy()
            if 'published_date' in paper_copy and isinstance(paper_copy['published_date'], datetime):
                paper_copy['published_date'] = paper_copy['published_date'].isoformat()
            all_years_json.append(paper_copy)
        json.dump(all_years_json, f, ensure_ascii=False, indent=2)
    
    # Save filtering statistics
    stats_filename = os.path.join(output_dir, "arxiv_filtering_stats.json")
    with open(stats_filename, 'w', encoding='utf-8') as f:
        json.dump(filtering_stats, f, ensure_ascii=False, indent=2)
    
    # Analyze overall trends
    trends = analyze_trends(all_years_results)
    trends_filename = os.path.join(output_dir, "arxiv_inequality_trends.json")
    with open(trends_filename, 'w', encoding='utf-8') as f:
        json.dump(trends, f, ensure_ascii=False, indent=2)
    
    # Create and save top 50 author statistics
    author_stats = analyze_top_authors(all_years_results, top_n=50)
    authors_filename = os.path.join(output_dir, "arxiv_top_authors.json")
    with open(authors_filename, 'w', encoding='utf-8') as f:
        json.dump(author_stats, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    logger.info("\n===== Final Collection Results =====")
    logger.info(f"Total social inequality related papers collected: {len(all_years_results)}")
    
    # Calculate AI-related paper count (for reference)
    ai_papers = [p for p in all_years_results if p.get('is_ai_related', False)]
    
    # Prevent division by zero
    if all_years_results:
        ai_percentage = len(ai_papers)/len(all_years_results)*100
        logger.info(f"Total AI-related papers among social inequality papers: {len(ai_papers)} ({ai_percentage:.1f}%)")
    else:
        logger.info(f"Total AI-related papers among social inequality papers: 0 (0.0%)")
    
    # Print statistics by year (only if results exist)
    if trends.get('year_counts'):
        logger.info("\nSocial inequality related papers by year:")
        for y in sorted(trends['year_counts'].keys()):
            logger.info(f"{y}: {trends['year_counts'][y]}")
    
    if trends.get('ai_year_counts'):
        logger.info("\nAI-related papers among social inequality papers by year:")
        for y in sorted(trends['ai_year_counts'].keys()):
            logger.info(f"{y}: {trends['ai_year_counts'][y]}")
    
    if trends.get('top_categories'):
        logger.info("\nMost common categories:")
        for cat, count in trends['top_categories']:
            logger.info(f"{cat}: {count}")
    
    if author_stats.get('top_authors'):
        logger.info("\nMost active authors Top 10:")
        for author, count in author_stats['top_authors'][:10]:
            logger.info(f"{author}: {count} papers")
    
    return all_years_results


if __name__ == "__main__":
    # Import required packages
    import argparse
    
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Collect social inequality related papers from arXiv')
    parser.add_argument('--start_year', type=int, default=2015, help='Start year for search (default: 2015)')
    parser.add_argument('--end_year', type=int, default=2025, help='End year for search (default: 2025)')
    parser.add_argument('--max_results', type=int, default=MAX_RESULTS_PER_QUERY, help=f'Maximum papers to collect per year (default: {MAX_RESULTS_PER_QUERY}, API limit)')
    parser.add_argument("--output_folder", type=str, default="arxiv_papers", help='Results directory (default: arxiv_papers)')
    
    # Parse arguments
    args = parser.parse_args()
    
    collect_all_papers(
        start_year=args.start_year, 
        end_year=args.end_year, 
        max_results_per_year=args.max_results,
        output_dir=args.output_folder
    )