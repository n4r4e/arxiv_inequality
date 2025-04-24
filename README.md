# Analysis of Social Inequality Research in Computer Science (ArXiv Publications 2015 - Apr.2025)
An analysis of social inequality research trends in Computer Science publications on ArXiv from 2015 to April 2025.

# Project Overview
This project explores how research on social inequality has evolved in the Computer Science field over the past decade. Using ArXiv—a platform where researchers voluntarily share their work before formal publication in journals or conferences—as the data source, we aimed to understand:

- Which types of social inequality are predominantly studied
- Which methodologies are commonly employed
- Whether studies are AI-related, and if so, how AI is viewed in relation to inequality
- Temporal and geographical trends in inequality research

# Process
The project followed a three-stage process:

### 1. Data Collection
Using the ArXiv API to collect CS papers published between 2015 and April 23, 2025, containing inequality-related keywords in their titles or abstracts.

### 2. Content Analysis
Analyzing each paper's title and abstract using a language model (OpenAI GPT-4.1-nano) to extract structured information about:

- Whether it addresses social inequality; If so:
    - What type of social inequality it examines (racial, gender, class, etc.)
    - Research methodologies employed
    - How AI is perceived in relation to inequality
    - Geographic locations discussed in the paper

### 3. Data Analysis
Performing statistical analysis, visualization, and correlation studies on the extracted data to identify patterns and trends in inequality research.

# Repository Structure
```
arxiv_inequality/
├── src/                      # Source code
│   ├── collect_papers.py     # Paper collection script
│   ├── extract_info.py       # LLM-based paper analysis 
│   ├── basic_stats.py        # Basic statistical analysis
│   └── analysis.py           # Advanced analysis and visualization
├── data/                      # Collected and processed data
│   ├── collect_papers/       # Collected papers
│   ├── extracted_info/       # Extracted structured data
├── results/                  # Analysis outputs
│   ├── basic/                # Basic statistical results
│   ├── advanced/             # Advanced analysis results
├── docs/                     # Documentation and reports
│   └── report.md             # Analysis report (to be added)
└── README.md                 # Project documentation
```

# Requirements
```
matplotlib==3.10.1
numpy==2.2.5
openai==1.76.0
pandas==2.2.3
python-dotenv==1.1.0
Requests==2.32.3
seaborn==0.13.2
tqdm==4.67.1
```

# Installation

### 1. Clone the repository:

```bash
git clone https://github.com/n4r4e/arxiv_inequality.git
cd arxiv_inequality
```

### 2. Install required packages:

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Edit the .env file in the .src directory

# Usage

### 1. Collect papers from ArXiv:
```bash
python src/collect_papers.py --start_year 2015 --end_year 2025 --output_folder data/arxiv_papers
```

### 2. Extract structured information using LLM:

```bash
python src/extract_info.py --input_folder data/arxiv_papers --output_folder data/analyzed_texts
```

### 3. Generate analysis and visualizations:

#### 3.1. Basic statistics:
```bash
python src/basic_stats.py
```

#### 3.2. Advanced analysis:

```bash
python src/analysis.py
```

# Limitations
- The arXiv API does not provide researcher affiliation data, limiting geographic analysis at the author or institution level. This could be addressed in future work.
- The initial collection of inequality-related papers relied on keyword matching, potentially missing papers that discuss inequality without using these exact keywords.
- Data extraction was performed using a language model, so results may vary across runs and depend on the model version used.