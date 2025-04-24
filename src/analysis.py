import json
import os
import csv
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
import pandas as pd

# Function to load and combine all JSON files
def load_data(base_path="analyzed_texts/"):
    all_papers = []
    years = range(2015, 2026)
    
    for year in years:
        file_path = os.path.join(base_path, f"{year}.json")
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    papers = json.load(f)
                    print(f"{file_path}: Loaded {len(papers)} papers")
                    all_papers.extend(papers)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    print(f"Total papers loaded: {len(all_papers)}")
    return all_papers

# Function to apply category mapping to a value
def apply_mapping(value, category_mapping):
    if not value:
        return value
        
    for target, sources in category_mapping.items():
        if value in sources:
            return target
    
    return value

# Function to extract and count frequencies of nested list items with category mapping
def count_nested_list_items_with_mapping(papers, field_path, category_mapping=None):
    all_items = []
    path_parts = field_path.split('.')
    
    for paper in papers:
        current = paper
        valid_path = True
        
        # Navigate to nested fields
        for part in path_parts[:-1]:
            if part in current:
                current = current[part]
            else:
                valid_path = False
                break
        
        if valid_path and path_parts[-1] in current:
            items = current[path_parts[-1]]
            if items is not None:
                if isinstance(items, list):
                    all_items.extend(items)
                else:
                    all_items.append(items)
    
    # Remove None and empty values
    all_items = [item for item in all_items if item]
    
    # Apply category mapping if provided
    if category_mapping:
        mapped_items = []
        for item in all_items:
            found = False
            for target, sources in category_mapping.items():
                if item in sources:
                    mapped_items.append(target)
                    found = True
                    break
            if not found:
                mapped_items.append(item)
        all_items = mapped_items
    
    # Calculate frequencies
    counter = Counter(all_items)
    return counter

# Function to create a directory if it doesn't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Define category mappings
inequality_type_mapping = {
    "gender": ["sexual orientation", "sexuality"],
    "racial/ethnic": ["racial", "ethnic", "race", "nationality"],
    "info/digital": ["informational", "digital"],
    "socioeconomic": ["socioeconomic", "economic", "income", "class", "wealth"],
    "social": ["social", "social bias", "social discrimination", "social fairness"],
    "geographic": ["geographic", "urban-rural"],
    "religious": ["religion", "religious"],
}

geographic_focus_mapping = {
    "UK": ["UK", "United Kingdom"],
    "Europe": ["Europe", "European Union"],
    "US": ["United States", "US", "USA", "U.S."],
    "Global": ["Global", "global"],

}

ai_relationship_mapping = {
    "AI as cause/amplifier": [
        "AI as amplifier", 
        "AI as cause/amplifier", 
        "AI as amplifier of social bias",
        "AI as amplifier of social biases",
        "AI as amplifier of bias",
        "AI as amplifier of social inequalities",
        "AI as amplifier of social fairness issues",
        "AI as amplifier/discriminator",
        "AI as amplifier of disparities",
        "AI as amplifier of societal bias",
        "AI as perpetuator of social inequality",
        "AI as amplifier of systemic injustice",
        "AI as amplifier of existing biases",
        "AI as amplifier of discrimination",
        "AI as amplifier of social inequality",
        "AI as mechanism perpetuating inequality",
        "AI as amplifier of fairness issues",
        "AI as amplifier of societal biases"
    ],
    "AI as measurement tool": [
        "AI as measurement tool", 
        "Measurement tool", 
    ],
    "AI as solution": [
        "AI as solution", 
        "AI as fairness mechanism", 
        "AI as amplifier and tool for social good",
        "AI as mitigation tool"
    ],
    "AI as subject of regulation": [
        "AI as subject of regulation", 
        "AI as regulation subject", 
    ],
}

# Load all data
papers = load_data()

# Create output directories (Updated directory structure)
ensure_dir("results/advanced")
ensure_dir("results/advanced/csv")
ensure_dir("results/advanced/images")

# Helper function to replace year labels for plotting
def get_year_labels(years):
    return ['2025 (Apr)' if year == 2025 else str(year) for year in years]

# Helper function to extract items from a paper with mapping
def extract_items_from_paper(paper, field_path, category_mapping=None):
    items = []
    
    path_parts = field_path.split('.')
    current = paper
    
    for part in path_parts[:-1]:
        if part in current:
            current = current[part]
        else:
            return items
    
    if path_parts[-1] in current:
        values = current[path_parts[-1]]
        if values:
            if isinstance(values, list):
                items.extend(values)
            else:
                items.append(values)
    
    # Apply mapping if provided
    if category_mapping and items:
        mapped_items = []
        for item in items:
            found = False
            for target, sources in category_mapping.items():
                if item in sources:
                    mapped_items.append(target)
                    found = True
                    break
            if not found:
                mapped_items.append(item)
        return mapped_items
    
    return items

# Helper function to analyze yearly distribution of a specific field
def analyze_field_distribution_by_year(papers, field_path, top_n=10, category_mapping=None, selected_categories=None):
    # Get all unique years
    years = sorted(list(set(paper.get('year') for paper in papers if 2015 <= paper.get('year', 0) <= 2025)))
    
    # Count occurrences by year
    items_by_year = {}
    
    for year in years:
        year_papers = [p for p in papers if p.get('year') == year]
        items_counter = count_nested_list_items_with_mapping(year_papers, field_path, category_mapping)
        items_by_year[year] = items_counter
    
    # Identify top N categories across all years
    all_items = Counter()
    for year_counter in items_by_year.values():
        all_items.update(year_counter)
    
    if selected_categories:
        top_categories = selected_categories
    else:
        top_categories = [item for item, _ in all_items.most_common(top_n)]
    
    # Create data for visualization
    data = []
    for year in years:
        for category in top_categories:
            count = items_by_year[year].get(category, 0)
            data.append({
                'Year': year,
                'Category': category,
                'Count': count
            })
    
    df = pd.DataFrame(data)
    
    return df, top_categories, years

# 1. Number of papers per year
def analyze_papers_per_year(papers):
    papers_by_year = defaultdict(int)
    years = range(2015, 2026)
    
    for paper in papers:
        year = paper.get('year')
        if year in years:
            papers_by_year[year] += 1
    
    # Sort by year
    papers_by_year = {year: papers_by_year[year] for year in sorted(papers_by_year.keys())}
    
    # Create dataframe for visualization
    df = pd.DataFrame({
        'Year': list(papers_by_year.keys()),
        'Papers': list(papers_by_year.values())
    })
    
    # Create custom year labels
    year_labels = get_year_labels(df['Year'])
    
    # Visualization
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Year', y='Papers', data=df, color='dodgerblue')
    
    # Add value labels
    for i, v in enumerate(df['Papers']):
        ax.text(i, v + 10, f"{v:,}", ha='center')
    
    # Set x-axis ticks to show every year
    plt.xticks(range(len(year_labels)), year_labels)

    plt.title('Number of Inequality-Related Papers on arXiv per Year', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig('results/advanced/images/papers_per_year.png')
    plt.close()
    
    # Save to CSV
    df.to_csv('results/advanced/csv/papers_per_year.csv', index=False)
    
    print("Papers per year analysis completed")
    return papers_by_year

# 2. Average number of researchers per paper by year
def analyze_researchers_per_paper(papers):
    researchers_by_year = defaultdict(list)
    years = range(2015, 2026)
    
    for paper in papers:
        year = paper.get('year')
        if year in years:
            authors = paper.get('authors', [])
            researchers_by_year[year].append(len(authors))
    
    # Calculate average
    avg_researchers = {}
    for year in sorted(researchers_by_year.keys()):
        if researchers_by_year[year]:
            avg_researchers[year] = sum(researchers_by_year[year]) / len(researchers_by_year[year])
        else:
            avg_researchers[year] = 0
    
    # Create lists for plotting
    years_list = sorted(avg_researchers.keys())
    values_list = [avg_researchers[year] for year in years_list]
    
    # Create custom year labels
    year_labels = ['2025 (Apr)' if year == 2025 else str(year) for year in years_list]
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Use direct plot instead of seaborn
    plt.plot(range(len(years_list)), values_list, marker='o', linewidth=2.5, color='forestgreen')
    
    # Set x-axis ticks explicitly
    plt.xticks(range(len(years_list)), year_labels)
    
    plt.title('Average Number of Researchers per Inequality-Related Paper by Year', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Number of Researchers', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/advanced/images/avg_researchers_per_paper.png')
    plt.close()
    
    # Save to CSV
    df = pd.DataFrame({
        'Year': years_list,
        'Average Researchers': values_list
    })
    df.to_csv('results/advanced/csv/avg_researchers_per_paper.csv', index=False)
    
    print("Average researchers per paper analysis completed")
    return avg_researchers

# 3. Ratio of AI-related papers by year (using LLM classification)
def analyze_ai_related_ratio(papers):
    # Track papers by both classification methods
    llm_ai_related_by_year = defaultdict(int)
    keyword_ai_related_by_year = defaultdict(int)
    total_by_year = defaultdict(int)
    years = range(2015, 2026)
    
    for paper in papers:
        year = paper.get('year')
        if year in years:
            total_by_year[year] += 1
            
            # LLM-based classification (primary method)
            ai_relationship = paper.get('analysis', {}).get('ai_relationship')
            if ai_relationship not in [None, "Not AI-related"]:
                llm_ai_related_by_year[year] += 1
            
            # Original keyword-based classification (for comparison)
            if paper.get('is_ai_related_original', False):
                keyword_ai_related_by_year[year] += 1
    
    # Calculate ratios for both methods
    llm_ai_ratio = {}
    keyword_ai_ratio = {}
    for year in sorted(total_by_year.keys()):
        if total_by_year[year] > 0:
            llm_ai_ratio[year] = llm_ai_related_by_year[year] / total_by_year[year]
            keyword_ai_ratio[year] = keyword_ai_related_by_year[year] / total_by_year[year]
        else:
            llm_ai_ratio[year] = 0
            keyword_ai_ratio[year] = 0
    
    # Create lists for plotting
    years_list = sorted(llm_ai_ratio.keys())
    llm_ratio_values = [llm_ai_ratio[year] * 100 for year in years_list]
    keyword_ratio_values = [keyword_ai_ratio[year] * 100 for year in years_list]
    
    # Create custom year labels
    year_labels = ['2025 (Apr)' if year == 2025 else str(year) for year in years_list]
    
    # Visualization - both methods for comparison
    plt.figure(figsize=(12, 6))
    
    # Plot both lines
    plt.plot(range(len(years_list)), llm_ratio_values, marker='o', linewidth=2.5, color='purple', 
             label='LLM Classification (ai_relationship)')
    plt.plot(range(len(years_list)), keyword_ratio_values, marker='s', linewidth=2.5, linestyle='--', 
             color='gray', label='Keyword Classification (is_ai_related_original)')
    
    # Set x-axis ticks explicitly
    plt.xticks(range(len(years_list)), year_labels)
    
    plt.title('Percentage of AI-Related Inequality Papers by Year', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Percentage of Papers (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('results/advanced/images/ai_related_ratio_comparison.png')
    plt.close()
    
    # Create a separate visualization for just the LLM classification (main result)
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(years_list)), llm_ratio_values, marker='o', linewidth=2.5, color='purple')
    plt.xticks(range(len(years_list)), year_labels)
    plt.title('Percentage of AI-Related Inequality Papers by Year (LLM Classification)', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Percentage of Papers (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('results/advanced/images/ai_related_ratio.png')
    plt.close()
    
    # Save both sets of data to CSV
    df = pd.DataFrame({
        'Year': years_list,
        'LLM_AI_Related_Ratio': llm_ratio_values,
        'Keyword_AI_Related_Ratio': keyword_ratio_values,
        'LLM_AI_Related_Papers': [llm_ai_related_by_year[year] for year in years_list],
        'Keyword_AI_Related_Papers': [keyword_ai_related_by_year[year] for year in years_list],
        'Total_Papers': [total_by_year[year] for year in years_list]
    })
    df.to_csv('results/advanced/csv/ai_related_ratio.csv', index=False)
    
    print("AI-related paper ratio analysis completed (using both classification methods)")
    
    # Return the LLM-based classification (primary result)
    return llm_ai_ratio

# 4. Distribution of inequality types by year
def analyze_inequality_types_by_year(papers, top_n=10):
    df, top_categories, years = analyze_field_distribution_by_year(
        papers, 
        'analysis.inequality_type', 
        top_n=top_n,
        category_mapping=inequality_type_mapping
    )
    
    # Pivot for better visualization
    pivot_df = df.pivot(index='Year', columns='Category', values='Count').fillna(0)
    
    # Create custom year labels
    year_labels = get_year_labels(pivot_df.index)
    
    # Visualization
    plt.figure(figsize=(14, 8))
    
    # Use a colormap that's distinguishable even with many categories
    colors = plt.cm.tab10.colors
    
    for i, category in enumerate(top_categories):
        if category in pivot_df.columns:
            plt.plot(
                range(len(pivot_df.index)),
                pivot_df[category],
                marker='o',
                linewidth=2.5,
                color=colors[i % len(colors)],
                label=category
            )
    
    plt.title(f'Distribution of Top {top_n} Inequality Types by Year', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to show every year
    plt.xticks(range(len(year_labels)), year_labels)
    
    plt.tight_layout()
    plt.savefig(f'results/advanced/images/inequality_types_by_year_top{top_n}.png')
    plt.close()
    
    # Save to CSV
    pivot_df.to_csv(f'results/advanced/csv/inequality_types_by_year_top{top_n}.csv')
    
    print(f"Inequality types distribution by year analysis completed")
    return pivot_df

# 5. Distribution of methodology types by year
def analyze_methodology_types_by_year(papers, top_n=10):
    df, top_categories, years = analyze_field_distribution_by_year(
        papers, 
        'analysis.methodology', 
        top_n=top_n
    )
    
    # Pivot for better visualization
    pivot_df = df.pivot(index='Year', columns='Category', values='Count').fillna(0)
    
    # Create custom year labels
    year_labels = get_year_labels(pivot_df.index)
    
    # Visualization
    plt.figure(figsize=(14, 8))
    
    # Use a colormap that's distinguishable even with many categories
    colors = plt.cm.tab10.colors
    
    for i, category in enumerate(top_categories):
        if category in pivot_df.columns:
            plt.plot(
                range(len(pivot_df.index)),
                pivot_df[category],
                marker='o',
                linewidth=2.5,
                color=colors[i % len(colors)],
                label=category
            )
    
    plt.title(f'Distribution of Top {top_n} Methodology Types by Year', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to show every year
    plt.xticks(range(len(year_labels)), year_labels)
    
    plt.tight_layout()
    plt.savefig(f'results/advanced/images/methodology_types_by_year_top{top_n}.png')
    plt.close()
    
    # Save to CSV
    pivot_df.to_csv(f'results/advanced/csv/methodology_types_by_year_top{top_n}.csv')
    
    print(f"Methodology types distribution by year analysis completed")
    return pivot_df

# 6. Distribution of AI relationship types by year
def analyze_ai_relationship_by_year(papers):
    selected_categories = [
        "AI as measurement tool", 
        "AI as solution", 
        "Not AI-related", 
        "AI as cause/amplifier", 
        "AI as subject of regulation"
    ]
    
    df, top_categories, years = analyze_field_distribution_by_year(
        papers, 
        'analysis.ai_relationship', 
        selected_categories=selected_categories,
        category_mapping=ai_relationship_mapping
    )
    
    # Pivot for better visualization
    pivot_df = df.pivot(index='Year', columns='Category', values='Count').fillna(0)
    
    # Create custom year labels
    year_labels = get_year_labels(pivot_df.index)
    
    # Visualization
    plt.figure(figsize=(14, 8))
    
    # Use a colormap that's distinguishable even with many categories
    colors = plt.cm.tab10.colors
    
    for i, category in enumerate(selected_categories):
        if category in pivot_df.columns:
            plt.plot(
                range(len(pivot_df.index)),
                pivot_df[category],
                marker='o',
                linewidth=2.5,
                color=colors[i % len(colors)],
                label=category
            )
    
    plt.title('Distribution of AI Relationship Types by Year', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to show every year
    plt.xticks(range(len(year_labels)), year_labels)
    
    plt.tight_layout()
    plt.savefig('results/advanced/images/ai_relationship_by_year.png')
    plt.close()
    
    # Save to CSV
    pivot_df.to_csv('results/advanced/csv/ai_relationship_by_year.csv')
    
    print("AI relationship types distribution by year analysis completed")
    return pivot_df

# 7. Distribution of geographic focus by year
def analyze_geographic_focus_by_year(papers, top_n=10):
    # Filter papers with geographic focus
    papers_with_geo = [p for p in papers if p.get('analysis', {}).get('geographic_focus') 
                      and p.get('analysis', {}).get('geographic_focus') not in [None, []]]
    
    # Get all unique years
    years = sorted(list(set(paper.get('year') for paper in papers if 2015 <= paper.get('year', 0) <= 2025)))
    
    # Count geographic focuses by year
    geo_by_year = {}
    for year in years:
        year_papers = [p for p in papers_with_geo if p.get('year') == year]
        
        # Get all geographic focuses mentioned in this year
        all_geos = []
        for paper in year_papers:
            geos = paper.get('analysis', {}).get('geographic_focus', [])
            if isinstance(geos, list):
                all_geos.extend(geos)
            else:
                all_geos.append(geos)
        
        # Apply mapping and count
        mapped_geos = []
        for geo in all_geos:
            if geo:
                found = False
                for target, sources in geographic_focus_mapping.items():
                    if geo in sources:
                        mapped_geos.append(target)
                        found = True
                        break
                if not found:
                    mapped_geos.append(geo)
        
        geo_counter = Counter(mapped_geos)
        geo_by_year[year] = geo_counter
    
    # Identify top N regions across all years
    all_geos = Counter()
    for year_counter in geo_by_year.values():
        all_geos.update(year_counter)
    
    top_regions = [geo for geo, _ in all_geos.most_common(top_n)]
    
    # Create data for visualization
    data = []
    for year in years:
        for region in top_regions:
            count = geo_by_year[year].get(region, 0)
            data.append({
                'Year': year,
                'Region': region,
                'Count': count
            })
    
    df = pd.DataFrame(data)
    
    # Pivot for better visualization
    pivot_df = df.pivot(index='Year', columns='Region', values='Count').fillna(0)
    
    # Create custom year labels
    year_labels = get_year_labels(pivot_df.index)
    
    # Visualization
    plt.figure(figsize=(14, 8))
    
    # Use a colormap that's distinguishable
    colors = plt.cm.tab10.colors
    
    for i, region in enumerate(top_regions):
        if region in pivot_df.columns:
            plt.plot(
                range(len(pivot_df.index)),
                pivot_df[region],
                marker='o',
                linewidth=2.5,
                color=colors[i % len(colors)],
                label=region
            )
    
    plt.title(f'Distribution of Top {top_n} Geographic Focuses by Year', fontsize=15)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Papers', fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to show every year
    plt.xticks(range(len(year_labels)), year_labels)
    
    plt.tight_layout()
    plt.savefig(f'results/advanced/images/geographic_focus_by_year_top{top_n}.png')
    plt.close()
    
    # Save to CSV
    pivot_df.to_csv(f'results/advanced/csv/geographic_focus_by_year_top{top_n}.csv')
    
    print(f"Geographic focus distribution by year analysis completed")
    return pivot_df


# 8. Correlation between inequality types and methodology types
def analyze_inequality_methodology_correlation(papers, top_n=10):
    # Get top inequality types
    all_inequality_types = count_nested_list_items_with_mapping(
        papers, 
        'analysis.inequality_type',
        category_mapping=inequality_type_mapping
    )
    top_inequality_types = [t for t, _ in all_inequality_types.most_common(top_n)]
    
    # Get top methodology types
    all_methodology_types = count_nested_list_items_with_mapping(
        papers, 
        'analysis.methodology'
    )
    top_methodology_types = [m for m, _ in all_methodology_types.most_common(top_n)]
    
    # Create correlation matrix
    correlation_matrix = np.zeros((len(top_inequality_types), len(top_methodology_types)))
    
    for paper in papers:
        inequality_types = extract_items_from_paper(paper, 'analysis.inequality_type', inequality_type_mapping)
        methodology_types = extract_items_from_paper(paper, 'analysis.methodology')
        
        for i_type in inequality_types:
            if i_type in top_inequality_types:
                i_idx = top_inequality_types.index(i_type)
                
                for m_type in methodology_types:
                    if m_type in top_methodology_types:
                        m_idx = top_methodology_types.index(m_type)
                        correlation_matrix[i_idx, m_idx] += 1
    
    # Visualization - Heatmap
    plt.figure(figsize=(16, 10))
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(
        correlation_matrix,
        index=top_inequality_types,
        columns=top_methodology_types
    )
    
    # Calculate percentages for each inequality type
    row_sums = heatmap_df.sum(axis=1)
    percentage_df = heatmap_df.div(row_sums, axis=0) * 100
    
    # Create heatmap
    sns.heatmap(
        percentage_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        annot_kws={"size": 11}
    )
    
    plt.title('Correlation Between Inequality Types and Methodology Types (%)', fontsize=15)
    plt.ylabel('Inequality Type', fontsize=14)
    plt.xlabel('Methodology Type', fontsize=14)
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)  
    plt.tight_layout()
    plt.savefig(f'results/advanced/images/inequality_methodology_correlation_top{top_n}.png')
    plt.close()
    
    # Also save absolute numbers
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        linewidths=0.5,
        annot_kws={"size": 11}
    )
    
    plt.title('Correlation Between Inequality Types and Methodology Types (Counts)', fontsize=15)
    plt.ylabel('Inequality Type', fontsize=14)
    plt.xlabel('Methodology Type', fontsize=14)
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/advanced/images/inequality_methodology_correlation_counts_top{top_n}.png')
    plt.close()
    
    # Save to CSV
    heatmap_df.to_csv(f'results/advanced/csv/inequality_methodology_correlation_counts_top{top_n}.csv')
    percentage_df.to_csv(f'results/advanced/csv/inequality_methodology_correlation_percent_top{top_n}.csv')
    
    print("Inequality-methodology correlation analysis completed")
    return heatmap_df, percentage_df

# 9. Correlation between inequality types and AI relationship types
def analyze_inequality_ai_correlation(papers, top_n=10):
    # Get top inequality types
    all_inequality_types = count_nested_list_items_with_mapping(
        papers, 
        'analysis.inequality_type',
        category_mapping=inequality_type_mapping
    )
    top_inequality_types = [t for t, _ in all_inequality_types.most_common(top_n)]
    
    # Selected AI relationship types
    selected_ai_relationships = [
        "AI as measurement tool", 
        "AI as solution", 
        "Not AI-related", 
        "AI as cause/amplifier", 
        "AI as subject of regulation"
    ]
    
    # Create correlation matrix
    correlation_matrix = np.zeros((len(top_inequality_types), len(selected_ai_relationships)))
    
    for paper in papers:
        inequality_types = extract_items_from_paper(paper, 'analysis.inequality_type', inequality_type_mapping)
        ai_relationship = extract_items_from_paper(paper, 'analysis.ai_relationship', ai_relationship_mapping)
        
        # There should be only one AI relationship per paper, but just in case
        for ai_rel in ai_relationship:
            if ai_rel in selected_ai_relationships:
                ai_idx = selected_ai_relationships.index(ai_rel)
                
                for i_type in inequality_types:
                    if i_type in top_inequality_types:
                        i_idx = top_inequality_types.index(i_type)
                        correlation_matrix[i_idx, ai_idx] += 1
    
    # Visualization - Heatmap
    plt.figure(figsize=(16, 10))
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(
        correlation_matrix,
        index=top_inequality_types,
        columns=selected_ai_relationships
    )
    
    # Calculate percentages for each inequality type
    row_sums = heatmap_df.sum(axis=1)
    percentage_df = heatmap_df.div(row_sums, axis=0) * 100
    
    # Create heatmap
    sns.heatmap(
        percentage_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        annot_kws={"size": 11}
    )
    
    plt.title('Correlation Between Inequality Types and AI Relationship Types (%)', fontsize=15)
    plt.ylabel('Inequality Type', fontsize=14)
    plt.xlabel('AI Relationship Type', fontsize=14)
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/advanced/images/inequality_ai_correlation_top{top_n}.png')
    plt.close()
    
    # Also save absolute numbers
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        linewidths=0.5,
        annot_kws={"size": 11}
    )
    
    plt.title('Correlation Between Inequality Types and AI Relationship Types (Counts)', fontsize=15)
    plt.ylabel('Inequality Type', fontsize=14)
    plt.xlabel('AI Relationship Type', fontsize=14)
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/advanced/images/inequality_ai_correlation_counts_top{top_n}.png')
    plt.close()
    
    # Save to CSV
    heatmap_df.to_csv(f'results/advanced/csv/inequality_ai_correlation_counts_top{top_n}.csv')
    percentage_df.to_csv(f'results/advanced/csv/inequality_ai_correlation_percent_top{top_n}.csv')
    
    print("Inequality-AI relationship correlation analysis completed")
    return heatmap_df, percentage_df

# 10. Correlation between geographic focus and inequality types
def analyze_geographic_inequality_correlation(papers, top_geo_n=10, top_inequality_n=10):
    # Filter papers with geographic focus
    papers_with_geo = [p for p in papers if p.get('analysis', {}).get('geographic_focus') 
                      and p.get('analysis', {}).get('geographic_focus') not in [None, []]]
    
    # Get top geographic focuses
    all_geos = []
    for paper in papers_with_geo:
        geos = paper.get('analysis', {}).get('geographic_focus', [])
        if isinstance(geos, list):
            all_geos.extend(geos)
        else:
            all_geos.append(geos)
    
    # Apply mapping
    mapped_geos = []
    for geo in all_geos:
        if geo:
            found = False
            for target, sources in geographic_focus_mapping.items():
                if geo in sources:
                    mapped_geos.append(target)
                    found = True
                    break
            if not found:
                mapped_geos.append(geo)
    
    geo_counter = Counter(mapped_geos)
    top_geos = [geo for geo, _ in geo_counter.most_common(top_geo_n)]
    
    # Get top inequality types
    all_inequality_types = count_nested_list_items_with_mapping(
        papers, 
        'analysis.inequality_type',
        category_mapping=inequality_type_mapping
    )
    top_inequality_types = [t for t, _ in all_inequality_types.most_common(top_inequality_n)]
    
    # Create correlation matrix
    correlation_matrix = np.zeros((len(top_geos), len(top_inequality_types)))
    
    for paper in papers_with_geo:
        geos = paper.get('analysis', {}).get('geographic_focus', [])
        if not isinstance(geos, list):
            geos = [geos]
        
        inequality_types = extract_items_from_paper(paper, 'analysis.inequality_type', inequality_type_mapping)
        
        # Map geos
        mapped_paper_geos = []
        for geo in geos:
            if geo:
                found = False
                for target, sources in geographic_focus_mapping.items():
                    if geo in sources:
                        mapped_paper_geos.append(target)
                        found = True
                        break
                if not found:
                    mapped_paper_geos.append(geo)
        
        for geo in mapped_paper_geos:
            if geo in top_geos:
                g_idx = top_geos.index(geo)
                
                for i_type in inequality_types:
                    if i_type in top_inequality_types:
                        i_idx = top_inequality_types.index(i_type)
                        correlation_matrix[g_idx, i_idx] += 1
    
    # Visualization - Heatmap
    plt.figure(figsize=(16, 10))
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame(
        correlation_matrix,
        index=top_geos,
        columns=top_inequality_types
    )
    
    # Calculate percentages for each geographic focus
    row_sums = heatmap_df.sum(axis=1)
    percentage_df = heatmap_df.div(row_sums, axis=0) * 100
    
    # Create heatmap
    sns.heatmap(
        percentage_df,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        linewidths=0.5,
        annot_kws={"size": 11}
    )
    
    plt.title('Correlation Between Geographic Focus and Inequality Types (%)', fontsize=15)
    plt.ylabel('Geographic Focus', fontsize=14)
    plt.xlabel('Inequality Type', fontsize=14)
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)  
    plt.tight_layout()
    plt.savefig(f'results/advanced/images/geographic_inequality_correlation_top{top_geo_n}.png')
    plt.close()
    
    # Also save absolute numbers
    plt.figure(figsize=(16, 10))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt="g",
        cmap="YlGnBu",
        linewidths=0.5,
        annot_kws={"size": 11}
    )
    
    plt.title('Correlation Between Geographic Focus and Inequality Types (Counts)', fontsize=15)
    plt.ylabel('Geographic Focus', fontsize=14)
    plt.xlabel('Inequality Type', fontsize=14)
    plt.xticks(fontsize=12)  
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/advanced/images/geographic_inequality_correlation_counts_top{top_geo_n}.png')
    plt.close()
    
    # Save to CSV
    heatmap_df.to_csv(f'results/advanced/csv/geographic_inequality_correlation_counts_top{top_geo_n}.csv')
    percentage_df.to_csv(f'results/advanced/csv/geographic_inequality_correlation_percent_top{top_geo_n}.csv')
    
    print("Geographic focus-inequality correlation analysis completed")
    return heatmap_df, percentage_df


# Run all analyses
def run_all_advanced_analyses(papers):
    print("\n===== RUNNING ADVANCED ANALYSES =====\n")
    
    # 1. Number of papers per year
    papers_per_year = analyze_papers_per_year(papers)
    
    # 2. Average number of researchers per paper by year
    avg_researchers = analyze_researchers_per_paper(papers)
    
    # 3. Ratio of AI-related papers by year
    ai_ratio = analyze_ai_related_ratio(papers)
    
    # 4. Distribution of inequality types by year
    inequality_by_year = analyze_inequality_types_by_year(papers, top_n=10)
    
    # 5. Distribution of methodology types by year
    methodology_by_year = analyze_methodology_types_by_year(papers, top_n=10)
    
    # 6. Distribution of AI relationship types by year
    ai_relationship_by_year = analyze_ai_relationship_by_year(papers)

    # 7. Distribution of geographic focus by year
    geographic_by_year = analyze_geographic_focus_by_year(papers, top_n=10)
    
    # 8. Correlation between inequality types and methodology types
    inequality_methodology_counts, inequality_methodology_percent = analyze_inequality_methodology_correlation(papers, top_n=10)
    
    # 9. Correlation between inequality types and AI relationship types
    inequality_ai_counts, inequality_ai_percent = analyze_inequality_ai_correlation(papers, top_n=10)
    
    # 10. Correlation between inequality types and geographic focus 
    geographic_inequality_counts, geographic_inequality_percent = analyze_geographic_inequality_correlation(papers, top_geo_n=10, top_inequality_n=10)

    advanced_summary = {
        "papers_per_year": papers_per_year,
        "avg_researchers_per_paper": avg_researchers,
        "ai_related_ratio": ai_ratio,
        "inequality_types_by_year_top10": inequality_by_year.to_dict(orient="index"),
        "methodology_types_by_year_top10": methodology_by_year.to_dict(orient="index"),
        "ai_relationship_by_year": ai_relationship_by_year.to_dict(orient="index"),
        "geographic_focus_by_year_top10": geographic_by_year.to_dict(orient="index"),
        "inequality_methodology_correlation_counts": inequality_methodology_counts.to_dict(),
        "inequality_methodology_correlation_percent": inequality_methodology_percent.to_dict(),
        "inequality_ai_correlation_counts": inequality_ai_counts.to_dict(),
        "inequality_ai_correlation_percent": inequality_ai_percent.to_dict(),
        "geographic_inequality_correlation_counts": geographic_inequality_counts.to_dict(),
        "geographic_inequality_correlation_percent": geographic_inequality_percent.to_dict(),
    }

    # Save summary JSON to the main results/advanced directory
    with open("results/advanced/summary_advanced.json", "w", encoding="utf-8") as f:
        json.dump(advanced_summary, f, indent=2, ensure_ascii=False)

    print("\nAll advanced analyses completed!")
    print("Results saved in:")
    print("- CSV files: 'results/advanced/csv/' directory")
    print("- Summary JSON: 'results/advanced/summary_advanced.json'")
    print("- Visualizations: 'results/advanced/images/' directory")

# Run all analyses
run_all_advanced_analyses(papers)