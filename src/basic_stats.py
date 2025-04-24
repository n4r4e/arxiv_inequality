import json
import os
import csv
from collections import Counter
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

# Function to find affected populations for papers with 'other' inequality type
def analyze_other_inequality_affected_populations(papers):
    other_affected_populations = []
    
    for paper in papers:
        if 'analysis' in paper and 'inequality_type' in paper['analysis']:
            inequality_types = paper['analysis']['inequality_type']
            if inequality_types and any(t.lower() == 'other' for t in inequality_types):
                if 'affected_populations' in paper['analysis'] and paper['analysis']['affected_populations']:
                    other_affected_populations.extend(paper['analysis']['affected_populations'])
    
    counter = Counter(other_affected_populations)
    return counter

# Function to save counter results to CSV
def save_counter_to_csv(counter, filename, total_count=None):
    # directory structure: results/basic/csv/
    result_dir = os.path.join("results", "basic", "csv")
    os.makedirs(result_dir, exist_ok=True)
    filepath = os.path.join(result_dir, filename)
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if total_count:
            writer.writerow(['Item', 'Frequency', 'Percentage(%)'])
        else:
            writer.writerow(['Item', 'Frequency'])
        
        for item, count in counter.most_common():
            if total_count:
                percentage = (count / total_count) * 100
                writer.writerow([item, count, f"{percentage:.2f}"])
            else:
                writer.writerow([item, count])
    
    print(f"Results saved to {filepath}")
    return filepath

# Function to save summary results to JSON
def save_summary_to_json(results):
    # Keep summary JSON in the main results/basic directory
    result_dir = os.path.join("results", "basic")
    os.makedirs(result_dir, exist_ok=True)
    filepath = os.path.join(result_dir, "summary_basic.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"Summary results saved to {filepath}")
    return filepath

# Function to plot top N counts and save the figure
def plot_top_n_counts(counter, n=15, title="", filename="", figsize=(12, 8), color='skyblue'):
    items, counts = zip(*counter.most_common(n))
    items = list(items)
    counts = list(counts)
    items.reverse()
    counts.reverse()
    
    plt.figure(figsize=figsize)
    bars = plt.barh(items, counts, color=color)
    
    for i, (bar, value) in enumerate(zip(bars, counts)):
        plt.text(
            bar.get_width() + (max(counts) * 0.01),
            bar.get_y() + bar.get_height()/2,
            f"{value:,}",
            va='center',
            fontweight='bold',
            fontsize=10,
            color='black'
        )
    
    plt.xlim(0, max(counts) * 1.1)
    plt.title(title, fontsize=15)
    plt.xlabel('Number of Papers', fontsize=12)
    plt.ylabel('', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # DSirectory structure: results/basic/images/
    viz_dir = os.path.join("results", "basic", "images")
    os.makedirs(viz_dir, exist_ok=True)
    filepath = os.path.join(viz_dir, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Visualization saved to {filepath}")

# Load all data
papers = load_data()
total_papers = len(papers)

# Basic analysis and result storage
print("\n===== BASIC ANALYSIS AND RESULT STORAGE =====\n")

# Dictionary to store summary results
summary_results = {
    "total_papers": total_papers,
    "statistics": {}
}

# Define category mappings
## These mappings were created after reviewing initial results to group similar terms together.
## You can add appropriate groupings below to standardize terminology in your analysis.
inequality_type_mapping = {
    # "gender": ["sexual orientation", "sexuality"],
    # "racial/ethnic": ["racial", "ethnic", "race", "nationality"],
    # "info/digital": ["informational", "digital"],
    # "socioeconomic": ["socioeconomic", "economic", "income", "class", "wealth"],
    # "social": ["social", "social bias", "social discrimination", "social fairness"],
    # "geographic": ["geographic", "urban-rural"],
    # "religious": ["religion", "religious"],
}

geographic_focus_mapping = {
    # "UK": ["UK", "United Kingdom"],
    # "Europe": ["Europe", "European Union"],
    # "US": ["United States", "US", "USA", "U.S."],
    # "Global": ["Global", "global"],
}

ai_relationship_mapping = {
    # "AI as cause/amplifier": [
    #     "AI as amplifier", 
    #     "AI as cause/amplifier", 
    #     "AI as amplifier of social bias",
    #     "AI as amplifier of social biases",
    #     "AI as amplifier of bias",
    #     "AI as amplifier of social inequalities",
    #     "AI as amplifier of social fairness issues",
    #     "AI as amplifier/discriminator",
    #     "AI as amplifier of disparities",
    #     "AI as amplifier of societal bias",
    #     "AI as perpetuator of social inequality",
    #     "AI as amplifier of systemic injustice",
    #     "AI as amplifier of existing biases",
    #     "AI as amplifier of discrimination",
    #     "AI as amplifier of social inequality",
    #     "AI as mechanism perpetuating inequality",
    #     "AI as amplifier of fairness issues",
    #     "AI as amplifier of societal biases"
    # ],
    # "AI as measurement tool": [
    #     "AI as measurement tool", 
    #     "Measurement tool", 
    # ],
    # "AI as solution": [
    #     "AI as solution", 
    #     "AI as fairness mechanism", 
    #     "AI as amplifier and tool for social good",
    #     "AI as mitigation tool"
    # ],
    # "AI as subject of regulation": [
    #     "AI as subject of regulation", 
    #     "AI as regulation subject", 
    # ],
}

# 1. Inequality type analysis with mapping
inequality_types = count_nested_list_items_with_mapping(
    papers, 
    'analysis.inequality_type',
    inequality_type_mapping
)
print(f"Number of unique inequality types (after grouping): {len(inequality_types)}")
print(f"Top 10 inequality types: {dict(inequality_types.most_common(10))}")

# Save to CSV
inequality_csv = save_counter_to_csv(inequality_types, "inequality_types_grouped.csv", total_count=sum(inequality_types.values()))

# Store summary information
summary_results["statistics"]["inequality_types_grouped"] = {
    "unique_count": len(inequality_types),
    "total_mentions": sum(inequality_types.values()),
    "top_10": dict(inequality_types.most_common(10))
}

# 2. Affected populations analysis (no mapping needed)
affected_populations = count_nested_list_items_with_mapping(papers, 'analysis.affected_populations')
print(f"\nNumber of unique affected populations: {len(affected_populations)}")
print(f"Top 10 affected populations: {dict(affected_populations.most_common(10))}")

# Save to CSV
populations_csv = save_counter_to_csv(affected_populations, "affected_populations.csv", total_count=sum(affected_populations.values()))

# Store summary information
summary_results["statistics"]["affected_populations"] = {
    "unique_count": len(affected_populations),
    "total_mentions": sum(affected_populations.values()),
    "top_10": dict(affected_populations.most_common(10))
}

# 3. Methodology analysis (no mapping needed)
methodologies = count_nested_list_items_with_mapping(papers, 'analysis.methodology')
print(f"\nNumber of unique methodologies: {len(methodologies)}")
print(f"Top 10 methodologies: {dict(methodologies.most_common(10))}")

# Save to CSV
methodologies_csv = save_counter_to_csv(methodologies, "methodologies.csv", total_count=sum(methodologies.values()))

# Store summary information
summary_results["statistics"]["methodologies"] = {
    "unique_count": len(methodologies),
    "total_mentions": sum(methodologies.values()),
    "top_10": dict(methodologies.most_common(10))
}

# 4. AI relationship analysis with mapping
ai_relationships = count_nested_list_items_with_mapping(
    papers, 
    'analysis.ai_relationship',
    ai_relationship_mapping
)
print(f"\nNumber of unique AI relationships (after grouping): {len(ai_relationships)}")
print(f"All AI relationships: {dict(ai_relationships.most_common())}")

# Save to CSV
ai_csv = save_counter_to_csv(ai_relationships, "ai_relationships_grouped.csv", total_count=sum(ai_relationships.values()))

# Store summary information
summary_results["statistics"]["ai_relationships_grouped"] = {
    "unique_count": len(ai_relationships),
    "total_mentions": sum(ai_relationships.values()),
    "all_relationships": dict(ai_relationships.most_common())
}

# 5. Geographic focus analysis with mapping
geographic_focuses = count_nested_list_items_with_mapping(
    papers, 
    'analysis.geographic_focus',
    geographic_focus_mapping
)
print(f"\nNumber of unique geographic focuses (after grouping): {len(geographic_focuses)}")
print(f"Top 10 geographic focuses: {dict(geographic_focuses.most_common(10))}")

# Save to CSV
geo_csv = save_counter_to_csv(geographic_focuses, "geographic_focuses_grouped.csv", total_count=sum(geographic_focuses.values()))

# Store summary information
summary_results["statistics"]["geographic_focuses_grouped"] = {
    "unique_count": len(geographic_focuses),
    "total_mentions": sum(geographic_focuses.values()),
    "top_10": dict(geographic_focuses.most_common(10))
}

# 6. Analyze 'other' inequality type's affected populations
other_inequality_populations = analyze_other_inequality_affected_populations(papers)
print(f"\nNumber of unique affected populations in 'other' inequality types: {len(other_inequality_populations)}")
print(f"Top 10 affected populations in 'other' inequality types: {dict(other_inequality_populations.most_common(10))}")

# Save to CSV
other_inequality_csv = save_counter_to_csv(other_inequality_populations, "other_inequality_affected_populations.csv")

# Store summary information
summary_results["statistics"]["other_inequality_affected_populations"] = {
    "unique_count": len(other_inequality_populations),
    "total_mentions": sum(other_inequality_populations.values()),
    "top_10": dict(other_inequality_populations.most_common(10))
}

# 7. Calculate AI-related paper ratio using LLM classification
# Use the LLM classification to determine AI-related papers
ai_related_llm_count = sum(1 for paper in papers 
                        if paper.get('analysis', {}).get('ai_relationship') not in [None, "Not AI-related"])
ai_related_llm_ratio = ai_related_llm_count / total_papers
print(f"\nNumber of AI-related papers (LLM classification): {ai_related_llm_count}")
print(f"Ratio of AI-related papers (LLM classification): {ai_related_llm_ratio:.4f}")

# Use keyword-based classification for comparison
ai_related_keyword_count = sum(1 for paper in papers if paper.get('is_ai_related_original', False))
ai_related_keyword_ratio = ai_related_keyword_count / total_papers
print(f"Number of AI-related papers (keyword method): {ai_related_keyword_count}")
print(f"Ratio of AI-related papers (keyword method): {ai_related_keyword_ratio:.4f}")

# Store summary information using the improved LLM classification
summary_results["statistics"]["ai_related"] = {
    "llm_classification": {
        "ai_related_papers": ai_related_llm_count,
        "ratio": ai_related_llm_ratio
    },
    "keyword_classification": {
        "ai_related_papers": ai_related_keyword_count,
        "ratio": ai_related_keyword_ratio
    }
}

# 7.5 Analyze relationship between keyword-based and LLM-based AI classification
print("\n===== COMPARISON OF AI CLASSIFICATION METHODS =====")

# Count papers in each category
true_positive = 0  # Both methods say it's AI-related
false_positive = 0  # Keyword says it's AI-related, but LLM says it's not
false_negative = 0  # Keyword says it's not AI-related, but LLM says it is
true_negative = 0  # Both methods say it's not AI-related

# Examples from each category (for analysis)
disagreement_examples = {
    "false_positive": [],
    "false_negative": []
}

for paper in papers:
    keyword_result = paper.get('is_ai_related_original', False)
    llm_result = paper.get('analysis', {}).get('ai_relationship') not in [None, "Not AI-related"]
    
    if keyword_result and llm_result:
        true_positive += 1
    elif keyword_result and not llm_result:
        false_positive += 1
        # Store a few examples of disagreement
        if len(disagreement_examples["false_positive"]) < 5:
            disagreement_examples["false_positive"].append({
                "title": paper.get('title', ''),
                "ai_relationship": paper.get('analysis', {}).get('ai_relationship', '')
            })
    elif not keyword_result and llm_result:
        false_negative += 1
        # Store a few examples of disagreement
        if len(disagreement_examples["false_negative"]) < 5:
            disagreement_examples["false_negative"].append({
                "title": paper.get('title', ''),
                "ai_relationship": paper.get('analysis', {}).get('ai_relationship', '')
            })
    else:
        true_negative += 1

# Calculate metrics
total = true_positive + false_positive + false_negative + true_negative
agreement_rate = (true_positive + true_negative) / total if total > 0 else 0
disagreement_rate = (false_positive + false_negative) / total if total > 0 else 0

# If treating LLM as ground truth:
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print comparison results
print(f"Agreement rate between methods: {agreement_rate:.4f} ({(agreement_rate*100):.1f}%)")
print(f"Disagreement rate between methods: {disagreement_rate:.4f} ({(disagreement_rate*100):.1f}%)")
print("\nConfusion matrix (treating LLM classification as reference):")
print(f"True Positive: {true_positive} papers (both methods classify as AI-related)")
print(f"False Positive: {false_positive} papers (keyword says AI-related, LLM says not)")
print(f"False Negative: {false_negative} papers (keyword says not AI-related, LLM says it is)")
print(f"True Negative: {true_negative} papers (both methods classify as not AI-related)")

print(f"\nKeyword method precision: {precision:.4f}")
print(f"Keyword method recall: {recall:.4f}")
print(f"Keyword method F1-score: {f1_score:.4f}")

# Print a few examples of disagreements
print("\nExamples of False Positives (keyword method wrongly classified as AI-related):")
for i, example in enumerate(disagreement_examples["false_positive"]):
    print(f"{i+1}. Title: {example['title']}")
    print(f"   LLM classification: {example['ai_relationship']}")

print("\nExamples of False Negatives (keyword method missed AI-related papers):")
for i, example in enumerate(disagreement_examples["false_negative"]):
    print(f"{i+1}. Title: {example['title']}")
    print(f"   LLM classification: {example['ai_relationship']}")

# Store comparison results in summary
summary_results["statistics"]["ai_classification_comparison"] = {
    "agreement_rate": agreement_rate,
    "disagreement_rate": disagreement_rate,
    "confusion_matrix": {
        "true_positive": true_positive,
        "false_positive": false_positive, 
        "false_negative": false_negative,
        "true_negative": true_negative
    },
    "keyword_method_metrics": {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    },
    "disagreement_examples": disagreement_examples
}

# Create visualizations for AI classification comparison
fig, ax = plt.subplots(figsize=(10, 6))
confusion_data = [
    [true_negative, false_negative],
    [false_positive, true_positive]
]
sns.heatmap(confusion_data, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["LLM: Not AI", "LLM: AI-related"],
            yticklabels=["Keyword: Not AI", "Keyword: AI-related"],
            cbar=False, ax=ax)
plt.title("Confusion Matrix of AI Classification Methods")
plt.tight_layout()

# Save visualization
viz_dir = os.path.join("results", "basic", "images")
os.makedirs(viz_dir, exist_ok=True)
confusion_filepath = os.path.join(viz_dir, "ai_classification_comparison.png")
plt.savefig(confusion_filepath)
plt.close()
print(f"AI classification comparison visualization saved to {confusion_filepath}")

# 7.5 ends

# Create bar chart showing AI relations from LLM classification
ai_relation_types = {}
for paper in papers:
    relation = paper.get('analysis', {}).get('ai_relationship')
    if relation:
        ai_relation_types[relation] = ai_relation_types.get(relation, 0) + 1

# Sort by count (descending)
sorted_relations = sorted(ai_relation_types.items(), key=lambda x: x[1], reverse=True)
relation_names, relation_counts = zip(*sorted_relations)

plt.figure(figsize=(12, 6))
bars = plt.bar(relation_names, relation_counts, color='cornflowerblue')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{height}', ha='center', va='bottom')

plt.title('Distribution of AI Relationships in Papers (LLM Classification)')
plt.xlabel('AI Relationship Type')
plt.ylabel('Number of Papers')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save visualization
relation_filepath = os.path.join(viz_dir, "ai_relationship_distribution.png")
plt.savefig(relation_filepath)
plt.close()
print(f"AI relationship distribution visualization saved to {relation_filepath}")

# 8. Calculate papers with geographic focus ratio
geo_focus_count = sum(1 for paper in papers 
                      if paper.get('analysis', {}).get('geographic_focus') 
                      and paper.get('analysis', {}).get('geographic_focus') not in [None, []])
geo_focus_ratio = geo_focus_count / total_papers
print(f"Number of papers with geographic focus: {geo_focus_count}")
print(f"Ratio of papers with geographic focus: {geo_focus_ratio:.4f}")

# Store summary information
summary_results["statistics"]["geographic_focus_presence"] = {
    "papers_with_geographic_focus": geo_focus_count,
    "ratio": geo_focus_ratio
}

# 9. Calculate average confidence value
confidence_values = [paper.get('analysis', {}).get('confidence', 0) for paper in papers]
confidence_values = [v for v in confidence_values if v is not None]  # Remove None values
avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
print(f"Average confidence value: {avg_confidence:.4f}")

# Store summary information
summary_results["statistics"]["confidence"] = {
    "average_value": avg_confidence,
    "papers_with_confidence_value": len(confidence_values)
}

# Save summary information to JSON
summary_json = save_summary_to_json(summary_results)

# Create visualizations
print("\n===== CREATING VISUALIZATIONS =====\n")

# Top 10 inequality types visualization
plot_top_n_counts(
    inequality_types, 
    n=10, 
    title="Top 10 Inequality Types in arXiv Papers (2015-2025)",
    filename="top_inequality_types_grouped.png"
)

# Top 10 affected populations visualization
plot_top_n_counts(
    affected_populations, 
    n=10, 
    title="Top 10 Affected Populations in Inequality Research (2015-2025)",
    filename="top_affected_populations.png",
    color='lightgreen'
)

# Top 10 methodologies visualization
plot_top_n_counts(
    methodologies, 
    n=10, 
    title="Top 10 Methodologies in Inequality Research (2015-2025)",
    filename="top_methodologies.png",
    color='salmon'
)

# Top 10 geographic focuses visualization
plot_top_n_counts(
    geographic_focuses, 
    n=10, 
    title="Top 10 Geographic Focuses in arXiv Inequality Papers (2015-2025)",
    filename="top_geographic_focuses_grouped.png",
    color='purple'
)

# Top 10 affected populations in 'other' inequality types
plot_top_n_counts(
    other_inequality_populations, 
    n=10, 
    title="Top 10 Affected Populations in 'Other' Inequality Types (2015-2025)",
    filename="other_inequality_affected_populations.png",
    color='orange'
)

print("\nAnalysis complete!")
print(f"- CSV result files are saved in the 'results/basic/csv' directory")
print(f"- Summary JSON file: {summary_json}")
print(f"- Visualizations are saved in the 'results/basic/images' directory")