import streamlit as st

st.set_page_config(layout="wide")

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
@st.cache_data
def load_data():
    with open('../results/advanced/summary_advanced.json') as f:
        data = json.load(f)
    return data

data = load_data()

st.title("📚 Trends in Social Inequality Research in Computer Science on arXiv (2015–2025)")


# Sidebar
st.sidebar.title("Navigation")
section = st.sidebar.radio("Select a section:", [
    "Papers per Year",
    "Inequality Types",
    "Methodologies",
    "AI Relevance",
    "AI Role",
    "Geographic Focus",
    "Geo Focus vs. Inequality",
    "Inequality vs. Methodology",
    "Inequality vs AI Role",
    "Authorship Trends"
])

if section == "Papers per Year":
    st.subheader("📈 Number of Papers per Year")
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> Inequality-related research in CS on arXiv has increased significantly over the last decade, with a steep rise since 2022.
    </div>
    """, unsafe_allow_html=True)

    papers_per_year = data["papers_per_year"]
    df = pd.DataFrame(papers_per_year.items(), columns=["Year", "Count"]).sort_values("Year")
    df["Year"] = df["Year"].astype(int)

    fig, ax = plt.subplots()
    ax.plot(df["Year"], df["Count"], marker='o')
    ax.set_title("Number of Papers per Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Paper Count")
    ax.grid(True)
    
    ax.tick_params(axis='both', labelsize=8)

    # add 2025 annotation
    ax.annotate("2025: up to April",
        xy=(2025, df[df["Year"] == 2025]["Count"].values[0]),
        xytext=(2024.2, df[df["Year"] == 2025]["Count"].values[0] + 30),
        fontsize=8, color="gray")
    
    st.pyplot(fig)
    st.caption("Note: 2025 data includes only up to April.")

elif section == "Inequality Types":
    st.subheader("⚖️ Inequality Types (Top 10)")
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> Racial/ethnic and gender inequality have been dominant topics overall, 
                followed by a steady rise in socioeconomic inequality. Health inequality also saw an increase in 2024.
    </div>
    """, unsafe_allow_html=True)

    inequality_data = data["inequality_types_by_year_top10"]
    df = pd.DataFrame(inequality_data).T.fillna(0).astype(int)  # Transpose and clean
    df.index = df.index.astype(int)
    df = df.sort_index()

    # Sort columns by total count across all years
    sorted_columns = df.sum().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in sorted_columns:
        ax.plot(df.index, df[col], marker='o', label=col)

    ax.set_title("Trends in Inequality Types by Year (Top 10)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    ax.grid(True)
    ax.legend(title="Inequality Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.annotate("2025: up to April",
                xy=(2025, df.loc[2025].max()),
                xytext=(2024.2, df.loc[2025].max() + 30),

                fontsize=9, color="gray")

    st.pyplot(fig)

elif section == "Methodologies":
    st.subheader("🛠️ Methodologies Used")
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> Quantitative and experimental methods has been the most commonly used approaches. 
                Experiment methods have overtaken quantitative methods since 2023.
                ML, NLP, and dataset creation methods have also increased over the past two years.
    </div>
    """, unsafe_allow_html=True)

    method_data = data["methodology_types_by_year_top10"]
    df = pd.DataFrame(method_data).T.fillna(0).astype(int)
    df.index = df.index.astype(int)
    df = df.sort_index()

    sorted_columns = df.sum().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in sorted_columns:
        ax.plot(df.index, df[col], marker='o', label=col)

    ax.set_title("Trends in Methodologies by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    ax.grid(True)
    ax.legend(title="Methodology", bbox_to_anchor=(1.05, 1), loc="upper left")

    if 2025 in df.index:
        ax.annotate("2025: up to April",
                    xy=(2025, df.loc[2025].max()),
                    xytext=(2024.2, df.loc[2025].max() + 30),
                    fontsize=9, color="gray")

    st.pyplot(fig)

elif section == "AI Relevance":
    st.subheader("🤖 AI Relevance per Year")
    
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> The proportion of inequality-related CS papers involving AI has steadily increased—rising until 2019 and remaining around 80–90% since then.
    </div>
    """, unsafe_allow_html=True)

    ai_ratio = data["ai_related_ratio"]
    df_ai = pd.DataFrame(list(ai_ratio.items()), columns=["Year", "AI Relevance"])
    df_ai["Year"] = df_ai["Year"].astype(int)
    df_ai = df_ai.sort_values("Year")
    df_ai["AI Relevance (%)"] = df_ai["AI Relevance"] * 100

    fig, ax = plt.subplots()
    ax.plot(df_ai["Year"], df_ai["AI Relevance (%)"], marker='o', color='teal')
    ax.set_title("AI Relevance in Inequality Research (% of Papers)", fontsize=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("AI Relevance (%)")
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=8)

    # add 2025 annotation
    ax.annotate("2025: up to April",
        xy=(2025, df_ai[df_ai["Year"] == 2025]["AI Relevance (%)"].values[0]),
        xytext=(2024.2, df_ai[df_ai["Year"] == 2025]["AI Relevance (%)"].values[0] + 5),
        fontsize=8, color="gray")
    
    st.pyplot(fig)
    st.caption("Note: AI relevance was classified using a large language model (LLM).")

elif section == "AI Role":
    st.subheader("🔍 AI Framing in Inequality Research")
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> AI is most often framed as a cause or amplifier of inequality since 2019, with a sharp rise in 2023 and 2024. 
                AS as Measurement tool and solution roles follow.
    </div>
    """, unsafe_allow_html=True)

    ai_data = data["ai_relationship_by_year"]
    df = pd.DataFrame(ai_data).T.fillna(0).astype(int)
    df.index = df.index.astype(int)
    df = df.sort_index()

    sorted_columns = df.sum().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in sorted_columns:
        ax.plot(df.index, df[col], marker='o', label=col)

    ax.set_title("Trends in AI Role Framing by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    ax.grid(True)
    ax.legend(title="AI Role", bbox_to_anchor=(1.05, 1), loc="upper left")

    if 2025 in df.index:
        ax.annotate("2025: up to April",
                    xy=(2025, df.loc[2025].max()),
                    xytext=(2024.2, df.loc[2025].max() + 30),
                    fontsize=9, color="gray")

    st.pyplot(fig)

elif section == "Geographic Focus":
    st.subheader("🌍 Geographic Focus in Papers (Top 10)")
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> Among studies with identified geographic focus, the US is most frequently mentioned, 
                followed by India.
    </div>
    """, unsafe_allow_html=True)

    geo_data = data["geographic_focus_by_year_top10"]
    df = pd.DataFrame(geo_data).T.fillna(0).astype(int)
    df.index = df.index.astype(int)
    df = df.sort_index()

    sorted_columns = df.sum().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in sorted_columns:
        ax.plot(df.index, df[col], marker='o', label=col)

    ax.set_title("Trends in Geographic Focus by Year (Top 10)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Mentions")
    ax.grid(True)
    ax.legend(title="Region", bbox_to_anchor=(1.05, 1), loc="upper left")

    if 2025 in df.index:
        ax.annotate("2025: up to April",
                    xy=(2025, df.loc[2025].max()),
                    xytext=(2024.2, df.loc[2025].max() + 2),
                    fontsize=9, color="gray")

    st.pyplot(fig)
    st.caption("Note: Although New York City is part of the US, it is retained as a separate geographic reference.")

elif section == "Geo Focus vs. Inequality":
    st.subheader("🔗 Geographic Focus vs. Inequality Type")
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> Geographic focus correlates with different inequality concerns: 
                the US emphasizes racial/ethnic and socioeconomic issues, Europe on racial/ethnic and gender, 
                India on gender and socioeconomic, and Africa on racial/ethnic.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Percentage View (within each region)")
    geo_ineq_pct = data["geographic_inequality_correlation_percent"]
    df_geo_ineq_pct = pd.DataFrame(geo_ineq_pct).fillna(0).T

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_geo_ineq_pct, annot=True, fmt=".1f", cmap="Greens", cbar=False, annot_kws={"size": 8}, ax=ax1)
    ax1.set_title("Geographic Region vs. Inequality Type (Percent)", fontsize=12)
    st.pyplot(fig1)

    st.markdown("#### Count View")
    geo_ineq_data = data["geographic_inequality_correlation_counts"]
    df_geo_ineq = pd.DataFrame(geo_ineq_data).fillna(0).astype(int).T

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_geo_ineq, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 8}, ax=ax2)
    ax2.set_title("Geographic Region vs. Inequality Type (Counts)", fontsize=12)
    st.pyplot(fig2)

elif section == "Inequality vs. Methodology":
    st.subheader("🔬 Inequality Type vs. Methodology")
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> Quantitative analysis and experiments are the most common methods across inequality types. 
                Geographic, info/digital, and socioeconomic inequality studies rely more on quantitative analysis, 
                while racial/ethnic, gender, and social inequality studies more often use experiments. 
                Health inequality studies show a relatively higher use of machine learning.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Percentage View (within each inequality type)")
    ineq_method_pct = data["inequality_methodology_correlation_percent"]
    df_ineq_method_pct = pd.DataFrame(ineq_method_pct).fillna(0).T

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_ineq_method_pct, annot=True, fmt=".1f", cmap="Oranges", cbar=False, annot_kws={"size": 8}, ax=ax1)
    ax1.set_title("Inequality Type vs. Methodology (Percent)", fontsize=12)
    st.pyplot(fig1)

    st.markdown("#### Count View")
    ineq_method_data = data["inequality_methodology_correlation_counts"]
    df_ineq_method = pd.DataFrame(ineq_method_data).fillna(0).astype(int).T

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_ineq_method, annot=True, fmt="d", cmap="Purples", cbar=False, annot_kws={"size": 8}, ax=ax2)
    ax2.set_title("Inequality Type vs. Methodology (Counts)", fontsize=12)
    st.pyplot(fig2)

elif section == "Inequality vs AI Role":
    st.subheader("🔄 Inequality Type vs. AI Role")
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> Correlations between inequality types and AI roles show that 
                AI is most often seen as a cause or amplifier in studies on general social inequality. 
                In age-related inequality, AI is frequently viewed as a measurement tool, 
                while geographic inequality is often seen as unrelated to AI.
    </div>
    """, unsafe_allow_html=True)

    # Percentage View (within each inequality type. row-wise normalization)
    st.markdown("#### Percentage View (within each Inequality Type)")
    ineq_ai_pct = data["inequality_ai_correlation_percent"]
    df_ineq_ai_pct = pd.DataFrame(ineq_ai_pct).fillna(0).T
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    sns.heatmap(df_ineq_ai_pct, annot=True, fmt=".1f", cmap="BuGn", cbar=False, annot_kws={"size": 8}, ax=ax1)
    ax1.set_title("Inequality Type by AI Role (Percentage within each AI Role)", fontsize=14)
    ax1.set_xlabel("Inequality Type", fontsize=10)
    ax1.set_ylabel("AI Role", fontsize=10)
    ax1.tick_params(axis='x', labelsize=8, rotation=0)
    ax1.tick_params(axis='y', labelsize=9)
    st.pyplot(fig1)

    st.markdown("#### Count View")
    ai_ineq_counts = data["inequality_ai_correlation_counts"]
    df_ineq_ai = pd.DataFrame(ai_ineq_counts).fillna(0).astype(int).T

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_ineq_ai, annot=True, fmt="d", cmap="Oranges", cbar=False, annot_kws={"size": 8}, ax=ax2)
    ax2.set_title("Inequality Type vs. AI Role (Counts)", fontsize=14)
    ax2.set_xlabel("Inequality Type", fontsize=10)
    ax2.set_ylabel("AI Role", fontsize=10)
    ax2.tick_params(axis='x', labelsize=8, rotation=0)
    ax2.tick_params(axis='y', labelsize=9)
    st.pyplot(fig2)

elif section == "Authorship Trends":
    st.subheader("👥 Authorship Trends Over Time")
    st.markdown("""
    <div style='font-size:20px'>
        <strong>Finding:</strong> Number of researchers per paper steadily increased—suggesting more collaborative work.
    </div>
    """, unsafe_allow_html=True)

    author_data = data["avg_researchers_per_paper"]
    df = pd.DataFrame(author_data.items(), columns=["Year", "AvgAuthors"]).sort_values("Year")
    df["Year"] = df["Year"].astype(int)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Year"], df["AvgAuthors"], marker='o', color="teal")
    ax.set_title("Average Number of Authors per Paper by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Average Author Count")
    ax.grid(True)

    # add 2025 annotation
    if 2025 in df["Year"].values:
        y_2025 = df[df["Year"] == 2025]["AvgAuthors"].values[0]
        ax.annotate("2025: up to April",
                    xy=(2025, y_2025),
                    xytext=(2024.2, y_2025 + 0.1),
                    fontsize=9, color="gray")

    st.pyplot(fig)