import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama
import os

# ===============================
# AI-Powered Insights using Ollama
# ===============================
def generate_ai_insights(df_summary):
    try:
        prompt = f"""
Analyze the dataset summary and provide insights in bullet points.
Focus on patterns, anomalies, and suggestions.

{df_summary}
"""
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"❌ Ollama error: {e}"

# ===============================
# Generate Visualizations
# ===============================
def generate_visualizations(df):
    plot_paths = []

    # Histograms for numeric columns
    for col in df.select_dtypes(include="number").columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f"Distribution of {col}")
        path = f"{col}_distribution.png"
        plt.savefig(path)
        plt.close()
        plot_paths.append(path)

    # Correlation heatmap
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        plt.figure(figsize=(8, 5))
        sns.heatmap(
            numeric_df.corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5
        )
        plt.title("Correlation Heatmap")
        path = "correlation_heatmap.png"
        plt.savefig(path)
        plt.close()
        plot_paths.append(path)

    return plot_paths

# ===============================
# Main EDA Function
# ===============================
def eda_analysis(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Summary + missing values
    summary = df.describe(include="all").to_string()
    missing_values = df.isnull().sum().to_string()

    # AI insights
    insights = generate_ai_insights(summary)

    # Visualizations
    plot_paths = generate_visualizations(df)

    report = (
        "✅ Data Loaded Successfully\n\n"
        "📊 DATA SUMMARY:\n"
        f"{summary}\n\n"
        "❓ MISSING VALUES:\n"
        f"{missing_values}\n\n"
        "🤖 AI INSIGHTS:\n"
        f"{insights}"
    )

    return report, plot_paths

# ===============================
# Gradio Interface
# ===============================
demo = gr.Interface(
    fn=eda_analysis,
    inputs=gr.File(type="filepath", label="Upload CSV File"),
    outputs=[
        gr.Textbox(label="EDA Report", lines=25),
        gr.Gallery(label="📈 Data Visualizations")
    ],
    title="📊 LLM-Powered Exploratory Data Analysis (EDA)",
    description="Upload a CSV file to get automated EDA with AI insights and visualizations."
)

# ===============================
# Launch App
# ===============================
demo.launch(share=True)
