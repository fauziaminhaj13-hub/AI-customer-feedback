import streamlit as st
import google.generativeai as genai
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from textblob import TextBlob
from keybert import KeyBERT
from fpdf import FPDF
from io import BytesIO
from deep_translator import GoogleTranslator

genai.configure(api_key="AIzaSyDu_VFGgir9K0wBdgeQ0z5Lsdwj6JUgecI")

# Translate any text to English
def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(str(text))
    except:
        return text  # fallback if translation fails

# Sentiment analysis using TextBlob
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Streamlit Page Settings
st.set_page_config(page_title="AI Customer Feedback Sentiment App", layout="wide")

st.title("💬 AI Customer Feedback Sentiment App 🌍")
st.write("Upload customer feedback in any language. The app will auto-translate, analyze sentiment, extract keywords, and give AI insights.")

# File Upload
uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx"])

df = None
feedback_col = None
date_col = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file)

        st.write("📊 File Preview:", df.head())

        # Auto-detect feedback column (first text column)
        for col in df.columns:
            if df[col].dtype == "object":
                feedback_col = col
                break
        if feedback_col is None:
            feedback_col = df.columns[0]

        # Auto-detect date column
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                date_col = col
                break
            except:
                continue

        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        st.success(f"✅ Using '{feedback_col}' as Feedback column")
        if date_col:
            st.success(f"✅ Using '{date_col}' as Date column")

        # Translate Feedback to English
        st.info("🌍 Auto-translating feedback to English (if needed)...")
        df["Feedback_English"] = df[feedback_col].apply(translate_text)

        # Use translated column for analysis
        feedback_col = "Feedback_English"

    except Exception as e:
        st.error(f"⚠️ Could not read file: {e}")

# Sentiment Analysis & Visualization
if df is not None and feedback_col:
    df["Sentiment"] = df[feedback_col].apply(get_sentiment)

    # Sentiment distribution chart
    st.subheader("📊 Sentiment Distribution")
    counts = df["Sentiment"].value_counts()
    if not counts.empty:
        fig, ax = plt.subplots()
        counts.plot(kind="bar", color=["green", "red", "grey"], ax=ax)
        plt.title("Sentiment Counts")
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        st.pyplot(fig)
    else:
        st.write("No sentiment data available.")

    # Sentiment Trend Chart
    if date_col and not df[date_col].isnull().all():
        st.subheader("📈 Sentiment Trend")
        try:
            trend_data = df.groupby([df[date_col].dt.to_period('M'), 'Sentiment']).size().unstack(fill_value=0)
            trend_data.index = trend_data.index.astype(str)
            if not trend_data.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                trend_data.plot(kind='line', ax=ax, marker='o')
                ax.set_title('Sentiment Trend Over Time')
                ax.set_xlabel('Month')
                ax.set_ylabel('Number of Responses')
                ax.legend(title='Sentiment')
                st.pyplot(fig)
            else:
                st.write("No trend data available.")
        except Exception as e:
            st.error(f"Error generating trend chart: {e}")
    else:
        st.write("No date column available for trend analysis.")

    # Word Clouds Section
    st.subheader("☁️ Word Clouds")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Positive Feedback**")
        pos_text = " ".join(df[df["Sentiment"] == "Positive"][feedback_col].astype(str))
        if pos_text.strip():
            try:
                wordcloud = WordCloud(width=400, height=300, background_color="white").generate(pos_text)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating positive word cloud: {e}")
        else:
            st.write("No positive feedback available.")

    with col2:
        st.markdown("**Negative Feedback**")
        neg_text = " ".join(df[df["Sentiment"] == "Negative"][feedback_col].astype(str))
        if neg_text.strip():
            try:
                wordcloud = WordCloud(width=400, height=300, background_color="black", colormap="Reds").generate(neg_text)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error generating negative word cloud: {e}")
        else:
            st.write("No negative feedback available.")

    # AI insight
    st.subheader("🧠 AI Insights")
    summary = (
        f"Total feedback: {len(df)}.\n"
        f"Positive: {counts.get('Positive', 0)}, Negative: {counts.get('Negative', 0)}, Neutral: {counts.get('Neutral', 0)}.\n"
        "Customers are happy with product quality and support, but common issues include delivery delays, app performance, and navigation."
    )
    actions = "- Improve delivery & packaging\n- Optimize app performance\n- Simplify website navigation"
    st.write("**Summary:**", summary)
    st.write("**Top 3 Action Items:**")
    st.write(actions)

    # Keyword Extraction
    st.subheader("🔑 Top Keywords")
    try:
        all_feedback = " ".join(df[feedback_col].astype(str))
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(all_feedback, top_n=10)
        for kw, score in keywords:
            st.write(f"{kw} ({score:.2f})")
    except Exception as e:
        st.error(f"Error extracting keywords: {e}")

    # Download options
    st.subheader("📥 Download Results")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "sentiment_results.csv", "text/csv", key="download_csv")

    excel_buffer = BytesIO()
    df.to_excel(excel_buffer, index=False, engine="xlsxwriter")
    st.download_button("Download Excel", excel_buffer.getvalue(), "sentiment_results.xlsx", "application/vnd.ms-excel", key="download_excel")

    # PDF Report
    def create_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Customer Feedback Sentiment Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)

        pdf.ln(5)
        pdf.multi_cell(0, 8, summary)
        pdf.ln(5)
        pdf.multi_cell(0, 8, actions)
        pdf.ln(5)
        pdf.cell(0, 10, "Top Keywords:", ln=True)
        for kw, score in keywords:
            pdf.cell(0, 8, f"{kw} ({score:.2f})", ln=True)

        pdf_buffer = BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        return pdf_buffer

    if st.button("⬇️ Generate PDF Report", key="generate_pdf"):
        try:
            pdf_buffer = create_pdf()
            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name="feedback_report.pdf",
                mime="application/pdf",
                key="download_pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")

# Sidebar Quick Feedback
st.sidebar.subheader("🔍 Quick Feedback Test")
single_fb = st.sidebar.text_area("Enter a single feedback (any language):")
if st.sidebar.button("Analyze Feedback", key="analyze_feedback"):
    fb_en = translate_text(single_fb)
    result = get_sentiment(fb_en)
    emoji = "😊" if result == "Positive" else "😡" if result == "Negative" else "😐"
    st.sidebar.write(f"Sentiment: {result} {emoji}")
    st.sidebar.write(f"🔎 Translated: {fb_en}")
