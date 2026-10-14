
import streamlit as st

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔍",
    layout="centered"
)

st.title("🔍 AI Research Agent")

st.write(
    "An AI-powered research assistant "
    "using CrewAI, Groq and web search."
)

topic = st.text_input(
    "Enter your research topic:",
    placeholder="How is Generative AI transforming Civil Engineering?"
)

if st.button("Check Topic"):
    if not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        st.success("Research topic accepted!")
        st.write("Your topic:", topic)
        st.info(
            "AI research functionality will be added "
            "in the upcoming steps."
        )
