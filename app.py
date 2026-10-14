
import streamlit as st
from researcher import research_topic


# 1. Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide"
)


# 2. Application heading
st.title("🔍 AI Research Assistant")

st.write(
    "Generate research reports using CrewAI, "
    "Groq GPT-OSS 120B and web search."
)

st.divider()


# 3. Store generated reports
if "report" not in st.session_state:
    st.session_state.report = ""

if "sources" not in st.session_state:
    st.session_state.sources = []


# 4. Research topic input
with st.form("research_form"):

    topic = st.text_area(
        "Enter your research topic:",
        placeholder=(
            "How is Generative AI transforming "
            "Civil Engineering?"
        ),
        height=100
    )

    submitted = st.form_submit_button(
        "Generate Research Report",
        type="primary"
    )


# 5. Process research request
if submitted:

    st.session_state.report = ""
    st.session_state.sources = []

    if not topic.strip():
        st.warning("Please enter a research topic.")

    else:

        # Read API key securely
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except (FileNotFoundError, KeyError):
            api_key = None

        if not api_key:
            st.error(
                "Groq API key is missing. "
                "Configure GROQ_API_KEY in "
                "Streamlit Cloud Secrets."
            )

        else:

            with st.spinner(
                "Researching your topic..."
            ):

                try:
                    report, sources = research_topic(
                        topic,
                        api_key
                    )

                except Exception:
                    st.error(
                        "Research failed. Please check "
                        "the Groq API configuration, "
                        "usage limits and web search."
                    )

                else:
                    st.session_state.report = report
                    st.session_state.sources = sources

                    st.success(
                        "Research report generated!"
                    )


# 6. Display generated report
if st.session_state.report:

    st.divider()

    st.header("Research Report")

    st.markdown(st.session_state.report)
    
    # Check the available research evidence
    references_section = st.session_state.report.rsplit(
        "## Retrieved Web Sources", 1
    )[-1]

    snippet_count = references_section.count(
        "Evidence Type: SNIPPET_ONLY"
    )

    article_count = references_section.count(
        "Evidence Type: ARTICLE_EXCERPT"
    )

    # Display evidence quality information
    st.info(
        f"Research Evidence: {article_count} article excerpts "
        f"and {snippet_count} search snippets."
    )

    if snippet_count > 0 and article_count == 0:
        st.warning(
            "This report is based only on search snippets. "
            "The full article contents were not retrieved. "
            "Please verify important claims from original sources."
        )

    st.caption(
        f"Retrieved sources: "
        f"{len(st.session_state.sources)}. "
        "Verify important claims against "
        "the original sources."
    )

    # 7. Download research report
    st.download_button(
        label="Download Research Report",
        data=st.session_state.report,
        file_name="research_report.md",
        mime="text/markdown",
        on_click="ignore"
    )


# 8. Footer
st.divider()

st.caption(
    "AI Research Assistant | "
    "CrewAI + Groq + Streamlit"
)
