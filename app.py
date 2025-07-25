import streamlit as st
import os
from resolution_reviewer import ResolutionReviewer

def select_or_upload_file(res_dir):
    st.subheader("Upload a Resolution Document")
    uploaded_file = st.file_uploader("Upload a .docx file:", type=['docx'])
    temp_file_path = None
    file_to_review = None
    file_source = None
    if uploaded_file is not None:
        temp_file_path = "temp_resolution.docx"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        file_to_review = temp_file_path
        file_source = "uploaded"
    return file_to_review, file_source, temp_file_path

def show_document_preview(reviewer, file_to_review):
    with st.expander("Preview Extracted Text from Document", expanded=False):
        try:
            extracted_text = reviewer.read_document(file_to_review)
            st.text_area("Extracted Text", extracted_text, height=300)
        except Exception as e:
            st.error(f"Could not extract text: {e}")

def analyze_and_display_results(reviewer, file_to_review, file_source, temp_file_path):
    if st.button("Analyze Resolution", type="primary"):
        try:
            with st.spinner("Analyzing resolution..."):
                results = reviewer.review_resolution(file_to_review)
            st.header("Review Results")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Template Violations")
                if results["template_violations"]:
                    for violation in results["template_violations"]:
                        with st.expander(f"Violation at {violation['location']}", expanded=True):
                            st.markdown(f"**Rule:** {violation['rule']}")
                            st.markdown(f"**Issue:** {violation['description']}")
                            st.markdown(f"**Suggestion:** {violation['suggestion']}")
                else:
                    st.success("No template violations found.")
            with col2:
                st.subheader("Formatting Violations")
                if results["formatting_violations"]:
                    for violation in results["formatting_violations"]:
                        with st.expander(f"Violation at {violation['location']}", expanded=True):
                            st.markdown(f"**Rule:** {violation['rule']}")
                            st.markdown(f"**Issue:** {violation['description']}")
                            st.markdown(f"**Suggestion:** {violation['suggestion']}")
                else:
                    st.success("No formatting violations found.")
            st.subheader("Overall Assessment")
            st.info(results["overall_assessment"])
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            if file_source == "uploaded" and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

def main():
    st.set_page_config(
        page_title="CUNY Resolution Reviewer",
        page_icon="📄",
        layout="wide"
    )
    st.title("CUNY Resolution Reviewer")
    st.markdown("""
This tool analyzes CUNY Board of Trustees resolutions for compliance with templates and rules.\n
""")
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        st.error("Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    res_dir = os.path.join(os.getcwd(), "reinaccurateboardresolutions")
    file_to_review, file_source, temp_file_path = select_or_upload_file(res_dir)
    if file_to_review:
        reviewer = ResolutionReviewer(OPENAI_API_KEY)
        show_document_preview(reviewer, file_to_review)
        analyze_and_display_results(reviewer, file_to_review, file_source, temp_file_path)
    else:
        st.info("Please upload a .docx file to begin.")

if __name__ == "__main__":
    main() 