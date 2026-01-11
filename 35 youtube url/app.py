import validators
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from youtube_transcript_api import YouTubeTranscriptApi
import re
import requests
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

# Streamlit UI Config
st.set_page_config(page_title="Universal Summarizer", page_icon="📝")
st.title("📝 YouTube & Website Summarizer")
st.subheader('Enter a YouTube or Website URL to Generate Summary')

# Sidebar for API key and options
with st.sidebar:
    default_groq_api_key = os.getenv('GROQ_API_KEY', '')
    
    groq_api_key = st.text_input(
        "🔑 Groq API Key", 
        value=default_groq_api_key, 
        type="password",
        help="Enter your Groq API key from https://console.groq.com"
    )
    st.markdown("---")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3)
    max_tokens = st.slider("Max Tokens", 100, 2048, 512)

# URL input
input_url = st.text_input("🌐 Enter YouTube or Website URL")

# Function to extract YouTube video ID
def extract_video_id(url):
    patterns = [
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([^&]+)',
        r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([^&]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Summarization function
def summarize_text(full_text):
    llm = ChatGroq(
        model="Llama3-8b-8192",
        groq_api_key=groq_api_key.strip(),
        temperature=temperature,
        max_tokens=max_tokens
    )
    if len(full_text) > 10000:
        full_text = full_text[:10000]
    docs = [Document(page_content=full_text)]
    chain = load_summarize_chain(llm, chain_type="stuff")
    output_summary = chain.invoke({"input_documents": docs})
    return output_summary["output_text"]

# Main summarization
if st.button("🚀 Summarize", disabled=not (groq_api_key.strip() and input_url.strip())):
    if not validators.url(input_url):
        st.error("❗ Invalid URL")
    else:
        try:
            # Detect YouTube or Website
            video_id = extract_video_id(input_url)
            if video_id:
                # --- YouTube Flow ---
                with st.spinner("⏳ Fetching video transcript..."):
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        full_text = " ".join([entry['text'] for entry in transcript])
                        summary = summarize_text(full_text)
                        st.success("✅ YouTube Video Summary:")
                        st.write(summary)

                    except Exception as transcript_error:
                        st.error(f"Error fetching transcript: {transcript_error}")
                        st.info(
                            "Possible reasons:\n"
                            "- No transcript available\n"
                            "- Video might be private or age-restricted\n"
                            "- Transcript generation failed"
                        )
            else:
                # --- Website Flow ---
                with st.spinner("⏳ Fetching website content..."):
                    response = requests.get(input_url, timeout=10)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    paragraphs = soup.find_all('p')
                    full_text = " ".join([para.get_text() for para in paragraphs])
                    if not full_text.strip():
                        st.error("❗ Could not extract readable text from the website.")
                        st.stop()
                    summary = summarize_text(full_text)
                    st.success("✅ Website Summary:")
                    st.write(summary)

        except Exception as e:
            st.error(f"❌ Unexpected Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Powered by LangChain & Groq</p>",
    unsafe_allow_html=True
)
