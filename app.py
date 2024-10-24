import streamlit as st
from image_search import image_search_page
from text_search import text_search_page
from audio_search import audio_search_page
from video_search import video_search_page

# Custom CSS to style the app
st.markdown("""
    <style>
    /* Style the green buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px;
        cursor: pointer;
        border-radius: 12px;
        width: 180px; /* Fixed width for all buttons */
        height: 80px; /* Fixed height for all buttons */
        white-space: normal; /* Allow text to wrap inside the button */
        word-wrap: break-word; /* Break long words if necessary */
    }

    /* Center the buttons */
    .stButton {
        display: flex;
        justify-content: center;
    }

    /* Center the main heading text */
    .centered-text {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🌿 Working with Vector Databases and Performing Semantic Searches")

# Initialize session state for Pinecone API key and navigation
if 'pinecone_api_key' not in st.session_state:
    st.session_state.pinecone_api_key = None

if 'page' not in st.session_state:
    st.session_state.page = 'home'  # Default page is 'home'

# Function to handle API key input and save
def enter_api_key():
    api_key_input = st.text_input("Enter your Pinecone API Key:", type="password")
    if api_key_input:
        st.session_state.pinecone_api_key = api_key_input
        st.success("API Key saved successfully!")
        st.rerun()  # Force a rerun after the API key is saved

# Function to switch between pages
def switch_page(page_name):
    st.session_state.page = page_name
    st.rerun()  # Forces rerun of the script to render new page

# Step 1: Enter Pinecone API Key if not provided
if st.session_state.pinecone_api_key is None:
    enter_api_key()

# Step 2: Display navigation buttons only after API key is entered
if st.session_state.pinecone_api_key:
    st.markdown('<div class="centered-text">Choose the type of search you\'d like to perform:</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🖼️ Image Search"):
            switch_page("image")
    with col2:
        if st.button("✍️ Text Search"):
            switch_page("text")
    with col3:
        if st.button("🎥 Video Search"):
            switch_page("video")
    with col4:
        if st.button("🎵 Audio Search"):
            switch_page("audio")

# Step 3: Render the appropriate page based on user selection
if st.session_state.page == "image":
    image_search_page(st.session_state.pinecone_api_key)

elif st.session_state.page == "text":
    text_search_page(st.session_state.pinecone_api_key)

elif st.session_state.page == "video":
    video_search_page(st.session_state.pinecone_api_key)

elif st.session_state.page == "audio":
    audio_search_page(st.session_state.pinecone_api_key)

elif st.session_state.page == "home":
    st.write("Welcome! Please select a search type to begin.")
