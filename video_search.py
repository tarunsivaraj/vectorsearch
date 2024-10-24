# video_search.py
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import cv2
import os
import tempfile
import numpy as np
from pinecone import Pinecone, ServerlessSpec, PineconeException

def video_search_page(api_key):
    st.sidebar.title("⚙️ Video Search Options")
    
    pinecone_options = ["API Key", "Environment", "Index Name"]
    selected_pinecone_option = st.sidebar.selectbox("Pinecone Settings", pinecone_options)

    # Display Pinecone settings
    if selected_pinecone_option == "API Key":
        st.sidebar.write(f"Current API Key: {api_key}")
    elif selected_pinecone_option == "Environment":
        st.sidebar.write("Environment: us-east-1")
    elif selected_pinecone_option == "Index Name":
        st.sidebar.write("Index Name: video-search-index")

    model_options = ["CLIP"]
    selected_model_option = st.sidebar.selectbox("Models Used", model_options)

    if selected_model_option == "CLIP":
        st.sidebar.write("Model: CLIP by OpenAI")

    # Initialize Pinecone client for video
    if 'pinecone_initialized_video' not in st.session_state:
        try:
            pc_video = Pinecone(
                api_key=api_key,
                environment='us-east-1'
            )
            st.session_state.pc = pc_video
            st.session_state.pinecone_initialized_video = True
        except PineconeException as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.stop()

    # Step 2: Interactive step to create Pinecone index
    index_name = "video-search-index"

    if st.button("🛠️ Create a Vector Index"):
        if 'index_created_video' not in st.session_state or not st.session_state.index_created_video:
            try:
                existing_indexes = st.session_state.pc.list_indexes()
                if index_name not in existing_indexes:
                    st.write(f"Creating a new index '{index_name}'...")
                    st.session_state.pc.create_index(
                        name=index_name,
                        dimension=512,  # CLIP outputs 512-dimensional embeddings
                        metric="cosine",
                        spec=ServerlessSpec(cloud='aws', region='us-east-1')
                    )
                st.session_state.index_created_video = True
                st.session_state.index_video = st.session_state.pc.Index(index_name)
                st.write(f"Index '{index_name}' created or connected successfully.")
            except PineconeException as e:
                st.error(f"Error creating or connecting to the index: {str(e)}")
                st.stop()
        else:
            st.write(f"Index '{index_name}' is already created.")

    # Delete button for the video index
    if st.button("❌ Delete Video Index"):
        try:
            st.write(f"Attempting to delete index '{index_name}'...")
            st.session_state.pc.delete_index(index_name)
            st.write(f"Index '{index_name}' deleted successfully.")

            # Reset session state flags
            st.session_state.index_created_video = False
            if 'index_video' in st.session_state:
                del st.session_state.index_video
            st.success(f"All session state related to '{index_name}' has been cleared.")
        except Exception as e:
            st.error(f"Error deleting index: {str(e)}")

    # Load CLIP model
    if 'clip_model' not in st.session_state:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        st.session_state.clip_model = clip_model
        st.session_state.clip_processor = clip_processor
    else:
        clip_model = st.session_state.clip_model
        clip_processor = st.session_state.clip_processor

    # Function to extract frames from video by interval
    def get_single_frame_from_video(video_capture, time_sec):
        video_capture.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
        success, frame = video_capture.read()
        if success:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return None

    def get_frames_from_video_by_interval(video_path, interval_sec=10):
        frames = []
        video_capture = cv2.VideoCapture(video_path)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_duration_sec = total_frames / fps

        for time_sec in np.arange(0, total_duration_sec, interval_sec):
            frame = get_single_frame_from_video(video_capture, time_sec)
            if frame is not None:
                frames.append(frame)
        
        video_capture.release()
        return frames

    # Function to create image embedding using CLIP
    def get_image_embedding(image):
        inputs = clip_processor(images=image, return_tensors="pt")
        image_embeddings = clip_model.get_image_features(**inputs)
        return list(map(float, image_embeddings[0].detach().numpy().astype(np.float32)))

    # Function to create text embedding using CLIP
    def get_text_embedding(text):
        inputs = clip_processor(text=[text], return_tensors="pt")
        text_embedding = clip_model.get_text_features(**inputs)
        return list(map(float, text_embedding[0].detach().numpy().astype(np.float32)))

    # Main process to extract frames, create embeddings, and store in Pinecone
    def process_video_for_embedding(video_path, interval_sec=10):
        frames = get_frames_from_video_by_interval(video_path, interval_sec)
        
        image_embeddings = []
        image_ids = []
        for i, frame in enumerate(frames):
            embedding = get_image_embedding(frame)
            image_embeddings.append(embedding)
            image_ids.append(str(i))

        pinecone_vectors = [
            (image_ids[i], image_embeddings[i])
            for i in range(len(image_embeddings))
        ]
        
        index = st.session_state.pc.Index(index_name)
        index.upsert(vectors=pinecone_vectors)
        st.success(f"Inserted {len(pinecone_vectors)} vectors into Pinecone.")
        
        return pinecone_vectors

    # Function to search for similar video frames based on text query
    def search_video_by_text(query_text):
        query_embedding = get_text_embedding(query_text)
        
        index = st.session_state.pc.Index(index_name)
        result = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        
        if 'matches' in result and len(result['matches']) > 0:
            closest_match = result['matches'][0]
            frame_id = closest_match['id']
            similarity_score = closest_match['score']
            
            return frame_id, similarity_score
        else:
            return None, None  # No matching results found

    # Function to extract and play video segment using moviepy
    def play_video_segment(video_path, frame_id, interval_sec=10, segment_duration=5):
        if frame_id is None:
            st.error("No frame ID provided. Cannot play video.")
            return None

        frame_time_sec = int(frame_id) * interval_sec
        start_time_sec = max(frame_time_sec - segment_duration // 2, 0)
        end_time_sec = start_time_sec + segment_duration

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_path = temp_video_file.name

        ffmpeg_extract_subclip(video_path, start_time_sec, end_time_sec, targetname=temp_video_path)

        if os.path.exists(temp_video_path):
            return temp_video_path
        else:
            st.error("Failed to create video segment.")
            return None

    uploaded_video = st.file_uploader("📤 Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
            temp_video_file.write(uploaded_video.read())
            temp_video_path = temp_video_file.name

        st.video(temp_video_path)

        # Capture the button click event
        if st.button("Preprocessing and Vectorization", key="process_button_video"):
            process_video_for_embedding(temp_video_path)

        query_text = st.text_input("Enter a semantic search string:")

        if query_text and st.button("🔎 Search for Videos"):
            frame_id, score = search_video_by_text(query_text)

            if frame_id is not None:
                st.write(f"Closest frame ID: {frame_id} with similarity score: {score}")
                segment_video_path = play_video_segment(temp_video_path, frame_id)

                if segment_video_path:
                    st.video(segment_video_path)
                else:
                    st.error("No matching video segment found for the query.")
            else:
                st.write("No matching results found for the query.")
