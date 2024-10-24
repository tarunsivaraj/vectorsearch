import streamlit as st
import wave
from io import BytesIO
from transformers import ClapProcessor, ClapModel
import numpy as np
from pinecone import Pinecone, ServerlessSpec, PineconeException
import pandas as pd
import re
from nltk.corpus import stopwords
import nltk
import os
import torch

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def audio_search_page(api_key):
    st.sidebar.title("⚙️ Audio Search Options")
    
    pinecone_options = ["API Key", "Environment", "Index Name"]
    selected_pinecone_option = st.sidebar.selectbox("Pinecone Settings", pinecone_options)

    # Display Pinecone settings
    if selected_pinecone_option == "API Key":
        st.sidebar.write(f"Current API Key: {api_key}")
    elif selected_pinecone_option == "Environment":
        st.sidebar.write("Environment: us-east-1")
    elif selected_pinecone_option == "Index Name":
        st.sidebar.write("Index Name: audio-search-index")

    model_options = ["CLAP"]
    selected_model_option = st.sidebar.selectbox("Models Used", model_options)

    if selected_model_option == "CLAP":
        st.sidebar.write("Model: CLAP by LAION")

    # Initialize Pinecone client for audio
    if 'pinecone_initialized_audio' not in st.session_state:
        try:
            pc_audio = Pinecone(
                api_key=api_key,
                environment='us-east-1'
            )
            st.session_state.pc = pc_audio
            st.session_state.pinecone_initialized_audio = True
        except PineconeException as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            st.stop()

    # Step 2: Interactive step to create Pinecone index
    index_name = "audio-search-index"

    if st.button("🛠️ Create a Vector Index"):
        if 'index_created_audio' not in st.session_state or not st.session_state.index_created_audio:
            try:
                existing_indexes = st.session_state.pc.list_indexes()
                if index_name not in existing_indexes:
                    st.write(f"Creating a new index '{index_name}'...")
                    st.session_state.pc.create_index(
                        name=index_name,
                        dimension=512,  # CLAP audio/text outputs 512-dimensional embeddings
                        metric="cosine",
                        spec=ServerlessSpec(cloud='aws', region='us-east-1')
                    )
                st.session_state.index_created_audio = True
                st.session_state.index_audio = st.session_state.pc.Index(index_name)
                st.write(f"Index '{index_name}' created or connected successfully.")
            except PineconeException as e:
                st.error(f"Error creating or connecting to the index: {str(e)}")
                st.stop()
        else:
            st.write(f"Index '{index_name}' is already created.")

    # Delete button for the audio index
    if st.button("❌ Delete Audio Index"):
        try:
            st.write(f"Attempting to delete index '{index_name}'...")
            st.session_state.pc.delete_index(index_name)
            st.write(f"Index '{index_name}' deleted successfully.")

            # Reset session state flags
            st.session_state.index_created_audio = False
            if 'index_audio' in st.session_state:
                del st.session_state.index_audio
            st.success(f"All session state related to '{index_name}' has been cleared.")
        except Exception as e:
            st.error(f"Error deleting index: {str(e)}")

    # Accept file path input for the Parquet file
    parquet_file_path = st.text_input("📁 Enter the path of the Parquet file containing audio data", key="parquet_audio_path")

    # Define utility functions
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)

    def normalize_embeddings(embeddings):
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return embeddings / (norm + 1e-8)

    def extract_audio_array_from_bytes(audio_bytes):
        with wave.open(BytesIO(audio_bytes), 'rb') as wav_file:
            sampling_rate = wav_file.getframerate()
            frames = wav_file.getnframes()
            audio_frames = wav_file.readframes(frames)
        audio_array = np.frombuffer(audio_frames, dtype=np.int16).astype(np.float32)
        return audio_array, sampling_rate

    def create_audio_embeddings(audio_array, sampling_rate):
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
        inputs = processor(audios=audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = model.get_audio_features(**inputs).cpu().numpy()
        return normalize_embeddings(embeddings).squeeze().tolist()

    def create_text_embeddings(text):
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
        preprocessed_text = preprocess_text(text)
        inputs = processor(text=preprocessed_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs).cpu().numpy()
        return normalize_embeddings(embeddings).squeeze().tolist()

    def store_audio_and_text_embeddings(audio_embedding, text_embedding, audio_id, metadata):
        index = st.session_state.pc.Index(index_name)  # Use st.session_state.pc to get Pinecone client
        index.upsert(vectors=[
            {"id": f"audio_{audio_id}", "values": audio_embedding, "metadata": metadata},
            {"id": f"text_{audio_id}", "values": text_embedding, "metadata": metadata}
        ])

    def play_audio_and_get_text_by_id(audio_id, df):
        row = df[df['line_id'] == audio_id].iloc[0]
        audio_bytes = row['audio']['bytes']
        text = row['text']

        if audio_bytes is not None and len(audio_bytes) > 0:
            audio_array, sampling_rate = extract_audio_array_from_bytes(audio_bytes)
            return audio_array, sampling_rate, text
        else:
            return None, None, None    

    def display_first_5_audios(df):
        st.write("Displaying first 5 audio samples and associated text:")
        for idx, row in df.iterrows():
            if idx >= 5:
                break  # Limit to the first 5 audios
            audio_id = row['line_id']
            audio_bytes = row['audio']['bytes']
            text_associated = row['text']

            if audio_bytes:
                audio_array, sampling_rate = extract_audio_array_from_bytes(audio_bytes)
                st.audio(audio_array, format="audio/wav", sample_rate=sampling_rate)
                st.write(f"Text associated with audio {audio_id}: {text_associated}")
            else:
                st.write(f"Audio for row {audio_id} is not available.")

    def filter_results_by_relevance(results, query_text, df):
        relevant_results = []
        query_terms = set(query_text.lower().split())

        for result in results:
            audio_id = result['id'].replace('text_', '').replace('audio_', '')
            audio_array, sampling_rate, audio_text = play_audio_and_get_text_by_id(audio_id, df)

            if audio_array is not None and audio_text is not None:
                audio_text_lower = audio_text.lower()
                if any(term in audio_text_lower for term in query_terms):
                    relevant_results.append({
                        'id': result['id'],
                        'score': result['score'],
                        'audio_array': audio_array,
                        'sampling_rate': sampling_rate,
                        'text': audio_text
                    })

        return relevant_results

    def search_similar_audios(query_text, df, top_k=10):
        text_embedding = create_text_embeddings(query_text)
        index = st.session_state.pc.Index(index_name)  # Use st.session_state.pc to get Pinecone client
        search_results = index.query(vector=text_embedding, top_k=top_k, include_metadata=True)

        if 'matches' in search_results and search_results['matches']:
            matches = search_results['matches']
            st.write(f"Found {len(matches)} results for your query.")

            unique_results = {match['id']: match for match in matches}.values()
            sorted_results = sorted(unique_results, key=lambda x: x['score'], reverse=True)

            relevant_results = filter_results_by_relevance(sorted_results, query_text, df)

            if relevant_results:
                for match in relevant_results[:5]:
                    st.audio(match['audio_array'], format='audio/wav', sample_rate=match['sampling_rate'])
                    st.write(f"Text: {match['text']}")
                    st.write(f"Score: {match['score']}")
            else:
                st.write("No relevant matches found.")
        else:
            st.write("No results found for your query.")

    # Handle Parquet file loading and embedding creation based on file path
    if parquet_file_path:
        try:
            # Strip any quotes from the path and handle Windows-style paths
            parquet_file_path = parquet_file_path.strip().strip('"')
            
            # Validate if the path exists
            if os.path.exists(parquet_file_path):
                # Load the Parquet file from the path
                df = pd.read_parquet(parquet_file_path).head(75)
                st.session_state.df = df  # Store the DataFrame in session state
                st.write("Dataset loaded successfully.")
            else:
                st.error(f"File does not exist at the given path: {parquet_file_path}")

            if st.button("🔍 Preview Data"):
                display_first_5_audios(df)

            if st.button("Preprocessing and Vectorization"):
                for idx, row in df.iterrows():
                    audio_id = row['line_id']
                    audio_bytes = row['audio']['bytes']
                    text_associated = row['text']
                    if audio_bytes:
                        audio_array, sampling_rate = extract_audio_array_from_bytes(audio_bytes)
                        audio_embeddings = create_audio_embeddings(audio_array, sampling_rate)
                        text_embeddings = create_text_embeddings(text_associated)
                        metadata = {"text": text_associated}
                        store_audio_and_text_embeddings(audio_embeddings, text_embeddings, audio_id, metadata)
                st.write("All audio and text embeddings stored successfully.")

        except Exception as e:
            st.error(f"Error loading file: {e}")

    # Check if 'df' exists in session state before allowing search
    if 'df' in st.session_state:
        query_text = st.text_input("Enter a query to search similar audio:", key="query_audio")
        if query_text and st.button("🔎 Search Similar Audio"):
            search_similar_audios(query_text, st.session_state.df)
    else:
        st.write("Please provide the file path and process it before searching for similar audio.")
