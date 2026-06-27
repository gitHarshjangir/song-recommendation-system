import streamlit as st
import pandas as pd
import pickle
#for page
st.set_page_config(page_title="Song Recommendation System", layout="centered")

#Load data
@st.cache_data
def load_data():
    try:
        # Load the dataframe and similarity matrix
        df = pd.read_pickle('model_artifacts/hindi_songs_df.pkl')
        sim_matrix = pickle.load(open('model_artifacts/cosine_sim.pkl', 'rb'))
        return df, sim_matrix
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return None, None

df, cosine_sim = load_data()

# Helper function to get recommendations
def recommend_songs(song_title, df, sim_matrix):
    try:
        # Get the index of the selected song
        idx = df[df['Song Title'] == song_title].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top 5 similar songs (excluding the song itself)
        sim_scores = sim_scores[1:6]
        
        # Get the indices of the recommended songs
        song_indices = [i[0] for i in sim_scores]
        
        # Return a DataFrame with relevant columns
        return df.iloc[song_indices][['Song Title', 'Movie', 'Singer']]
    
    except Exception as e:
        st.error(f"An error occurred during recommendation: {e}")
        return None

#Main Application interfact of my application
st.title("Old Melodies Recommendation System")
st.write("Welcome! Select a song below to get similar recommendations based on our machine learning model.")

if df is not None:
    # Reset index to avoid positional indexing issues during recommendation
    df = df.reset_index(drop=True)
    song_list = df['Song Title'].values
    
    # Search Bar
    selected_song = st.selectbox("Search your favorite old melody:", song_list)
    
    # Recommendation Button
    if st.button("Recommend Songs"):
        st.write(f"### Recommendations for '{selected_song}':")
        
        # Fetch recommendations
        recommendations = recommend_songs(selected_song, df, cosine_sim)
        
        if recommendations is not None and not recommendations.empty:
            # Display recommendations cleanly as a dataframe table
            # reset index and adjust to start from 1 for clean display
            recommendations.index = range(1, len(recommendations) + 1)
            st.table(recommendations)
        else:
            st.warning("Could not find any recommendations.")
else:
    st.error("Data files are missing. Please ensure 'model_artifacts' folder is present in the same directory.")
