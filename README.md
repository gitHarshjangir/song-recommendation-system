# Song Recommendation System

This project is a content-based recommendation system that suggests similar old Hindi songs based on a user's selection. 

### Dataset & Preprocessing
The dataset used contains information about classic Hindi songs, including the track title, movie, and singer. Text features were processed using a TF-IDF vectorizer (saved as `tfidf_vectorizer.pkl`) to convert the textual metadata into a matrix of TF-IDF features. 

### Modeling
I used Cosine Similarity to calculate the distance between the song vectors. The resulting similarity matrix (`cosine_sim.pkl`) is loaded at runtime to quickly retrieve the top 5 most similar tracks for any given song in the dataset.

### Project Structure
- `app.py`: Streamlit script for the frontend.
- `model_artifacts/`: Directory containing the pandas dataframe and pickle files for the TF-IDF vectorizer and cosine similarity matrix.
- `requirements.txt`: Python package dependencies.

### How to Run

1. Clone the repo and navigate to the project directory:
   ```bash
   git clone https://github.com/gitHarshjangir/song-recommendation-system.git
   cd song-recommendation-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
