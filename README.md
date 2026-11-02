# 🎬 Content-Based Movie Recommendation System

An end-to-end ML + deployment project: data preprocessing → TF-IDF vectorization → cosine similarity → Streamlit app → GitHub → Render.

## Dataset
- Name: TMDB 5000 Movie Dataset
- Kaggle link: https://www.kaggle.com/datasets/tmdb/tmdb-5000-movie-dataset
- File used: 'tmdb_5000_movies.csv'
- Item identifier: 'title'
- Text features: 'overview', 'genres', 'keywords'

## Approach
1. Load & explore the dataset (shape, columns, head, missing values).
2. Preprocess text: lowercase → remove punctuation/special characters → remove stopwords → fill missing values with empty string. Result stored in 'clean_text'.
3. TF-IDF ('max_features=5000', 'ngram_range=(1, 2)') converts text to vectors.
4. Cosine similarity between all movies (measures the angle between vectors, so it is independent of text length and works well on sparse text vectors).
5. 'recommend(item_name, top_n=5)' finds the movie, ranks all others by similarity and returns the top N.
6. Streamlit UI: dropdown to select a movie + button to get recommendations.

## Project Structure
'''
├── app.py                 # Streamlit app
├── recommender.py         # Preprocessing, TF-IDF, similarity, recommend()
├── run_tasks.py           # Prints output for Tasks 1-5
├── tmdb_5000_movies.csv   # Dataset
├── requirements.txt
└── README.md
'''

## Run Locally
'''bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
python run_tasks.py        # see Task 1-5 outputs
streamlit run app.py       # launch the app
'''

## Deployment (Render)
- Build command: 'pip install -r requirements.txt'
- Start command: 'streamlit run app.py --server.port $PORT --server.address 0.0.0.0'
- Deployed URL: '<paste your Render URL here>'

## Tech Stack
Python, Pandas, NumPy, scikit-learn, Streamlit, Git/GitHub, Render
