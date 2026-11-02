"""
Task 6: Simple Streamlit UI for the content-based movie recommender.
Run locally:  streamlit run app.py
"""

import streamlit as st

import recommender as r

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Building recommendation model...")
def load_model():
    return r.get_model()


df, _, _ = load_model()

st.title("🎬 Movie Recommendation System")
st.caption("Content-based recommender using TF-IDF + Cosine Similarity (TMDB 5000 Movies)")

# Dropdown to select an item
movie = st.selectbox("Select a movie you like:", sorted(df["title"].tolist()))
top_n = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

# Button to generate recommendations
if st.button("Recommend"):
    results = r.recommend(movie, top_n=top_n)
    st.subheader(f"Because you liked **{movie}**:")
    for i, row in results.iterrows():
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {row['title']}**  —  similarity: `{row['similarity']}`")
            if row["genres_text"]:
                st.caption(f"Genres: {row['genres_text']}")
            overview = row["overview"] or ""
            st.write(overview[:250] + ("..." if len(overview) > 250 else ""))
