import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import urllib.request
import zipfile

st.set_page_config(
    page_title="Non-Personalized Movie Recommender",
    layout="wide"
)

#Data Loading
DATA_DIR = "movielens_data/ml-latest-small"
ratings_path = os.path.join(DATA_DIR, "ratings.csv")
movies_path = os.path.join(DATA_DIR, "movies.csv")

@st.cache_data
def download_data():
    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        os.makedirs(DATA_DIR, exist_ok=True)
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        zip_path = "movielens_data/ml-latest-small.zip"
        
        with st.spinner("Downloading MovieLens dataset..."):
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("movielens_data")
            os.remove(zip_path)

@st.cache_data
def load_data():
    download_data()
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    #Extract year from title
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

    return ratings, movies

ratings, movies = load_data()

#Popularity Metrics
pop = ratings.groupby("movieId").agg(
    rating_count=("rating", "count"),
    avg_rating=("rating", "mean")
).reset_index()

pop = pop.merge(movies, on="movieId", how="left")

#Weighted score (IMDB formula)
m = pop["rating_count"].quantile(0.80)
C = pop["avg_rating"].mean()

pop["weighted_score"] = (
    (pop["rating_count"] / (pop["rating_count"] + m)) * pop["avg_rating"] +
    (m / (pop["rating_count"] + m)) * C
)

#Dataset Statistics
st.title("🎬 Non-Personalized Movie Recommender")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Movies", f"{len(movies):,}")
with col2:
    st.metric("Total Ratings", f"{len(ratings):,}")
with col3:
    st.metric("Avg Rating", f"{ratings['rating'].mean():.2f}")
with col4:
    st.metric("Unique Genres", len(pop["genres"].dropna().str.split("|").explode().unique()))

st.markdown("---")

#Sidebar
st.sidebar.title("Controls")

TOP_N = st.sidebar.slider("Top N", min_value=5, max_value=50, value=10)

#Genre list for filters
genres_series = pop["genres"].dropna().str.split("|").explode().unique()
genres_list = sorted([g for g in genres_series if isinstance(g, str)])

selected_genre = st.sidebar.selectbox("Filter by Genre", ["None"] + genres_list)
selected_year = st.sidebar.slider("Minimum Year", min_value=1900, max_value=2024, value=2000)

#Main app tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Popularity", 
    "⭐ Top Rated", 
    "🏆 Weighted Score", 
    "🎭 Genre Recommender", 
    "⏳ Recent Movies",
    "🔍 Compare Algorithms"
])

#TAB1 — Popularity
with tab1:
    st.header("Popularity-Based Recommender (Most Rated Movies)")
    
    with st.expander("ℹ️ How this works"):
        st.write("Recommends movies based on the number of ratings they received. More ratings = more popular. Simple but effective for finding widely-watched movies.")

    top_popular = pop.sort_values(by="rating_count", ascending=False).head(TOP_N)
    st.dataframe(top_popular[["title", "rating_count", "avg_rating", "genres"]])

    st.subheader("Rating Count Distribution")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(pop['rating_count'], bins=50, color='skyblue', edgecolor='black')
    ax.set_title("Distribution of Rating Counts")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Movies")
    st.pyplot(fig)

#TAB2 — Top-rated
with tab2:
    st.header("Top Rated Movies (Min Ratings ≥ 50)")
    
    with st.expander("ℹ️ How this works"):
        st.write("Recommends movies with the highest average ratings, but only considers movies with at least 50 ratings to avoid obscure movies with few perfect scores.")

    top_rated = pop[pop["rating_count"] >= 50].sort_values(by="avg_rating", ascending=False).head(TOP_N)
    st.dataframe(top_rated[["title", "rating_count", "avg_rating", "genres"]])

    st.subheader("Genre Distribution")

    genres_exp = pop['genres'].str.split("|").explode().value_counts().head(20)

    fig2, ax2 = plt.subplots(figsize=(12,6))
    genres_exp.plot(kind="bar", ax=ax2, color="salmon")
    ax2.set_title("Top 20 Genres in MovieLens")
    st.pyplot(fig2)

#TAB3 — Weighted
with tab3:
    st.header("Weighted Score Recommender (IMDB Formula)")
    
    with st.expander("ℹ️ How this works"):
        st.write("Uses IMDB's weighted rating formula to balance popularity and quality. Movies need both good ratings AND sufficient votes to rank high. This prevents obscure movies with few perfect ratings from dominating.")
        st.latex(r"Weighted\ Score = \frac{v}{v+m} \times R + \frac{m}{v+m} \times C")
        st.write("Where: v = number of votes, m = minimum votes threshold, R = average rating, C = mean rating across all movies")

    top_weighted = pop.sort_values(by="weighted_score", ascending=False).head(TOP_N)
    st.dataframe(top_weighted[["title", "rating_count", "avg_rating", "weighted_score", "genres"]])

    st.subheader("Top Weighted Movies (Bar Chart)")

    fig3, ax3 = plt.subplots(figsize=(12,6))
    ax3.barh(top_weighted["title"], top_weighted["weighted_score"], color="orange")
    ax3.set_title("Top Movies by Weighted Score")
    ax3.set_xlabel("Weighted Score")
    ax3.invert_yaxis()
    st.pyplot(fig3)

#TAB4 — Genre filter
with tab4:
    st.header("Genre-Based Non-Personalized Recommender")
    
    with st.expander("ℹ️ How this works"):
        st.write("A demographic-based approach that filters movies by genre and ranks by popularity. Useful for finding popular movies within specific categories.")

    if selected_genre != "None":
        mask = pop["genres"].str.contains(selected_genre, na=False)
        genre_filtered = pop[mask].sort_values(by="rating_count", ascending=False).head(TOP_N)

        st.subheader(f"Top {TOP_N} Movies in {selected_genre}")
        st.dataframe(genre_filtered[["title", "rating_count", "avg_rating", "genres"]])

    else:
        st.info("Select a genre from the sidebar to view recommendations.")

#TAB5 — Recent movies
with tab5:
    st.header("Recent Movies (Non-Personalized)")
    
    with st.expander("ℹ️ How this works"):
        st.write("Filters movies by release year and ranks by popularity. Helps discover newer popular movies rather than just classics.")

    recent = pop[pop["year"] >= selected_year].sort_values(
        by="rating_count", ascending=False
    ).head(TOP_N)

    st.subheader(f"Top {TOP_N} Movies Released After {selected_year}")
    st.dataframe(recent[["title", "year", "rating_count", "avg_rating", "genres"]])

#TAB6 — Comparison
with tab6:
    st.header("Algorithm Comparison")
    
    with st.expander("ℹ️ Why compare?"):
        st.write("Different algorithms produce different results. This comparison shows how the same movie ranks across different recommendation approaches.")
    
    # Get top 10 from each method
    top_pop = set(pop.sort_values(by="rating_count", ascending=False).head(10)["title"])
    top_rated = set(pop[pop["rating_count"] >= 50].sort_values(by="avg_rating", ascending=False).head(10)["title"])
    top_weight = set(pop.sort_values(by="weighted_score", ascending=False).head(10)["title"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Movies in Multiple Top 10s")
        common_all = top_pop & top_rated & top_weight
        if common_all:
            st.success(f"**In all 3 algorithms:** {len(common_all)} movies")
            for movie in common_all:
                st.write(f"- {movie}")
        
        common_two = (top_pop & top_rated) | (top_pop & top_weight) | (top_rated & top_weight)
        common_two = common_two - common_all
        if common_two:
            st.info(f"**In 2 algorithms:** {len(common_two)} movies")
    
    with col2:
        st.subheader("Algorithm Overlap")
        st.write(f"**Popularity only:** {len(top_pop - top_rated - top_weight)} movies")
        st.write(f"**Top Rated only:** {len(top_rated - top_pop - top_weight)} movies")
        st.write(f"**Weighted only:** {len(top_weight - top_pop - top_rated)} movies")
    
    st.subheader("Side-by-Side Comparison")
    comparison_df = pd.DataFrame({
        "Popularity": pop.sort_values(by="rating_count", ascending=False).head(10)["title"].values,
        "Top Rated": pop[pop["rating_count"] >= 50].sort_values(by="avg_rating", ascending=False).head(10)["title"].values,
        "Weighted Score": pop.sort_values(by="weighted_score", ascending=False).head(10)["title"].values
    })
    st.dataframe(comparison_df, use_container_width=True)

st.markdown("---")
st.write("This is a **non-personalized** recommender. It is based on item popularity, average rating, and simple genre filtering.")
