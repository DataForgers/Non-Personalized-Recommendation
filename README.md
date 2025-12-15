# Non-Personalized Movie Recommender by GROUP1

A Streamlit-based movie recommendation system using non-personalized approaches.

## Features

- **Popularity-Based**: Recommends most rated movies
- **Top Rated**: Shows highest-rated movies with minimum rating threshold
- **Weighted Score**: Uses IMDB formula for balanced recommendations
- **Genre Filter**: Filter recommendations by genre
- **Recent Movies**: Discover recent popular movies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Data

This project uses the MovieLens dataset. Download it from [MovieLens](https://grouplens.org/datasets/movielens/) and place it in `movielens_data/ml-latest-small/`.

## Deployment
We deployed the recommendation system using streamlit since it is fast and simple. View demo here https://non-personalized-recommendation-3qnkcgeghznepiihclv9fp.streamlit.app/
