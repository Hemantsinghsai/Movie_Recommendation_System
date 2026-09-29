import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")
st.write("Personalized movie recommendations using Collaborative Filtering")


# Load dataset
df = pd.read_excel(
    "data/Movie_Recommendation_System.xlsx",
    sheet_name="Movie_Data"
)


# Data cleaning
df["Genre"] = df["Genre"].fillna("Unknown")
df["Language"] = df["Language"].fillna("Unknown")

df = df.dropna(subset=["Rating"])

df = df.drop_duplicates(
    subset=["User_ID", "Movie_Title"]
)


# Create User-Movie Matrix
user_movie_matrix = df.pivot_table(
    index="User_ID",
    columns="Movie_Title",
    values="Rating",
    aggfunc="mean"
)


# Replace NaN with 0
user_movie_matrix_filled = user_movie_matrix.fillna(0)


# Calculate similarity
user_similarity = cosine_similarity(
    user_movie_matrix_filled
)


# Convert similarity matrix into DataFrame
similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)


# User selection
user_id = st.selectbox(
    "Select User ID",
    user_movie_matrix.index
)


if st.button("🎯 Recommend Movies"):

    # Find similar users
    similar_users = similarity_df[user_id].sort_values(
        ascending=False
    )

    # Remove current user
    similar_users = similar_users.drop(user_id)

    # Top 5 similar users
    top_users = similar_users.head(5).index

    # Get ratings of similar users
    similar_users_ratings = user_movie_matrix.loc[top_users]

    # Average ratings
    movie_scores = similar_users_ratings.mean(axis=0)

    # Movies already rated by user
    user_movies = user_movie_matrix.loc[user_id].dropna()

    # Remove already rated movies
    movie_scores = movie_scores.drop(
        user_movies.index,
        errors="ignore"
    )

    # Sort recommendations
    movie_scores = movie_scores.sort_values(
        ascending=False
    )

    # Top 5 recommendations
    recommendations = movie_scores.head(5)

    st.subheader("🎬 Recommended Movies")

    for movie, score in recommendations.items():

        st.write(
            f"**{movie}** — ⭐ {score:.2f}"
        )