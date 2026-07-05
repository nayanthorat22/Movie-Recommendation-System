import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('tmdb_5000_movies.csv')

movies = movies[['title', 'genres', 'keywords']]

movies.dropna(inplace=True)

movies['tags'] = movies['genres'] + movies['keywords']

new_df = movies[['title', 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]

    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x:x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(new_df.iloc[i[0]].title)

    return recommended_movies

st.title('Movie Recommendation System')

selected_movie = st.selectbox(
    'Select a movie',
    new_df['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)

    for movie in recommendations:
        st.write(movie)