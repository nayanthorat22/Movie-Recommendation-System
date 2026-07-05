import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎥",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 3rem;
    color: #FF4B4B;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #BBBBBB;
    font-size: 1.2rem;
    margin-bottom: 30px;
}

.movie-card {
    background: #1E1E1E;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    margin-top: 10px;
    color: white;
    min-height: 80px;
}

.stButton > button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 50px;
    font-size: 18px;
    border: none;
}

.stButton > button:hover {
    background-color: #ff3333;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")

    df = df[['title', 'genres', 'keywords']]
    df.dropna(inplace=True)

    df['tags'] = df['genres'] + " " + df['keywords']
    df['tags'] = df['tags'].str.lower()

    cv = CountVectorizer(
        max_features=5000,
        stop_words='english'
    )

    vectors = cv.fit_transform(df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    return df, similarity


movies, similarity = load_data()

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(movie):

    try:
        movie_index = movies[
            movies['title'] == movie
        ].index[0]

        distances = similarity[movie_index]

        movie_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:6]

        recommendations = []

        for i in movie_list:
            recommendations.append(
                movies.iloc[i[0]].title
            )

        return recommendations

    except:
        return []


# ---------------- HEADER ----------------
st.markdown(
    "<div class='title'>🎬 CineMatch AI</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Find Movies Similar To Your Favorites</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- SEARCH ----------------
col1, col2 = st.columns([4, 1])

with col1:
    selected_movie = st.selectbox(
        "Select a Movie",
        movies['title'].values
    )

with col2:
    st.write("")
    st.write("")
    recommend_btn = st.button("Recommend")

st.markdown("---")

# ---------------- TRENDING ----------------
st.subheader("🔥 Popular Movies")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.image("https://picsum.photos/200/300?1")
    st.caption("Avatar")

with c2:
    st.image("https://picsum.photos/200/300?2")
    st.caption("Inception")

with c3:
    st.image("https://picsum.photos/200/300?3")
    st.caption("Interstellar")

with c4:
    st.image("https://picsum.photos/200/300?4")
    st.caption("The Dark Knight")

with c5:
    st.image("https://picsum.photos/200/300?5")
    st.caption("Avengers")

st.markdown("---")

# ---------------- RECOMMENDATIONS ----------------
if recommend_btn:

    recommended_movies = recommend(selected_movie)

    st.subheader("🎯 Recommended Movies")

    if len(recommended_movies) == 0:
        st.error("No recommendations found.")

    else:

        cols = st.columns(len(recommended_movies))

        for idx, movie in enumerate(recommended_movies):

            with cols[idx]:

                st.image(
                    f"https://picsum.photos/250/350?random={idx}",
                    width=220
                )

                st.markdown(
                    f"""
                    <div class="movie-card">
                        <h4>{movie}</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")

st.markdown(
    """
    <center>
    <p style='color:gray'>
    Built using Streamlit • Scikit-Learn • TMDB Dataset
    </p>
    </center>
    """,
    unsafe_allow_html=True
)