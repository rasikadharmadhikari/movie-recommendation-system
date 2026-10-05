
import os
import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/calci/bollywood-movie-dataset/"
    "master/BollywoodMovieDetail.csv"
)
DATA_DIR = "dataset"
DATA_FILE = os.path.join(DATA_DIR, "hindi_movies.csv")

@st.cache_data
def load_movies():
    if not os.path.exists(DATA_FILE):
        try:
            response = requests.get(DATA_URL, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"Could not download the Hindi movie dataset: {exc}")
            st.stop()

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(DATA_FILE, "wb") as data_file:
            data_file.write(response.content)

    movies = pd.read_csv(DATA_FILE)
    movies = movies.rename(columns={"genre": "genres", "releaseYear": "year"})
    movies["genres"] = movies["genres"].fillna("Unknown")
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    return movies

@st.cache_resource
def build_model(movies):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(movies["genres_clean"])
    similarity = cosine_similarity(matrix)
    return vectorizer, matrix, similarity

def get_recommendations(movie_index, movies, similarity, n=5):
    scores = list(enumerate(similarity[movie_index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scores:
        if idx == movie_index:
            continue
        results.append({
            "index": idx,
            "title": movies.iloc[idx]["title"],
            "genres": movies.iloc[idx]["genres"],
            "score": float(score)
        })
        if len(results) == n:
            break
    return pd.DataFrame(results)

def find_movie(search_text, movies):
    text = search_text.strip().lower()

    exact = movies[movies["title"].str.lower() == text]
    if not exact.empty:
        return exact.index[0]

    starts = movies[movies["title"].str.lower().str.startswith(text, na=False)]
    if not starts.empty:
        return starts.index[0]

    contains = movies[movies["title"].str.lower().str.contains(text, regex=False, na=False)]
    if not contains.empty:
        return contains.index[0]

    return None

movies = load_movies()
vectorizer, tfidf_matrix, similarity = build_model(movies)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0b1020, #17152d 55%, #24162e);
        color: #f4f5fb; font-family: "Inter", "Segoe UI", Arial, sans-serif; }
    html, body, [class*="css"] { font-family: "Inter", "Segoe UI", Arial, sans-serif; }
    .block-container { max-width: 1180px; padding: .8rem 1.5rem 2rem; }
    [data-testid="stHeader"] { background: transparent; }
    .hero { padding: .85rem 1.25rem .8rem; border: 1px solid rgba(255,255,255,.12);
        border-radius: 16px; background: radial-gradient(circle at 85% 20%,
        rgba(234,88,121,.28), transparent 35%), linear-gradient(110deg,
        rgba(31,41,79,.95), rgba(57,28,63,.9)); box-shadow: 0 10px 28px rgba(0,0,0,.2);
        margin-bottom: .65rem; }
    .eyebrow, .section-label { color: #ffb4c0; font-size: .68rem; font-weight: 800;
        letter-spacing: .16em; }
    .hero h1 { color: #fff; font-size: clamp(1.45rem, 3vw, 2.35rem); margin: .18rem 0 .22rem; }
    .hero p { color: #e4e6f1; font-size: .82rem; max-width: 680px; margin: 0; line-height: 1.35; }
    .metric { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.1);
        border-radius: 10px; padding: .42rem .55rem; text-align: center; }
    .metric strong { display: block; color: #fff; font-size: .95rem; }
    .metric span { color: #d5d8e8; font-size: .68rem; }
    .movie-card { padding: .75rem .85rem; min-height: 7.2rem; border: 1px solid rgba(255,255,255,.12);
        border-radius: 14px; background: rgba(255,255,255,.06); }
    .movie-card h4 { color: #fff; margin: 0 0 .45rem; font-size: .95rem; line-height: 1.3; }
    .movie-card p { color: #d4d7e5; margin: .25rem 0; font-size: .8rem; line-height: 1.35; }
    .selected-card { padding: 1.2rem; min-height: 12rem; border: 1px solid rgba(255,255,255,.14);
        border-radius: 18px; background: linear-gradient(145deg, rgba(232,93,117,.18),
        rgba(168,85,247,.12)); }
    .selected-card .label { color: #ffb4c0; text-transform: uppercase; font-size: .72rem;
        font-weight: 800; letter-spacing: .13em; }
    .selected-card h2 { color: #fff; font-size: 1.45rem; line-height: 1.25; margin: .55rem 0; }
    .selected-card p { color: #e1e3ee; font-size: .9rem; line-height: 1.5; margin: .35rem 0; }
    .stSelectbox label, .stSlider label { color: #f4f5fb !important; font-weight: 700 !important; }
    .stCaption, small { color: #d0d3e1 !important; }
    div[data-testid="stHorizontalBlock"] { gap: .8rem; }
    div[data-testid="stVerticalBlock"] > div { gap: .35rem; }
    .insight-title { color: #fff; font-size: 1.05rem; font-weight: 700; margin: .3rem 0; }
    div.stButton > button[kind="primary"] { background: linear-gradient(90deg, #e85d75,
        #a855f7); border: 0; border-radius: 12px; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">BOLLYWOOD • DISCOVER SOMETHING NEW</div>
        <h1>Your next favourite film awaits.</h1>
        <p>Find Hindi movies with the same mood, style, and genre as the stories you already love.</p>
    </div>
    """,
    unsafe_allow_html=True
)

metric_cols = st.columns(3)
for col, value, label in zip(
    metric_cols,
    [f"{len(movies):,}", "TF-IDF", "Cosine"],
    ["Hindi films", "Recommendation engine", "Similarity model"],
):
    with col:
        st.markdown(
            f'<div class="metric"><strong>{value}</strong><span>{label}</span></div>',
            unsafe_allow_html=True
        )

movie_titles = movies["title"].sort_values().tolist()
st.markdown('<p class="section-label">START YOUR DISCOVERY</p>', unsafe_allow_html=True)
controls = st.columns([3.2, 1.1, 1.25], vertical_alignment="bottom")
with controls[0]:
    selected_movie = st.selectbox(
        "Choose a movie",
        movie_titles,
        index=(
            movie_titles.index("Lagaan: Once Upon a Time in India")
            if "Lagaan: Once Upon a Time in India" in movie_titles
            else 0
        ),
    )
with controls[1]:
    number = st.slider("Results", 3, 10, 5)
with controls[2]:
    recommend_clicked = st.button("🎯 Discover", type="primary", use_container_width=True)

if recommend_clicked:
    movie_index = movies.index[movies["title"] == selected_movie][0]
    recommendations = get_recommendations(movie_index, movies, similarity, number)

    st.markdown(
        f'<p class="section-label">CURATED FOR YOU • {selected_movie.upper()}</p>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.05, 2.95], gap="large")
    with left:
        selected_year = movies.iloc[movie_index]["year"]
        st.markdown(
            f'<div class="selected-card"><div class="label">Now playing</div>'
            f'<h2>{selected_movie}</h2>'
            f'<p><strong>Genres:</strong> {movies.iloc[movie_index]["genres"]}</p>'
            f'<p><strong>Release year:</strong> {int(selected_year)}</p></div>',
            unsafe_allow_html=True
        )

    with right:
        st.markdown('<p class="section-label">NOW PLAYING</p>', unsafe_allow_html=True)
        st.markdown(
            "Choose a title to discover films with similar genres and themes.",
            unsafe_allow_html=True
        )

        st.markdown('<p class="section-label">SIMILAR STORIES</p>', unsafe_allow_html=True)
        cols = st.columns(min(5, len(recommendations)), gap="small")

        for col, (_, row) in zip(cols, recommendations.iterrows()):
            with col:
                st.markdown(
                    f'<div class="movie-card"><h4>{row["title"]}</h4>'
                    f'<p>Match score · {row["score"]:.0%}</p>'
                    f'<p>{row["genres"]}</p></div>',
                    unsafe_allow_html=True
                )

    st.divider()
    st.markdown('<p class="section-label">WHY THESE PICKS</p>', unsafe_allow_html=True)

    chart_df = recommendations[["title", "score"]].copy()
    chart_df["title"] = chart_df["title"].str.replace(
        r"\s+", " ", regex=True
    ).str.slice(0, 32)
    chart_df = chart_df.sort_values("score")

    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.barh(chart_df["title"], chart_df["score"], color="#58a6d9", height=.62)
    ax.set_xlabel("Similarity score", color="#e5e7f2", labelpad=8)
    ax.set_title(
        "Similarity score of recommended movies",
        color="#ffffff", fontsize=14, pad=14, loc="left"
    )
    ax.set_xlim(0, 1.08)
    ax.tick_params(axis="y", colors="#e5e7f2", labelsize=9, pad=6)
    ax.tick_params(axis="x", colors="#b8bdd5", labelsize=8)
    ax.grid(axis="x", color="#ffffff", alpha=.12, linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for value, label in zip(chart_df["score"], chart_df["title"]):
        ax.text(
            min(value + .015, 1.02), label, f"{value:.0%}",
            va="center", color="#ffffff", fontsize=8, fontweight="bold"
        )
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

st.divider()
st.markdown('<p class="section-label">EXPLORE THE COLLECTION</p>', unsafe_allow_html=True)

genre_counts = (
    movies["genres_clean"]
    .str.split()
    .explode()
    .value_counts()
    .head(10)
)

insight_tabs = st.tabs(["Genre map", "About the project"])
with insight_tabs[0]:
    genre_chart = genre_counts.sort_values()
    fig2, ax2 = plt.subplots(figsize=(8, 4.2))
    fig2.patch.set_alpha(0)
    ax2.set_facecolor("none")
    ax2.barh(genre_chart.index, genre_chart.values, color="#e85d75", height=.62)
    ax2.set_xlabel("Number of movies", color="#e5e7f2", labelpad=8)
    ax2.set_title("Most common genres", color="#ffffff", fontsize=14, pad=14, loc="left")
    ax2.tick_params(axis="y", colors="#e5e7f2", labelsize=9, pad=6)
    ax2.tick_params(axis="x", colors="#b8bdd5", labelsize=8)
    ax2.grid(axis="x", color="#ffffff", alpha=.12, linewidth=.8)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for value, label in zip(genre_chart.values, genre_chart.index):
        ax2.text(value + max(genre_chart.values) * .02, label, f"{value:,}",
                 va="center", color="#ffffff", fontsize=8, fontweight="bold")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
with insight_tabs[1]:
    st.markdown(
        "This compact discovery engine uses **TF-IDF** and **cosine similarity** "
        "to compare Hindi movie genres and find similar titles.",
    )

st.caption(
    "Educational mini-project using a Hindi Bollywood dataset and content-based filtering."
)
