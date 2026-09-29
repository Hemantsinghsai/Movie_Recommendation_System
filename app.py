import html

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


# Page configuration
st.set_page_config(
    page_title="MovieMind",
    page_icon="🎬",
    layout="wide"
)


# ============================================================
# UI THEME (presentation only)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg: #07070D;
    --glass: rgba(255,255,255,.055);
    --glass-strong: rgba(255,255,255,.09);
    --border: rgba(255,255,255,.12);
    --text: #F3F2FA;
    --muted: #A5A3BD;
    --violet: #7C5CFF;
    --pink: #FF4D8D;
    --gold: #FFC857;
    --grad: linear-gradient(135deg, #7C5CFF 0%, #FF4D8D 100%);
}

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; color: var(--text); }
h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; color: var(--text); }

.stApp {
    background:
        radial-gradient(700px 420px at 12% -5%, rgba(124,92,255,.28), transparent 60%),
        radial-gradient(640px 420px at 92% 8%, rgba(255,77,141,.20), transparent 60%),
        var(--bg);
}
header[data-testid="stHeader"], #MainMenu, footer { visibility: hidden; height: 0; }
.block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
.fade { animation: fadeUp .6s ease both; }

/* ---------- Navigation ---------- */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; border: 1px solid var(--border); border-radius: 16px;
    background: var(--glass); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
.nav .brand { font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.25rem; }
.nav .links { display: flex; gap: 6px; }
.nav .links a {
    color: var(--muted) !important; text-decoration: none; font-size: .92rem; font-weight: 500;
    padding: 7px 14px; border-radius: 10px; transition: background .2s, color .2s;
}
.nav .links a:hover { background: var(--glass-strong); color: var(--text) !important; }
.nav .links a.active { color: var(--text) !important; background: var(--glass-strong); }

/* ---------- Hero ---------- */
.hero {
    position: relative; overflow: hidden; margin: 22px 0 26px 0; padding: 64px 40px;
    border: 1px solid var(--border); border-radius: 26px; text-align: center;
    background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02));
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
}
.hero::before, .hero::after {
    content: ""; position: absolute; left: 0; right: 0; height: 14px; opacity: .35;
    background: repeating-linear-gradient(90deg, rgba(255,255,255,.55) 0 16px, transparent 16px 34px);
}
.hero::before { top: 10px; }
.hero::after { bottom: 10px; }
.hero h1 {
    font-size: clamp(2.8rem, 7vw, 5rem); font-weight: 800; letter-spacing: -0.03em; line-height: 1.05; margin: 0 0 .5rem 0;
}
.grad-text { background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero .sub { font-family: 'Outfit', sans-serif; font-size: clamp(1.05rem, 2.2vw, 1.4rem); font-weight: 600; margin: 0 0 .9rem 0; }
.hero .desc { color: var(--muted); max-width: 52ch; margin: 0 auto; font-size: 1.02rem; line-height: 1.6; }

/* ---------- Glass cards ---------- */
.glass {
    background: var(--glass); border: 1px solid var(--border); border-radius: 20px; padding: 10px 24px;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    box-shadow: 0 12px 40px rgba(0,0,0,.35);
}
.section-title { font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 700; margin: 34px 0 14px 0; }

/* Selection card (Streamlit container with key="select_card") */
.st-key-select_card {
    background: var(--glass); border: 1px solid var(--border); border-radius: 20px; padding: 26px 28px;
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); box-shadow: 0 12px 40px rgba(0,0,0,.35);
}
.select-head h3 { margin: 0 0 4px 0; font-size: 1.5rem; }
.select-head p { color: var(--muted); margin: 0 0 6px 0; }
div[data-baseweb="select"] > div {
    background: rgba(0,0,0,.35) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
}
label p { color: var(--muted) !important; }

/* Button */
.stButton > button {
    background: var(--grad); color: #fff; border: none; border-radius: 14px; font-weight: 600; font-size: 1.02rem;
    padding: .75rem 1.6rem; width: 100%; transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
    box-shadow: 0 8px 26px rgba(124,92,255,.35);
}
.stButton > button:hover { color: #fff; transform: translateY(-2px); filter: brightness(1.08); box-shadow: 0 14px 34px rgba(255,77,141,.35); }
.stButton > button:focus-visible { outline: 3px solid #fff; outline-offset: 2px; }

/* ---------- Empty state ---------- */
.empty {
    margin-top: 26px; text-align: center; padding: 54px 24px; border-radius: 22px;
    border: 1px dashed rgba(255,255,255,.22);
    background: radial-gradient(500px 220px at 50% 0%, rgba(124,92,255,.16), transparent 70%), var(--glass);
}
.empty .reel { font-size: 3rem; margin-bottom: 6px; }
.empty h3 { font-size: 1.6rem; margin: 0 0 6px 0; }
.empty p { color: var(--muted); margin: 0; }

/* ---------- Metrics ---------- */
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 26px; }
.metric { padding: 20px; border-radius: 18px; background: var(--glass); border: 1px solid var(--border); transition: transform .25s, border-color .25s; }
.metric:hover { transform: translateY(-4px); border-color: rgba(255,255,255,.28); }
.metric .ic {
    width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; font-size: 1.15rem;
    background: rgba(124,92,255,.18); margin-bottom: 12px;
}
.metric b { font-family: 'Outfit', sans-serif; font-size: 1.9rem; display: block; line-height: 1.1; }
.metric small { color: var(--muted); }

/* ---------- Recommendation cards ---------- */
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }
.card {
    position: relative; padding: 24px; border-radius: 20px; min-height: 210px;
    display: flex; flex-direction: column; justify-content: space-between;
    background: var(--glass); border: 1px solid var(--border); backdrop-filter: blur(14px);
    box-shadow: 0 12px 34px rgba(0,0,0,.35);
    animation: fadeUp .55s ease both; transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.card:hover { transform: translateY(-6px); border-color: rgba(255,255,255,.3); box-shadow: 0 20px 44px rgba(0,0,0,.5); }
.card .rank { font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.05rem; color: var(--muted); }
.card h3 { font-size: 1.4rem; line-height: 1.2; margin: 10px 0; overflow-wrap: anywhere; }
.card .score { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; color: var(--gold); font-size: 1.1rem; }
.card .note { color: var(--muted); font-size: .9rem; margin-top: 8px; }
.card.first {
    background: linear-gradient(160deg, rgba(124,92,255,.30), rgba(255,77,141,.18));
    border-color: rgba(255,120,190,.55); box-shadow: 0 0 0 1px rgba(255,77,141,.25), 0 18px 50px rgba(124,92,255,.35);
}
.card.first .rank { color: #fff; }
.badge { position: absolute; top: 18px; right: 18px; font-size: .75rem; font-weight: 600; padding: 4px 10px; border-radius: 99px; background: var(--grad); color: #fff; }

/* ---------- Similar users ---------- */
.user-row { display: flex; align-items: center; gap: 16px; padding: 12px 0; border-bottom: 1px solid var(--border); }
.user-row:last-child { border-bottom: none; }
.user-row .av { width: 38px; height: 38px; border-radius: 50%; background: var(--grad); display: grid; place-items: center; font-weight: 700; flex-shrink: 0; }
.user-row .nm { width: 120px; font-weight: 600; flex-shrink: 0; }
.user-row .bar { flex: 1; height: 10px; border-radius: 99px; background: rgba(255,255,255,.1); overflow: hidden; }
.user-row .bar i { display: block; height: 100%; border-radius: 99px; background: var(--grad); }
.user-row .pct { width: 54px; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }

/* ---------- How it works ---------- */
.flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; }
.step { position: relative; text-align: center; padding: 26px 18px; border-radius: 20px; background: var(--glass); border: 1px solid var(--border); }
.step .ic { font-size: 2rem; margin-bottom: 10px; }
.step h4 { margin: 0 0 6px 0; font-size: 1.1rem; }
.step p { color: var(--muted); font-size: .9rem; margin: 0; line-height: 1.5; }
.step:not(:last-child)::after {
    content: "→"; position: absolute; right: -16px; top: 50%; transform: translateY(-50%);
    width: 14px; text-align: center; color: var(--pink); font-weight: 700; z-index: 2;
}

/* ---------- Footer ---------- */
.footer { margin-top: 60px; padding: 30px 10px 6px; text-align: center; border-top: 1px solid var(--border); }
.footer .b { font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.2rem; }
.footer p { color: var(--muted); margin: 4px 0; font-size: .9rem; }

/* ---------- Responsive ---------- */
@media (max-width: 980px) {
    .cards { grid-template-columns: repeat(2, 1fr); }
    .metrics { grid-template-columns: repeat(2, 1fr); }
    .flow { grid-template-columns: repeat(2, 1fr); }
    .step:not(:last-child)::after { display: none; }
}
@media (max-width: 620px) {
    .hero { padding: 48px 20px; }
    .cards, .metrics, .flow { grid-template-columns: 1fr; }
    .nav .links { display: none; }
    .user-row .nm { width: 84px; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION + HERO
# ============================================================
st.markdown(
    '<div class="nav fade"><div class="brand">🎬 <span class="grad-text">MovieMind</span></div>'
    '<div class="links"><a class="active" href="#home">Home</a>'
    '<a href="#recommendations">Recommendations</a>'
    '<a href="#how-it-works">How It Works</a></div></div>'
    '<div id="home"></div>'
    '<div class="hero fade">'
    '<h1>🎬 <span class="grad-text">MovieMind</span></h1>'
    '<p class="sub">Personalized Movie Recommendations Powered by Collaborative Filtering</p>'
    '<p class="desc">Discover movies tailored to your taste using intelligent user-based collaborative filtering.</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND (unchanged)
# ============================================================

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


# ============================================================
# USER SELECTION CARD
# ============================================================
with st.container(key="select_card"):
    st.markdown(
        '<div class="select-head"><h3>👤 Find Movies For You</h3>'
        '<p>Select a user profile to generate personalized recommendations based on similar viewing preferences.</p></div>',
        unsafe_allow_html=True,
    )
    sel_col, btn_col = st.columns([2, 1], vertical_alignment="bottom")
    with sel_col:
        # User selection
        user_id = st.selectbox(
            "Select User ID",
            user_movie_matrix.index
        )
    with btn_col:
        clicked = st.button("✨ Get My Recommendations")

st.markdown('<div id="recommendations"></div>', unsafe_allow_html=True)


# ============================================================
# RESULTS / EMPTY STATE
# ============================================================
if clicked:

    with st.spinner("Finding movies you'll love..."):

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

    # ---------- Statistics ----------
    metrics = [
        ("👤", html.escape(str(user_id)), "Selected User"),
        ("🎬", len(user_movies), "Movies Rated"),
        ("👥", len(top_users), "Similar Users"),
        ("🎯", len(recommendations), "Recommendations"),
    ]
    st.markdown(
        '<div class="metrics fade">'
        + "".join(
            f'<div class="metric"><div class="ic">{ic}</div><b>{val}</b><small>{label}</small></div>'
            for ic, val, label in metrics
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ---------- Recommendation cards ----------
    st.markdown('<div class="section-title">🎯 Your Personalized Recommendations</div>', unsafe_allow_html=True)

    notes = {1: "Highly recommended for you", 2: "Great match", 3: "Great match"}
    cards = []
    for i, (movie, score) in enumerate(recommendations.items(), start=1):
        score_txt = "—" if pd.isna(score) else f"{score:.2f}"
        first = " first" if i == 1 else ""
        badge = '<span class="badge">Top pick</span>' if i == 1 else ""
        cards.append(
            f'<div class="card{first}" style="animation-delay:{i * 0.08:.2f}s">{badge}'
            f'<div><div class="rank">#{i}</div><h3>🎬 {html.escape(str(movie))}</h3></div>'
            f'<div><div class="score">⭐ {score_txt}</div>'
            f'<div class="note">{notes.get(i, "Recommended for you")}</div></div></div>'
        )
    st.markdown('<div class="cards">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    # ---------- Similar users ----------
    st.markdown('<div class="section-title">👥 Users With Similar Taste</div>', unsafe_allow_html=True)

    rows = []
    for uid, sim in similar_users.head(5).items():
        pct = max(0.0, min(100.0, float(sim) * 100))
        label = html.escape(str(uid))
        rows.append(
            f'<div class="user-row"><div class="av">👤</div><div class="nm">User {label}</div>'
            f'<div class="bar"><i style="width:{pct:.0f}%"></i></div><div class="pct">{pct:.0f}%</div></div>'
        )
    st.markdown('<div class="glass fade">' + "".join(rows) + "</div>", unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="empty fade"><div class="reel">🎞️</div>'
        "<h3>🎬 Discover Your Next Favorite Movie</h3>"
        "<p>Select your User ID and let MovieMind discover movies that match your taste.</p></div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HOW IT WORKS
# ============================================================
st.markdown('<div id="how-it-works"></div><div class="section-title">🧠 How MovieMind Works</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="flow">'
    '<div class="step"><div class="ic">👤</div><h4>Your Ratings</h4><p>We start with the movies you have already rated.</p></div>'
    '<div class="step"><div class="ic">🤝</div><h4>Similar Users</h4><p>Cosine similarity finds people whose taste matches yours.</p></div>'
    '<div class="step"><div class="ic">📊</div><h4>Preference Analysis</h4><p>Their ratings are averaged for movies you have not seen.</p></div>'
    '<div class="step"><div class="ic">🎬</div><h4>Movie Recommendations</h4><p>The highest-scoring movies become your top picks.</p></div>'
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer"><div class="b">🎬 <span class="grad-text">MovieMind</span></div>'
    "<p>Personalized Movie Recommendation System</p>"
    "<p>Built with Python • Pandas • Scikit-learn • Streamlit</p>"
    "<p>Developed by Hemant Singh</p></div>",
    unsafe_allow_html=True,
)