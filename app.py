import html

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="MovieMind", page_icon="🎬", layout="wide")


# ============================================================
# ROYAL THEME  (oxblood velvet, antique gold, ivory)
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel+Decorative:wght@700;900&family=Cinzel:wght@500;600;700&family=Cormorant+Garamond:ital,wght@0,500;0,600;1,500&display=swap');

:root {
    --wine: #1a0508;
    --velvet: #4a0d1a;
    --velvet-hi: #7a1a2c;
    --navy: #0c1226;
    --gold: #d4af37;
    --gold-hi: #f6e27a;
    --gold-lo: #8a6d1d;
    --ivory: #f5ecd7;
    --muted: #c9b88f;
    --gold-grad: linear-gradient(120deg, #8a6d1d 0%, #f6e27a 35%, #d4af37 55%, #f6e27a 75%, #8a6d1d 100%);
    --panel: linear-gradient(160deg, rgba(74,13,26,.72), rgba(12,18,38,.78));
}

html, body, [class*="css"], .stApp { font-family: 'Cormorant Garamond', serif; font-size: 1.12rem; color: var(--ivory); }
h1, h2, h3, h4 { font-family: 'Cinzel', serif !important; color: var(--ivory); letter-spacing: .04em; }

/* Velvet + damask texture */
.stApp {
    background:
        radial-gradient(900px 500px at 50% -10%, rgba(212,175,55,.16), transparent 65%),
        radial-gradient(circle at 25% 30%, rgba(212,175,55,.05) 0 2px, transparent 3px) 0 0 / 46px 46px,
        radial-gradient(circle at 75% 70%, rgba(212,175,55,.05) 0 2px, transparent 3px) 0 0 / 46px 46px,
        repeating-linear-gradient(45deg, rgba(255,255,255,.012) 0 2px, transparent 2px 6px),
        linear-gradient(180deg, var(--wine), #0b0a14 60%, var(--navy));
    background-attachment: fixed;
}
header[data-testid="stHeader"], #MainMenu, footer { visibility: hidden; height: 0; }
.block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem; }

.gold-text { background: var(--gold-grad); background-size: 220% auto; -webkit-background-clip: text; background-clip: text; color: transparent; animation: shimmer 6s linear infinite; }
@keyframes shimmer { to { background-position: 220% center; } }
@keyframes rise { from { opacity: 0; transform: translateY(18px); } to { opacity: 1; transform: none; } }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes glow { 0%,100% { box-shadow: 0 0 0 1px rgba(212,175,55,.4), 0 0 26px rgba(212,175,55,.18); } 50% { box-shadow: 0 0 0 1px rgba(246,226,122,.7), 0 0 44px rgba(212,175,55,.38); } }
@keyframes curtainL { to { transform: translateX(-102%); } }
@keyframes curtainR { to { transform: translateX(102%); } }
@keyframes flicker { 0%,100% { opacity: .55; } 50% { opacity: .8; } }

/* ---------- Navigation ---------- */
.nav { display: flex; align-items: center; justify-content: space-between; padding: 12px 24px; border: 1px solid var(--gold-lo); border-radius: 4px; background: var(--panel); box-shadow: inset 0 0 0 4px rgba(0,0,0,.25), inset 0 0 0 5px rgba(212,175,55,.35); }
.nav .brand { font-family: 'Cinzel Decorative', serif; font-weight: 700; font-size: 1.3rem; }
.nav .links { display: flex; gap: 4px; }
.nav .links a { font-family: 'Cinzel', serif; color: var(--muted) !important; text-decoration: none; font-size: .82rem; letter-spacing: .12em; padding: 7px 14px; border-bottom: 1px solid transparent; transition: color .25s, border-color .25s; }
.nav .links a:hover, .nav .links a.active { color: var(--gold-hi) !important; border-color: var(--gold); }

/* ---------- Hero with opening curtains ---------- */
.hero { position: relative; overflow: hidden; margin: 22px 0 28px; padding: 84px 40px 72px; text-align: center; border: 1px solid var(--gold-lo); border-radius: 6px;
    background: radial-gradient(520px 300px at 50% 0%, rgba(246,226,122,.22), transparent 70%), var(--panel);
    box-shadow: inset 0 0 0 6px rgba(0,0,0,.3), inset 0 0 0 7px rgba(212,175,55,.5), 0 24px 60px rgba(0,0,0,.55); }
.hero .curtain { position: absolute; top: 0; bottom: 0; width: 51%; z-index: 5; pointer-events: none;
    background: repeating-linear-gradient(90deg, rgba(0,0,0,.35) 0 6px, transparent 6px 34px, rgba(255,255,255,.07) 34px 40px, transparent 40px 70px), linear-gradient(90deg, var(--velvet), var(--velvet-hi) 50%, var(--velvet));
    box-shadow: 0 0 40px rgba(0,0,0,.7); animation-duration: 2.2s; animation-delay: .35s; animation-timing-function: cubic-bezier(.7,0,.2,1); animation-fill-mode: forwards; }
.hero .curtain.l { left: 0; border-right: 3px solid var(--gold); animation-name: curtainL; }
.hero .curtain.r { right: 0; border-left: 3px solid var(--gold); animation-name: curtainR; }
.hero .crest { font-size: 2.4rem; animation: flicker 3s ease-in-out infinite; }
.hero h1 { font-family: 'Cinzel Decorative', serif !important; font-weight: 900; font-size: clamp(2.8rem, 8vw, 5.6rem); letter-spacing: .03em; line-height: 1.05; margin: .2rem 0 .6rem; }
.hero .sub { font-family: 'Cinzel', serif; letter-spacing: .28em; font-size: clamp(.8rem, 1.6vw, 1.05rem); color: var(--gold-hi); margin: 0 0 1rem; }
.hero .rule { width: 220px; height: 2px; margin: 0 auto 1.1rem; background: linear-gradient(90deg, transparent, var(--gold), transparent); }
.hero .desc { color: var(--muted); font-style: italic; max-width: 54ch; margin: 0 auto; font-size: 1.3rem; line-height: 1.55; }

/* ---------- Selection panel ---------- */
.st-key-select_card { background: var(--panel); border: 1px solid var(--gold-lo); border-radius: 6px; padding: 26px 30px; box-shadow: inset 0 0 0 4px rgba(0,0,0,.25), inset 0 0 0 5px rgba(212,175,55,.35), 0 14px 40px rgba(0,0,0,.5); }
.select-head h3 { margin: 0 0 4px; font-size: 1.5rem; }
.select-head p { color: var(--muted); margin: 0 0 8px; font-style: italic; }
div[data-baseweb="select"] > div, div[data-baseweb="base-input"] { background: rgba(0,0,0,.4) !important; border: 1px solid var(--gold-lo) !important; border-radius: 3px !important; color: var(--ivory) !important; }
label p { color: var(--gold-hi) !important; font-family: 'Cinzel', serif; font-size: .82rem !important; letter-spacing: .1em; }
div[data-testid="stExpander"] { border: 1px solid rgba(212,175,55,.3); border-radius: 4px; background: rgba(0,0,0,.2); }
div[data-testid="stExpander"] summary p { font-family: 'Cinzel', serif; color: var(--gold-hi); }

.stButton > button { font-family: 'Cinzel', serif; font-weight: 700; letter-spacing: .1em; color: #2a0a10; width: 100%; padding: .8rem 1.4rem; border: 1px solid var(--gold-hi); border-radius: 3px; background: var(--gold-grad); background-size: 220% auto; box-shadow: 0 8px 26px rgba(212,175,55,.3); transition: background-position .6s, transform .2s, box-shadow .2s; }
.stButton > button:hover { color: #2a0a10; background-position: 100% center; transform: translateY(-2px); box-shadow: 0 14px 34px rgba(246,226,122,.4); }
.stButton > button:focus-visible { outline: 3px solid var(--gold-hi); outline-offset: 2px; }

.section-title { font-family: 'Cinzel', serif; font-size: 1.55rem; font-weight: 700; margin: 40px 0 6px; text-align: center; }
.section-sub { text-align: center; color: var(--muted); font-style: italic; margin-bottom: 22px; }
.ornament { text-align: center; color: var(--gold); letter-spacing: .6em; margin: 0 0 4px; }

/* ---------- Empty state ---------- */
.empty { margin-top: 28px; text-align: center; padding: 56px 24px; border: 1px dashed var(--gold-lo); border-radius: 6px; background: radial-gradient(480px 200px at 50% 0%, rgba(212,175,55,.14), transparent 70%), rgba(0,0,0,.25); animation: rise .8s ease both; }
.empty .reel { font-size: 3.2rem; }
.empty h3 { font-size: 1.5rem; margin: 6px 0; }
.empty p { color: var(--muted); font-style: italic; margin: 0; }

/* ---------- Plaques (metrics) ---------- */
.metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 28px; }
.metric { text-align: center; padding: 20px 10px; border: 1px solid var(--gold-lo); border-radius: 4px; background: var(--panel); box-shadow: inset 0 0 0 3px rgba(0,0,0,.25), inset 0 0 0 4px rgba(212,175,55,.3); animation: rise .6s ease both; }
.metric .ic { font-size: 1.4rem; }
.metric b { font-family: 'Cinzel', serif; font-size: 2rem; display: block; color: var(--gold-hi); line-height: 1.2; }
.metric small { color: var(--muted); letter-spacing: .08em; }

/* ---------- Recommendation tickets ---------- */
.cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; }
.card { position: relative; padding: 26px 24px 22px; min-height: 250px; display: flex; flex-direction: column; justify-content: space-between; border: 1px solid var(--gold-lo); border-radius: 4px; background: linear-gradient(165deg, rgba(122,26,44,.55), rgba(12,18,38,.85)); box-shadow: inset 0 0 0 4px rgba(0,0,0,.25), inset 0 0 0 5px rgba(212,175,55,.32), 0 14px 34px rgba(0,0,0,.5); animation: rise .7s ease both; transition: transform .3s ease, box-shadow .3s ease; overflow: hidden; }
.card::after { content: ""; position: absolute; top: 0; left: -80%; width: 50%; height: 100%; background: linear-gradient(100deg, transparent, rgba(246,226,122,.22), transparent); transform: skewX(-18deg); transition: left .8s ease; }
.card:hover { transform: translateY(-8px) scale(1.015); box-shadow: inset 0 0 0 4px rgba(0,0,0,.25), inset 0 0 0 5px rgba(246,226,122,.7), 0 26px 50px rgba(0,0,0,.6); }
.card:hover::after { left: 130%; }
.card .rank { font-family: 'Cinzel Decorative', serif; font-size: 1.9rem; font-weight: 900; color: var(--gold); line-height: 1; }
.card h3 { font-size: 1.28rem; line-height: 1.25; margin: 12px 0 8px; overflow-wrap: anywhere; }
.card .tags { display: flex; flex-wrap: wrap; gap: 6px; }
.card .tag { font-family: 'Cinzel', serif; font-size: .68rem; letter-spacing: .1em; padding: 3px 9px; border: 1px solid var(--gold-lo); border-radius: 99px; color: var(--gold-hi); background: rgba(0,0,0,.3); }
.card .stars { color: var(--gold-hi); font-size: 1.25rem; letter-spacing: .1em; margin-top: 16px; }
.card .score { font-family: 'Cinzel', serif; font-weight: 700; color: var(--ivory); }
.card .note { color: var(--muted); font-style: italic; margin-top: 4px; }
.card.first { border-color: var(--gold-hi); animation: rise .7s ease both, glow 3.5s ease-in-out infinite; background: linear-gradient(165deg, rgba(150,40,60,.7), rgba(12,18,38,.85)); }
.badge { position: absolute; top: 16px; right: -34px; transform: rotate(38deg); width: 130px; text-align: center; font-family: 'Cinzel', serif; font-size: .62rem; font-weight: 700; letter-spacing: .12em; padding: 4px 0; background: var(--gold-grad); color: #2a0a10; }

/* ---------- Similar users ---------- */
.glass { background: var(--panel); border: 1px solid var(--gold-lo); border-radius: 4px; padding: 10px 26px; box-shadow: inset 0 0 0 3px rgba(0,0,0,.25), inset 0 0 0 4px rgba(212,175,55,.3), 0 12px 36px rgba(0,0,0,.45); }
.user-row { display: flex; align-items: center; gap: 16px; padding: 14px 0; border-bottom: 1px solid rgba(212,175,55,.2); }
.user-row:last-child { border-bottom: none; }
.user-row .av { width: 40px; height: 40px; border-radius: 50%; border: 1px solid var(--gold); background: radial-gradient(circle, var(--velvet-hi), var(--wine)); display: grid; place-items: center; font-family: 'Cinzel', serif; font-weight: 700; color: var(--gold-hi); flex-shrink: 0; }
.user-row .nm { width: 130px; font-family: 'Cinzel', serif; font-size: .9rem; flex-shrink: 0; }
.user-row .bar { flex: 1; height: 10px; border-radius: 99px; background: rgba(0,0,0,.45); border: 1px solid rgba(212,175,55,.25); overflow: hidden; }
.user-row .bar i { display: block; height: 100%; transform-origin: left; background: var(--gold-grad); animation: grow 1.2s cubic-bezier(.2,.8,.2,1) both; }
.user-row .pct { width: 56px; text-align: right; font-family: 'Cinzel', serif; color: var(--gold-hi); font-variant-numeric: tabular-nums; }

/* ---------- How it works ---------- */
.flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; }
.step { position: relative; text-align: center; padding: 28px 18px; border: 1px solid var(--gold-lo); border-radius: 4px; background: var(--panel); transition: transform .3s, border-color .3s; }
.step:hover { transform: translateY(-5px); border-color: var(--gold-hi); }
.step .ic { font-size: 2rem; margin-bottom: 8px; }
.step h4 { margin: 0 0 6px; font-size: 1rem; color: var(--gold-hi); }
.step p { color: var(--muted); font-size: 1rem; margin: 0; line-height: 1.45; }
.step:not(:last-child)::after { content: "❖"; position: absolute; right: -17px; top: 50%; transform: translateY(-50%); width: 14px; color: var(--gold); z-index: 2; }

/* ---------- Footer ---------- */
.footer { margin-top: 64px; padding: 30px 10px 6px; text-align: center; border-top: 1px solid var(--gold-lo); }
.footer .b { font-family: 'Cinzel Decorative', serif; font-weight: 700; font-size: 1.25rem; }
.footer p { color: var(--muted); margin: 4px 0; font-size: 1rem; }

@media (max-width: 980px) {
    .cards, .metrics, .flow { grid-template-columns: repeat(2, 1fr); }
    .step:not(:last-child)::after { display: none; }
}
@media (max-width: 620px) {
    .hero { padding: 60px 18px 50px; }
    .cards, .metrics, .flow { grid-template-columns: 1fr; }
    .nav .links { display: none; }
    .user-row .nm { width: 90px; }
}
@media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
    .hero .curtain { display: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# NAVIGATION + HERO
# ============================================================
st.markdown(
    '<div class="nav"><div class="brand">🎬 <span class="gold-text">MovieMind</span></div>'
    '<div class="links"><a class="active" href="#home">HOME</a>'
    '<a href="#recommendations">RECOMMENDATIONS</a>'
    '<a href="#how-it-works">HOW IT WORKS</a></div></div>'
    '<div id="home"></div>'
    '<div class="hero"><div class="curtain l"></div><div class="curtain r"></div>'
    '<div class="crest">👑</div>'
    '<h1 class="gold-text">MovieMind</h1>'
    '<p class="sub">THE ROYAL CINEMA OF YOUR TASTE</p>'
    '<div class="rule"></div>'
    '<p class="desc">Take your seat in the royal box. MovieMind studies viewers who love what you love, '
    'and presents the films you have yet to discover.</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# BACKEND (same logic, now cached)
# ============================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_excel("data/Movie_Recommendation_System.xlsx", sheet_name="Movie_Data")
    df["Genre"] = df["Genre"].fillna("Unknown")
    df["Language"] = df["Language"].fillna("Unknown")
    df = df.dropna(subset=["Rating"])
    df = df.drop_duplicates(subset=["User_ID", "Movie_Title"])
    return df


@st.cache_data(show_spinner=False)
def build_model(df):
    matrix = df.pivot_table(index="User_ID", columns="Movie_Title", values="Rating", aggfunc="mean")
    sim = cosine_similarity(matrix.fillna(0))
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    info = df.drop_duplicates("Movie_Title").set_index("Movie_Title")[["Genre", "Language"]]
    return matrix, sim_df, info


df = load_data()
user_movie_matrix, similarity_df, movie_info = build_model(df)


# ============================================================
# SELECTION PANEL
# ============================================================
with st.container(key="select_card"):
    st.markdown(
        '<div class="select-head"><h3>🎟️ Reserve Your Private Screening</h3>'
        '<p>Choose a viewer profile and we will curate films from the tastes closest to yours.</p></div>',
        unsafe_allow_html=True,
    )
    sel_col, btn_col = st.columns([2, 1], vertical_alignment="bottom")
    with sel_col:
        user_id = st.selectbox("Select User ID", user_movie_matrix.index)
    with btn_col:
        clicked = st.button("✨ Present My Films")

    with st.expander("Fine-tune your screening"):
        c1, c2, c3 = st.columns(3)
        with c1:
            n_users = st.slider("Similar users to consult", 1, 15, 5)
        with c2:
            n_recs = st.slider("Films to recommend", 3, 12, 6)
        with c3:
            genres = st.multiselect("Limit to genres", sorted(df["Genre"].unique()))

st.markdown('<div id="recommendations"></div>', unsafe_allow_html=True)


# ============================================================
# RESULTS / EMPTY STATE
# ============================================================
if clicked:
    with st.spinner("The projectionist is threading the reel..."):
        similar_users = similarity_df[user_id].sort_values(ascending=False).drop(user_id)
        top_users = similar_users.head(n_users).index
        similar_users_ratings = user_movie_matrix.loc[top_users]
        movie_scores = similar_users_ratings.mean(axis=0)
        support = similar_users_ratings.notna().sum(axis=0)

        user_movies = user_movie_matrix.loc[user_id].dropna()
        movie_scores = movie_scores.drop(user_movies.index, errors="ignore").dropna()

        if genres:
            allowed = movie_info.index[movie_info["Genre"].isin(genres)]
            movie_scores = movie_scores[movie_scores.index.isin(allowed)]

        recommendations = movie_scores.sort_values(ascending=False).head(n_recs)

    metrics = [
        ("👤", html.escape(str(user_id)), "Selected User"),
        ("🎞️", len(user_movies), "Movies Rated"),
        ("👥", len(top_users), "Similar Users"),
        ("🏆", len(recommendations), "Recommendations"),
    ]
    st.markdown(
        '<div class="metrics">'
        + "".join(
            f'<div class="metric" style="animation-delay:{k * 0.1:.1f}s"><div class="ic">{ic}</div><b>{val}</b><small>{label}</small></div>'
            for k, (ic, val, label) in enumerate(metrics)
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ornament">❖ ❖ ❖</div><div class="section-title">Tonight\'s Royal Selection</div>'
        '<div class="section-sub">Films loved by viewers most like you</div>',
        unsafe_allow_html=True,
    )

    if recommendations.empty:
        st.info("No unseen films match these filters. Try more genres or consult more similar users.")
    else:
        notes = {1: "The crown jewel of your evening", 2: "A splendid match", 3: "A splendid match"}
        cards = []
        for i, (movie, score) in enumerate(recommendations.items(), start=1):
            first = " first" if i == 1 else ""
            badge = '<span class="badge">TOP PICK</span>' if i == 1 else ""
            genre = html.escape(str(movie_info["Genre"].get(movie, "Unknown")))
            lang = html.escape(str(movie_info["Language"].get(movie, "Unknown")))
            stars = int(round(min(float(score), 5)))
            star_txt = "★" * stars + "☆" * (5 - stars)
            backers = int(support.get(movie, 0))
            cards.append(
                f'<div class="card{first}" style="animation-delay:{0.15 + i * 0.1:.2f}s">{badge}'
                f'<div><div class="rank">{i:02d}</div><h3>{html.escape(str(movie))}</h3>'
                f'<div class="tags"><span class="tag">{genre}</span><span class="tag">{lang}</span></div></div>'
                f'<div><div class="stars">{star_txt} <span class="score">{float(score):.2f}</span></div>'
                f'<div class="note">{notes.get(i, "Chosen for you")} · rated by {backers} similar viewer{"s" if backers != 1 else ""}</div></div></div>'
            )
        st.markdown('<div class="cards">' + "".join(cards) + "</div>", unsafe_allow_html=True)

        st.download_button(
            "Download my list (CSV)",
            recommendations.rename("Predicted_Rating").round(2).to_csv().encode("utf-8"),
            file_name=f"moviemind_{user_id}.csv",
            mime="text/csv",
        )

    st.markdown(
        '<div class="ornament">❖ ❖ ❖</div><div class="section-title">Your Kindred Viewers</div>'
        '<div class="section-sub">The audience members whose taste is closest to yours</div>',
        unsafe_allow_html=True,
    )
    rows = []
    for uid, sim in similar_users.head(n_users).items():
        pct = max(0.0, min(100.0, float(sim) * 100))
        label = html.escape(str(uid))
        rows.append(
            f'<div class="user-row"><div class="av">{label[:1].upper()}</div><div class="nm">User {label}</div>'
            f'<div class="bar"><i style="width:{pct:.0f}%"></i></div><div class="pct">{pct:.0f}%</div></div>'
        )
    st.markdown('<div class="glass">' + "".join(rows) + "</div>", unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="empty"><div class="reel">🎞️</div>'
        "<h3>The Screen Awaits</h3>"
        "<p>Select your User ID above and press “Present My Films” to begin your screening.</p></div>",
        unsafe_allow_html=True,
    )


# ============================================================
# HOW IT WORKS
# ============================================================
st.markdown(
    '<div id="how-it-works"></div><div class="ornament">❖ ❖ ❖</div>'
    '<div class="section-title">How MovieMind Works</div>'
    '<div class="section-sub">Four steps from your ratings to your next favourite film</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="flow">'
    '<div class="step"><div class="ic">📜</div><h4>Your Ratings</h4><p>We begin with the films you have already rated.</p></div>'
    '<div class="step"><div class="ic">🤝</div><h4>Similar Users</h4><p>Cosine similarity finds viewers whose taste matches yours.</p></div>'
    '<div class="step"><div class="ic">⚖️</div><h4>Preference Analysis</h4><p>Their ratings are averaged for films you have not seen.</p></div>'
    '<div class="step"><div class="ic">🏆</div><h4>Recommendations</h4><p>The highest-scoring films become your top picks.</p></div>'
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer"><div class="b">🎬 <span class="gold-text">MovieMind</span></div>'
    "<p>Personalized Movie Recommendation System</p>"
    "<p>Built with Python • Pandas • Scikit-learn • Streamlit</p>"
    "<p>Developed by Hemant Singh</p></div>",
    unsafe_allow_html=True,
)