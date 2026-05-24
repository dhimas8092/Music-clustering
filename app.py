# app.py ──────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import unicodedata
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import squareform

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Song Clustering System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('df_sample.csv')
    df['tokens'] = df['tokens'].apply(lambda x: str(x).split())
    matrices    = np.load('matrices.npz')
    sim_matrix  = matrices['similarity_matrix']
    dist_matrix = matrices['distance_matrix']
    Z           = np.load('linkage_matrix.npy')
    with open('model_config.json') as f:
        config = json.load(f)
    return df, sim_matrix, dist_matrix, Z, config

df, sim_matrix, dist_matrix, Z, config = load_data()

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
STOPWORDS = set([
    'yang','di','ke','dari','dan','atau','ini','itu','dengan','untuk','pada',
    'adalah','dalam','tidak','akan','juga','ada','saya','aku','kamu','dia',
    'kita','kami','mereka','nya','ku','mu','lah','pun','kan','tak','ya','oh',
    'ah','hai','hei','si','sang','para','se','ter','me','ber',
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','from','is','it','its','be','as','this','that','are','was','were',
    'been','have','has','had','do','does','did','will','would','could',
    'should','may','might','shall','can','not','no','my','your','his','her',
    'our','their','i','you','he','she','we','they','me','him','us','them',
    'so','if','up','out','about','into','than','then','when','where','who',
    'what','how','all','just','like','get','got','go','oh','im','ive',
    'dont','cant','wont',
])

MOOD_COLORS = {
    'Happy'      : '#FFD700', 'Energetic' : '#FF6B35',
    'Groovy'     : '#9B59B6', 'Dark'      : '#2C3E50',
    'Sad'        : '#5DADE2', 'Melancholic': '#85929E',
    'Romantic'   : '#E91E8C', 'Calm'      : '#27AE60',
}

GENRE_COLORS = {
    'Pop'               : '#FF6B6B', 'Rock'          : '#4ECDC4',
    'Metal'             : '#2C3E50', 'Electronic'    : '#45B7D1',
    'Hip-Hop'           : '#96CEB4', 'Jazz & Blues'  : '#FFEAA7',
    'Classical & Ambient': '#DDA0DD', 'Folk & Country': '#98D8C8',
    'Latin'             : '#F7DC6F', 'World Music'   : '#BB8FCE',
}

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def preprocess_judul(judul):
    judul = unicodedata.normalize('NFKD', judul)
    judul = judul.encode('ascii', 'ignore').decode('ascii')
    judul = judul.lower()
    judul = re.sub(r'[^a-z\s]', ' ', judul)
    tokens = judul.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

def jaccard_similarity(tokens_a, tokens_b):
    set_a, set_b = set(tokens_a), set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)

def query_lagu(judul_input, top_n=5):
    tokens_input = preprocess_judul(judul_input)
    if not tokens_input:
        return None, None, None, None
    sims = [jaccard_similarity(tokens_input, row) for row in df['tokens']]
    df_result             = df.copy()
    df_result['similarity'] = sims
    df_top = df_result.sort_values('similarity', ascending=False).head(top_n)
    pred_genre   = df_top['genre_utama'].value_counts().index[0]
    pred_mood    = df_top['mood'].value_counts().index[0]
    pred_cluster = int(df_top['cluster'].value_counts().index[0])
    return df_top[['judul_lagu','artis','genre_utama','mood','cluster','similarity']].reset_index(drop=True), \
           pred_genre, pred_mood, pred_cluster

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎵 Song Clustering")
    st.markdown("*Jaccard Similarity + Hierarchical Clustering*")
    st.divider()
    page = st.radio("Navigation", [
        "🔍 Search", "📊 Exploration", "🌿 Dendrogram",
        "🔥 Heatmap", "📋 Dataset"
    ])
    st.divider()
    st.markdown("**Filter**")
    genre_filter = st.selectbox("Genre", ["All Genres"] + sorted(df['genre_utama'].unique()))
    st.divider()
    st.markdown(f"**Model Info**")
    st.markdown(f"- Clusters: `{config['best_k']}`")
    st.markdown(f"- Silhouette: `{config['silhouette']:.4f}`")
    st.markdown(f"- Total songs: `{config['total_sample']:,}`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: SEARCH
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Search":
    st.title("🔍 Song Search & Analysis")
    st.markdown("Enter a song title to find similar songs based on **Jaccard Similarity**.")

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        judul_input = st.text_input("", placeholder="e.g. Love Story, Dark Night, Cinta Dalam Diam")
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("🔍 Analyze", use_container_width=True)

    if analyze and judul_input:
        df_top, pred_genre, pred_mood, pred_cluster = query_lagu(judul_input)

        if df_top is None:
            st.warning("⚠️ Title too short or contains no meaningful words.")
        else:
            # Prediksi badges
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Genre", pred_genre)
            c2.metric("Mood", pred_mood)
            c3.metric("Cluster", f"#{pred_cluster}")
            c4.metric("Tokens", str(preprocess_judul(judul_input)))

            st.markdown("---")
            col_left, col_right = st.columns([3, 2])

            # Top-5 rekomendasi
            with col_left:
                st.markdown("#### 🎯 Similar Songs (Top-5)")
                for i, row in df_top.iterrows():
                    mood_color  = MOOD_COLORS.get(row['mood'], '#888')
                    genre_color = GENRE_COLORS.get(row['genre_utama'], '#888')
                    st.markdown(
                        f"""
                        <div style='padding:10px;margin:5px 0;border-radius:8px;
                                    background:#1e1e2e;border-left:4px solid {genre_color}'>
                            <b style='font-size:15px'>{i+1}. {row['judul_lagu']}</b><br>
                            <span style='color:#aaa;font-size:13px'>{row['artis']}</span><br>
                            <span style='background:{genre_color};color:#000;padding:2px 8px;
                                         border-radius:10px;font-size:12px'>{row['genre_utama']}</span>
                            <span style='background:{mood_color};color:#000;padding:2px 8px;
                                         border-radius:10px;font-size:12px;margin-left:5px'>{row['mood']}</span>
                            <span style='float:right;color:#fff;font-weight:bold'>{row['similarity']:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True
                    )

            # Mini similarity matrix
            with col_right:
                st.markdown("#### 🔢 Similarity Matrix (5×5)")
                top5_idx = df_top.index.tolist()
                if len(top5_idx) >= 2:
                    mini_sim = sim_matrix[np.ix_(top5_idx, top5_idx)]
                    fig, ax  = plt.subplots(figsize=(4, 3.5))
                    fig.patch.set_facecolor('#0e1117')
                    ax.set_facecolor('#0e1117')
                    sns.heatmap(
                        mini_sim, ax=ax, annot=True, fmt='.2f',
                        cmap='Blues', linewidths=0.5,
                        xticklabels=[f"L{i+1}" for i in range(len(top5_idx))],
                        yticklabels=[f"L{i+1}" for i in range(len(top5_idx))],
                        cbar=False, annot_kws={'size': 9, 'color': 'white'},
                    )
                    ax.tick_params(colors='white')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.info("Not enough similar songs to display matrix.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Exploration":
    st.title("📊 Data Exploration")

    df_show = df if genre_filter == "All Genres" else df[df['genre_utama'] == genre_filter]

    st.markdown(f"Showing **{len(df_show):,}** songs · Filter: `{genre_filter}`")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Genre Distribution")
        genre_counts = df_show['genre_utama'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        bars = ax.barh(genre_counts.index, genre_counts.values,
                       color=[GENRE_COLORS.get(g, '#888') for g in genre_counts.index])
        ax.set_xlabel('Count', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_visible(False)
        for bar, val in zip(bars, genre_counts.values):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color='white', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Mood Distribution")
        mood_counts = df_show['mood'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        wedges, texts, autotexts = ax.pie(
            mood_counts.values,
            labels=mood_counts.index,
            colors=[MOOD_COLORS.get(m, '#888') for m in mood_counts.index],
            autopct='%1.1f%%', startangle=90,
            textprops={'color': 'white', 'fontsize': 9},
        )
        for at in autotexts:
            at.set_color('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown("#### Genre × Mood Heatmap")
    pivot = df_show.groupby(['genre_utama', 'mood']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='d',
                linewidths=0.3, cbar_kws={'shrink': 0.8})
    ax.tick_params(colors='white', axis='both')
    ax.set_xlabel('Mood', color='white')
    ax.set_ylabel('Genre', color='white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: DENDROGRAM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🌿 Dendrogram":
    st.title("🌿 Hierarchical Clustering Dendrogram")
    st.markdown("Visualization of song grouping hierarchy using **Ward Linkage**.")

    p_val = st.slider("Number of leaf nodes shown", min_value=10, max_value=60, value=40, step=5)

    fig, ax = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    dendrogram(
        Z, ax=ax, truncate_mode='lastp', p=p_val,
        leaf_rotation=90, leaf_font_size=8,
        show_contracted=True,
        color_threshold=0.7 * max(Z[:, 2]),
    )
    ax.set_title('Dendrogram — Agglomerative Clustering (Ward)',
                 color='white', fontsize=14, pad=15)
    ax.set_xlabel('Songs (summarized)', color='white')
    ax.set_ylabel('Distance', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.markdown("#### Cluster Summary")
    summary_data = config['cluster_summary']
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔥 Heatmap":
    st.title("🔥 Jaccard Similarity Heatmap")
    st.markdown("Pairwise similarity between sampled songs.")

    n_show = st.slider("Sample size", min_value=20, max_value=100, value=50, step=10)

    idx_show = list(range(n_show))
    sim_show = sim_matrix[np.ix_(idx_show, idx_show)]
    labels_show = [f"{df['genre_utama'].iloc[i][:3]}{i}" for i in idx_show]

    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.heatmap(
        sim_show, ax=ax, cmap='YlOrRd',
        xticklabels=labels_show, yticklabels=labels_show,
        linewidths=0.1, linecolor='#333',
        cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.8},
    )
    ax.tick_params(colors='white', labelsize=7)
    ax.set_title(f'Jaccard Similarity Matrix ({n_show}×{n_show})',
                 color='white', fontsize=14, pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset":
    st.title("📋 Dataset Browser")

    df_show = df if genre_filter == "All Genres" else df[df['genre_utama'] == genre_filter]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Songs", f"{len(df_show):,}")
    col2.metric("Genres", df_show['genre_utama'].nunique())
    col3.metric("Moods", df_show['mood'].nunique())

    st.divider()

    mood_filter = st.selectbox("Filter by Mood", ["All Moods"] + sorted(df['mood'].unique()))
    if mood_filter != "All Moods":
        df_show = df_show[df_show['mood'] == mood_filter]

    search_title = st.text_input("🔎 Search song title", "")
    if search_title:
        df_show = df_show[df_show['judul_lagu'].str.contains(search_title, case=False, na=False)]

    st.dataframe(
        df_show[['judul_lagu','artis','genre_utama','mood','cluster','valence','energy','popularity']]
        .rename(columns={
            'judul_lagu' : 'Title', 'artis'      : 'Artist',
            'genre_utama': 'Genre', 'mood'        : 'Mood',
            'cluster'    : 'Cluster',
        }),
        use_container_width=True, height=500,
    )

    st.download_button(
        "⬇️ Download filtered data",
        df_show.to_csv(index=False),
        file_name="filtered_songs.csv",
        mime="text/csv",
    )
