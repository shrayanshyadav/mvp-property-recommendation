
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load data + artifacts ----------
@st.cache_resource
def load_artifacts():
    with open("property_recommender_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

@st.cache_data
def load_data():
    # For local testing replace with your path or use an uploaded file section.
    return pd.read_csv("enhanced_property_data_with_rich_descriptions.csv")

art = load_artifacts()
df = load_data()

ID_COL = art["id_col"]
TEXT_COL = art["text_col"]
NUM_COLS = art["num_cols"]
CAT_COLS = art["cat_cols"]
scaler = art["scaler"]
tfidf = art["tfidf"]

# Precompute vectors
X_num = scaler.transform(df[NUM_COLS])
X_text = tfidf.transform(df[TEXT_COL].fillna(""))

st.set_page_config(page_title="PropertyAI – Hybrid Recommender", page_icon="🏠", layout="wide")

st.title("🏠 PropertyAI — AI-Powered Property Recommendations")
st.caption("Hybrid model: semantic search (description) + ML similarity on structured data")

with st.expander("Upload your CSV (optional) and reload", expanded=False):
    up = st.file_uploader("Upload a CSV with the same schema", type=["csv"])
    if up is not None:
        new_df = pd.read_csv(up)
        st.session_state["df"] = new_df
        st.success("Uploaded! Click 'Use uploaded data' to proceed.")
        if st.button("Use uploaded data"):
            df = st.session_state["df"]
            X_num = scaler.transform(df[NUM_COLS])
            X_text = tfidf.transform(df[TEXT_COL].fillna(""))
    st.write("Using rows:", len(df))

# ---------- Sidebar / Inputs ----------
st.subheader("🎯 Find Your Perfect Property")

c1, c2 = st.columns(2)
with c1:
    budget = st.slider("Budget ($)", min_value=int(df["price"].min()), max_value=int(df["price"].max()), value=int(np.percentile(df["price"], 50)), step=1000)
    min_bedrooms = st.selectbox("Minimum Bedrooms", sorted(df["bedrooms"].unique()))
    max_commute = st.slider("Max Commute Time (minutes)", min_value=int(df["commute_time_min"].min()), max_value=int(df["commute_time_min"].max()), value=int(np.percentile(df["commute_time_min"], 50)))
    min_school = st.slider("Minimum School Rating", min_value=int(df["school_rating"].min()), max_value=int(df["school_rating"].max()), value=int(np.percentile(df["school_rating"], 50)))
with c2:
    city = st.selectbox("Preferred City (optional)", ["Any"] + sorted(df["city"].unique().tolist()))
    min_bath = st.selectbox("Minimum Bathrooms", sorted(df["bathrooms"].unique()))
    min_size = st.slider("Minimum Size (sqft)", min_value=int(df["size_sqft"].min()), max_value=int(df["size_sqft"].max()), value=int(np.percentile(df["size_sqft"], 25)))
    pool_pref = st.selectbox("Pool Preference", ["No Preference", "Needs Pool", "No Pool"])

st.text_area("Special Priorities / Description (semantic search):", key="desc", placeholder="e.g., modern kitchen, walkable neighborhood, quiet cul-de-sac, near good schools...")

alpha = st.slider("Weight: Structured vs Text (α)", 0.0, 1.0, 0.6, 0.05, help="Final score = α * structured_score + (1-α) * text_score")

# ---------- Scoring helpers ----------
def structured_score(df: pd.DataFrame) -> np.ndarray:
    # Rule-based desirability vector then cosine against properties
    # Build a synthetic "ideal" vector in the same scaled space as NUM_COLS
    # Signals:
    ideal = {
        "price": budget,  # closer is better
        "bedrooms": max(min_bedrooms, df["bedrooms"].min()),  # at least
        "size_sqft": max(min_size, df["size_sqft"].min()),
        "school_rating": max(min_school, df["school_rating"].min()),
        "commute_time_min": df["commute_time_min"].min(),  # smaller is better
        "bathrooms": max(min_bath, df["bathrooms"].min()),
        "year_built": df["year_built"].quantile(0.75),  # newer preferred
        "garage_spaces": df["garage_spaces"].quantile(0.5),  # moderate
        "lot_size_sqft": df["lot_size_sqft"].quantile(0.5),
    }
    v = np.array([ideal[c] for c in NUM_COLS]).reshape(1, -1)
    v_scaled = scaler.transform(v)
    # Cosine similarity in scaled numeric feature space
    sim = cosine_similarity(v_scaled, X_num)[0]

    # Apply hard constraints/filters
    mask = np.ones(len(df), dtype=bool)
    mask &= df["bedrooms"] >= min_bedrooms
    mask &= df["bathrooms"] >= min_bath
    mask &= df["size_sqft"] >= min_size
    mask &= df["school_rating"] >= min_school
    mask &= df["commute_time_min"] <= max_commute
    if city != "Any":
        mask &= df["city"] == city
    if pool_pref == "Needs Pool":
        mask &= df["has_pool"] == 1
    elif pool_pref == "No Pool":
        mask &= df["has_pool"] == 0

    # Penalize those outside budget by distance; reward inside
    price = df["price"].values.astype(float)
    price_penalty = 1 - np.minimum(np.abs(price - budget) / (price.max() - price.min() + 1e-9), 1.0)
    score = 0.7 * sim + 0.3 * price_penalty
    score[~mask] *= 0.2  # strong down-weight if violates filters
    return score

def text_score(query: str) -> np.ndarray:
    if not query or not query.strip():
        return np.zeros(len(df))
    q_vec = tfidf.transform([query])
    sim = cosine_similarity(q_vec, X_text)[0]
    return sim

if st.button("🔎 Get Recommendations", use_container_width=True):
    s_struct = structured_score(df)
    s_text = text_score(st.session_state.get("desc", ""))

    # Normalize both to 0-1
    def norm(x):
        x = np.nan_to_num(x, nan=0.0)
        mi, ma = x.min(), x.max()
        return (x - mi) / (ma - mi + 1e-9) if ma > mi else np.zeros_like(x)

    s_struct_n = norm(s_struct)
    s_text_n = norm(s_text)

    final = alpha * s_struct_n + (1 - alpha) * s_text_n
    top_idx = np.argsort(-final)[:3]
    results = df.iloc[top_idx].copy()
    results["Structured Score"] = s_struct_n[top_idx]
    results["Text Score"] = s_text_n[top_idx]
    results["Final Score"] = final[top_idx]
    results = results[[ID_COL, "city", "price", "bedrooms", "bathrooms", "size_sqft", "school_rating", "commute_time_min", "year_built", "has_pool", "garage_spaces", "lot_size_sqft", "description", "Structured Score", "Text Score", "Final Score"]]

    st.success(f"Found {len(results)} top matches for you.")
    for i, row in results.reset_index(drop=True).iterrows():
        with st.container(border=True):
            st.markdown(f"### {i+1}. {row[ID_COL]}  —  **${int(row['price']):,}**")
            st.caption(f"{row['city']} • {row['bedrooms']} bed • {row['bathrooms']} bath • {row['size_sqft']:,} sqft • School {row['school_rating']} • Commute {row['commute_time_min']} min")
            st.progress(float(row["Final Score"]))
            st.markdown("**AI Analysis**")
            st.write(row["description"])

    st.dataframe(results.reset_index(drop=True))
else:
    st.info("Set your preferences and click **Get Recommendations**.")

st.divider()
st.caption("Prototype — hybrid recommender (TF-IDF + numeric similarity). Place this file alongside your dataset and artifacts when deploying to Streamlit Cloud.")
