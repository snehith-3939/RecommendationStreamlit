import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("your_combined_data.csv")  # Replace with your cleaned metadata + reviews file
    return df

df = load_data()

# Prepare TF-IDF Matrix
@st.cache_resource
def prepare_tfidf(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['about_product'].fillna(''))
    return tfidf_matrix, tfidf

tfidf_matrix, tfidf = prepare_tfidf(df)
product_ids = df['product_id'].tolist()
product_id_to_index = {pid: idx for idx, pid in enumerate(product_ids)}

# Collaborative Filtering (SVD)
@st.cache_resource
def train_collab_model(df):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'product_id', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    param_grid = {'n_factors': [50], 'n_epochs': [20], 'lr_all': [0.005]}
    gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)
    best_model = gs.best_estimator['rmse']
    best_model.fit(trainset)

    user_mapping = {uid: i for i, uid in enumerate(df['user_id'].unique())}
    return best_model, user_mapping

model, user_mapping = train_collab_model(df)

# Hybrid Recommendation Function
def hybrid_recommendation(user_id, product_id, top_n=5):
    if product_id not in product_id_to_index:
        st.error(f"Product {product_id} not found.")
        return []
    if user_id not in df['user_id'].values:
        st.error(f"User {user_id} not found.")
        return []

    idx = product_id_to_index[product_id]
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    content_scores = list(enumerate(cosine_similarities))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i, score in content_scores[1:100]:
        pid = df.iloc[i]['product_id']
        try:
            pred = model.predict(user_id, pid).est
            final_score = score + pred
            recommendations.append((pid, final_score))
        except:
            continue

    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
    return recommendations

# Streamlit Interface
st.title("📦 Hybrid Product Recommendation System")

user_id = st.text_input("Enter your User ID:")
product_id = st.text_input("Enter a Product ID you liked:")

if st.button("Get Recommendations"):
    if user_id and product_id:
        recs = hybrid_recommendation(user_id, product_id, top_n=5)
        if recs:
            st.success("Here are your recommendations:")
            for pid, score in recs:
                name = df[df['product_id'] == pid]['product_name'].values[0]
                st.markdown(f"**{name}** (Product ID: `{pid}`) - Hybrid Score: `{score:.2f}`")
        else:
            st.warning("No recommendations found.")
    else:
        st.warning("Please enter both User ID and Product ID.")
