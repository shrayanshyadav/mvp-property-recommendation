# 🏠 PropertyAI – MVP Property Recommendation App

An **AI-powered property recommender** built with **Streamlit**, combining **traditional ML + NLP**.

## ✨ Features
- Hybrid recommendation:
  - **Structured ML scoring** (price, size, bedrooms, bathrooms, commute, school rating, etc.)
  - **Semantic NLP matching** using property descriptions
- Returns **Top 3 best-matching properties** ranked by score
- Interactive single-page Streamlit UI
- Supports **CSV upload** (with same schema) for testing on new datasets

## 📊 Dataset Schema
The app expects a dataset with the following columns:

- `address` (unique property identifier)  
- `city`  
- `price`  
- `bedrooms`  
- `size_sqft`  
- `school_rating`  
- `commute_time_min`  
- `bathrooms`  
- `year_built`  
- `has_pool` (0/1)  
- `garage_spaces`  
- `lot_size_sqft`  
- `description` (rich text description of property)  

## 🚀 Run Locally
Clone this repo and run:

```bash
pip install -r requirements.txt
streamlit run app.py
