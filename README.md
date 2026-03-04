


#  Product Recommendation System

This project implements a ** Recommendation System** for an e-commerce platform similar to Amazon or Myntra.  
The system recommends products to users based on **product similarity and user behavior**.

---

##  Features

- Product search and recommendations
- Hybrid recommendation algorithm
- Trending products display
- Category filtering
- Add to Cart functionality
- Cart total calculation
- Web interface using Streamlit

---

##  Recommendation Techniques

### 1️ Content-Based Filtering
Uses **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert product names into vectors and compute similarity between products using **Cosine Similarity**.

### 2️ Collaborative Filtering
Creates a **User-Item Matrix** and recommends products based on similar user preferences.

### 3️ Hybrid Model
The final recommendation score combines both approaches.

```
Hybrid Score = (0.7 × Content Score) + (0.3 × Collaborative Score)
```

---

##  Project Structure

```
Ai-e-commerce-
│
├── amazon.csv
├── hybrid_app.py
├── hybrid.py
├── collaborative.py
├── recommendation.py
├── model_training.py
├── similarity.pkl
├── matrix.pkl
└── templates/
    ├── index.html
    ├── cart.html
    ├── login.html
    └── product.html
```

---

##  Evaluation Metrics

The model performance is evaluated using:

- Precision
- Recall
- F1 Score

Example results:

```
Precision@5 : 82%
Recall@5    : 79%
F1 Score    : 80%
```

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- HTML / CSS
- Git & GitHub

---

##  How to Run the Project

### Clone the repository

```
git clone https://github.com/Devipriya8/Ai-e-commerce-.git
```

### Install dependencies

```
pip install pandas numpy scikit-learn streamlit
```

### Run the application

```
streamlit run hybrid_app.py
```

The web application will open in your browser.

---

##  Future Improvements

- Real user behavior dataset
- Deep learning recommendation models
- Payment gateway integration
- Cloud deployment
- User login system

---

##  Author

**Devi Priya Yannam**

Hybrid Product Recommendation System
