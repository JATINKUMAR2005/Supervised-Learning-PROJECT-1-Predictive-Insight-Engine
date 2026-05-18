# 🏠 Predictive Insight Engine
### Supervised Learning — Real Estate House Price Prediction
**Red & White Skill Education | Duration: 6 Hours**

---

## 📌 Project Overview

This project implements and evaluates multiple **supervised learning regression models** to predict house prices based on property features such as area, location score, number of rooms, and age. It covers the full ML workflow — from data exploration and model building to gradient descent optimization and bias-variance diagnostics.

> **Role:** Junior Data Scientist at a real estate analytics firm  
> **Goal:** Design, implement, analyze, and evaluate models that accurately predict house prices and explain why certain models perform better than others.

---

## 📁 Repository Structure

```
Predictive-Insight-Engine/
│
├── Predictive_Insight_Engine.ipynb   # Main Jupyter Notebook (all parts)
├── RealEstate_HousePrice_Dataset_4200.csv  # Dataset
├── README.md                         # This file
│
└── outputs/                          # Generated plots & charts
    ├── heatmap.png
    ├── scatter_plots.png
    ├── price_distribution.png
    ├── slr_regression_line.png
    ├── slr_assumptions.png
    ├── slr_actual_vs_predicted.png
    ├── mlr_coefficients.png
    ├── poly_comparison.png
    ├── gd_convergence.png
    ├── bias_variance.png
    ├── learning_curve.png
    └── final_comparison.png
```

---

## 📊 Dataset

| Feature | Description |
|---|---|
| `area_sqft` | House area in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `location_score` | Location desirability score (1–10) |
| `age_years` | Age of the property in years |
| `distance_city_km` | Distance from city centre (km) |
| `lot_size_sqft` | Total lot size in square feet |
| `has_garage` | Binary — 1 if garage present |
| `has_pool` | Binary — 1 if pool present |
| `renovation_years_ago` | Years since last renovation |
| `house_price_inr` | **Target** — House price in INR |

- **Total Records:** 4,200  
- **Train / Test Split:** 80% / 20%

---

## 🗂️ Project Parts

### Part A — Conceptual Understanding (Theory)
- Supervised Learning Algorithms
- Regression vs Classification
- Simple Linear Regression
- Assumptions of Linear Regression
- Bias–Variance Trade-Off
- Overfitting and Underfitting

### Part B — Dataset Understanding & Preparation
- Variable identification (independent / dependent)
- Feature–target visualizations (scatter plots, correlation heatmap)
- Train-test split

### Part C — Simple Linear Regression
- SLR using `area_sqft` as predictor
- Regression line plotted with slope and intercept
- Assumption validation (Residuals vs Fitted, Q-Q Plot, Residual Histogram)

### Part D — Model Evaluation Metrics
| Metric | Formula |
|---|---|
| MSE | Mean of squared errors |
| MAE | Mean of absolute errors |
| RMSE | √MSE |
| R² | Proportion of variance explained |
| Adjusted R² | R² penalised for number of features |

### Part E — Multiple Linear Regression
- MLR using all 10 features
- Feature coefficient analysis
- Compared against SLR

### Part F — Polynomial Regression
- Degree 2 and Degree 3 polynomial regression
- Visual and numerical comparison with linear regression
- Overfitting / underfitting diagnosis

### Part G — Gradient Descent Optimization
Three variants implemented **from scratch** (no sklearn):

| Method | Update Per Iteration | Convergence |
|---|---|---|
| Batch GD | Full dataset | Smooth, slow |
| Stochastic GD | 1 random sample | Fast, noisy |
| Mini-Batch GD | 64 samples | Balanced |

- Learning Rate: `0.05`
- Iterations: `800`
- Features normalised with `StandardScaler`

### Part H — Bias–Variance & Model Diagnostics
- Bias and variance proxies computed for all models
- Train vs Test R² comparison
- Learning curve for MLR
- Best model identified based on bias-variance balance

### Part I — Final Analysis & Reporting
- Complete model comparison table
- Best model recommendation
- Business interpretation of results

---

## 🏆 Results Summary

| Model | R² Score | RMSE (₹) |
|---|---|---|
| Multiple Linear Regression | **Highest** | **Lowest** |
| Polynomial Regression (deg=3) | Mid | Mid |
| Polynomial Regression (deg=2) | Mid | Mid |
| Simple Linear Regression | Lowest | Highest |

> ✅ **Best Model: Multiple Linear Regression (all features)**  
> Reason: Leverages all available predictors, lowest bias, minimal train-test gap, best generalisation.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

### Steps
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Predictive-Insight-Engine.git
cd Predictive-Insight-Engine

# 2. Place the dataset in the root folder (same level as the notebook)

# 3. Launch Jupyter Notebook
jupyter notebook Predictive_Insight_Engine.ipynb

# 4. Run All Cells: Kernel → Restart & Run All
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| NumPy | Numerical computations, GD from scratch |
| Pandas | Data loading and manipulation |
| Matplotlib / Seaborn | Visualisations |
| Scikit-learn | ML models, metrics, pipelines |
| SciPy | Q-Q plot for assumption validation |
| Jupyter Notebook | Interactive development & reporting |

---

## 📝 Key Learnings

- MLR outperforms SLR when multiple informative features are available
- Polynomial regression on a single feature is still limited by the information bottleneck
- Mini-Batch Gradient Descent provides the best convergence speed vs stability trade-off
- Feature scaling is critical for gradient descent convergence
- Adjusted R² is a better model selection metric than plain R² when comparing models with different numbers of features

---

## 👤 Author

**Jatin**  
B.Tech — Artificial Intelligence & Machine Learning  
JG University  

---

*Predictive Insight Engine | Supervised Learning | Red & White Skill Education*
