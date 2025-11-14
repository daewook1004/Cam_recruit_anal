# Campus Recruitment Analysis & Visualization

This project analyzes and visualizes factors influencing campus recruitment success among students using data mining and machine learning.
Through extensive exploratory data analysis (EDA) and model interpretability using Logistic Regression and Random Forest, it identifies the most significant predictors of student placement outcomes.


---

## Technologies Used

**Python**, **Pandas**, **Matplotlib**, **Seaborn**, **Scikit-learn**, **NumPy**

---
## 📂 Project Structure
```
├── CAMRECRUIT_Anal_Visualization.ipynb   # Main notebook (EDA + Model + Visualization)
├── graphfolder/                           # Saved plots (categorical & numerical distribution, coefficients)
└── README.md                              # Project documentation
```
---
## 🚀 How to Run
### Clone the repository
```
git clone https://github.com/daewook1004/CampusRecruit_Analysis.git
cd Cam_recruit_anal
```
### Install dependencies
```
pip install numpy pandas matplotlib seaborn scikit-learn scipy kagglehub
```

### Run the notebook
```
jupyter notebook CAMRECRUIT_Anal_Visualization.ipynb
```
---
## 🧩 Problem Definition

The goal is to predict whether a student will be successfully placed in a company based on individual attributes such as academic performance, specialization, and work experience.
> **Campus Placement**:
> A hiring system where companies visit universities to recruit students before graduation, common in India and other Asian countries.

Dataset Overview:
- **Records:** 215 student entries
- **Attributes:** 15 features + 1 target (status)
- **Target Variable:** status — Placed or Not Placed
- **Note:** Salary data exists only for placed students and was excluded from training.

---
## 📊 Exploratory Data Analysis (EDA)
### Categorical Variables Distribution

Visualized the proportion of categorical features such as gender, educational board, specialization, and work experience to understand dataset composition.

<p align="left"> <img src="graphfolder/categorical_counts_20251113_163329.png" width="850"> </p>


Insights

- Commerce and Science majors dominate, while Arts is underrepresented.
- Most students have no prior work experience, though those who do tend to have higher placement rates.
- Specialization in Marketing & Finance is slightly more common than Marketing & HR.
- Majority of records are Placed, suggesting successful campus recruitment outcomes overall.

### Numerical Variables Distribution

Examined continuous attributes such as ssc_p, hsc_p, degree_p, etest_p, and mba_p.
<p align="left"> <img src="graphfolder/numerical_hist_20251113_164841.png" width="850"> </p>

Insights

- Most scores cluster between 50–80%, indicating generally average-to-good academic performance.
- Employment test scores (etest_p) show a wide range, suggesting differences in test difficulty or preparation.

### Correlation & Feature Relationships

Correlation analysis was conducted to detect associations among academic and employability factors.
<p align="left"> <img src="graphfolder/corr_matrix_20251113_165256.png" width="850"> </p>

Findings

- Strong positive correlations among academic grades: ssc_p, hsc_p, and degree_p.
- Moderate positive relationship between etest_p and placement success.
- mba_p shows a negative correlation with placement status, implying that higher MBA scores might not guarantee better employability outcomes

### Monotonicity & PDP (Partial Dependence)
To verify directional influence, monotonicity checks using Spearman correlation and Partial Dependence Plots were performed.
<p align="left"> <img src="graphfolder/pdp_20251113_173151.png" width="800"> </p>

Observations
- ssc_p, hsc_p, and degree_p display clear monotonic increases with placement probability.
- etest_p acts as a supporting variable—important but not dominant.
- Interestingly, mba_p shows an inverse pattern where high-scoring MBA students were less likely to be placed, potentially reflecting employer preference for balanced profiles over high academic achievers

### Summary of EDA
- **Academic performance** (secondary → higher → degree) remains the most consistent predictor of placement success.
- **Work experience** positively contributes to employability outcomes
- **MBA performance** introduces a unique reverse effect that warrants deeper domain-level interpretation
- Interestingly, mba_p shows an inverse pattern where high-scoring MBA students were less likely to be placed, potentially reflecting employer preference for balanced profiles over high academic achievers








