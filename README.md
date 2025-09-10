# 🌱 Sowing Success: How Machine Learning Helps Farmers Select the Best Crops

This project tackles a critical challenge in modern agriculture: **helping farmers choose the optimal crop to plant based on soil conditions**. Using a dataset containing soil measurements (Nitrogen, Phosphorus, Potassium, and pH), a machine learning model is built to predict the best-suited crop for a given field. The core objective is not only to build a predictive model but to **identify the single most important soil feature** that drives crop selection, empowering farmers to make data-driven decisions even under budget constraints.

This project was completed using DataCamp’s Datalab environment.

---

## 🎯 Project Objectives

- Build a **multi-class classification model** to predict the optimal crop based on soil metrics.
- Evaluate the **individual predictive power** of each soil feature (N, P, K, pH) using Logistic Regression.
- Identify the **single most important feature** for crop prediction.
- Store the result in a dictionary:
  ```python
  best_predictive_feature = {"K": 0.3018}
  ```

---

## 🗃️ Dataset Overview

The data comes from a single CSV file provided by a farmer seeking data-driven advice:

| File                                | Description                                   |
|-------------------------------------|-----------------------------------------------|
| [`soil_measures.csv`](./soil_measures.csv)     | Soil measurements and corresponding optimal crops for various fields |

### Key Columns

| Column           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `N`              | Nitrogen content ratio in the soil                                          |
| `P`              | Phosphorous content ratio in the soil                                       |
| `K`              | Potassium content ratio in the soil                                         |
| `pH`             | pH value of the soil                                                        |
| `crop`           | **Target variable**: The optimal crop for the given soil conditions         |

---

## 🔍 Methodology: Feature Performance Evaluation

### 1. Exploratory Data Analysis (EDA)

Loaded the dataset and confirmed:
- No missing values.
- Target variable (`crop`) is categorical.
- Features are numerical, ready for scikit-learn’s LogisticRegression.

```python
crops = pd.read_csv("soil_measures.csv")
# Confirmed data integrity with .info(), .isna().sum(), .describe()
```

### 2. Isolating Feature Impact

Instead of building one complex model, the approach was to **evaluate each feature individually** to understand its standalone predictive power. This is crucial for farmers who may only afford to test one soil metric.

For each feature (`N`, `P`, `K`, `pH`):
- Split the data into training and test sets.
- Train a Logistic Regression model using **only that single feature**.
- Record the model’s accuracy score on the test set.

```python
results = {}
for feature in ['N', 'P', 'K', 'pH']:
    X_f = crops[feature].values.reshape(-1, 1)
    y = crops["crop"].values
    X_train, X_test, y_train, y_test = train_test_split(X_f, y, random_state=12)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    results[feature] = score
```

### 3. Identifying the Best Feature

The scores for each feature were visualized using a bar chart. The feature with the highest score was declared the most predictive.

```python
plt.bar(results.keys(), results.values())
plt.title("Predictive Power of Individual Soil Features")
plt.ylabel("Accuracy Score")
plt.show()
```

### 4. Final Output

The single most important feature and its score were stored in the required dictionary format.

```python
best_predictive_feature = {"K": results["K"]}
# Output: {'K': 0.3018181818181818}
```

---

## 📊 Key Findings

- ✅ **Most Predictive Feature:** **Potassium (K)**
- 📌 **Predictive Score:** **~30.18% Accuracy**
- 📈 **Conclusion:** While the overall accuracy is modest (indicating that a single feature is not sufficient for high-confidence predictions), **Potassium (K) is the most informative single metric** for determining the optimal crop. This suggests that if a farmer can only test one soil component due to budget constraints, **testing Potassium levels will provide the most valuable insight**.

> 💡 Final Output:
> ```python
> best_predictive_feature = {"K": 0.3018181818181818}
> ```

---

## 🛠️ Tools Used

- **Python**
- **pandas** – for data loading and manipulation
- **scikit-learn** – for Logistic Regression and train/test splitting
- **matplotlib** – for visualizing feature performance
- **Jupyter Notebook / DataCamp Datalab** – for analysis and reporting

---

## 📌 How to Use

This project was completed in **DataCamp’s Datalab environment**. To reproduce:

1. Upload `soil_measures.csv` to your workspace.
2. Open the notebook.
3. Run all cells to reproduce the full analysis.

> 🔗 This project was created by **DataCamp** as part of their machine learning curriculum. You can find the original exercise on their platform.

---

## ✍️ Author

Completed by **Achraf Salimi** — applying machine learning to solve real-world agricultural challenges as part of a structured learning path in data science.