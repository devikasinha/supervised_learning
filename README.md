# Supervised Machine Learning Techniques

Welcome to the Supervised Machine Learning repository! This repository serves as a comprehensive guide and reference for selecting, understanding, and implementing core supervised learning algorithms based on dataset characteristics. It includes conceptual workflows, key evaluation metrics, and ready-to-run implementation notebooks.

---

## 🚀 Algorithm Selection Workflow

Choosing the right machine learning model depends directly on your dataset characteristics. Refer to the decision logic below to guide your selection:
[ Start: Your Dataset ]
                         │
           What is your target variable?
                         │
          ├──────────────────────────────┤
     (Categorical)       (Continuous)
          │                              │
     Dataset Size?        Relationship Type?
     ├───────────┤               ├────────────────┤
  (Small)     (Large)         (Linear)        (Non-Linear)
                
     │           │               │                 │
     ### 📊 Key Factors Influencing Model Choice

| Factor | Strategy & Selection Guidelines |
| :--- | :--- |
| **Dataset Size** | **Small:** KNN, Decision Tree (DT), SVM<br>**Large:** Random Forest (RF), XGBoost, Logistic Regression (LR) |
| **Feature Count** | **High-Dimensional:** SVM<br>**Low-Dimensional:** KNN, Decision Tree (DT) |
| **Training Speed** | **Fast:** Linear / Logistic Regression<br>**Slow OK:** Random Forest (RF), XGBoost |
| **Interpretability** | **Needed:** Decision Tree (DT), Linear Regression<br>**Not Needed:** Random Forest (RF), XGBoost |
| **Outliers** | **Sensitive:** SVM, Linear Regression<br>**Robust:** Random Forest (RF), XGBoost |
| **Missing Data** | Handle natively using **XGBoost** or **Random Forest (RF)** |
| **Overfitting Risk** | **High Risk:** Decision Tree (DT), KNN<br>**Low Risk:** Regularized Regression, SVM |
| **Feature Scaling** | **Required:** SVM, KNN<br>**Not Needed:** Random Forest (RF), Decision Tree (DT) |

---

## 🧠 Algorithms & Notebooks

### 1. Regression Techniques (Continuous Target)
* **Simple Linear Regression:** Models a linear relationship between a dependent target variable and one independent feature variable using the line formula parameters $m=\frac{\sum_{i=1}^{n}(y_{i}-\overline{y})(x_{i}-\overline{x})}{\sum_{i=1}^{n}(x_{i}-\overline{x})^{2}}$ and $c=\overline{y}-m\overline{x}$.
    * 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1VWBJuWtSS41CT2gvUkQYIEY7ewmpRg_Y?usp=sharing)
* **Multiple Linear Regression:** Maps a linear relationship between a dependent target variable and multiple independent feature variables via the formula: 
    $$y = \alpha + \beta_1(x_1) + \beta_2(x_2) + \dots + \beta_n(x_n)$$
    * 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1q-KszS1k0l5N29yHgDTv3BKL3FOE0y5x?usp=sharing)
* **Gradient Descent:** An iterative optimization algorithm used to minimize a cost function (error) by updating model parameters—like weights and biases—in the direction of the steepest decrease defined by the negative gradient.
    * 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1S5u1HpTl5j7-LnUhW4E2Q6XEIMhmkkdn?usp=sharing)

### 2. Classification Techniques (Categorical Target)
* **Logistic Regression:** Used for classification problems, especially binary outcomes (0 or 1). It utilizes a sigmoid function to convert output into probability values between 0 and 1 to predict class likelihood.
    * 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1QgULyTiXnBVaCg30GBXI0T2WUCSlu5-R?usp=sharing)
* **Support Vector Machine (SVM):** Finds the optimal hyperplane that separates data into different classes with a maximum margin. Support vectors represent the closest data points to the boundary.
    * 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1O0ni-d84TunMiLZGTJkGH24V0CjEh9mQ?usp=sharing)
* **K-Nearest Neighbors (KNN):** Classifies a data point based on the majority group class of its nearest neighbor points ($K$) using Euclidean distance.
    * 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1bJvyqv5IAZPY8vGmwkpo2cLRgvW9iK3m?usp=sharing)
* **Decision Tree:** Splitting data into branches based on condition paths (like yes/no questions) to build hierarchical structures of nodes and outcome leaves.
    * 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1ePdP81r5Xv6S4JZendLiHIOXwCJGSTf?usp=sharing)
* **Naive Bayes Classification:** A probabilistic algorithm that leverages Bayes' Theorem under the assumption that all dataset features are independent.
    * 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1lciV94uW74u5See3fN9wP9Nd5Ki9tgJq?usp=sharing)
* **Random Forest:** An ensemble learning method that aggregates predictions from multiple constructed decision trees, relying on majority voting or averaging for accurate outputs.
    * 🔗 [Google Colab Notebook](https://colab.research.google.com/drive/1201729_E_INTSDMQfAI0ytfIK-qx004w?usp=sharing)

---

## 📈 Performance Evaluation

### Confusion Matrix Breakdown
A confusion matrix evaluates the classification model performance by directly comparing ground truth actual classes against predicted classes:

| | Predicted Positive | Predicted Negative | Validation Summary Metrics |
|---|---|---|---|
| **Actual Positive** | **True Positive (TP)** | **False Negative (FN)** <br>*(Type II Error)* | **Sensitivity / Recall:** <br> $\frac{TP}{TP + FN}$ |
| **Actual Negative** | **False Positive (FP)** <br>*(Type I Error)* | **True Negative (TN)** | **Specificity:** <br> $\frac{TN}{TN + FP}$ |
| **Validation Summary** | **Precision:** <br> $\frac{TP}{TP + FP}$ | **Negative Predictive Value (NPV):** <br> $\frac{TN}{TN + FN}$ | **Accuracy:** <br> $\frac{TP + TN}{TP + TN + FP + FN}$ |

---

## 🏆 Model Comparison Benchmark

To review how all of these models perform when executed against an identical dataset, check out our baseline accuracy benchmark:
