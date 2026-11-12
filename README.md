# 🚗 Car Price Prediction Model

A machine learning project built to predict vehicle prices (MSRP) based on technical specifications, fuel efficiency, and brand attributes. This project covers end-to-end data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation using **Linear Regression** and **Decision Tree Regressor**[cite: 1].

---

## 📊 Dataset Overview

The dataset is sourced from the [ML Bookcamp Repository](https://github.com/alexeygrigorev/mlbookcamp-code) (`chapter-02-car-price/data.csv`)[cite: 1].
* **Total Records:** 11,914 vehicles[cite: 1]
* **Original Features:** 16 columns (e.g., `Make`, `Model`, `Year`, `Engine HP`, `Engine Cylinders`, `Highway MPG`, `City MPG`, `Popularity`, `MSRP`)[cite: 1]
* **Target Variable:** `price` (originally `MSRP`, transformed using $\log(1 + \text{price})$ to normalize right-skewness and reduce bias)[cite: 1]

---

## 🛠️ Project Workflow

### 1. Data Exploration & Standardization
* Inspected dataset dimensions, data types, and summary statistics[cite: 1].
* Standardized all column names to lowercase with underscores (`snake_case`) and renamed `msrp` to `price`[cite: 1].

### 2. Exploratory Data Analysis (EDA)
* Visualized the distribution of vehicle prices using `Seaborn` histograms[cite: 1].
* Applied a logarithmic transformation (`np.log1p`) to handle extreme price outliers and achieve a more normal distribution[cite: 1].

### 3. Data Cleaning & Preprocessing
* **Dropped Columns:** Removed `market_category` (high missing values: 3,742 nulls), `engine_fuel_type`, `transmission_type`, `driven_wheels`, and `vehicle_style`[cite: 1].
* **Imputation:** Filled missing numerical values in `engine_hp` (69 nulls), `engine_cylinders` (30 nulls), and `number_of_doors` (6 nulls) using their respective column means[cite: 1].

### 4. Feature Engineering
* Applied **One-Hot Encoding** (`pd.get_dummies`) to categorical variables (`make`, `model`, and `vehicle_size`), expanding the feature space to **973 features**[cite: 1].
* Split the dataset into **80% training** and **20% testing** sets (`random_state=42`)[cite: 1].

### 5. Model Training & Evaluation
Two regression models were trained and evaluated using **Root Mean Squared Error (RMSE)** and **$R^2$ Score**:

| Model | Train RMSE | Train $R^2$ Score | Test RMSE | Test $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | `0.1740`[cite: 1] | `0.9750`[cite: 1] | — | — |
| **Decision Tree Regressor** | `0.0615`[cite: 1] | `0.9969`[cite: 1] | `0.1110`[cite: 1] | `0.9903`[cite: 1] |

> **Result:** The **Decision Tree Regressor** demonstrated superior performance, achieving an **$R^2$ score of 99.03%** and a low **RMSE of 0.1110** on unseen test data[cite: 1].

### 6. Model Serialization
* Exported the trained Decision Tree model using `joblib` as `Ammar2_dt_model.pkl` for future inference and deployment[cite: 1].

---

## 💻 Technologies & Libraries Used

* **Python 3**[cite: 1]
* **Data Manipulation:** `Pandas`, `NumPy`[cite: 1]
* **Data Visualization:** `Matplotlib`, `Seaborn`[cite: 1]
* **Machine Learning:** `Scikit-Learn`[cite: 1]
* **Model Persistence:** `Joblib`[cite: 1]

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
