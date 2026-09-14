# FIFA Players: Exploratory Data Analysis & Preprocessing Pipeline

An end-to-end data analysis and machine learning preprocessing pipeline built on the **FIFA Players dataset**. This project covers robust exploratory data analysis (EDA), data quality checks, statistical summaries, correlation diagnostics, and rigorous train-test splitting in Python.

---

## Project Structure
- `Assignment_2.ipynb`: The main Jupyter Notebook containing all data cleaning, EDA steps, visualisations, and preprocessing workflows.
- `Fifa.csv`: The underlying dataset containing player attributes, ratings, values, and team statistics.

---

## Exploratory Data Analysis (EDA) Highlights

1. **Data Quality & Hygiene:**
   - Checked dataset dimensions (**1,967 rows**, **9 columns** after sampling).
   - Verified zero missing values and zero duplicate rows.
   - Inspected data types (`str`, `int64`, `float64`).

2. **Univariate Analysis:**
   - **Target Variable (`Value Per M$`):** Visualized using histograms and KDE plots. Revealed a strong right-skew ($\text{skewness} \approx 7.98$), indicating extreme high-value outliers (max value of $190.5M vs mean of $2.58M).
   - **Player Ratings & Attributes:** Analyzed distributions across `Age`, `Overall_Rating`, `Future Potential`, and `Total_Stats Score`.

3. **Bivariate & Multivariate Analysis:**
   - **Correlation Matrix:** Generated a heatmap highlighting strong positive linear relationships between `Overall_Rating`, `Future Potential`, `Total_Stats Score`, and player market value (`Value Per M$`).
   - **Positional Performance:** Grouped data by player `Position` to compare average overall ratings across different tactical roles on the field.

---

## Data Preprocessing & Modeling Setup

To prepare the data for predictive modeling without data leakage:
- **Feature Separation:** Dropped non-predictive string identifiers (`Name`). Separated features ($X$) from the target variable ($y = \text{Value Per M\$}$).
- **Train/Test Split:** Implemented an **80/20 stratified split** (`train_size=1573`, `test_size=394`) *before* applying any feature scaling or encoding.

---

## Requirements & Libraries
The analysis is executed using standard Python data science libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
