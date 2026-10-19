# User Behavior & Monetization Regression Pipeline

This pipeline leverages a Random Forest architecture and advanced interpretability tools (SHAP, PDP) to evaluate user behavioral telemetry and its impact on revenue. It implements custom winsorization for outlier handling, a unique stable-mode heuristic for hyperparameter selection, and rigorous non-parametric statistical diagnostics (Cohen's d, KS-Test) to map behavioral cohorts directly to monetization scale.

## 🚀 Key Framework Features

* **Outlier Mitigation Engine:** Implements custom automated winsorization tailored for heavy-tailed telemetry distributions (e.g., in-game power actions or whale monetization spikes) to stabilize regression training.
* **Stable-Heuristic Hyperparameter Tuning:** Explores the search space via an iterative meta-tuning loop, filters for high-generalization models (positive test R² with a <0.2 train-test gap), and computes the statistical mode across stable runs to lock in robust parameters rather than chasing single-split overfitted maximums.
* **Distributional Diagnostics:** Applies Cohen's d effect sizes to calculate variance magnitude, alongside non-parametric Kolmogorov-Smirnov (KS) testing to validate population shifts and statistical significance across high vs. low usage cohorts.
* **Explainable AI Foundations:** Cross-validates features using global Random Forest importances, vector-optimized side-by-side Partial Dependence Plots (PDPs) mapping broad and zoomed trends, and cooperative game-theory SHAP attributions (Beeswarm, Absolute Scales, and Interaction Dependencies).

## 📊 Dataset Information
* **`synthetic_users.csv`**: A synthetic player/user dataset generated specifically for this demonstration pipeline to ensure data privacy while maintaining realistic telemetry variance and heavy-tailed monetization distributions. 

## 🛠️ Stack & Dependencies
* **Core Processing:** `pandas`, `numpy`
* **Modeling & Metrics:** `scikit-learn` (`RandomForestRegressor`, `root_mean_squared_error`, `RandomizedSearchCV`)
* **Statistical Engines:** `scipy` (`ks_2samp`)
* **Visualization & XAI:** `matplotlib`, `seaborn`, `plotly`, `shap`

## 📂 Project Structure
* `user_behavior_analysis.ipynb`: The main workflow notebook containing data loading, preparation, model screening, tuning, and evaluation cells with detailed analytical commentary.
* `regression_model_setup.py`: Core production module containing clean, performance-optimized implementations of all helper engines.
* `synthetic_users.csv`: Baseline user telemetry file.
