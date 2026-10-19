def analyze_multiple_stats(data, columns=None):
    """
    Calculates key descriptive statistics and high-end percentiles 
    for specified dataframe columns.
    """
    if columns is None:
        columns = data.columns
        
    # Filter data for selected columns
    sub_data = data[columns]
    
    # Calculate base descriptive statistics across all columns at once
    summary = sub_data.describe().T
    
    # Calculate custom high-end percentiles efficiently
    summary['95th Percentile'] = sub_data.quantile(0.95)
    summary['99th Percentile'] = sub_data.quantile(0.99)
    summary['999th Percentile'] = sub_data.quantile(0.999)
    
    # Rename and reorder columns to match your exact output layout
    summary = summary.reset_index().rename(columns={
        'index': 'Feature', 'min': 'Min', 'max': 'Max', 
        'mean': 'Mean', '50%': 'Median', '25%': '25th Percentile', 
        '75%': '75th Percentile'
    })
    
    # Keep only the columns you requested in the exact order
    output_cols = ['Feature', 'Min', 'Max', 'Mean', 'Median', 
                   '25th Percentile', '75th Percentile', 
                   '95th Percentile', '99th Percentile', '999th Percentile']
    
    return summary[output_cols]


import numpy as np

def winsorize_df(df, lower=0.01, upper=0.99):
    """
    Clips extreme outliers in numerical columns to specified lower and upper percentiles.
    """
    df_wins = df.copy()
    
    # Select only numerical features to prevent errors on string/datetime data
    numerical_cols = df_wins.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        low_val = df_wins[col].quantile(lower)
        high_val = df_wins[col].quantile(upper)
        df_wins[col] = np.clip(df_wins[col], low_val, high_val)
        
    return df_wins


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

def quick_rf_screening(df, target, test_size=0.3):
    """
    Fits a baseline Random Forest Regressor to screen feature performance 
    and evaluate baseline training and testing metrics.
    """
    # Separate features and target while ensuring dataframes are passed cleanly
    y = df[target]
    X = df.drop(columns=[target])

    # Partition dataset into train and test groups
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Initialize and fit the random forest baseline model
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)

    # Generate predictions across both subsets
    y_pred_tr = rf.predict(X_train)
    y_pred_te = rf.predict(X_test)

    # Compute evaluation metrics (using scikit-learn's modern RMSE calculator)
    rmse_tr = root_mean_squared_error(y_train, y_pred_tr)
    rmse_te = root_mean_squared_error(y_test, y_pred_te)
    r2_tr = rf.score(X_train, y_train)
    r2_te = rf.score(X_test, y_test)

    # Output formatted performance strings
    print(f"Train RMSE: {rmse_tr:.2f}, R²: {r2_tr:.2f}")
    print(f"Test RMSE: {rmse_te:.2f}, R²: {r2_te:.2f}")

    return rmse_tr, rmse_te, r2_tr, r2_te

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

def final_model(df, target, test_size=0.3, rf_params=None):
    """
    Trains the final Random Forest Regressor using optimized hyperparameters 
    and outputs evaluation performance across train and test sets.
    """
    # Separate independent features and dependent target
    y = df[target]
    X = df.drop(columns=[target])

    # Partition dataset into train and test sets using fixed random state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Initialize parameters to an empty dict if none are explicitly provided
    if rf_params is None:
        rf_params = {}

    # Build and fit the optimized random forest regressor
    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, y_train)

    # Evaluate predictions across subsets
    y_pred_tr = rf.predict(X_train)
    y_pred_te = rf.predict(X_test)

    # Calculate performance metrics
    rmse_tr = root_mean_squared_error(y_train, y_pred_tr)
    rmse_te = root_mean_squared_error(y_test, y_pred_te)
    r2_tr = rf.score(X_train, y_train)
    r2_te = rf.score(X_test, y_test)

    # Output formatted performance strings
    print(f"Train RMSE: {rmse_tr:.2f}, R²: {r2_tr:.2f}")
    print(f"Test RMSE: {rmse_te:.2f}, R²: {r2_te:.2f}")

    return rmse_tr, rmse_te, r2_tr, r2_te


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

def plot_feature_importances(X, y, feature_names, best_params, max_features=None, save_path=None):
    """
    Trains a Random Forest Regressor to extract, sort, and plot 
    the most significant predictive features.
    """
    # Train the model locally with optimal hyperparameters
    model = RandomForestRegressor(**best_params)
    model.fit(X, y)

    # Extract and sort features by their predictive split weights
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_df = feature_df.sort_values(by='importance', ascending=False)

    # Apply maximum feature window if requested
    if max_features:
        feature_df = feature_df.head(max_features)

    # Initialize a clean horizontal bar plot layout
    plt.figure(figsize=(10, 6))
    plt.barh(feature_df['feature'][::-1], feature_df['importance'][::-1], color='teal')
    plt.xlabel('Importance')
    plt.title('Feature Importances')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Export figure file to disk if an output path is provided
    if save_path:
        plt.savefig(save_path, format="jpeg", dpi=300)

    plt.show()
    plt.close()  # Free memory allocation from matplotlib backend

    return feature_df

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def get_feature_importance_table(X, y, feature_names, best_params, top_n=None):
    """
    Fits a Random Forest Regressor using optimal hyperparameters and returns 
    a sorted pandas DataFrame of the feature importances.
    """
    # Initialize and train the random forest regressor
    model = RandomForestRegressor(**best_params)
    model.fit(X, y)

    # Extract importances and pair them with their feature names
    importances = model.feature_importances_
    feature_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    
    # Sort features from highest impact to lowest impact
    feature_df = feature_df.sort_values(by='importance', ascending=False)

    # Restrict output rows if a limit is specified
    if top_n:
        feature_df = feature_df.head(top_n)

    return feature_df


import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def analyze_group_difference(df, df_name, target_col):
    """
    Computes key statistical comparisons (Mean Ratio, Cohen's d, KS-Test) 
    between high and low-value feature groups.
    """
    # Isolate group target vectors safely
    high_values = df[df['grouped'] == 'High'][target_col]
    low_values = df[df['grouped'] == 'Low'][target_col]

    # Calculate Mean Revenue Ratio safely to protect against zero-division
    mean_low = low_values.mean()
    mean_ratio = high_values.mean() / mean_low if mean_low != 0 else 0

    # Compute Cohen's d Effect Size
    # Use standard deviation with ddof=1 (sample std) to align with standard pandas rules
    pooled_std = np.sqrt((high_values.std() ** 2 + low_values.std() ** 2) / 2)
    cohen_d = (high_values.mean() - low_values.mean()) / pooled_std if pooled_std != 0 else 0

    # Perform Kolmogorov-Smirnov statistical comparison
    ks_stat, ks_p_value = ks_2samp(high_values, low_values)

    # Output clean, single-row evaluation metrics summary
    results_df = pd.DataFrame({
        'Feature': [df_name],
        'Target Column': [target_col],
        'Mean Revenue Ratio': [round(mean_ratio, 2)],
        'Cohen\'s d Effect Size': [round(cohen_d, 2)],
        'KS Statistic': [round(ks_stat, 2)],
        'KS p-value': [round(ks_p_value, 4)]
    })

    return results_df

def analyze_multiple_by_quantile(data, columns, target_col, quantile=0.95):
    """
    Iterates over multiple behavior metrics, sections populations into high/low usage 
    cohorts at a custom threshold, and returns combined cross-distribution comparisons.
    """
    results = []

    for col in columns:
        if col == target_col:
            continue
            
        # Isolate target and current feature efficiently without heavy dataframe replication
        sub_df = data[[col, target_col]].copy()
        
        cutoff = sub_df[col].quantile(quantile)
        sub_df['grouped'] = np.where(sub_df[col] > cutoff, 'High', 'Low')

        # Run distribution diagnostics
        result_row = analyze_group_difference(sub_df, col, target_col=target_col)
        results.append(result_row)

    # Stack metrics into one master frame
    return pd.concat(results, ignore_index=True)


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

def run_random_forest_analysis(model_df, target_col, best_params, top_n=10):
    """
    Trains a final RandomForestRegressor, evaluates training/full-fit performance, 
    and plots the top feature importances.
    """
    # Separate features and target cleanly without data structure warnings
    features = model_df.drop(columns=[target_col])
    target = model_df[target_col]

    # Initialize and fit the final model using your optimal parameters
    rf_model = RandomForestRegressor(**best_params)
    rf_model.fit(features, target.values.ravel())

    # Generate full-fit predictions
    y_pred = rf_model.predict(features)

    # Calculate model fit diagnostics
    rmse = root_mean_squared_error(target, y_pred)
    r2 = rf_model.score(features, target)

    # Corrected terminology: changed 'Test set' to 'Model fit' to keep logic precise
    print(f'Model fit RMSE of Random Forest Regressor: {rmse:.2f}')
    print(f'Model fit R² score of Random Forest Regressor: {r2:.4f}')

    # Extract, index, and sort feature importance metrics
    importances = pd.Series(rf_model.feature_importances_, index=features.columns)
    top_importances = importances.sort_values(ascending=False).head(top_n)

    # Build a clean horizontal bar plot layout
    plt.figure(figsize=(10, 6))
    top_importances.plot(kind='barh', color='lightgreen')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} Feature Importances (Random Forest)')
    plt.gca().invert_yaxis()  # Keeps the highest importance item at the very top
    plt.tight_layout()
    plt.show()
    plt.close()

    return rf_model, importances, rmse, r2, features, target


import os
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def plot_partial_dependence_grid(model, features, importances, run_id, top_n=10, zoom_fraction=0.3, save_path=None):
    """
    Generates side-by-side Partial Dependence Plots showing full and 
    zoomed views for the top N most important features.
    """
    # Configure export directory structure
    export_dir = os.path.join("exports", run_id)
    os.makedirs(export_dir, exist_ok=True)

    # Isolate the targeted top-tier feature list
    top_features = importances.sort_values(ascending=False).head(top_n).index.tolist()

    for feature in top_features:
        # Create a single side-by-side vector layout directly
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Render the complete global behavior distribution
        PartialDependenceDisplay.from_estimator(
            model, features, [feature], ax=axes[0]
        )
        axes[0].set_ylabel("Predicted target")
        axes[0].set_title(f"Global View: {feature}")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Render the baseline graph onto the second axis, then adjust bounds dynamically
        PartialDependenceDisplay.from_estimator(
            model, features, [feature], ax=axes[1]
        )
        
        # Calculate dynamic clipping bounds using axis coordinates
        xmin, xmax = axes[1].get_xlim()
        zoom_max = xmin + zoom_fraction * (xmax - xmin)
        axes[1].set_xlim(xmin, zoom_max)
        
        axes[1].set_ylabel("Predicted target")
        axes[1].set_title(f"Zoomed View ({int(zoom_fraction*100)}% Lower Range)")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        fig.suptitle(f"Partial Dependence Profiles — Feature: {feature}", fontsize=14, y=1.02)
        plt.tight_layout()

        # Save clean vector-rendered figure if output is configured
        # current_save_path = os.path.join(export_dir, f"partial_dependence_{feature}.jpeg")
        # fig.savefig(current_save_path, format="jpeg", dpi=300, bbox_inches='tight')

        plt.show()
        plt.close(fig)  # Safely drop figure memory allocations


import seaborn as sns
# import matplotlib.pyplot as plt

def apply_theme():
    sns.set(font_scale=1.5)
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import ks_2samp

def meta_tuning(X_full, y_full, model_class, param_grid, n_iterations=10, sample_size=1000):
    from sklearn.ensemble import RandomForestRegressor
    results = []

    for i in range(n_iterations):
        sample_idx = np.random.choice(len(X_full), size=sample_size, replace=False)
        X_sample = X_full.iloc[sample_idx]
        y_sample = y_full.iloc[sample_idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_sample, y_sample, test_size=0.2, random_state=i
        )

        model = model_class()
        search = RandomizedSearchCV(
            model, param_grid, n_iter=20, cv=5, scoring='r2', random_state=42
        ) #start 10, 3
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        train_r2 = r2_score(y_train, best_model.predict(X_train))
        test_r2 = r2_score(y_test, best_model.predict(X_test))

        results.append({
            'iteration': i,
            'train_r2': train_r2,
            'test_r2': test_r2,
            **search.best_params_
        })

    df = pd.DataFrame(results)

    return df



