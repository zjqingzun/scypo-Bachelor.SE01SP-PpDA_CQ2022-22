import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import zscore

# Set random seed for reproducibility
np.random.seed(42)

# Data Loading
def load_data(train_path='data/train.csv', test_path='data/test.csv'):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    print("Train data info:")
    print(train_data.info())
    print("\nFirst 5 rows of train data:")
    print(train_data.head())
    return train_data, test_data

# Data Preprocessing
def preprocess_data(train_data, test_data=None):
    def extract_value_and_rpm(val):
        try:
            parts = re.findall(r"(\d+\.?\d*)", str(val))
            return float(parts[0]), float(parts[1]) if len(parts) >= 2 else np.nan
        except:
            return np.nan, np.nan

    for data in [train_data, test_data] if test_data is not None else [train_data]:
        data[['Max_Power_Value', 'Max_Power_RPM']] = data['Max Power'].apply(
            lambda x: pd.Series(extract_value_and_rpm(x))
        )
        data[['Max_Torque_Value', 'Max_Torque_RPM']] = data['Max Torque'].apply(
            lambda x: pd.Series(extract_value_and_rpm(x))
        )
        data['Engine'] = data['Engine'].str.replace('cc', '').astype(float)

    numeric_cols = [
        'Year', 'Kilometer', 'Engine', 'Length', 'Width', 'Height',
        'Seating Capacity', 'Fuel Tank Capacity', 'Max_Power_Value',
        'Max_Power_RPM', 'Max_Torque_Value', 'Max_Torque_RPM'
    ]
    categorical_cols = ['Make', 'Location', 'Fuel Type', 'Transmission', 'Seller Type', 'Owner', 'Drivetrain']

    for col in numeric_cols:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        if test_data is not None:
            test_data[col] = test_data[col].fillna(train_data[col].mean())

    mask = train_data.drop(columns=['Price']).notna().all(axis=1) & train_data['Price'].notna()
    train_data = train_data[mask].reset_index(drop=True)
    
    z_scores = zscore(train_data['Price'])
    mask_outlier = np.abs(z_scores) < 3
    train_data = train_data[mask_outlier].reset_index(drop=True)

    plt.hist(train_data['Price'], bins=50)
    plt.title("Distribution of Price after Outlier Removal")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.show()

    owner_order = ['First', 'Second', 'Third', 'Fourth', '4 or More', 'UnRegistered Car']
    owner_map = {v: i for i, v in enumerate(owner_order)}
    train_data['Owner'] = train_data['Owner'].map(owner_map)
    if test_data is not None:
        test_data['Owner'] = test_data['Owner'].map(owner_map).fillna(0)

    for col in ['Make', 'Location']:
        freq_map = train_data[col].value_counts(normalize=True).to_dict()
        train_data[f'{col}_Freq'] = train_data[col].map(freq_map)
        if test_data is not None:
            test_data[f'{col}_Freq'] = test_data[col].map(freq_map).fillna(0)

    one_hot_cols = ['Fuel Type', 'Transmission', 'Seller Type', 'Drivetrain']
    train_data = pd.get_dummies(train_data, columns=one_hot_cols)
    if test_data is not None:
        test_data = pd.get_dummies(test_data, columns=one_hot_cols)
        missing_cols = set(train_data.columns) - set(test_data.columns)
        for col in missing_cols:
            test_data[col] = 0
        test_data = test_data[train_data.columns]

    drop_cols = ['Model', 'Max Power', 'Max Torque', 'Color']
    train_data = train_data.drop(columns=drop_cols, axis=1)
    if test_data is not None:
        test_data = test_data.drop(columns=drop_cols, axis=1)

    return train_data, test_data, numeric_cols

def select_features(train_data, numeric_cols, threshold_corr=0.35, threshold_anova=120):
    def correlation_scores(X, y):
        corr = [np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]
        return np.array(corr)

    def anova_f_test_onehot(X, y):
        scores = {}
        y_mean = np.mean(y)
        for col in X.columns:
            if set(X[col].unique()) <= {0, 1}:
                group1 = y[X[col] == 1]
                group0 = y[X[col] == 0]
                if len(group1) < 2 or len(group0) < 2:
                    continue
                n1, n0 = len(group1), len(group0)
                mu1, mu0 = group1.mean(), group0.mean()
                ss_between = n1 * (mu1 - y_mean)**2 + n0 * (mu0 - y_mean)**2
                ss_within = np.sum((group1 - mu1)**2) + np.sum((group0 - mu0)**2)
                f_score = ss_between / (ss_within / (n1 + n0 - 2))
                scores[col] = f_score
        return pd.Series(scores).sort_values(ascending=False)

    def anova_f_test_score(X_cat, y):
        groups = {}
        for xi, yi in zip(X_cat, y):
            groups.setdefault(xi, []).append(yi)
        group_means = {k: np.mean(v) for k, v in groups.items()}
        group_sizes = {k: len(v) for k, v in groups.items()}
        overall_mean = np.mean(y)
        ss_between = sum(group_sizes[k] * (group_means[k] - overall_mean)**2 for k in groups)
        ss_within = sum(sum((yi - group_means[k])**2 for yi in v) for k, v in groups.items())
        df_between = len(groups) - 1
        df_within = len(y) - len(groups)
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        return ms_between / ms_within if ms_within != 0 else 0

    y = train_data['Price'].to_numpy()
    x = train_data[numeric_cols].to_numpy()
    x_scaled = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    correlations = correlation_scores(x_scaled, y)
    correlations_freq = correlation_scores(train_data[['Make_Freq', 'Location_Freq']].to_numpy(), y)

    feature_scores = pd.DataFrame({
        'Feature': numeric_cols + ['Make_Freq', 'Location_Freq'],
        'Correlation_with_Price': np.concatenate((correlations, correlations_freq))
    }).sort_values(by='Correlation_with_Price', ascending=False)
    print("\nFeature Correlation Scores:")
    print(feature_scores)

    anova_scores = anova_f_test_onehot(train_data, train_data['Price'])
    owner_fscore = anova_f_test_score(train_data['Owner'].to_numpy(), train_data['Price'])
    anova_scores = pd.concat([anova_scores, pd.Series({'Owner': owner_fscore})])
    onehot_scores = pd.DataFrame({
        'Feature': anova_scores.index,
        'ANOVA_F_Score': anova_scores.values
    }).sort_values(by='ANOVA_F_Score', ascending=False)
    print("\nANOVA F-Scores for Categorical Features:")
    print(onehot_scores)

    selected_features = feature_scores[np.abs(feature_scores['Correlation_with_Price']) > threshold_corr]['Feature'].tolist()
    selected_features += onehot_scores[onehot_scores['ANOVA_F_Score'] > threshold_anova]['Feature'].tolist()
    print("\nSelected features:", selected_features)
    print("Number of selected features:", len(selected_features))

    return selected_features

def standardize_data(train_data, test_data, numeric_cols):
    train_mean = np.mean(train_data[numeric_cols].to_numpy(), axis=0)
    train_std = np.std(train_data[numeric_cols].to_numpy(), axis=0)
    train_data[numeric_cols] = (train_data[numeric_cols] - train_mean) / train_std
    if test_data is not None:
        test_data[numeric_cols] = (test_data[numeric_cols] - train_mean) / train_std
    return train_data, test_data, train_mean, train_std

# Linear Models
def make_transform_func(selected_features, poly_degree=1):
    def transform_func(X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        X_sel = X[selected_features].astype(float).to_numpy()
        if X_sel.ndim == 1:
            X_sel = X_sel.reshape(1, -1)
        if isinstance(poly_degree, int):
            X_poly = X_sel
            for d in range(2, poly_degree + 1):
                X_poly = np.hstack([X_poly, X_sel ** d])
            return X_poly
        elif isinstance(poly_degree, (list, np.ndarray)):
            X_poly = []
            for i, deg in enumerate(poly_degree):
                col = X_sel[:, i].reshape(-1, 1)
                col_poly = np.hstack([col ** d for d in range(1, deg + 1)])
                X_poly.append(col_poly)
            return np.hstack(X_poly)
    return transform_func

class LinearRegression:
    def __init__(self, alpha=0.11, l2_lambda=0.1, num_iterations=1000, transform_func=None):
        self.alpha = alpha
        self.l2_lambda = l2_lambda
        self.num_iterations = num_iterations
        self.transform_func = transform_func
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        X_trans = self.transform_func(X) if self.transform_func else X.to_numpy()
        if self.weights is None:
            self.weights = np.zeros(X_trans.shape[1], dtype=float)
        for _ in range(self.num_iterations):
            predictions = np.dot(X_trans, self.weights) + self.bias
            errors = predictions - y
            gradient_w = (X_trans.T.dot(errors) + 2 * self.l2_lambda * self.weights) / len(y)
            gradient_b = np.mean(errors)
            self.weights -= self.alpha * gradient_w
            self.bias -= self.alpha * gradient_b
        return self

    def predict(self, X):
        X_trans = self.transform_func(X) if self.transform_func else X.to_numpy()
        return np.dot(X_trans, self.weights) + self.bias

    def get_weights(self):
        return self.weights, self.bias

def evaluate_regression(model, X, y, is_log=True, mechanism=None, epsilon=None, delta=None, sensitivity=None):
    y_pred_log = model.predict(X)
    if is_log:
        y_true = y
        y_pred = y_pred_log
    else:
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y)

    # Thêm nhiễu nếu có cơ chế được chỉ định
    if mechanism and epsilon is not None and sensitivity is not None:
        if mechanism == 'gaussian' and delta is not None:
            y_pred_noisy = gaussian_mechanism(y_pred, sensitivity, epsilon, delta)
        elif mechanism == 'laplace':
            y_pred_noisy = laplace_mechanism(y_pred, sensitivity, epsilon)
        elif mechanism == 'exponential':
            y_pred_noisy = exponential_mechanism(y_pred, sensitivity, epsilon, utility_range=10.0)
        else:
            y_pred_noisy = y_pred
    else:
        y_pred_noisy = y_pred

    # Tính các chỉ số
    mse = np.mean((y_pred_noisy - y_true) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred_noisy - y_true))
    ss_res = np.sum((y_true - y_pred_noisy) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = max(min(1 - ss_res / ss_tot if ss_tot != 0 else 0, 1), -1)

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}, y_pred_noisy

# Differential Privacy Mechanisms
def gaussian_mechanism(preds, sensitivity, epsilon, delta):
    """Gaussian mechanism with L2 sensitivity"""
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(loc=0.0, scale=sigma, size=preds.shape)
    return preds + noise

def laplace_mechanism(preds, sensitivity, epsilon):
    """Laplace mechanism with L1 sensitivity"""
    b = sensitivity / epsilon
    noise = np.random.laplace(loc=0.0, scale=b, size=preds.shape)
    return preds + noise

def exponential_mechanism(preds, sensitivity, epsilon, utility_range=None):
    """Fixed exponential mechanism - đơn giản hóa và ổn định hơn"""
    n_samples = len(preds)
    
    # Tính utility_range dựa trên độ lệch chuẩn thay vì range
    if utility_range is None:
        utility_range = np.std(preds) * 2  # 2 standard deviations
        utility_range = max(utility_range, 1000.0)  # Minimum threshold
    
    # Tạo candidates gần với predictions
    noisy_preds = np.zeros_like(preds)
    
    for i in range(n_samples):
        # Tạo candidates trong khoảng hẹp hơn xung quanh prediction
        candidate_range = np.linspace(
            preds[i] - utility_range, 
            preds[i] + utility_range, 
            100  # Giảm số candidates
        )
        
        # Utility function: khoảng cách Euclidean
        utilities = -np.abs(candidate_range - preds[i])
        
        # Normalize utilities để tránh overflow
        utilities = utilities - np.max(utilities)  # Max utility = 0
        
        # Tính probabilities với sensitivity scaling
        probs = np.exp(epsilon * utilities / (2 * sensitivity))
        probs = probs / np.sum(probs)
        
        # Chọn candidate
        noisy_preds[i] = np.random.choice(candidate_range, p=probs)
    
    return noisy_preds

def advanced_comp_epsilon(epsilon, delta, k, delta_prime):
    eps_tot = np.sqrt(2 * k * np.log(1 / delta_prime)) * epsilon + k * epsilon * (np.exp(epsilon) - 1)
    delta_tot = k * delta + delta_prime
    return eps_tot, delta_tot

class DPLinearRegression(LinearRegression):
    def __init__(self, alpha=0.11, l2_lambda=0.1, num_iterations=1000, transform_func=None, 
                 epsilon=1, delta=1e-5, clip_val=None, mechanism='gaussian'):
        super().__init__(alpha, l2_lambda, num_iterations, transform_func)
        self.epsilon = epsilon
        self.delta = delta
        self.clip_val = clip_val
        self.l1_sensitivity = None
        self.l2_sensitivity = None
        self.mechanism = mechanism
        self.l1_norm = None
        self.l2_norm = None

    def compute_sensitivity(self, X):
        """Compute both L1 and L2 sensitivity based on mathematical theory and prediction range"""
        if self.weights is None:
            raise ValueError("Model must be fitted to compute weights.")

        # Apply the same transformation as used in training/prediction
        X_trans = self.transform_func(X) if self.transform_func else X.to_numpy()

        # Ensure X_trans is 2D
        if X_trans.ndim == 1:
            X_trans = X_trans.reshape(1, -1)
    
        norms = np.linalg.norm(X_trans, axis=1)
        self.clip_val = np.percentile(norms, 90) if self.clip_val is None else self.clip_val

        # Clip transformed data
        X_clipped = X_trans.copy()
        for i in range(len(X_clipped)):
            if norms[i] > self.clip_val:
                X_clipped[i] = X_clipped[i] * (self.clip_val / norms[i])

        # Tính norm của weights
        self.l1_norm = np.sum(np.abs(self.weights))
        self.l2_norm = np.linalg.norm(self.weights)

        # Dự đoán sạch để ước lượng phạm vi
        y_pred_log = np.dot(X_clipped, self.weights) + self.bias
        y_pred = np.expm1(y_pred_log)  # Chuyển sang miền gốc
        data_range = np.max(y_pred) - np.min(y_pred)  # Phạm vi dự đoán

        # Tính sensitivity dựa trên phạm vi dữ liệu và norm weights
        self.l1_sensitivity = self.l1_norm * self.clip_val + data_range * 0.01  # Thêm yếu tố phạm vi
        self.l2_sensitivity = self.l2_norm * self.clip_val + data_range * 0.01  # Thêm yếu tố phạm vi

        print(f"L1 Norm of weights: {self.l1_norm:.6f}")
        print(f"L2 Norm of weights: {self.l2_norm:.6f}")
        print(f"L1 Sensitivity: {self.l1_sensitivity:.6f}")
        print(f"L2 Sensitivity: {self.l2_sensitivity:.6f}")
        print(f"Data Range (post-expm1): {data_range:.6f}")

        return self.l1_sensitivity, self.l2_sensitivity

    def fit(self, X, y):
        super().fit(X, y)
        self.compute_sensitivity(X)
        return self

    def predict(self, X):
        X_trans = self.transform_func(X) if self.transform_func else X.to_numpy()
        predictions = np.dot(X_trans, self.weights) + self.bias
        predictions = np.clip(predictions, np.log1p(1e3), np.log1p(3e6))  # Giới hạn trước khi trả về
        return predictions

    def compute_privacy_loss(self, num_queries):
        k = min(num_queries, 100)
        return advanced_comp_epsilon(self.epsilon, self.delta, k, self.delta)

# Plotting Function
def plot_combined_metrics(results):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Performance Metrics vs Privacy Budget', fontsize=16)

    metrics = ['MSE', 'RMSE', 'MAE', 'R2']
    titles = ['Mean Squared Error', 'Root Mean Squared Error', 'Mean Absolute Error', 'R2 Score']
    ylabels = ['MSE', 'RMSE', 'MAE', 'R2']

    for ax, metric, title, ylabel in zip(axs.flat, metrics, titles, ylabels):
        for mechanism in results['mechanism'].unique():
            data = results[results['mechanism'] == mechanism]
            ax.plot(data['epsilon'], data[metric], marker='o', label=mechanism.capitalize())
        ax.set_title(title)
        ax.set_xlabel('Privacy Budget (ε)')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

def print_all_metrics(metrics, title):
    """Print all 4 metrics in a formatted way"""
    print(f"{title}:")
    print(f"  MSE:  {metrics['MSE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  MAE:  {metrics['MAE']:.2f}")
    print(f"  R2:   {metrics['R2']:.4f}")

def main():
    # Data Loading
    train_data, test_data = load_data()

    # Data Preprocessing
    train_data, test_data, numeric_cols = preprocess_data(train_data, test_data)
    selected_features = select_features(train_data, numeric_cols)
    train_data, test_data, _, _ = standardize_data(train_data, test_data, numeric_cols)

    X_train = train_data[selected_features]
    y_train = np.log1p(train_data['Price'].to_numpy())
    X_test = test_data[selected_features]
    y_test = np.log1p(test_data['Price'].to_numpy())

    # Linear Models (No DP)
    poly_degree_2 = [2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    model_1 = LinearRegression(alpha=0.11, l2_lambda=0.1, num_iterations=1000, transform_func=make_transform_func(selected_features, poly_degree=1))
    model_1.fit(X_train, y_train)
    print("\n" + "="*50)
    print("BASELINE MODELS EVALUATION (NO DP)")
    print("="*50)
    result_train_1, _ = evaluate_regression(model_1, X_train, y_train)  # Đảm bảo nhận 2 giá trị
    print_all_metrics(result_train_1, "Model 1 (Linear) - Train")
    result_test_1, _ = evaluate_regression(model_1, X_test, y_test)  # Đảm bảo nhận 2 giá trị
    print_all_metrics(result_test_1, "Model 1 (Linear) - Test")

    model_2 = LinearRegression(alpha=0.11, l2_lambda=0.1, num_iterations=1000, transform_func=make_transform_func(selected_features, poly_degree=poly_degree_2))
    model_2.fit(X_train, y_train)
    result_train_2, _ = evaluate_regression(model_2, X_train, y_train)
    print_all_metrics(result_train_2, "Model 2 (Polynomial) - Train")
    result_test_2, _ = evaluate_regression(model_2, X_test, y_test)
    print_all_metrics(result_test_2, "Model 2 (Polynomial) - Test")

    def transform_func_3(X):
        X_trans = make_transform_func(selected_features, poly_degree=[1, 1, 2] + [1] * (len(selected_features) - 3))(X)
        X_trans[:, 0] += X_trans[:, 1]
        return X_trans
    model_3 = LinearRegression(alpha=0.11, l2_lambda=0.1, num_iterations=1000, transform_func=transform_func_3)
    model_3.fit(X_train, y_train)
    result_train_3, _ = evaluate_regression(model_3, X_train, y_train)
    print_all_metrics(result_train_3, "Model 3 (Combined Features) - Train")
    result_test_3, _ = evaluate_regression(model_3, X_test, y_test)
    print_all_metrics(result_test_3, "Model 3 (Combined Features) - Test")

    def transform_func_4(X):
        X_trans = make_transform_func(selected_features, poly_degree=[1, 1, 2] + [1] * (len(selected_features) - 3))(X)
        X_trans[:, 0] *= X_trans[:, 1]
        return X_trans
    model_4 = LinearRegression(alpha=0.11, l2_lambda=0.1, num_iterations=1000, transform_func=transform_func_4)
    model_4.fit(X_train, y_train)
    result_train_4, _ = evaluate_regression(model_4, X_train, y_train)
    print_all_metrics(result_train_4, "Model 4 (Interaction Term) - Train")
    result_test_4, _ = evaluate_regression(model_4, X_test, y_test)
    print_all_metrics(result_test_4, "Model 4 (Interaction Term) - Test")

    # Differential Privacy Models (phần này giữ nguyên, không liên quan lỗi)
    print("\n" + "="*50)
    print("DIFFERENTIAL PRIVACY EVALUATION")
    print("="*50)
    epsilon_values = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]
    mechanisms = ['gaussian', 'laplace', 'exponential']
    results = []

    for mechanism in mechanisms:
        print(f"\n--- {mechanism.upper()} MECHANISM ---")
        for epsilon in epsilon_values:
            np.random.seed(42)
            dp_model = DPLinearRegression(
                alpha=0.11, l2_lambda=0.1, num_iterations=1000,
                transform_func=make_transform_func(selected_features, poly_degree=poly_degree_2),
                epsilon=epsilon,
                delta=1e-5,
                mechanism=mechanism
            )
            dp_model.fit(X_train, y_train)
            dp_model.compute_sensitivity(X_train)  # Cập nhật sensitivity
     
            result_test_dp, y_pred_noisy_test = evaluate_regression(
                dp_model, X_test, y_test, is_log=False,
                mechanism=mechanism, epsilon=epsilon, delta=1e-5,
                sensitivity=dp_model.l2_sensitivity if mechanism == 'gaussian' else dp_model.l1_sensitivity
            )
            print(f"\nDP Model (ε={epsilon}):")
            print_all_metrics(result_test_dp, f"Test Metrics")
            epsilon_prime, delta_prime = dp_model.compute_privacy_loss(num_queries=len(X_test))
            print(f"Privacy Loss: ε'={epsilon_prime:.4f}, δ'={delta_prime:.4e}\n")

            results.append({
                'mechanism': mechanism,
                'epsilon': epsilon,
                'MSE': result_test_dp['MSE'],
                'RMSE': result_test_dp['RMSE'],
                'MAE': result_test_dp['MAE'],
                'R2': result_test_dp['R2'],
                'epsilon_prime': epsilon_prime,
                'delta_prime': delta_prime
            })

    results_df = pd.DataFrame(results)
    plot_combined_metrics(results_df)

    print("\n" + "="*50)
    print("SUMMARY TABLE")
    print("="*50)
    summary_table = results_df.pivot_table(
        index='epsilon', 
        columns='mechanism', 
        values=['MSE', 'RMSE', 'MAE', 'R2'], 
        aggfunc='mean'
    )
    print(summary_table.round(4))

    train_data.to_csv('train_data_preprocessed_new.csv', index=False)
    test_data.to_csv('test_data_preprocessed_new.csv', index=False)

if __name__ == "__main__":
    main()