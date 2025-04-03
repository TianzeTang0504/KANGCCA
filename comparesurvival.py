import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os

def compare_models(X, T, E, name="experiment"):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=32)
    X_reduced = pca.fit_transform(X_scaled)

    X_train, X_test, T_train, T_test, E_train, E_test = train_test_split(
        X_reduced, T, E, test_size=0.2, random_state=42
    )
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    T_train = pd.Series(T_train).reset_index(drop=True)
    T_test = pd.Series(T_test).reset_index(drop=True)
    E_train = pd.Series(E_train).reset_index(drop=True)
    E_test = pd.Series(E_test).reset_index(drop=True)

    results = {}

    cox_train = pd.DataFrame(X_train)
    cox_train["T"] = T_train
    cox_train["E"] = E_train
    cox_train = cox_train.dropna()

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_train, duration_col="T", event_col="E")

    risk_score = cph.predict_partial_hazard(pd.DataFrame(X_test))
    c_index = concordance_index(T_test, -risk_score, E_test)
    results["cox"] = {"c_index": c_index, "risk_score": risk_score}

    os.makedirs("km_curves", exist_ok=True)

    for model_name, result in results.items():
        score = result["risk_score"]
        threshold = np.median(score)
        high_risk = score > threshold if model_name == "cox" else score < threshold
        low_risk = ~high_risk

        kmf = KaplanMeierFitter()
        plt.figure(figsize=(8, 6))
        kmf.fit(T_test[high_risk], E_test[high_risk], label="High Risk")
        ax = kmf.plot(ci_show=False)
        kmf.fit(T_test[low_risk], E_test[low_risk], label="Low Risk")
        kmf.plot(ax=ax, ci_show=False)

        if model_name == "cox":
            aic = cph.AIC_partial_
            log_p = -np.log2(cph.log_likelihood_ratio_test().p_value)
            title = (f"{model_name.upper()} KM Curve (Reduced {name})\n"
                     f"C-index = {result['c_index']:.4f} | AIC = {aic:.2f} | -log2(p) = {log_p:.2f}")
        else:
            title = f"{model_name.upper()} KM Curve (Reduced {name})\n" \
                    f"C-index = {result['c_index']:.4f}"

        plt.title(title)
        plt.xlabel("Days")
        plt.ylabel("Survival Probability")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"km_curves/{name}_{model_name}_km.png")
        plt.close()



    for model_name, result in results.items():
        print(f"[{model_name.upper()}] C-index = {result['c_index']:.4f}")

X = pd.read_csv("./fused_multiview_features.csv")
T = pd.read_csv("./data/1.csv", header=None).iloc[:, 0]
E = pd.read_csv("./data/2.csv", header=None).iloc[:, 0]

compare_models(X, T, E, name="kan0.002")