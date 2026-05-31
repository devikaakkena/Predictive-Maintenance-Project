# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================
# CONFIGURATION
# ==============================
DATA_PATH = "ml/data/raw/ai4i.csv"
MODEL_PATH = "ml/models/trained_model.pkl"

FEATURE_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

TARGET_COLUMN = 'Machine failure'


# ==============================
# LOAD DATA
# ==============================
def load_data(path):
    data = pd.read_csv(path)
    print("✅ Data loaded successfully")
    return data


# ==============================
# PREPROCESS DATA
# ==============================
def preprocess_data(data):
    data = data.drop(['UDI', 'Product ID', 'Type'], axis=1)

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    print("✅ Data preprocessing completed")
    return X, y


# ==============================
# TRAIN MODEL
# ==============================
def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("✅ Model training completed")
    return model


# ==============================
# SAVE MODEL
# ==============================
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved at: {path}")


# ==============================
# MAIN FUNCTION
# ==============================
def main():
    print("🚀 Training Started...\n")

    data = load_data(DATA_PATH)
    X, y = preprocess_data(data)

    model = train_model(X, y)

    # ==============================
    # GRAPH 1: FEATURE IMPORTANCE
    # ==============================
    print("\n📊 Feature Importance Graph")

    importances = model.feature_importances_

    plt.figure()
    plt.barh(FEATURE_COLUMNS, importances)
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig("ml/analysis/feature_importance.png")
    plt.savefig("backend/app/static/graphs/feature_importance.png")
    plt.show()

    # ==============================
    # GRAPH 2: FAILURE vs SAFE COUNT
    # ==============================
    print("\n📊 Failure vs Safe Count")

    y_pred = model.predict(X)

    counts = pd.Series(y_pred).value_counts()

    plt.figure()
    counts.plot(kind='bar', color=['green', 'red'])
    plt.title("Safe vs Failure Count")
    plt.xticks([0, 1], ["Safe", "Failure"], rotation=0)
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("ml/analysis/failure_vs_safe.png")
    plt.savefig("backend/app/static/graphs/failure_vs_safe.png")
    plt.show()

    # ==============================
    # GRAPH 3: CORRELATION HEATMAP
    # ==============================
    print("\n📊 Correlation Heatmap")

    plt.figure(figsize=(8, 6))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("ml/analysis/correlation_heatmap.png")
    plt.savefig("backend/app/static/graphs/correlation_heatmap.png")
    plt.show()

    # ==============================
    # ACCURACY
    # ==============================
    accuracy = accuracy_score(y, y_pred)
    print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%")

    # ==============================
    # SAVE MODEL
    # ==============================
    save_model(model, MODEL_PATH)

    print("\n🎉 Training Completed Successfully!")


# ==============================
# RUN PROGRAM
# ==============================
if __name__ == "__main__":
    main()