import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

model_scores = {}

for name, model in models.items():
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    model_scores[name] = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred) * 100, 2),
        "recall": round(recall_score(y_test, y_pred) * 100, 2),
        "f1": round(f1_score(y_test, y_pred) * 100, 2),
    }

voting_model = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression(max_iter=2000, random_state=42)),
        ("dt", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("svc", SVC(probability=True, random_state=42)),
    ],
    voting="soft"
)

voting_model.fit(X_train_scaled, y_train)
voting_pred = voting_model.predict(X_test_scaled)

voting_scores = {
    "accuracy": round(accuracy_score(y_test, voting_pred) * 100, 2),
    "precision": round(precision_score(y_test, voting_pred) * 100, 2),
    "recall": round(recall_score(y_test, voting_pred) * 100, 2),
    "f1": round(f1_score(y_test, voting_pred) * 100, 2),
}

rf_for_importance = RandomForestClassifier(n_estimators=200, random_state=42)
rf_for_importance.fit(X_train, y_train)
feature_importance = dict(
    zip(X.columns, [round(v, 4) for v in rf_for_importance.feature_importances_])
)

cm = confusion_matrix(y_test, voting_pred).tolist()

joblib.dump(voting_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

metadata = {
    "best_model_name": "Soft Voting Classifier",
    "best_model_accuracy": voting_scores["accuracy"],
    "all_model_accuracies": {
        "Logistic Regression": model_scores["Logistic Regression"]["accuracy"],
        "Decision Tree": model_scores["Decision Tree"]["accuracy"],
        "SVM": model_scores["SVM"]["accuracy"],
        "Random Forest": model_scores["Random Forest"]["accuracy"],
        "AdaBoost": model_scores["AdaBoost"]["accuracy"],
        "Gradient Boosting": model_scores["Gradient Boosting"]["accuracy"],
        "Voting Classifier": voting_scores["accuracy"],
    },
    "all_model_metrics": {
        "Logistic Regression": model_scores["Logistic Regression"],
        "Decision Tree": model_scores["Decision Tree"],
        "SVM": model_scores["SVM"],
        "Random Forest": model_scores["Random Forest"],
        "AdaBoost": model_scores["AdaBoost"],
        "Gradient Boosting": model_scores["Gradient Boosting"],
        "Voting Classifier": voting_scores,
    },
    "feature_importance": feature_importance,
    "feature_names": list(X.columns),
    "confusion_matrix": cm,
}

joblib.dump(metadata, "metadata.pkl")

print("Training completed successfully.")
print("Best Model:", metadata["best_model_name"])
print("Best Accuracy:", metadata["best_model_accuracy"])