import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

dfs = []
for project in ["tensorflow", "pytorch", "keras", "incubator-mxnet", "caffe"]:
    df = pd.read_csv(f"data/{project}.csv")
    df = df.drop(columns=["Unnamed: 0"])

    df["text"] = ( #TO DO: Potentially add more columns
        df["Title"].fillna("") + " " +
        df["Body"].fillna("")
    )

    X = df["text"]
    y = df["class"]

    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []

    for seed in range(30):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=seed
        )

        vectorizer = TfidfVectorizer(stop_words="english") #This removes "not", is this bad?

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        nb = MultinomialNB(alpha=0.01)
        nb.fit(X_train_tfidf, y_train)

        y_pred = nb.predict(X_test_tfidf)

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average="macro", zero_division=0))
        recalls.append(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        false_positive_rates, true_positive_rates, thresholds = roc_curve(y_test, y_pred, pos_label=1)
        aucs.append(auc(false_positive_rates, true_positive_rates))

    print(f"\nProject: {project}")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall:    {np.mean(recalls):.4f}")
    print(f"Average F1 Score:  {np.mean(f1s):.4f}")
    print(f"Average AUC Score:  {np.mean(aucs):.4f}")

    print(np.unique(y_pred, return_counts=True))