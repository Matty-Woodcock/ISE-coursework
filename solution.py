import pandas as pd
import numpy as np
import torch

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW   

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Runs on GPU or CPU if no GPU available
print(f"Using {device}")

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

for project in ["pytorch"]:#, "tensorflow", "keras", "incubator-mxnet", "caffe"]:
    df = pd.read_csv(f"data/{project}.csv")
    df = df.drop(columns=["Unnamed: 0"])

    df["text"] = ( #TO DO: Potentially add more columns
        df["Title"].fillna("") + " " +
        df["Body"].fillna("")
    )

    X = df["text"]
    y = df["class"]

    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []

    for seed in range(1):
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased") #TO DO: Slow?
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=seed
        )

        encoded_train = tokenizer(list(X_train), return_tensors="pt", padding=True, truncation=True, max_length=512)

        train_input_ids = encoded_train["input_ids"]
        train_attention_mask = encoded_train["attention_mask"]
        #labels = torch.tensor(y_train.values)
        train_labels = torch.tensor(list(y_train))
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        for epoch in range(3):
            for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad() #Reset gradients for each batch
                batch_outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
                batch_loss = batch_outputs.loss
                batch_loss.backward()
                optimizer.step()

        model.eval()

        with torch.no_grad():
            encoded_test = tokenizer(list(X_test), return_tensors="pt", padding=True, truncation=True, max_length=512)

            test_outputs = model(input_ids=encoded_test["input_ids"].to(device), attention_mask=encoded_test["attention_mask"].to(device))

            y_pred = torch.argmax(test_outputs.logits, dim=1).cpu().numpy() #Convert model logits to predicted class labels

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