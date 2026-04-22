import pandas as pd
import numpy as np
import torch
import re
import os

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, logging, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW   
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from datetime import datetime

#The two text preprocessing functions are taken directly from the baseline for a fair comparison
def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"  # enclosed characters
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Runs on GPU or CPU if no GPU available
print(f"Using {device}\n")

logging.set_verbosity_error()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

projects = ["pytorch", "tensorflow", "keras", "incubator-mxnet", "caffe"]

for project in projects:
    df = pd.read_csv(f"data/{project}.csv")
    df = df.drop(columns=["Unnamed: 0"])

    df["text"] = (
        df["Title"].fillna("") + " " +
        df["Body"].fillna("")
    )

    df["text"] = df["text"].apply(remove_html)
    df["text"] = df["text"].apply(remove_emoji)

    #Code used to carry out token length analysis, not used in standard run
    # lengths = df["text"].apply(lambda x: len(tokenizer.encode(x)))
    # print(f"Average tokens: {lengths.mean():.0f}")
    # print(f"Max tokens: {lengths.max()}")
    # print(f"90th percentile: {lengths.quantile(0.9):.0f}")   

    X = df["text"]
    y = df["class"]

    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []

    n_runs = 30
    n_epochs = 3

    run_start_time = datetime.now()
    for seed in range(n_runs):
        print(f"Project: {project}, Repeat: {seed + 1}/{n_runs}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        
        #Train test split, stratification removed to align with baseline
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            #stratify=y,
            random_state=seed
        )

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

        encoded_train = tokenizer(list(X_train), return_tensors="pt", padding=True, truncation=True, max_length=512) #Change maximum context window length here

        train_input_ids = encoded_train["input_ids"]
        train_attention_mask = encoded_train["attention_mask"]
        train_labels = torch.tensor(list(y_train))
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        #Code for learning rate scheduler, however this resulted in weaker performance
        # total_steps = len(train_dataloader) * n_epochs
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        for epoch in range(n_epochs):
            for batch_input_ids, batch_attention_mask, batch_labels in train_dataloader:
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad() #Reset gradients for each batch
                batch_outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                batch_loss = loss_function(batch_outputs.logits, batch_labels) #Manually compute weighted loss
                batch_loss.backward()
                optimizer.step()
                # scheduler.step()

        model.eval()

        with torch.no_grad():
            encoded_test = tokenizer(list(X_test), return_tensors="pt", padding=True, truncation=True, max_length=512) #Change maximum context window length here

            test_outputs = model(input_ids=encoded_test["input_ids"].to(device), attention_mask=encoded_test["attention_mask"].to(device))

            y_pred = torch.argmax(test_outputs.logits, dim=1).cpu().numpy() #Convert model logits to predicted class labels
            y_prob = torch.softmax(test_outputs.logits, dim=1)[:, 1].cpu().numpy()

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average="macro", zero_division=0))  
        recalls.append(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        false_positive_rates, true_positive_rates, thresholds = roc_curve(y_test, y_prob, pos_label=1)
        aucs.append(auc(false_positive_rates, true_positive_rates))

        #Prevents memory leakage
        del model
        del optimizer
        del train_dataset
        del train_dataloader
        del encoded_train
        del encoded_test
        del batch_outputs
        del test_outputs
        torch.cuda.empty_cache()

        #print(f"{run_duration:.1f} Seconds, VRAM Used: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    run_end_time = datetime.now()

    print(f"\nProject: {project}")
    print(f"Median Accuracy: {np.median(accuracies):.4f}")
    print(f"Median Precision: {np.median(precisions):.4f}")
    print(f"Median Recall: {np.median(recalls):.4f}")
    print(f"Median F1 Score: {np.median(f1s):.4f}")
    print(f"Median AUC Score: {np.median(aucs):.4f}")

    run_duration = (run_end_time - run_start_time).total_seconds()
    print(f"30 Runs Execution Time: {run_duration:.1f} Seconds")

    os.makedirs("solution_results", exist_ok = True)

    df_log = pd.DataFrame({
        'seed': list(range(n_runs)),
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1': f1s,
        'AUC': aucs
    })

    df_log.to_csv(f'solution_results/{project}.csv', index=False)
    print(f"Results saved to solution_results/{project}.csv")