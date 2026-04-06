import pandas as pd
import numpy as np
import torch
import re

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, logging, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW   
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from datetime import datetime

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
#projects = ["tensorflow"]

for project in projects:
    df = pd.read_csv(f"data/{project}.csv")
    df = df.drop(columns=["Unnamed: 0"]) #TO DO: This is ID?

    df["text"] = ( #TO DO: Potentially add more columns
        df["Title"].fillna("") + " " +
        df["Body"].fillna("")
    )

    # for i in range(len(df)):
    #     df["text"][i] = remove_html(df["text"][i])
    #     df["text"][i] = remove_emoji(df["text"][i])

    df["text"] = df["text"].apply(remove_html)
    df["text"] = df["text"].apply(remove_emoji)

    # lengths = df["text"].apply(lambda x: len(tokenizer.encode(x)))
    # print(f"Average tokens: {lengths.mean():.0f}")
    # print(f"Max tokens: {lengths.max()}")
    # print(f"90th percentile: {lengths.quantile(0.9):.0f}")

    X = df["text"]
    y = df["class"]

    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []

    n_runs = 30
    n_epochs = 3

    for seed in range(n_runs):
        run_start_time = datetime.now()
        print(f"Project: {project}, Repeat: {seed + 1}/{n_runs}")

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2) #TO DO: Slow? Does num_labels work?
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            stratify=y,
            random_state=seed
        )

        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        loss_function = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

        encoded_train = tokenizer(list(X_train), return_tensors="pt", padding=True, truncation=True, max_length=128)

        train_input_ids = encoded_train["input_ids"]
        train_attention_mask = encoded_train["attention_mask"]
        #labels = torch.tensor(y_train.values)
        train_labels = torch.tensor(list(y_train))
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

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
            encoded_test = tokenizer(list(X_test), return_tensors="pt", padding=True, truncation=True, max_length=128)

            test_outputs = model(input_ids=encoded_test["input_ids"].to(device), attention_mask=encoded_test["attention_mask"].to(device))

            y_pred = torch.argmax(test_outputs.logits, dim=1).cpu().numpy() #Convert model logits to predicted class labels

        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average="macro", zero_division=0))  
        recalls.append(recall_score(y_test, y_pred, average="macro", zero_division=0))
        f1s.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        false_positive_rates, true_positive_rates, thresholds = roc_curve(y_test, y_pred, pos_label=1)
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

        run_end_time = datetime.now()
        run_duration = (run_end_time - run_start_time).total_seconds()
        print(f"{run_duration:.1f} Seconds, VRAM Used: {torch.cuda.memory_allocated()/1024**2:.0f}MB")

    print(f"\nProject: {project}")
    print(f"Average Accuracy: {np.mean(accuracies):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall:    {np.mean(recalls):.4f}")
    print(f"Average F1 Score:  {np.mean(f1s):.4f}")
    print(f"Average AUC Score:  {np.mean(aucs):.4f}")

    print(f"{np.unique(y_pred, return_counts=True)}\n")

#TO DO LIST
#1. Scheduler made it worse?
#2. Slow model / fast model

# Project: pytorch
# Average Accuracy: 0.8965
# Average Precision: 0.7914
# Average Recall:    0.7201
# Average F1 Score:  0.7419
# Average AUC Score:  0.7201

# Project: pytorch
# Average Accuracy: 0.8982
# Average Precision: 0.7638
# Average Recall:    0.7681
# Average F1 Score:  0.7590
# Average AUC Score:  0.7681

#Added class weights - better

# Project: pytorch
# Average Accuracy: 0.8841
# Average Precision: 0.7739
# Average Recall:    0.7953
# Average F1 Score:  0.7693
# Average AUC Score:  0.7953

# Project: pytorch
# Average Accuracy: 0.8841
# Average Precision: 0.7487
# Average Recall:    0.8100
# Average F1 Score:  0.7710
# Average AUC Score:  0.8100

#With cleaning

# Project: pytorch
# Average Accuracy: 0.8867
# Average Precision: 0.7629
# Average Recall:    0.7821
# Average F1 Score:  0.7643
# Average AUC Score:  0.7821

#First all run - 5 runs per project

# Project: pytorch
# Average Accuracy: 0.8619
# Average Precision: 0.7297
# Average Recall:    0.8149
# Average F1 Score:  0.7494
# Average AUC Score:  0.8149
# (array([0, 1]), array([188,  38]))

# Project: tensorflow
# Average Accuracy: 0.9275
# Average Precision: 0.8809
# Average Recall:    0.8986
# Average F1 Score:  0.8862
# Average AUC Score:  0.8986
# (array([0, 1]), array([372,  75]))

# Project: keras
# Average Accuracy: 0.8876
# Average Precision: 0.8283
# Average Recall:    0.8441
# Average F1 Score:  0.8337
# Average AUC Score:  0.8441
# (array([0, 1]), array([146,  55]))

# Project: incubator-mxnet
# Average Accuracy: 0.8852
# Average Precision: 0.7676
# Average Recall:    0.8617
# Average F1 Score:  0.7897
# Average AUC Score:  0.8617
# (array([0, 1]), array([108,  47]))

# Project: caffe
# Average Accuracy: 0.8767
# Average Precision: 0.7135
# Average Recall:    0.6871
# Average F1 Score:  0.6816
# Average AUC Score:  0.6871
# (array([0, 1]), array([80,  6]))

#30 runs

# Project: pytorch
# Average Accuracy: 0.8724
# Average Precision: 0.7458
# Average Recall:    0.8004
# Average F1 Score:  0.7572
# Average AUC Score:  0.8004

# Project: tensorflow
# Average Accuracy: 0.9124
# Average Precision: 0.8564
# Average Recall:    0.8782
# Average F1 Score:  0.8627
# Average AUC Score:  0.8782

# Project: keras
# Average Accuracy: 0.8658
# Average Precision: 0.8044
# Average Recall:    0.8335
# Average F1 Score:  0.8072
# Average AUC Score:  0.8335

# Project: incubator-mxnet
# Average Accuracy: 0.9112
# Average Precision: 0.8097
# Average Recall:    0.8432
# Average F1 Score:  0.8145
# Average AUC Score:  0.8432

# Project: caffe
# Average Accuracy: 0.8550
# Average Precision: 0.6894
# Average Recall:    0.6763
# Average F1 Score:  0.6604
# Average AUC Score:  0.6763

#30 runs, seeds 1-30

# Project: pytorch
# Average Accuracy: 0.8670
# Average Precision: 0.7409
# Average Recall:    0.7923
# Average F1 Score:  0.7453
# Average AUC Score:  0.7923

# Project: tensorflow
# Average Accuracy: 0.9164
# Average Precision: 0.8657
# Average Recall:    0.8691
# Average F1 Score:  0.8644
# Average AUC Score:  0.8691

# Project: keras
# Average Accuracy: 0.8624
# Average Precision: 0.8019
# Average Recall:    0.8371
# Average F1 Score:  0.8066
# Average AUC Score:  0.8371

# Project: incubator-mxnet
# Average Accuracy: 0.8976
# Average Precision: 0.7988
# Average Recall:    0.8390
# Average F1 Score:  0.7987
# Average AUC Score:  0.8390

# Project: caffe
# Average Accuracy: 0.8058
# Average Precision: 0.6610
# Average Recall:    0.7049
# Average F1 Score:  0.6379
# Average AUC Score:  0.7049

#Seeded 30 scheduler

# Project: pytorch
# Average Accuracy: 0.8640
# Average Precision: 0.7184
# Average Recall:    0.7759
# Average F1 Score:  0.7368
# Average AUC Score:  0.7759

# Project: tensorflow
# Average Accuracy: 0.9139
# Average Precision: 0.8536
# Average Recall:    0.8784
# Average F1 Score:  0.8642
# Average AUC Score:  0.8784

# Project: keras
# Average Accuracy: 0.8498
# Average Precision: 0.7739
# Average Recall:    0.8207
# Average F1 Score:  0.7897
# Average AUC Score:  0.8207

# Project: incubator-mxnet
# Average Accuracy: 0.8963
# Average Precision: 0.7763
# Average Recall:    0.7829
# Average F1 Score:  0.7738
# Average AUC Score:  0.7829

# Project: caffe
# Average Accuracy: 0.8705
# Average Precision: 0.6223
# Average Recall:    0.5577
# Average F1 Score:  0.5475
# Average AUC Score:  0.5577

#Seed 30 256 testing only

# Project: pytorch
# Average Accuracy: 0.8732
# Average Precision: 0.7495
# Average Recall:    0.7900
# Average F1 Score:  0.7502
# Average AUC Score:  0.7900

# Project: tensorflow
# Average Accuracy: 0.9132
# Average Precision: 0.8642
# Average Recall:    0.8534
# Average F1 Score:  0.8557
# Average AUC Score:  0.8534

# Project: keras
# Average Accuracy: 0.8627
# Average Precision: 0.8000
# Average Recall:    0.8303
# Average F1 Score:  0.8041
# Average AUC Score:  0.8303

# Project: incubator-mxnet
# Average Accuracy: 0.8966
# Average Precision: 0.8007
# Average Recall:    0.8327
# Average F1 Score:  0.7950
# Average AUC Score:  0.8327

# Project: caffe
# Average Accuracy: 0.8178
# Average Precision: 0.6586
# Average Recall:    0.6900
# Average F1 Score:  0.6331
# Average AUC Score:  0.6900

#Seed 30 128

# Project: pytorch
# Average Accuracy: 0.8720
# Average Precision: 0.7552
# Average Recall:    0.7815
# Average F1 Score:  0.7477
# Average AUC Score:  0.7815

# Project: tensorflow
# Average Accuracy: 0.9029
# Average Precision: 0.8472
# Average Recall:    0.8664
# Average F1 Score:  0.8498
# Average AUC Score:  0.8664

# Project: keras
# Average Accuracy: 0.8496
# Average Precision: 0.7845
# Average Recall:    0.8051
# Average F1 Score:  0.7826
# Average AUC Score:  0.8051

# Project: incubator-mxnet
# Average Accuracy: 0.8940
# Average Precision: 0.7987
# Average Recall:    0.8348
# Average F1 Score:  0.7930
# Average AUC Score:  0.8348

# Project: caffe
# Average Accuracy: 0.8132
# Average Precision: 0.6633
# Average Recall:    0.6757
# Average F1 Score:  0.6304
# Average AUC Score:  0.6757