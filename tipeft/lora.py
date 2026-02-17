import argparse
import os
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    LoraModel, 
    LoraConfig,
    PeftType,
    PeftConfig,
    PeftModel)
from evaluate import load
from torch.utils.data import DataLoader, Dataset
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from tqdm.notebook import tqdm_notebook
import gc
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)
from multiprocessing import Pool
from functools import lru_cache
import json
import numpy as np
import shutil



def train_tabular_classification(data, label, model_name_or_path,lr=0.0001):
    data = data.copy()

    class ClinicalDatasetForTabularPreopClassification(Dataset):
        def __init__(self, dataframe, tokenizer,label_to_id):
            self.dataframe = dataframe
            self.tokenizer = tokenizer
            self.label_to_id=label_to_id

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            sentence = self.dataframe.iloc[idx]['text']
            label = self.dataframe.iloc[idx]['label']

            # Tokenize the sentence
            inputs = self.tokenizer(sentence, truncation=True, max_length=512, return_tensors="pt")
            inputs = {key: val.squeeze(0) for key, val in inputs.items()}

            # Convert label to a numeric format if necessary
            label_to_id = self.label_to_id
            label_id = label_to_id[label]

            inputs['labels'] = torch.tensor(label_id, dtype=torch.long)

            return inputs
    torch.cuda.empty_cache()
    gc.collect()
    data["label"]=data[label]
    train=data[["text","label"]]
    labels=list(set(list(data[label])))
    device = "cuda"
    num_epochs = 2
    if model_name_or_path=="microsoft/biogpt":
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 target_modules=["q_proj","k_proj","v_proj","o_proj"],
                                 bias="none",r=8,lora_dropout=0.1, 
                                 lora_alpha=16)
    else:
        peft_config = LoraConfig(task_type="SEQ_CLS",target_modules=['query','value','key'],bias="none",r=4,lora_dropout=0.05, lora_alpha=16)

    # Initialize the tokenizer
    padding_side = "right"
    if model_name_or_path=="microsoft/biogpt":
        batch_size=8
    else:
        batch_size = 16


    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,padding_side=padding_side, use_auth_token=False)
    label_to_id = {label: index for index, label in enumerate(labels)}
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=len(label_to_id), use_auth_token=False)

    train_dataset = ClinicalDatasetForTabularPreopClassification(train, tokenizer,label_to_id)
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    model = get_peft_model(model, peft_config)
    optimizer = AdamW(params=model.parameters(), lr=lr)
    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm_notebook(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    mmm=(model_name_or_path.split("/"))[-1]
    model.save_pretrained(f"pretrained_model/{mmm}/{label}")
    tokenizer.save_pretrained(f"pretrained_model/{mmm}/{label}")
    del label
    return(model)



def train_tabular_regression(data, label,model_name_or_path,lr=0.0001):
    data = data.copy()
    data[label] = data[label].astype(float)

    class ClinicalDatasetForTabularPreopReg(Dataset):
        def __init__(self, dataframe, tokenizer):
            self.dataframe = dataframe
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            sentence = self.dataframe.iloc[idx]['text']
            # Label is now a continuous value
            label = self.dataframe.iloc[idx]['label']

            inputs = self.tokenizer(sentence, truncation=True, max_length=512, return_tensors="pt")
            inputs = {key: val.squeeze(0) for key, val in inputs.items()}
            inputs['labels'] = torch.tensor(label, dtype=torch.float)

            return inputs
    
    torch.cuda.empty_cache()
    gc.collect()
    data["label"]=data[label]
    train=data[["text","label"]]
    labels=list(set(list(data[label])))
    device = "cuda"
    num_epochs = 2
    if model_name_or_path=="microsoft/biogpt":
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 target_modules=["q_proj","k_proj","v_proj","o_proj"],
                                 bias="none",r=8,lora_dropout=0.1, 
                                 lora_alpha=16)
    else:
        peft_config = LoraConfig(task_type="SEQ_CLS",target_modules=['query','value','key'],bias="none",r=4,lora_dropout=0.05, lora_alpha=16)

    padding_side = "right"
    
    if model_name_or_path=="microsoft/biogpt":
        batch_size=8
    else:
        batch_size = 16
    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,padding_side=padding_side, use_auth_token=False)
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1, use_auth_token=False)

    train_dataset = ClinicalDatasetForTabularPreopReg(train, tokenizer)
    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    model = get_peft_model(model, peft_config)
    optimizer = AdamW(params=model.parameters(), lr=lr)
    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm_notebook(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    mmm=(model_name_or_path.split("/"))[-1]
    model.save_pretrained(f"pretrained_model/{mmm}/{label}")
    tokenizer.save_pretrained(f"pretrained_model/{mmm}/{label}")
    del label
    return(model)





def load_and_accumulate_lora(model_path, module_name, param_type, columns_unique_labels, base_model_name):
    path_base = os.path.basename(model_path)
    num_labels = columns_unique_labels[path_base]

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=num_labels, use_auth_token=False
    )

    try:
        lora_model = PeftModel.from_pretrained(base_model, model_path, use_auth_token=False)
        corresponding_module = dict(lora_model.named_modules()).get(module_name)

        if corresponding_module and hasattr(corresponding_module, param_type):
            param_container = getattr(corresponding_module, param_type)

            if isinstance(param_container, torch.nn.ModuleDict):
                param = param_container["default"].weight
                return param.data.clone(), 1
            elif isinstance(param_container, torch.nn.ParameterDict):
                param = param_container["default"]
                return param.data.clone(), 1
            elif isinstance(param_container, torch.nn.Parameter):
                return param_container.data.clone(), 1
            else:
                print(f"Skipped non-weight attribute {module_name}.{param_type}")
                return None, 0
        else:
            print(f"Warning: {module_name}.{param_type} not found.")
            return None, 0

    except Exception as e:
        print(f"Error loading LoRa from {model_path}: {e}")
        return None, 0


def parallel_advanced_initialization(model1, columns_unique_labels, name_of_lora_feature_based_model, base_model_name):
    model_dir = f'pretrained_model/{name_of_lora_feature_based_model}'
    models_to_average_paths = [
        os.path.join(model_dir, name) for name in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, name))
    ]

    lora_params = ['lora_A', 'lora_B']

    for module_name, module in tqdm_notebook(model1.named_modules()):
        for param_type in lora_params:
            if hasattr(module, param_type):
                args = [
                    (model_path, module_name, param_type, columns_unique_labels, base_model_name)
                    for model_path in models_to_average_paths
                ]

                with Pool(processes=10) as pool:
                    results = pool.starmap(load_and_accumulate_lora, args)

                params_sum = None
                models_count = 0
                for result, count in results:
                    if result is not None:
                        params_sum = result if params_sum is None else params_sum + result
                        models_count += count

                if models_count > 0:
                    averaged_param = params_sum / models_count
                    with torch.no_grad():
                        target_attr = getattr(module, param_type)

                        if isinstance(target_attr, torch.nn.ModuleDict):
                            target_attr["default"].weight.data.copy_(averaged_param)
                        elif isinstance(target_attr, torch.nn.ParameterDict):
                            target_attr["default"].data.copy_(averaged_param)
                        elif isinstance(target_attr, torch.nn.Parameter):
                            target_attr.data.copy_(averaged_param)
                        else:
                            print(f"Unhandled attribute type for {module_name}.{param_type}")
                else:
                    print(f"No successful parameter loaded for {module_name}.{param_type}")

    return model1





def train_tabular_infused_lora(train,val,pretrained_model_name,label_col,text_col,columns_unique_labels_of_tabular_features,lr=0.001,num_epochs=5,lr_of_tabular_infused_features=0.0001):

    """
    INSERT DOCUMENTATION HERE. 
    
    """
    train = train.copy()  # Add this
    val = val.copy()      # Add this
    
    train["label"] = train[label_col]
    train["text"] = train[text_col]
    val["label"] = val[label_col]
    val["text"] = val[text_col]

    train = train.drop(columns=[label_col, text_col])
    val   = val.drop(columns=[label_col, text_col])


    list_numerical   = [k for k, v in columns_unique_labels_of_tabular_features.items() if v == 1]
    list_categorical = [k for k, v in columns_unique_labels_of_tabular_features.items() if v > 1]

    for i in tqdm_notebook(list(list_numerical), desc="training numerical tabular-infused features"):
        train_tabular_regression(train, i, pretrained_model_name,lr_of_tabular_infused_features)

    for i in tqdm_notebook(list(list_categorical), desc="training categorical tabular-infused features"):
        train_tabular_classification(train, i, pretrained_model_name,lr_of_tabular_infused_features)

    new_model_name=f"IA3_{pretrained_model_name}_{label_col}"


    torch.manual_seed(42)

    label_to_id = {False: 0, True: 1}
    padding_side = "right"
    batch_size = 16
    lr = lr
    local_dir=f"trained_models/{new_model_name}"
    class clinicalDataset(Dataset):
        def __init__(self, dataframe, tokenizer):
            self.dataframe = dataframe
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            sentence = self.dataframe.iloc[idx]['text']
            label = self.dataframe.iloc[idx]['label']

            # Tokenize the sentence
            inputs = self.tokenizer(sentence, truncation=True, max_length=512, return_tensors="pt")
            inputs = {key: val.squeeze(0) for key, val in inputs.items()}

            # Convert label to a numeric format if necessary
            label_to_id = {False: 0, True: 1}
            label_id = label_to_id[label]

            inputs['labels'] = torch.tensor(label_id, dtype=torch.long)

            return inputs


    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    padding_side = "right"
    batch_size = 16
    lr = lr
    model_name_or_path = pretrained_model_name
    device = "cuda"
    num_epochs = num_epochs
    if pretrained_model_name=="microsoft/biogpt":
        peft_config = LoraConfig(task_type="SEQ_CLS",
                                 target_modules=["q_proj","k_proj","v_proj","o_proj"],
                                 bias="none",r=8,lora_dropout=0.1, 
                                 lora_alpha=16)
    else:
        peft_config = LoraConfig(task_type="SEQ_CLS",target_modules=['query','value','key'],bias="none",r=4,lora_dropout=0.05, lora_alpha=16)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,padding_side=padding_side, use_auth_token=False)

    label_to_id = {False: 0, True: 1}
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=len(label_to_id), use_auth_token=False)
    train_dataset = clinicalDataset(train, tokenizer)
    val_dataset = clinicalDataset(val, tokenizer)
    # Use the collate_fn in your DataLoaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    mmm=(pretrained_model_name.split("/"))[-1]
    print("Initializing the PEFT parameters!")
    model=parallel_advanced_initialization(model,columns_unique_labels_of_tabular_features,mmm, pretrained_model_name)
    model.save_pretrained(f"init_models/{mmm}/adv_init_model")
    torch.cuda.empty_cache()
    gc.collect()

    optimizer = AdamW(params=model.parameters(), lr=lr,weight_decay=0.01)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.06* (len(train_dataloader) * num_epochs),
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    model.to(device)
    f1_metric = load('f1', config_name='multiclass', average='weighted')
    accuracy_metric = load('accuracy')
    precision_metric = load('precision')
    recall_metric=load('recall')
    # Assuming binary classification, accumulate predictions and true labels
    all_predictions = []
    all_references = []
    all_scores = []  # For AUROC and AUPRC
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(tqdm_notebook(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for step, batch in enumerate(tqdm_notebook(val_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            # Assuming outputs.logits are raw scores for each class
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()  # Get probability for class '1'
            predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            references = batch["labels"].cpu().numpy()

            all_scores.extend(scores)
            all_predictions.extend(predictions)
            all_references.extend(references)

            # Your existing metric updates here
            accuracy_metric.add_batch(predictions=predictions, references=references)
            f1_metric.add_batch(predictions=predictions, references=references)
            recall_metric.add_batch(predictions=predictions, references=references)
            precision_metric.add_batch(predictions=predictions, references=references)

        # Compute final metric values
        final_accuracy = accuracy_metric.compute()
        final_f1 = f1_metric.compute()
        final_recall = recall_metric.compute()
        final_precision = precision_metric.compute()

        # Calculate AUROC and AUPRC
        final_auroc = roc_auc_score(all_references, all_scores)
        final_auprc = average_precision_score(all_references, all_scores)

        # Output the metrics
        print("="*20)
        print("VALIDATION METRICS:")
        print(f"Accuracy: {final_accuracy['accuracy']}")
        print(f"Precision: {final_precision['precision']}")
        print(f"Recall: {final_recall['recall']}")
        print(f"F1 Score: {final_f1['f1']}")
        print(f"AUROC: {final_auroc}")
        print(f"AUPRC: {final_auprc}")

    # Save your model and tokenizer
    model.save_pretrained(f"trained_models/{new_model_name}")
    tokenizer.save_pretrained(f"trained_models/{new_model_name}")

    shutil.rmtree("pretrained_model", ignore_errors=True)

    return model, tokenizer



