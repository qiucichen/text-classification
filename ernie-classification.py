# -*- coding: utf-8 -*-            
# Author: XieCheng
# datetime: 2026/10
# software: PyCharm


"""
description: 基于pytorch的使用ernie预训练模型进行文本分类

pip install datasets transformers
pip install accelerate -U
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


def data_load(data_path):
    """ 待训练数据集加载 """
    data = pd.read_csv(data_path, encoding="utf8")
    return data


def data_process(data, test_size=0.2):
    """ 数据集划分与转换 """
    # 训练集、测试集划分
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data["text"], data["label"], test_size=test_size,  random_state=2024, stratify=data["label"]
    )
    # 数据集转换
    train_dict = {"text": train_texts, "labels": train_labels}
    test_dict = {"text": test_texts, "labels": test_labels}
    train_dataset = Dataset.from_dict(train_dict)
    val_dataset = Dataset.from_dict(test_dict)
    return train_dataset, val_dataset


def compute_metrics(pred):
    """ 训练过程输出评估项 """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    weighted_f1 = f1_score(
        labels, preds, average="weighted", zero_division=0
    )
    return {"accuracy": acc, "macro_f1": f1, "macro_precision": precision, "macro_recall": recall, "weighted_f1": weighted_f1,}


def tokenize_fun(dataset, tokenizer, max_length=50):
    """ 文本处理 """
    model_inputs = tokenizer(dataset["text"], max_length=max_length, padding="max_length", truncation=True)
    return model_inputs


def train_model(model, train_dataset, val_dataset, output_dir="./ernie3-checkpoints-save"):
    """ model 训练定义 """
    # ERNIE模型中的所有参数的requires_grad属性设置为True，标记为需要计算梯度，
    # 以便进行反向传播和模型的参数更新。这通常是训练神经网络时需要执行的操作，因为通过计算梯度可以优化模型的权重和偏置，从而提高模型的准确性
    for param in model.ernie.parameters():
        param.requires_grad = True

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        num_train_epochs=6,
        eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_strategy="steps", logging_steps=10,
        warmup_steps=10,  weight_decay=0.01,
        save_total_limit=5, seed=1234,  data_seed=1234,
    )

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset, eval_dataset=val_dataset,
                      compute_metrics=compute_metrics)

    trainer.train()
    tokenizer.save_pretrained(save_directory=output_dir)
    trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    # 是否使用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model tokenizer 加载
    model_path = "/home/tllm01/models/ernie-3.0-base-zh"
    num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.to(device)

    # 数据集准备
    data_path = "./data_all.csv"
    data = data_load(data_path)

    # 数据集划分
    train_dataset, val_dataset = data_process(data, test_size=0.2)

    # 数据集转换处理
    train_dataset = train_dataset.map(tokenize_fun, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": 100})
    val_dataset = val_dataset.map(tokenize_fun, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": 100})

    # 训练
    train_model(model, train_dataset, val_dataset, output_dir="./ernie3-checkpoints-save-v2")
