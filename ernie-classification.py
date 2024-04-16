# -*- coding: utf-8 -*-
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
    train_texts, test_texts, train_labels, test_labels = train_test_split(data["text"], data["label"], test_size=test_size)
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
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def tokenize_fun(dataset, tokenizer, max_length=50):
    """ 文本处理 """
    model_inputs = tokenizer(dataset["text"], max_length=max_length, padding="max_length", truncation=True)
    return model_inputs


def train_model(model, train_dataset, val_dataset):
    """ model 训练定义 """
    # ERNIE模型中的所有参数的requires_grad属性设置为True，标记为需要计算梯度，
    # 以便进行反向传播和模型的参数更新。这通常是训练神经网络时需要执行的操作，因为通过计算梯度可以优化模型的权重和偏置，从而提高模型的准确性
    for param in model.ernie.parameters():
        param.requires_grad = True

    train_args = TrainingArguments(
        output_dir="ernie3-checkpoints",
        per_device_train_batch_size=32,
        learning_rate=5e-5,
        num_train_epochs=6,
        evaluation_strategy="epoch",
        warmup_ratio=0.2,
        save_total_limit=5,
        seed=2024,
    )

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=train_dataset, eval_dataset=val_dataset,
                      compute_metrics=compute_metrics)

    trainer.train()
    trainer.save_model(output_dir="/content/ernie3-checkpoints-save")


if __name__ == "__main__":
    # 是否使用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model tokenizer 加载
    model_path = "nghuyong/ernie-3.0-base-zh"
    num_labels = 5
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    model.to(device)

    # 数据集准备
    data_path = "/content/train.csv"
    data = data_load(data_path)

    # 数据集划分
    train_dataset, val_dataset = data_process(data, test_size=0.2)

    # 数据集转换处理
    train_dataset = train_dataset.map(tokenize_fun, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": 100})
    val_dataset = val_dataset.map(tokenize_fun, batched=True, fn_kwargs={"tokenizer": tokenizer, "max_length": 100})

    # 训练
    train_model(model, train_dataset, val_dataset)

    # 训练加载测试
    # model_reload = AutoModelForSequenceClassification.from_pretrained("/content/ernie3-checkpoints-save")

    # ----------------------------------------------------------------------------------------------------
    # # 测试使用
    # device = torch.device("cpu")
    # # 模型加载
    # # 保存tokenizer tokenizer.save_pretrained("/content/ernie3-checkpoints-save")
    # tokenizer = AutoTokenizer.from_pretrained("/content/ernie3-checkpoints-save")
    # model_reload = AutoModelForSequenceClassification.from_pretrained("/content/ernie3-checkpoints-save")
    # model_reload.to(device)
    #
    # # 待处理文本
    # word_tokenize = tokenizer("中国羽毛球队的小型庆功会就此拉开帷幕",return_tensors="pt")
    # input_ids=word_tokenize["input_ids"],
    # attention_mask=word_tokenize["attention_mask"]
    #
    # # 预测
    # pred = model_reload(input_ids=input_ids, attention_mask=attention_mask)
    # # 标签索引
    # np.argmax(pred.logits.detach().numpy(), axis=1)
