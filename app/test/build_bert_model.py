import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from app.common.logger import logger
from app.scripts.init_kb import load_multiclass_docs


# -------------------------- 配置参数 --------------------------
class Config:
    DATA_ROOT = r"app/data/init_docs"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MODEL_NAME = r"D:/py_models/huawei-noah_TinyBERT_General_4L_312D"   #改为本地模型地址
    SAVE_DIR = "app/modules/retrieval/model/bert_multiclass_final"
    MAX_LENGTH = 128
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 8
    EPOCHS = 5
    WEIGHT_DECAY = 0.01
    LOG_DIR = "logs/bert_training"


# -------------------------- 数据准备 --------------------------
def prepare_dataset():
    logger.info(f"从{Config.DATA_ROOT}加载多分类文档...")
    docs = load_multiclass_docs(Config.DATA_ROOT)

    if not docs:
        raise ValueError("未加载到任何文档，请检查DATA目录是否正确")

    train_data = []
    for doc in docs:
        text_sample = doc.text[:512].strip() if doc.text else ""
        if not text_sample:
            continue
        train_data.append({
            "text": text_sample,
            "label": doc.metadata["category"]
        })

    label_series = pd.Series([d['label'] for d in train_data])
    label_distribution = label_series.value_counts().to_dict()
    logger.info(f"构造完成{len(train_data)}条样本，类别分布：{label_distribution}")

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        [d["text"] for d in train_data],
        [d["label"] for d in train_data],
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=[d["label"] for d in train_data]
    )

    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
        "test": Dataset.from_dict({"text": test_texts, "label": test_labels})
    })

    return dataset


# -------------------------- 模型与训练（修复参数警告） --------------------------
def train():
    dataset = prepare_dataset()
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    labels = list(set(dataset["train"]["label"]))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    logger.info(f"类别映射：{label2id}")

    def preprocess_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=Config.MAX_LENGTH,
            padding="max_length"
        )
        tokenized["labels"] = [label2id[label] for label in examples["label"]]
        return tokenized

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
        return {
            "accuracy": accuracy_score(labels, predictions),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    model = AutoModelForSequenceClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir=Config.SAVE_DIR + "_temp",
        overwrite_output_dir=True,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        logging_dir=Config.LOG_DIR,
        logging_steps=10,
        evaluation_strategy="epoch",  # 4.54.1支持evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    # 修复FutureWarning：用processing_class替代tokenizer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        # processing_class=tokenizer,  # 替代原tokenizer参数
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    logger.info("开始训练小BERT多分类模型...（权重初始化提示为正常现象）")
    trainer.train()

    logger.info(f"训练完成，保存模型到{Config.SAVE_DIR}...")
    trainer.save_model(Config.SAVE_DIR)
    tokenizer.save_pretrained(Config.SAVE_DIR)
    logger.info("模型保存成功！")

    eval_results = trainer.evaluate()
    logger.info(f"测试集最终指标：{eval_results}")


def test_inference():
    if not os.path.exists(Config.SAVE_DIR):
        logger.warning("模型未找到，跳过推理测试")
        return

    from transformers import pipeline
    classifier = pipeline(
        "text-classification",
        model=Config.SAVE_DIR,
        tokenizer=Config.SAVE_DIR,
        device=0 if torch.cuda.is_available() else -1
    )

    test_samples = [
        "华为S5720如何配置VLAN？",
        "IP地址子网划分的方法是什么？",
        "RFC 791中定义的IP协议细节",
        "Cisco Catalyst 9300的端口配置命令"
    ]

    results = classifier(test_samples)
    for sample, res in zip(test_samples, results):
        logger.info(f"测试文本：{sample} → 预测分类：{res['label']}，置信度：{res['score']:.4f}")


if __name__ == "__main__":
    Path(Config.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    train()
    test_inference()
