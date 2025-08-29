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
    MODEL_NAME = r"D:/py_models/huawei-noah_TinyBERT_General_4L_312D"   # 本地模型地址
    SAVE_DIR = "app/modules/retrieval/model/bert_multiclass_final"
    MAX_LENGTH = 128
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 8
    EPOCHS = 5
    WEIGHT_DECAY = 0.01
    LOG_DIR = "logs/bert_training"
    # 新增：目标类别（需要提高占比的类别）
    TARGET_CATEGORIES = ["huawei_knowledge", "huawei_solutions"]  # 根据实际类别名称调整
    TARGET_RATIO = 0.8  # 目标类别总占比


# -------------------------- 数据准备（核心修改：调整类别比重） --------------------------
def prepare_dataset():
    logger.info(f"从{Config.DATA_ROOT}加载多分类文档...")
    docs = load_multiclass_docs(Config.DATA_ROOT)

    if not docs:
        raise ValueError("未加载到任何文档，请检查DATA目录是否正确")

    # 1. 原始样本收集
    raw_samples = []
    for doc in docs:
        text_sample = doc.text[:512].strip() if doc.text else ""
        if not text_sample:
            continue
        raw_samples.append({
            "text": text_sample,
            "label": doc.metadata["category"]
        })
    logger.info(f"原始样本总数：{len(raw_samples)}，原始类别分布：{pd.Series([d['label'] for d in raw_samples]).value_counts().to_dict()}")

    # 2. 分离目标类别（knowledge/solutions）和其他类别
    target_samples = [s for s in raw_samples if s["label"] in Config.TARGET_CATEGORIES]
    other_samples = [s for s in raw_samples if s["label"] not in Config.TARGET_CATEGORIES]

    if not target_samples:
        raise ValueError(f"未找到目标类别样本：{Config.TARGET_CATEGORIES}，请检查类别名称是否正确")

    # 3. 按目标比例计算样本数量（目标类别占80%）
    # 公式：目标样本数 / (目标样本数 + 其他样本数) = 0.8 → 其他样本数 = 目标样本数 * 0.2 / 0.8
    target_count = len(target_samples)
    max_other_count = int(target_count * (1 - Config.TARGET_RATIO) / Config.TARGET_RATIO)
    # 实际其他类别样本数 = min(计算值, 原始其他类别样本数)（避免超采）
    selected_other_count = min(max_other_count, len(other_samples))

    # 4. 随机采样其他类别样本（保持随机性）
    np.random.seed(Config.RANDOM_STATE)
    selected_other_samples = np.random.choice(
        other_samples,
        size=selected_other_count,
        replace=False  # 不重复采样
    ).tolist() if selected_other_count > 0 else []

    # 5. 合并样本并打乱顺序
    final_samples = target_samples + selected_other_samples
    np.random.shuffle(final_samples)  # 打乱顺序，避免类别集中

    # 6. 打印调整后的类别分布
    adjusted_distribution = pd.Series([s['label'] for s in final_samples]).value_counts().to_dict()
    target_total = sum(adjusted_distribution.get(cat, 0) for cat in Config.TARGET_CATEGORIES)
    total = len(final_samples)
    logger.info(f"调整后样本总数：{total}，目标类别总占比：{target_total/total:.2%}")
    logger.info(f"调整后类别分布：{adjusted_distribution}")

    # 7. 分割训练集和测试集（保持分层抽样）
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        [d["text"] for d in final_samples],
        [d["label"] for d in final_samples],
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=[d["label"] for d in final_samples]  # 按调整后的分布分层
    )

    dataset = DatasetDict({
        "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),
        "test": Dataset.from_dict({"text": test_texts, "label": test_labels})
    })

    return dataset


# -------------------------- 模型与训练 --------------------------
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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    logger.info("开始训练小BERT多分类模型...")
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
        "华为S5720如何配置VLAN？",  # 预期：huawei_solutions
        "IP地址子网划分的方法是什么？",  # 预期：huawei_knowledge
        "RFC 791中定义的IP协议细节",  # 预期：其他类别
        "Cisco Catalyst 9300的端口配置命令"  # 预期：其他类别
    ]

    results = classifier(test_samples)
    for sample, res in zip(test_samples, results):
        logger.info(f"测试文本：{sample} → 预测分类：{res['label']}，置信度：{res['score']:.4f}")


if __name__ == "__main__":
    Path(Config.SAVE_DIR).mkdir(parents=True, exist_ok=True)
    Path(Config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    train()
    test_inference()

