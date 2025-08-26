import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy
)
from langchain_openai import ChatOpenAI
from langchain_core.exceptions import LangChainException

from ragas.exceptions import RagasException
from langchain_huggingface import HuggingFaceEmbeddings
from app.services.rag_pipeline import RAGPipeline
from app.core.model import get_chat_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_evaluator")


# 配置参数 - 集中管理
class Config:
    TEST_QA_PATH = Path("app/data/test_cases")
    MODEL_CONFIG = {
        "type": "siliconflow",
        "name": "deepseek-ai/DeepSeek-V3"
    }
    EVALUATION_LLM = {
        "base_url": "https://api.302.ai",
        "api_key": "sk-wVdwd3IRZeVcB7FDMUch4xI1qYrYRLZiARqonowHvNPuJrwW",
        "model": "gpt-5-nano-2025-08-07"  # 替换为实际可用的模型名
    }
    TOP_K = 5
    MAX_TEST_CASES = 10  # 限制评估用例数量，可调整
    OUTPUT_DIR = Path("ragas_evaluation_results")


config = Config()


def load_test_qa_pairs(qa_path: Path) -> List[Dict]:
    """加载测试用例（包含问题、标准答案）"""
    try:
        qa_cases = []
        if not qa_path.exists():
            logger.error(f"测试用例路径不存在: {qa_path}")
            return []

        for domain_dir in qa_path.iterdir():
            if not domain_dir.is_dir():
                continue

            content_json = domain_dir / "content.json"
            if not content_json.exists():
                logger.warning(f"未找到content.json: {content_json}")
                continue

            with open(content_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                for qa in data.get("qa_pairs", []):
                    if "question" in qa and "answer" in qa:
                        qa_cases.append({
                            "question": qa["question"],
                            "ground_truth": qa["answer"],
                            "domain": domain_dir.name
                        })

        logger.info(f"已加载{len(qa_cases)}条测试用例")
        return qa_cases[:config.MAX_TEST_CASES]  # 限制数量

    except Exception as e:
        logger.error(f"加载测试用例失败: {str(e)}", exc_info=True)
        return []


def prepare_ragas_dataset(qa_cases: List[Dict], pipeline: RAGPipeline) -> Optional[Dataset]:
    """构建ragas评估所需的数据集（包含检索上下文）"""
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    try:
        # 加载模型
        model = get_chat_model(
            type=config.MODEL_CONFIG["type"],
            name=config.MODEL_CONFIG["name"]
        )
        logger.info(f"成功加载模型: {config.MODEL_CONFIG['name']}")

    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}", exc_info=True)
        return None

    for idx, case in enumerate(qa_cases, 1):
        question = case["question"]
        ground_truth = case["ground_truth"]
        logger.info(f"处理第{idx}/{len(qa_cases)}个问题: {question[:50]}...")

        try:
            # 获取检索上下文
            retrieved_docs = pipeline.retrieve_docs(question)
            contexts = [doc.get("text", "") for doc in retrieved_docs[:config.TOP_K]]

            # 生成答案
            prompt = pipeline.build_rag_prompt(
                query=question,
                history=[],
                model=config.MODEL_CONFIG
            )
            answer = model.invoke(prompt).content

            # 填充数据
            ragas_data["question"].append(question)
            ragas_data["answer"].append(answer)
            ragas_data["contexts"].append(contexts)
            ragas_data["ground_truth"].append(ground_truth)

        except LangChainException as e:
            logger.error(f"生成答案失败 (问题: {question[:30]}...): {str(e)}")
        except Exception as e:
            logger.error(f"处理用例失败: {str(e)}", exc_info=True)

    if not ragas_data["question"]:
        logger.error("没有生成有效的评估数据")
        return None

    return Dataset.from_pandas(pd.DataFrame(ragas_data))


def evaluate_with_ragas(dataset: Dataset) -> Optional[Dict]:
    """使用ragas评估检索精确率等指标"""

    try:
        # 1. 配置评估用LLM（直接使用LangChain的ChatOpenAI）
        custom_llm = ChatOpenAI(
            base_url=config.EVALUATION_LLM["base_url"],  # 302.ai的API地址
            api_key=config.EVALUATION_LLM["api_key"],    # 你的API密钥
            model=config.EVALUATION_LLM["model"],        # 302.ai支持的模型名（必填！）
            temperature=0.1,                             # 评估时建议低温度，减少随机性
            timeout=30                                   # 防止网络超时
        )
        logger.info(f"评估模型配置完成: {config.EVALUATION_LLM['model']}")

        local_embeddings = HuggingFaceEmbeddings(
            model_name=r"D:/py_models/BAAI_bge-large-zh",  # 你的本地 BGE 模型路径（raw字符串避免转义）
            model_kwargs={
                "device": "cuda"  # ✅ SentenceTransformer 支持的参数：指定设备（cuda=GPU，cpu=CPU）
                # 若没有GPU，直接写 "device": "cpu"，无需任何复杂配置
            },
            encode_kwargs={
                "normalize_embeddings": True  # ✅ BGE 模型必加：向量归一化，提升相似度计算准确性
            }
        )


        # 2. 定义评估指标（可按需扩展）
        metrics = [
            context_precision,   # 检索精确率：相关上下文占比
            context_recall,      # 检索召回率：覆盖标准答案的比例
            answer_relevancy     # 答案与问题的相关性
        ]

        # 3. 执行ragas评估（直接传LangChain模型实例）
        evaluation_result = evaluate(
            dataset=dataset,      # ragas数据集（包含question/contexts/ground_truth）
            metrics=metrics,      # 要评估的指标列表
            embeddings=local_embeddings,
            llm=custom_llm,       # 直接使用LangChain的LLM，无需包装
            show_progress=True,
            batch_size = 2,
            raise_exceptions = False
        )

        if evaluation_result is None or not hasattr(evaluation_result, "scores"):
            logger.error("❌ ragas 评估未返回有效结果！")
            return None

        logger.info("✅ ragas 评估完成，scores 类型: %s", type(evaluation_result.scores))
        return evaluation_result

    except Exception as e:
        logger.error(f"评估过程失败: {str(e)}", exc_info=True)
        return None


def save_evaluation_results(result) -> None:
    """保存评估结果到文件（修复KeyError）"""
    try:
        # 关键校验：确保 result 是有效评估结果，且 scores 是字典
        if not hasattr(result, "scores") or not isinstance(result.scores, dict):
            logger.error("❌ 无效评估结果：缺少 scores 属性，或 scores 不是字典！")
            return  # 直接终止，避免后续报错

        config.OUTPUT_DIR.mkdir(exist_ok=True)

        # 1. 保存汇总指标（用官方to_dict()方法）
        summary_path = config.OUTPUT_DIR / "summary.json"
        # 方式1：获取完整结果（含指标、用例数、模型信息）
        # full_result_dict = result.model_dump()
        scores = result.scores  # 已通过校验，确保是字典

        # 元数据（从 result 属性提取）
        metadata = {
            "num_examples": result.num_examples,  # 测试用例总数
            "model": result.model,  # 评估用 LLM 名称
            "evaluation_time": result.evaluation_time,  # 评估耗时（秒）
            "successful_examples": result.successful_examples,  # 成功用例数
            "failed_examples": result.failed_examples  # 失败用例数
        }

        # 合并为完整结果（字典解包，确保无类型冲突）
        full_result_dict = {**scores, **metadata}

        print(f"评估结果字典结构：{full_result_dict.keys()}")
        # 方式2：仅获取指标得分（简化版）
        # score_dict = {metric.name: score for metric, score in result.scores.items()}

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(full_result_dict, f, ensure_ascii=False, indent=2)  # 用to_dict()结果

        # 2. 保存详细用例得分（to_pandas()正常，无需修改）
        detailed_path = config.OUTPUT_DIR / "detailed_results.csv"
        result.to_pandas().to_csv(
            detailed_path,
            index=False,
            encoding="utf-8"
        )

        logger.info(f"评估结果已保存至: {config.OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"保存评估结果失败: {str(e)}", exc_info=True)


def main():
    try:
        # 初始化RAG管道
        rag_pipeline = RAGPipeline()
        logger.info("RAG管道初始化完成")

        # 加载测试用例
        test_cases = load_test_qa_pairs(config.TEST_QA_PATH)
        if not test_cases:
            logger.error("未找到有效测试用例，终止评估")
            return

        # 准备评估数据
        logger.info("开始准备评估数据...")
        ragas_dataset = prepare_ragas_dataset(test_cases, rag_pipeline)
        if ragas_dataset is None:
            logger.error("无法生成评估数据集，终止评估")
            return

        # 执行评估
        logger.info("开始ragas评估...")
        evaluation_result = evaluate_with_ragas(ragas_dataset)
        if evaluation_result is None:
            logger.error("评估未返回有效结果")
            return

        # 输出并保存结果
        logger.info("\n===== 检索指标评估结果 =====")
        logger.info(evaluation_result)
        save_evaluation_results(evaluation_result)

    except Exception as e:
        logger.critical(f"评估程序发生致命错误: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
