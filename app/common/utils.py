import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import re
from app.common.constants import HUAWEI_COMMANDS
import os
import base64
import tempfile
from pathlib import Path
import requests
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv
from app.common.logger import logger

API_ENV_PATH = Path(__file__).parent.parent / "API.env"  # 路径：app/API.env
if API_ENV_PATH.exists():
    load_dotenv(API_ENV_PATH, override=True)

def generate_uuid() -> str:
    """生成唯一UUID（用于请求ID、文档ID等）"""
    return str(uuid.uuid4()).replace("-", "")

def encrypt_string(s: str, salt: str = "huawei_tech_qa") -> str:
    """字符串加密（用于敏感信息存储）"""
    sha256 = hashlib.sha256()
    sha256.update((s + salt).encode("utf-8"))
    return sha256.hexdigest()

def format_datetime(dt: Optional[datetime] = None, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化日期时间（默认当前时间）"""
    if dt is None:
        dt = datetime.now()
    return dt.strftime(fmt)

def parse_datetime(s: str, fmt: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """解析字符串为日期时间对象"""
    return datetime.strptime(s, fmt)

def is_valid_device_model(model: str) -> bool:
    """验证设备型号是否合法（基于constants.DeviceModel）"""
    from app.common.constants import DeviceModel
    return model in [m.value for m in DeviceModel]

def extract_tech_terms(text: str) -> List[str]:
    """从文本中提取华为技术术语（命令、协议名等）"""
    # 提取命令关键字
    commands = [cmd for cmd in HUAWEI_COMMANDS if cmd in text.lower()]

    # 提取设备型号（如S5720、NE40E）
    model_pattern = r"S\d{4,5}|NE\d{2,3}E|MA\d{4}"
    models = re.findall(model_pattern, text)

    # 提取协议名（如OSPF、BGP）
    protocol_pattern = r"OSPF|BGP|MPLS|VLAN|QoS"
    protocols = re.findall(protocol_pattern, text)

    # 去重并合并
    terms = list(set(commands + models + protocols))
    return [term.upper() if term in ["ospf", "bgp"] else term for term in terms]

def deep_merge_dict(a: Dict[Any, Any], b: Dict[Any, Any]) -> Dict[Any, Any]:
    """深度合并两个字典（b覆盖a中的相同键）"""
    merged = a.copy()
    for key, value in b.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


class GPT4VOCRProcessor:
    """基于GPT-4V的OCR处理器，支持图片和PDF"""

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1")
        self.model = "gpt-4-vision-preview"  # GPT-4V模型名称

        # 校验密钥
        if not self.api_key:
            raise ValueError("未在app/API.env中找到OPENAI_API_KEY，请配置后重试")

    def _encode_image(self, image_path: str) -> str:
        """将图片转换为base64编码（GPT-4V要求的输入格式）"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _call_gpt4v(self, base64_images: list,
                    prompt: str = "提取图片中的所有文字，包括表格和命令行内容，保持原始格式。") -> str:
        """调用GPT-4V API识别文字"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构造消息（支持多图输入）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in
                      base64_images]
                ]
            }
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096  # 最大输出长度（技术文档可能较长）
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60  # 超时时间设长一些（处理多页PDF）
            )
            response.raise_for_status()  # 抛出HTTP错误
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"GPT-4V API调用失败：{str(e)}")
            return ""

    def extract_text(self, file_path: str) -> str:
        """
        提取图片或PDF中的文字（核心入口）
        :param file_path: 图片（.jpg/.png）或PDF路径
        :return: 识别的文字内容
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"文件不存在：{file_path}")
            return ""

        # 1. 处理PDF（转换为图片后识别）
        if file_path.suffix.lower() == ".pdf":
            try:
                # 将PDF每页转换为图片（300dpi确保清晰度）
                images = convert_from_path(str(file_path), 300)
                base64_images = []

                # 临时保存图片并编码
                for img in images:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        img.save(tmp, "JPEG")
                        tmp_path = tmp.name
                    base64_images.append(self._encode_image(tmp_path))
                    os.remove(tmp_path)  # 清理临时文件

                # 调用GPT-4V识别多页内容
                if base64_images:
                    logger.info(f"开始识别PDF：{file_path.name}（共{len(base64_images)}页）")
                    return self._call_gpt4v(base64_images)
                return ""
            except Exception as e:
                logger.error(f"PDF处理失败：{str(e)}")
                return ""

        # 2. 处理图片（直接识别）
        elif file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                logger.info(f"开始识别图片：{file_path.name}")
                base64_img = self._encode_image(str(file_path))
                return self._call_gpt4v([base64_img])
            except Exception as e:
                logger.error(f"图片处理失败：{str(e)}")
                return ""

        # 3. 不支持的格式
        else:
            logger.warning(f"不支持的OCR文件格式：{file_path.suffix}")
            return ""