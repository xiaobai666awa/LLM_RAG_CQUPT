import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import re
from app.common.constants import HUAWEI_COMMANDS

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
    