from enum import Enum
from typing import Dict, List

class DeviceModel(str, Enum):
    """华为网络设备型号枚举"""
    S5720 = "S5720"  # 接入层交换机
    S12700 = "S12700"  # 核心层交换机
    NE40E = "NE40E"  # 路由器
    MA5800 = "MA5800"  # OLT设备
    UNKNOWN = "unknown"  # 未知设备

class TechTag(str, Enum):
    """技术标签枚举（用于文档分类）"""
    VLAN = "交换机配置-VLAN"
    OSPF = "路由协议-OSPF"
    BGP = "路由协议-BGP"
    MPLS = "广域网-MPLS"
    QOS = "服务质量-QoS"
    SECURITY = "安全配置"
    UNKNOWN = "unknown"

# API路径常量
API_PATH: Dict[str, str] = {
    "USER_CHAT": "/api/v1/chat",
    "ADMIN_UPDATE_KB": "/api/v1/admin/update_kb",
    "ADMIN_MONITOR": "/api/v1/admin/monitor"
}

# 配置参数默认值
DEFAULT_CONFIG: Dict[str, any] = {
    "TOP_K": 5,  # 默认返回检索结果数量
    "EMBEDDING_MODEL": "BAAI/bge-large-zh-v1.5",
    "RERANKER_MODEL": "BAAI/bge-reranker-large",
    "LLM_TEMPERATURE": 0.3  # 技术场景默认温度参数
}

# 华为命令关键字（用于文本提取）
HUAWEI_COMMANDS: List[str] = [
    "system-view", "vlan", "interface", "ip address",
    "ospf", "bgp", "display", "quit", "return"
]
    