import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class LoggerConfig:
    """日志配置类，全局单例模式"""
    _instance: Optional[logging.Logger] = None

    @classmethod
    def get_logger(cls, name: str = "huawei_tech_qa") -> logging.Logger:
        """获取单例日志实例"""
        if cls._instance is None:
            cls._instance = cls._init_logger(name)
        return cls._instance

    @classmethod
    def _init_logger(cls, name: str) -> logging.Logger:
        """初始化日志配置"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # 基础级别设为DEBUG

        # 确保日志目录存在
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        # 日志格式（包含时间、模块、级别、消息）
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s"
        )

        # 控制台处理器（输出INFO及以上级别）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # 文件处理器（输出DEBUG及以上级别，按天滚动）
        today = datetime.now().strftime("%Y%m%d")
        file_handler = logging.FileHandler(f"logs/app_{today}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # 添加处理器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

# 全局日志实例（项目中直接导入使用）
logger = LoggerConfig.get_logger()
    