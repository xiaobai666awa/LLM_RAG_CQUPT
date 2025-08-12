import re
import hashlib
from typing import List, Set

class TextCleaner:
    """
    文本清洗工具，针对华为技术文档特点优化：
    - 统一命令格式（如system-view → system-view）
    - 去除冗余空行和空格
    - 处理特殊字符（如HTML转义符）
    - 去重重复内容片段
    """
    
    def __init__(self):
        # 技术文档常见冗余模式
        self.empty_line_pattern = re.compile(r'\n\s*\n')  # 连续空行
        self.multi_space_pattern = re.compile(r' {2,}')  # 多个空格
        self.command_pattern = re.compile(r'`([^`]+)`')  # 命令标记（如`system-view`）
        self.html_escape_pattern = re.compile(r'&[a-z]+;')  # HTML转义符
    
    def clean(self, text: str) -> str:
        """
        完整清洗流程：处理格式→去除冗余→统一命令格式
        :param text: 原始文本（来自LlamaIndex Node）
        :return: 清洗后的文本
        """
        if not text:
            return ""
            
        # 1. 处理HTML转义符（如&amp; → &）
        cleaned = self.html_escape_pattern.sub('', text)
        
        # 2. 统一命令格式（去除`包裹，保持命令原样）
        cleaned = self.command_pattern.sub(r'\1', cleaned)
        
        # 3. 处理空行和空格
        cleaned = self.empty_line_pattern.sub('\n\n', cleaned)  # 连续空行→单个空行
        cleaned = self.multi_space_pattern.sub(' ', cleaned)  # 多个空格→单个空格
        
        # 4. 去除首尾空白
        cleaned = cleaned.strip()
        
        # 5. 特殊处理华为命令格式（确保命令缩进一致）
        cleaned = self._normalize_command_indent(cleaned)
        
        return cleaned
    
    def _normalize_command_indent(self, text: str) -> str:
        """
        统一华为命令的缩进格式
        例如：
        system-view
          vlan 10 → 转换为 → system-view
                          vlan 10
        """
        lines = text.split('\n')
        normalized_lines = []
        for line in lines:
            # 去除行首多余空格，保留一个制表位（4空格）的缩进
            stripped = line.lstrip()
            if not stripped:
                normalized_lines.append('')
                continue
                
            indent_count = len(line) - len(stripped)
            if indent_count > 0:
                normalized_lines.append('    ' + stripped)  # 统一为4空格缩进
            else:
                normalized_lines.append(stripped)
        return '\n'.join(normalized_lines)
    
    def deduplicate(self, texts: List[str], min_length: int = 10) -> List[str]:
        """
        去除重复文本片段（针对批量文档）
        :param texts: 文本列表
        :param min_length: 最小文本长度（低于此值不参与去重）
        :return: 去重后的文本列表
        """
        seen_hashes: Set[str] = set()
        unique_texts = []
        
        for text in texts:
            # 短文本不参与去重（可能是通用术语）
            if len(text) < min_length:
                unique_texts.append(text)
                continue
                
            # 计算文本哈希（忽略空格和换行的差异）
            text_for_hash = re.sub(r'\s+', '', text).lower()
            text_hash = hashlib.md5(text_for_hash.encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)
        
        return unique_texts
    
    def extract_commands(self, text: str) -> List[str]:
        """
        从文本中提取华为设备命令（辅助功能）
        :param text: 清洗后的文本
        :return: 命令列表（去重后）
        """
        # 华为命令特征：通常为小写字母+连字符/空格，可能带参数
        command_pattern = re.compile(r'(?:^|\n)([a-z0-9\- ]+?)(?:$|\n| )')
        commands = command_pattern.findall(text)
        
        # 过滤无效命令（太短或包含中文）
        valid_commands = []
        for cmd in commands:
            cmd = cmd.strip()
            if 3 <= len(cmd) <= 100 and not re.search(r'[\u4e00-\u9fa5]', cmd):
                valid_commands.append(cmd)
        
        # 去重并保留顺序
        seen = set()
        return [cmd for cmd in valid_commands if not (cmd in seen or seen.add(cmd))]
