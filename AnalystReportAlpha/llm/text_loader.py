"""
text_loader.py — 研报文本文件 (txt/md) 批量加载器
===================================================
替代 pdf_parser.py，直接读取 txt/md 格式的研报文件。
支持批量扫描目录、读取文件内容、按文件名解析元信息。
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass

from core.config import TEXT_REPORT_DIR

logger = logging.getLogger(__name__)


@dataclass
class TextReport:
    """单份文本研报"""
    filename: str
    filepath: str
    content: str
    char_count: int
    load_success: bool
    error_message: str = ""


class TextReportLoader:
    """
    文本研报加载器。

    文件名建议格式：{股票代码}_{股票名称}_{日期}_{分析师}.txt
    例如：600519.SH_贵州茅台_2024-06-15_张明.txt

    这样可以从文件名自动提取元信息（回退到 LLM 提取）。
    """

    def __init__(self, report_dir: str = TEXT_REPORT_DIR):
        self.report_dir = report_dir

    def scan_files(self) -> List[str]:
        """扫描目录下所有 .txt / .md 文件"""
        if not os.path.isdir(self.report_dir):
            logger.warning(f"目录不存在: {self.report_dir}")
            return []
        files = []
        for f in sorted(os.listdir(self.report_dir)):
            if f.lower().endswith((".txt", ".md")):
                files.append(os.path.join(self.report_dir, f))
        logger.info(f"扫描到 {len(files)} 个文本研报文件")
        return files

    def load_single(self, filepath: str) -> TextReport:
        """读取单个文本文件"""
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            return TextReport(
                filename=filename,
                filepath=filepath,
                content=content,
                char_count=len(content),
                load_success=True,
            )
        except Exception as e:
            return TextReport(
                filename=filename,
                filepath=filepath,
                content="",
                char_count=0,
                load_success=False,
                error_message=str(e),
            )

    def batch_load(self, filepaths: Optional[List[str]] = None) -> List[TextReport]:
        """批量加载文本文件"""
        if filepaths is None:
            filepaths = self.scan_files()
        results = []
        for fp in filepaths:
            results.append(self.load_single(fp))
        ok = sum(1 for r in results if r.load_success)
        logger.info(f"批量加载: {ok}/{len(results)} 成功")
        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = TextReportLoader()
    files = loader.scan_files()
    for f in files[:3]:
        r = loader.load_single(f)
        print(f"  {'✅' if r.load_success else '❌'} {r.filename}: {r.char_count}字")
        if r.load_success:
            print(f"    前100字: {r.content[:100]}")
