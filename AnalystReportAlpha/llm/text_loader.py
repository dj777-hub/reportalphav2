"""
text_loader.py — 研报文件加载器（支持 PDF / txt / md）
=======================================================
功能：
  1. 批量扫描目录下的 PDF / txt / md 文件
  2. PDF 使用 PyMuPDF (fitz) 提取正文文本
  3. 从文件名解析元信息：股票代码、名称、日期、分析师
  4. 清洗文本（过滤页眉页脚、目录等噪音）
"""

import os
import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from core.config import TEXT_REPORT_DIR

logger = logging.getLogger(__name__)

# PDF 提取可用
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    logger.warning("PyMuPDF 未安装，PDF 文件将无法解析。安装: pip install pymupdf")


@dataclass
class TextReport:
    """单份研报"""
    filename: str
    filepath: str
    content: str
    char_count: int
    load_success: bool
    error_message: str = ""


# ── 文件名解析 ──────────────────────────────

# 文件名示例：
#   东北证券 - 精测电子(300567.SZ) - 结构优化兑现盈利拐点... - 2025-12-17_张禹,李玖.pdf
#   东吴证券 - 贵州茅台(600519.SH) - 2026加速营销转型... - 2025-12-31_孙瑜,苏铖.pdf
#   华泰证券 - 公司首次覆盖：... - 2025-06-15_李明.pdf

_re_filename = re.compile(
    r".*?"                          # 券商（非贪婪）
    r"[-\s]+"                       # 分隔符
    r".*?\((\d{6}\.[A-Z]+)\)"       # 股票代码 (600519.SH)
    r"[-\s]+"                       # 分隔符
    r".*?"                          # 标题
    r"[-\s]+"                       # 分隔符
    r"(\d{4}-\d{2}-\d{2})"          # 日期 2025-12-31
    r"_"                            # 下划线
    r"([^.]*)"                      # 分析师（不含扩展名）
)


def parse_filename(filename: str) -> dict:
    """
    从文件名解析元信息。

    Returns
    -------
    dict : {stock_code, stock_name, publish_date, analyst_name, title}
    """
    result = {
        "stock_code": "",
        "stock_name": "",
        "publish_date": "",
        "analyst_name": "",
        "title": "",
    }

    name_no_ext = os.path.splitext(filename)[0]

    # 先尝试正则提取
    m = _re_filename.search(name_no_ext)
    if m:
        result["stock_code"] = m.group(1)
        result["publish_date"] = m.group(2)
        result["analyst_name"] = m.group(3)

    # 提取股票名称（代码前的名称）
    code_match = re.search(r"\((\d{6}\.[A-Z]+)\)", name_no_ext)
    if code_match:
        code = code_match.group(1)
        result["stock_code"] = code
        # 代码前的名称
        before = name_no_ext[:code_match.start()].strip()
        # 提取最后一个分隔符后的名称
        parts = re.split(r"[-\s]+", before)
        if parts:
            result["stock_name"] = parts[-1].strip()

    # 提取日期（备选）
    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", name_no_ext)
    if date_match:
        result["publish_date"] = date_match.group(1)

    # 提取分析师（日期后的部分）
    if result["publish_date"]:
        after_date = name_no_ext.split(result["publish_date"])[-1].strip().lstrip("_")
        if after_date:
            result["analyst_name"] = after_date

    return result


# ── PDF 文本提取 ────────────────────────────

def extract_pdf_text(filepath: str) -> Tuple[str, str]:
    """
    使用 PyMuPDF 提取 PDF 正文文本。

    Returns
    -------
    (full_text, error_message)
    """
    if not HAS_FITZ:
        return "", "PyMuPDF 未安装"

    try:
        doc = fitz.open(filepath)
        pages_text = []
        for page in doc:
            text = page.get_text("text")
            pages_text.append(text)
        doc.close()

        full = "\n".join(pages_text)

        # 基础清洗：过滤过短的行（页眉页脚、目录页码等）
        cleaned_lines = []
        for line in full.split("\n"):
            stripped = line.strip()
            # 跳过页码、目录标记、过短行
            if stripped.isdigit() and len(stripped) <= 4:
                continue
            if len(stripped) < 3:
                continue
            if stripped.startswith("请务必阅读"):
                continue
            if "免责声明" in stripped or "评级说明" in stripped:
                continue
            cleaned_lines.append(stripped)

        return "\n".join(cleaned_lines), ""

    except Exception as e:
        return "", f"PDF 提取失败: {e}"


# ── 加载器 ──────────────────────────────────

class TextReportLoader:
    """
    研报加载器（支持 PDF / txt / md）。

    参数
    ----------
    report_dir : str
        研报文件目录
    """

    def __init__(self, report_dir: str = TEXT_REPORT_DIR):
        self.report_dir = report_dir

    def scan_files(self) -> List[str]:
        """扫描目录下所有 .pdf / .txt / .md 文件"""
        if not os.path.isdir(self.report_dir):
            logger.warning(f"目录不存在: {self.report_dir}")
            return []
        files = []
        for f in sorted(os.listdir(self.report_dir)):
            if f.lower().endswith((".pdf", ".txt", ".md")):
                files.append(os.path.join(self.report_dir, f))
        logger.info(f"扫描到 {len(files)} 个研报文件")
        return files

    def load_single(self, filepath: str) -> TextReport:
        """读取单个文件（PDF → fitz 提取，txt/md → 直接读）"""
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()

        try:
            if ext == ".pdf":
                text, err = extract_pdf_text(filepath)
                if err:
                    return TextReport(filename=filename, filepath=filepath,
                                      content="", char_count=0, load_success=False, error_message=err)
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

            return TextReport(
                filename=filename,
                filepath=filepath,
                content=text,
                char_count=len(text),
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
        """批量加载"""
        if filepaths is None:
            filepaths = self.scan_files()
        results = []
        for fp in filepaths:
            results.append(self.load_single(fp))
        ok = sum(1 for r in results if r.load_success)
        logger.info(f"批量加载: {ok}/{len(results)} 成功")
        return results

    def preview_pdf(self, filepath: str, max_chars: int = 3000) -> str:
        """预览 PDF 提取的文本内容（用于前端调试）"""
        text, err = extract_pdf_text(filepath)
        if err:
            return f"[错误] {err}"
        return text[:max_chars]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

    loader = TextReportLoader()
    files = loader.scan_files()

    print(f"\n共 {len(files)} 个文件\n")

    # 测试文件名解析
    print("=== 文件名解析示例 ===")
    for f in files[:5]:
        info = parse_filename(os.path.basename(f))
        print(f"  {os.path.basename(f)}")
        print(f"    → {info}")
        r = loader.load_single(f)
        print(f"    → 提取: {r.char_count} 字, {'✅' if r.load_success else '❌'}")
        if not r.load_success:
            print(f"      错误: {r.error_message}")
        print()
