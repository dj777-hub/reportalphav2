"""
llm_client.py — 多厂商 LLM API 封装（通义千问 / DeepSeek）
===========================================================
输入研报文本，输出结构化 JSON。
支持：本地缓存(MD5去重)、重试、超时、模型切换、厂商切换。
"""

import json, time, hashlib, logging, os
from typing import Optional, Dict, Any, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from core.config import (
    LLM_PROVIDER,
    QWEN_API_KEY, QWEN_BASE_URL, QWEN_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    LLM_TIMEOUT, LLM_MAX_RETRIES, LLM_CACHE_PATH,
)

logger = logging.getLogger(__name__)

REPORT_ANALYSIS_PROMPT = """你是一位专业的金融研报分析师。任务是从卖方研报文本中提取关键信息。

## 输出 JSON（严格）
{
    "has_positive_recommend": true,
    "target_stock_code_list": ["600519.SH", "000858.SZ"],
    "analyst_name": "张明",
    "report_publish_date": "2024-06-15",
    "reason": "研报明确给出买入评级"
}

## 规则
- has_positive_recommend：是否明确看多推荐（买入/增持/推荐/强烈推荐）
- target_stock_code_list：推荐标的代码（6位.SH/.SZ），没有则[]
- analyst_name：分析师姓名，无法识别填"未知分析师"
- report_publish_date：发布日期 YYYY-MM-DD，无法识别填"未知日期"
- reason：判断理由，不超过50字
"""


class LlmCache:
    def __init__(self, path: str = LLM_CACHE_PATH):
        self.path = path
        self._cache: Dict[str, dict] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
            except: self._cache = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=2)

    def get(self, text: str) -> Optional[dict]:
        return self._cache.get(hashlib.md5(text.encode()).hexdigest())

    def set(self, text: str, result: dict):
        self._cache[hashlib.md5(text.encode()).hexdigest()] = result

    def clear(self):
        self._cache = {}
        if os.path.exists(self.path): os.remove(self.path)


class LLMClient:
    """
    多厂商 LLM 客户端，支持 Qwen / DeepSeek。

    Parameters
    ----------
    provider : "qwen" | "deepseek"
    model : str
    use_cache : bool
    """

    def __init__(self, provider: str = "", model: str = "", use_cache: bool = True):
        self.use_cache = use_cache
        self._cache = LlmCache() if use_cache else None

        # 确定厂商和模型
        self.provider = provider or LLM_PROVIDER
        if model:
            self.model = model
        elif self.provider == "deepseek":
            self.model = DEEPSEEK_MODEL
        else:
            self.model = QWEN_MODEL

        # 构造客户端
        self._client = None
        if OpenAI is None:
            logger.warning("openai 库未安装")
            return

        if self.provider == "deepseek":
            self._client = OpenAI(
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                timeout=LLM_TIMEOUT, max_retries=0,
            )
        else:
            self._client = OpenAI(
                api_key=QWEN_API_KEY,
                base_url=QWEN_BASE_URL,
                timeout=LLM_TIMEOUT, max_retries=0,
            )

    @property
    def is_available(self) -> bool:
        return self._client is not None

    @property
    def display_name(self) -> str:
        return f"{'DeepSeek' if self.provider == 'deepseek' else '通义千问'} ({self.model})"

    def analyze_report(self, report_text: str) -> Dict[str, Any]:
        """分析单条研报。返回结构化 JSON + success/from_cache/error 字段。"""
        if not self.is_available:
            return {"has_positive_recommend": False, "target_stock_code_list": [],
                    "analyst_name": "未知分析师", "report_publish_date": "未知日期",
                    "reason": "LLM 不可用", "success": False, "from_cache": False, "error": "openai 未安装"}

        if self.use_cache and self._cache:
            cached = self._cache.get(report_text)
            if cached:
                cached["from_cache"] = True
                return cached

        text = report_text[:6000]
        messages = [
            {"role": "system", "content": REPORT_ANALYSIS_PROMPT},
            {"role": "user", "content": f"研报正文：\n{text}"},
        ]

        last_err = None
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                )
                result = json.loads(resp.choices[0].message.content.strip())
                result.setdefault("has_positive_recommend", False)
                result.setdefault("target_stock_code_list", [])
                result.setdefault("analyst_name", "未知分析师")
                result.setdefault("report_publish_date", "未知日期")
                result.setdefault("reason", "")
                result["success"] = True
                result["from_cache"] = False
                result["error"] = None
                if self.use_cache and self._cache:
                    self._cache.set(report_text, result)
                    self._cache.save()
                return result
            except Exception as e:
                last_err = str(e)
                logger.warning(f"LLM 失败(第{attempt}次): {last_err}")
                time.sleep(2 ** attempt)

        return {"has_positive_recommend": False, "target_stock_code_list": [],
                "analyst_name": "未知分析师", "report_publish_date": "未知日期",
                "reason": f"API失败:{last_err}", "success": False,
                "from_cache": False, "error": last_err}

    def batch_analyze(self, reports: List[Dict], show_progress: bool = True) -> "pd.DataFrame":
        import pandas as pd
        results = []
        total = len(reports)
        for i, item in enumerate(reports):
            result = self.analyze_report(item.get("report_text", ""))
            results.append({
                "report_id": item.get("report_id", ""),
                "filename": item.get("filename", ""),
                "analyst_name": result.get("analyst_name", "未知分析师"),
                "publish_date": result.get("report_publish_date", "未知日期"),
                "stock_code_list": json.dumps(result.get("target_stock_code_list", []), ensure_ascii=False),
                "report_content": item.get("report_text", "")[:500],
                "has_positive_recommend": result.get("has_positive_recommend", False),
                "reason": result.get("reason", ""),
                "llm_success": result.get("success", False),
                "from_cache": result.get("from_cache", False),
            })
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"LLM 进度: {i + 1}/{total}")
        return pd.DataFrame(results)

    def debug_analyze(self, report_text: str) -> Dict:
        start = time.time()
        r = self.analyze_report(report_text)
        r["debug_info"] = {
            "provider": self.provider, "model": self.model,
            "elapsed_seconds": round(time.time() - start, 2),
            "input_length": len(report_text),
            "cache_hit": r.get("from_cache", False),
        }
        return r

    def generate_text(self, prompt: str, system_prompt: str = "", temperature: float = 0.7) -> str:
        """生成文本（不强制 JSON 格式），用于报告生成等自由文本场景"""
        if not self.is_available:
            return "LLM 不可用"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                # 不使用 response_format，允许自由文本输出
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, max_tokens=4096,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = str(e)
                logger.warning(f"LLM 文本生成失败(第{attempt}次): {last_err}")
                time.sleep(2 ** attempt)
        return f"生成失败: {last_err}"

    def clear_cache(self):
        if self._cache: self._cache.clear()


def create_llm_client(provider: str = "", model: str = "", use_cache: bool = True) -> LLMClient:
    return LLMClient(provider=provider, model=model, use_cache=use_cache)
