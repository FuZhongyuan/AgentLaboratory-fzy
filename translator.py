"""
translator.py
---------------
独立的报告翻译模块，为 AgentLaboratory 提供统一的翻译接口。
目前实现基于 OpenAI ChatCompletion，可根据需要替换为其他翻译后端。
"""

import os
import re
from typing import Tuple, List
from inference import query_model


__all__ = [
    "translate_report",
]


def _detect_language(text: str) -> str:
    """简易语言检测，目前仅区分中文/英文。"""
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def _adjust_latex_macro(tex: str, target_lang: str) -> str:
    """根据目标语言替换/插入适当的 LaTeX 宏。"""
    zh_alias = {"zh", "zh-cn", "zh_cn", "chinese", "中文"}
    if target_lang.lower() in zh_alias:
        # 将 \documentclass{article} 替换为中文 LaTeX 宏
        tex = re.sub(r"\\documentclass\{[^}]*\}", r"\\documentclass[UTF8]{ctexart}", tex, count=1)
    else:
        # 若文件使用了 ctexart，改回 article
        tex = re.sub(r"\\documentclass\[[^]]*\]\{ctex[^}]*\}", r"\\documentclass{article}", tex, count=1)
    return tex


def _openai_translate(text: str, target_lang: str, model: str = "gpt-4o-mini", api_key: str | None = None) -> str:
    """通过 inference.query_model 调用大模型完成翻译。如果失败则返回原文。"""
    try:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY 环境变量未设置，跳过翻译。")

        system_prompt = (
            f"You are a professional academic translator. Please translate the given LaTeX/Markdown report to {target_lang}. "
            "Keep all formatting, equations, references and LaTeX commands untouched except for natural language parts."
        )

        translated = query_model(
            model_str=model,
            prompt=text,
            system_prompt=system_prompt,
            openai_api_key=api_key,
            temp=0.2,
            print_cost=False
        )

        return translated.strip()
    except Exception as e:
        print(f"[translator] 翻译失败，返回原文。错误信息: {e}")
        return text


def _split_sections(tex: str) -> List[str]:
    """将 LaTeX 文本按 \section 等结构拆分成若干段落，减少单次上下文长度。保留分隔符本身。"""
    # 找到所有 \section 标题的位置
    pattern = re.compile(r'(\\section\*?\{[^}]*\})', flags=re.MULTILINE)
    parts = pattern.split(tex)
    # pattern.split 会保留分隔符本身于奇数索引
    if len(parts) <= 1:
        return [tex]
    # 将标题与其后内容拼接形成独立段
    segments = []
    i = 0
    while i < len(parts):
        if pattern.match(parts[i]):
            # 标题
            header = parts[i]
            body = parts[i + 1] if i + 1 < len(parts) else ""
            segments.append(header + body)
            i += 2
        else:
            # 文档开头或其他无标题部分
            segments.append(parts[i])
            i += 1
    # 去除空段
    return [seg for seg in segments if seg.strip()]


def _iterative_translate_segment(original: str, target_lang: str, model: str, api_key: str) -> str:
    """对单个段落进行两轮翻译：初译 + 精修。"""
    # 第一轮：直接翻译
    first_pass = _openai_translate(original, target_lang, model=model, api_key=api_key)

    # 第二轮：带上下文的精修，让模型对比原文与初译进行改进
    system_prompt = (
        f"You are a professional academic translator. Your task is to improve an existing translation from English to {target_lang}. "
        "Keep all LaTeX commands / equations untouched. Ensure technical accuracy, academic tone, and fluent wording. "
        "Do NOT change the LaTeX structure. Output ONLY the improved translation."
    )
    improve_prompt = (
        "Here is the original LaTeX snippet:\n" + original + "\n\n" +
        "Here is the initial translation:\n" + first_pass + "\n\n" +
        "Please provide an improved translation below:"
    )
    improved = query_model(
        model_str=model,
        prompt=improve_prompt,
        system_prompt=system_prompt,
        openai_api_key=api_key,
        temp=0.3,
        print_cost=False
    ).strip()

    return improved if improved else first_pass


def _dedup_end_document(tex: str) -> str:
    """确保整篇 LaTeX 文档只保留最后一个 \end{document}，删除其他多余的出现。"""
    # 找到所有 \end{document} 的位置
    pattern = re.compile(r"\\end\{document\}")
    matches = list(pattern.finditer(tex))
    # 若出现 0 或 1 次，直接返回
    if len(matches) <= 1:
        return tex

    # 记录最后一次出现的位置
    last_start = matches[-1].start()
    # 删除最后一次之前的所有 \end{document}
    cleaned_head = pattern.sub("", tex[:last_start])
    cleaned_tex = cleaned_head + tex[last_start:]
    return cleaned_tex


def translate_report(report_text: str, target_lang: str = "中文", *, model_name: str | None = None, api_key: str | None = None) -> Tuple[str, bool]:
    """多轮高质量翻译接口，先按 section 拆分，再对每段进行两轮翻译，最后拼接与宏调整。"""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("必须显式提供 OpenAI API Key，或在环境变量 OPENAI_API_KEY 中设置。")

    model = model_name or "gpt-4o-mini"

    # 0. 快速判断是否已是目标语言
    if _detect_language(report_text) == _detect_language(target_lang):
        return report_text, False

    # 1. 拆分文档
    segments = _split_sections(report_text)

    # 2. 逐段翻译并精修
    translated_segments: List[str] = []
    for seg in segments:
        translated_seg = _iterative_translate_segment(seg, target_lang, model, api_key)
        translated_segments.append(translated_seg)

    translated_full = "\n".join(translated_segments)

    # 3. 调整 LaTeX 宏
    translated_full = _adjust_latex_macro(translated_full, target_lang)

    # 4. 移除多余的 \end{document}
    translated_full = _dedup_end_document(translated_full)

    success = translated_full.strip() != report_text.strip()

    return translated_full, success 