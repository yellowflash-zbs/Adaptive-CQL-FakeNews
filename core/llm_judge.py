# coding: utf-8
"""DeepSeek-backed structured judging helpers."""

import json
import os
import time

from core.evidence_selection import clean_spaced_text


PROMPT_VERSION = "defense_v1"


def get_deepseek_client():
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY，再运行需要 DeepSeek 的脚本。")
    return OpenAI(api_key=api_key, base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))


def label_calibration(dataset_name):
    if dataset_name == "LIAR-RAW":
        return """
[LIAR 六分类校准]
- pants-fire: 完全不准确，而且荒谬离谱。
- false: 不准确。
- barely-true: 只包含一点事实，但忽略关键事实。
- half-true: 部分准确，但遗漏重要细节或断章取义。
- mostly-true: 基本准确，但需要澄清或补充。
- true: 准确，且没有遗漏关键事实。
请避免把所有不确定样本都压到 false / barely-true；只有证据明确显示错误时才使用较低标签。
"""
    return """
[RAWFC 三分类校准]
- false: 声明不准确或虚假。
- half: 声明半真半假，夸大，或缺少关键上下文。
- true: 声明准确。
"""


def _chat_json(prompt, system, temperature=0.0, timeout=20, max_retries=3):
    client = get_deepseek_client()
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                timeout=timeout,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:
            if attempt == max_retries - 1:
                raise exc
            time.sleep(2)


def judge_stances(claim_text, selected_sentences, dataset_name):
    if not selected_sentences:
        return []
    evidence_text = "\n".join(
        f"{idx + 1}. {clean_spaced_text(sentence)}" for idx, sentence in enumerate(selected_sentences)
    )
    prompt = f"""
你是事实核查证据分析员。请逐条判断证据相对于新闻声明的立场。
{label_calibration(dataset_name)}

[新闻声明]: "{clean_spaced_text(claim_text)}"

[证据]:
{evidence_text}

每条证据只能标为 support、refute、neutral、irrelevant 之一。
请严格输出 JSON：
{{
  "sentence_stances": ["support", "refute", "neutral"]
}}
"""
    try:
        result = _chat_json(
            prompt,
            system="You classify evidence stances for fact-checking and return JSON only.",
            timeout=15,
        )
        stances = result.get("sentence_stances", [])
        return [str(s).strip().lower() for s in stances[: len(selected_sentences)]]
    except Exception:
        return ["error"] * len(selected_sentences)


def judge_verdict(claim_text, selected_sentences, dataset_name, stance_labels=None, prompt_version=PROMPT_VERSION):
    clean_claim = clean_spaced_text(claim_text)
    selected_sentences = [clean_spaced_text(s) for s in selected_sentences]
    stance_labels = stance_labels or []

    if selected_sentences:
        grouped = {"support": [], "refute": [], "neutral": [], "irrelevant": []}
        for idx, sentence in enumerate(selected_sentences):
            stance = stance_labels[idx] if idx < len(stance_labels) else "neutral"
            if stance not in grouped:
                stance = "neutral"
            grouped[stance].append(sentence)
        evidence_text = "\n".join(
            [
                "[支持方证据]\n" + "\n".join(f"- {s}" for s in grouped["support"]),
                "[反驳方证据]\n" + "\n".join(f"- {s}" for s in grouped["refute"]),
                "[中立/背景]\n" + "\n".join(f"- {s}" for s in grouped["neutral"]),
                "[无关噪音]\n" + "\n".join(f"- {s}" for s in grouped["irrelevant"]),
            ]
        )
    else:
        evidence_text = "No external evidence is available. Judge from the claim and calibrated label definitions only."

    options = "['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']"
    if dataset_name != "LIAR-RAW":
        options = "['false', 'true', 'half']"

    prompt = f"""
你是一位顶尖且谨慎的假新闻核查法官。请使用 defense-style 推理：先比较支持方与反驳方证据质量，再给出唯一判决。
{label_calibration(dataset_name)}

[Prompt Version]: {prompt_version}
[新闻声明]: "{clean_claim}"

[结构化证据]:
{evidence_text}

判决要求：
1. 不要把无关噪音当作证据。
2. 如果外部证据质量很低，请明确说明，并更多依赖声明语义与标签校准。
3. LIAR-RAW 六分类必须避免系统性偏向 false / barely-true；按“错误程度”细分。
4. 最终 prediction 必须是以下选项之一：{options}

请严格输出 JSON：
{{
  "step_by_step_analysis": "简短写出支持方、反驳方、噪音证据的权衡过程",
  "prediction": "填入唯一类别"
}}
"""
    try:
        return _chat_json(
            prompt,
            system="You are a logic-driven fact-checking judge. Return JSON only.",
            timeout=20,
        )
    except Exception as exc:
        return {"prediction": "unknown", "step_by_step_analysis": str(exc)}
