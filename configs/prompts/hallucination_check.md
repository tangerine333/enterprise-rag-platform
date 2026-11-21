# ============================================================
# Prompt: 幻觉检测（NLI逐句验证）
# ============================================================

system: |
  你需要判断以下回答中的每句话是否可以从提供的检索文档片段中推断出来。
  对每句话给出 entailment（蕴含）/ neutral（中立）/ contradiction（矛盾）的判断。
  如果 entailment score < 0.5，标记为可能的幻觉。

  ## 检索文档
  {% for chunk in chunks %}
  [{{ loop.index }}] {{ chunk.content }}
  {% endfor %}

  ## 待检测回答
  {{ answer }}

  ## 检测结果 (JSON)
