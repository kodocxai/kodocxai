"""원고 3.2절 코퍼스 단위 통계 — 유일 전거 (2026-08-20 신설).

08-15 산출이 1회용 스크립트라 소실되어 원고 절대치(어절·서브워드·형태소 수)의
전거가 없던 것을 바로잡는다. 이 스크립트의 출력이 원고 3.2절 수치의 유일 출처다.

정의(원고 서술과 1:1):
  - 어절     = 벤치마크 300건 docs.jsonl 의 words 전체. 빈 문자열은 토큰이 없어 제외.
  - 서브워드 = 어절별 tokenizer.tokenize(word). 독립 '▁' 토큰도 센다
               (원고 예시 ['▁','740','-5','번','지가'] = 5개와 동일 규칙).
  - 형태소   = kodx_units.Tagger(python-mecab-ko + mecab-ko-dic) 어절별 분해.
  - 경계 가로지름 = 문자 구간이 형태소 2개 이상과 겹치는 서브워드
               ('▁' 등 문자 구간이 없는 토큰은 판정 대상 아님 - 분모에는 포함).
  - 분할 형태소 = 서브워드 2개 이상과 겹치는 형태소.
  - 범주     = kodx_units.Morph 의 content/functional/mixed (전 성분 기준).

    D:\kodx-venv\Scripts\python.exe report_units_stats.py
      → results\단위통계_실측.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
from kodx_units import Tagger  # noqa: E402

PILOT = Path(r"D:\AIHub데이터\_pilot")
OUT = EXP / "results" / "단위통계_실측.md"


def word_token_spans(tok, word: str):
    """어절 하나를 tokenize 하고 각 토큰의 (문자 시작, 끝)을 복원한다.

    SentencePiece 조각에서 '▁'를 벗긴 표면형을 앞에서부터 되찾는다.
    표면형이 없으면(독립 '▁', <unk> 등) span 은 None.
    """
    pieces = tok.tokenize(word)
    spans = []
    cur = 0
    for p in pieces:
        surf = p.replace("▁", "")
        if not surf:
            spans.append(None)
            continue
        pos = word.find(surf, cur)
        if pos < 0:
            # 복원 실패(<unk> 등): 남은 구간 전체로 간주
            spans.append((cur, len(word)) if cur < len(word) else None)
            cur = len(word)
            continue
        spans.append((pos, pos + len(surf)))
        cur = pos + len(surf)
    return pieces, spans


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("microsoft/layoutxlm-base")
    tagger = Tagger()
    assert tagger.available, "MeCab-ko 필요"

    n_doc = 0
    n_eojeol = 0          # 비어 있지 않은 어절
    n_empty = 0
    n_sub = 0             # 서브워드 전체(독립 '▁' 포함)
    n_sub_nospan = 0      # 문자 구간 없는 토큰('▁' 등)
    n_cross = 0           # 형태소 경계 가로지름
    crossed_words = 0     # 그런 서브워드를 하나 이상 포함한 어절
    n_morph = 0
    n_split = 0           # 2개 이상 서브워드로 분할된 형태소
    n_con = n_fun = n_mix = 0
    recover_fail = 0

    for line in open(PILOT / "docs.jsonl", encoding="utf-8"):
        d = json.loads(line)
        n_doc += 1
        for w in d["words"]:
            word = w["text"]
            if not word.strip():
                n_empty += 1
                continue
            n_eojeol += 1
            pieces, spans = word_token_spans(tok, word)
            n_sub += len(pieces)
            n_sub_nospan += sum(1 for s in spans if s is None)

            morphs = tagger.morphs(word, 0)
            n_morph += len(morphs)
            for m in morphs:
                if m.tag == "UNK":
                    recover_fail += 1
                if m.is_mixed:
                    n_mix += 1
                elif m.is_functional:
                    n_fun += 1
                else:
                    n_con += 1

            word_crossed = False
            for s in spans:
                if s is None:
                    continue
                ts, te = s
                k = sum(1 for m in morphs if min(te, m.end) - max(ts, m.start) > 0)
                if k >= 2:
                    n_cross += 1
                    word_crossed = True
            if word_crossed:
                crossed_words += 1

            for m in morphs:
                k = sum(1 for s in spans if s and min(s[1], m.end) - max(s[0], m.start) > 0)
                if k >= 2:
                    n_split += 1

    lines = [
        "# 원고 3.2절 단위 통계 — 실측 (자동 생성)",
        "",
        "> `_exp\\report_units_stats.py` 가 만든다. **손으로 고치지 말 것.**",
        "> 정의는 스크립트 머리 주석 참조. 이 파일이 원고 3.2절 수치의 유일 출처다.",
        "",
        f"- 문서 {n_doc}건 · 빈 어절 제외 {n_empty}건 · 표면형 복원 실패 형태소 {recover_fail}건",
        "",
        "| 항목 | 값 |",
        "|---|---:|",
        f"| 어절 | {n_eojeol:,} |",
        f"| 서브워드 | {n_sub:,} |",
        f"| 형태소 | {n_morph:,} |",
        f"| 형태소 경계를 가로지르는 서브워드 | {n_cross:,} ({n_cross/n_sub*100:.1f}%) |",
        f"| 그런 서브워드를 포함한 어절 | {crossed_words:,} ({crossed_words/n_eojeol*100:.1f}%) |",
        f"| 둘 이상 서브워드로 분할된 형태소 | {n_split:,} ({n_split/n_morph*100:.1f}%) |",
        f"| 범주: 내용 | {n_con:,} ({n_con/n_morph*100:.1f}%) |",
        f"| 범주: 기능 | {n_fun:,} ({n_fun/n_morph*100:.1f}%) |",
        f"| 범주: 혼합 | {n_mix:,} ({n_mix/n_morph*100:.1f}%) |",
        "",
        f"(문자 구간 없는 토큰 - 독립 '▁' 등: {n_sub_nospan:,}개. 서브워드 총수에는 포함,",
        "경계 판정 대상에서는 제외)",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
