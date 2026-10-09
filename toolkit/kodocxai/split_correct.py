"""정확 예측 조건부 설명 지표 - 외부 검토 chatGPT #5 후반부 대응.

묻는 것: **표 3·4 의 낮은 값이 「설명이 나쁜 것」인가 「예측이 나빠서」인가.**

지금 표 3·4 는 인스턴스 1,166건을 정오 구분 없이 평균한다. 모델이 엉뚱한 곳을 예측한
문서에서는 귀속이 그 엉뚱한 예측의 근거를 가리키므로, 정답 영역 $R_k$ 와 안 맞는 것이
당연하다. 그것까지 섞어 평균하면 **설명 품질과 모델 성능이 뒤엉킨다.**

그래서 `eval_kie.py` 가 남긴 인스턴스별 정오 딱지로 표 3·4 를 둘로 가른다.
**모델을 다시 돌리지 않는다** - 이미 있는 두 산출물을 (doc_id, field) 로 붙일 뿐이다.

기준 두 가지를 함께 본다.
    exact : 예측 스팬이 정답과 **완전히 같음** (엄격)
    hit   : 정답과 **한 어절이라도 겹침** (느슨)

사용
    python split_correct.py [--seeds 1 2 3]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RES = Path(r"D:\AIHub데이터\_exp\results")
OUT = RES / "정확예측_조건부.md"
METHODS = [("ig", "IG"), ("rollout", "rollout"), ("random", "무작위")]


def load(seed: int):
    """(문서,필드) → 정오 딱지 · 그리고 평가 기록."""
    kie = json.loads((RES / f"kie_seed{seed}.json").read_text(encoding="utf-8"))
    tag = {(r["doc_id"], r["field"]): r for r in kie["instances"]}

    rows = []
    for line in (RES / f"seed{seed}_v2.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["level"] == "eojeol" and r["mask"] == "all":
            rows.append(r)
    return tag, rows


def agg(rows, tag, method, key, want):
    """want=None 전체 · True 맞힌 것 · False 틀린 것."""
    v = []
    for r in rows:
        if r["method"] != method:
            continue
        t = tag.get((r["doc_id"], r["field"]))
        if t is None:
            continue
        if want is not None and bool(t[key]) is not want:
            continue
        x = (r["plaus"].get("5") or {}).get("iou") if key == "_iou" else None
        v.append(x)
    return v


def cell(per_seed):
    per = [x for x in per_seed if x is not None]
    if not per:
        return "-", 0
    return f"{st.mean(per):.4f} ± {st.pstdev(per):.4f}", len(per)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    a = ap.parse_args()

    data = {s: load(s) for s in a.seeds}

    L, add = [], None
    L = []; add = L.append
    add("# 정확 예측 조건부 설명 지표")
    add("")
    add("> `_tools\\split_correct.py` 가 만든다. **손으로 고치지 말 것.**")
    add("> 외부 검토(chatGPT #5 후반)의 「정확/오예측을 분리해 보고하라」에 대한 답이다.")
    add("> 모델을 다시 돌리지 않았다 - `kie_seed*.json` 의 인스턴스별 정오 딱지와")
    add("> `seed*_v2.jsonl` 을 (문서, 필드)로 붙여 재집계했다. 집계 수준은 어절·마스킹 전체.")
    add("")

    for key, kname, desc in [("exact", "스팬 완전일치", "예측 스팬이 정답과 정확히 같은 경우"),
                             ("hit", "부분 일치", "정답과 한 어절이라도 겹치는 경우")]:
        add(f"## 기준: {kname} ({desc})")
        add("")

        # 표본 수
        ns = []
        for s in a.seeds:
            tag, rows = data[s]
            ig = [r for r in rows if r["method"] == "ig"]
            ok = sum(1 for r in ig if (tag.get((r["doc_id"], r["field"])) or {}).get(key))
            ns.append((len(ig), ok))
        tot = round(st.mean(n for n, _ in ns))
        okm = round(st.mean(k for _, k in ns))
        add(f"인스턴스 {tot}건 중 **맞힌 것 {okm}건({okm/tot*100:.1f}%)** · "
            f"틀린 것 {tot-okm}건 (시드 평균)")
        add("")

        for metric, label, path in [("iou", "타당성 IoU@5", ("plaus", "5", "iou")),
                                    ("aopc", "충실도 AOPC", ("faith", "aopc_comp"))]:
            add(f"### {label}")
            add("")
            add("| 기법 | 전체 | 맞힌 것만 | 틀린 것만 |")
            add("|---|---|---|---|")
            for mk, mko in METHODS:
                cells = []
                for want in (None, True, False):
                    per = []
                    for s in a.seeds:
                        tag, rows = data[s]
                        vals = []
                        for r in rows:
                            if r["method"] != mk:
                                continue
                            t = tag.get((r["doc_id"], r["field"]))
                            if t is None:
                                continue
                            if want is not None and bool(t[key]) is not want:
                                continue
                            o = r
                            for p in path:
                                o = (o or {}).get(p) if isinstance(o, dict) else None
                            if o is not None:
                                vals.append(o)
                        if vals:
                            per.append(st.mean(vals))
                    cells.append(cell(per)[0])
                add(f"| {mko} | {cells[0]} | **{cells[1]}** | {cells[2]} |")
            add("")

    add("## 읽는 법")
    add("")
    add("**맞힌 것만 놓고 봐도 IG 가 무작위를 크게 상회하면**, 기법 간 차이가 모델 성능과")
    add("무관하게 성립한다는 뜻이다. 표 3·4 의 결론이 예측 오류에 기댄 것이 아님을 보인다.")
    add("")
    add("**틀린 것에서 타당성이 낮은 것은 당연하다** - 귀속은 모델이 실제로 한 예측의 근거를")
    add("가리키는데, 그 예측이 엉뚱한 곳이면 정답 영역과 안 맞는다. 이 열은 「설명이 나쁘다」가")
    add("아니라 **「모델이 다른 곳을 봤다」**로 읽어야 한다.")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {OUT}")


if __name__ == "__main__":
    main()
