"""실험 결과 집계 — 원고 표 3·4·5, 그림 3, 괴리 사례 후보.

`_exp\\results\\seed{N}.jsonl`(전체 300건, NAOPC 생략)과 `seed{N}_naopc.jsonl`(부분표본
50건, NAOPC 포함)을 읽어 원고에 그대로 옮길 표를 만든다. **수치는 이 스크립트 출력에서만
가져온다. 손으로 계산 금지** — 주석 통계의 `report_kodx.py` 와 같은 규칙이다.

집계 규칙 (원고에 명시할 것)
  - 문서·필드를 가로질러 평균한 뒤, **시드 3회의 평균±표준편차**로 보고한다.
    시드 내에서 먼저 평균하고 시드 간 편차를 내야 「시드 3회 실행의 평균±표준편차」라는
    2.5절 서술과 일치한다. 전체 기록을 한 번에 평균하면 문서 수가 많은 시드에 가중된다
  - `None` 은 분모에서 뺀다. 정답이 없는 필드에서 타당성이 정의되지 않기 때문이다
  - NAOPC 는 부분표본 50건에서만 산출되므로 **표 3에 표본 크기를 함께 적는다**

사용
    python report_exp.py                    # → _exp\results\실험결과.md + 그림 3
    python report_exp.py --results <경로>   # 다른 디렉터리
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

EXP = Path(r"D:\AIHub데이터\_exp")
FIG = Path(r"D:\AIHub데이터\_manuscript\figures")
FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]
LEVELS = ["subword", "morpheme", "eojeol"]
LEVEL_KO = {"subword": "서브워드", "morpheme": "형태소", "eojeol": "어절"}
MASKS = ["all", "content", "functional"]
MASK_KO = {"all": "전체", "content": "내용", "functional": "기능"}
METHODS = ["ig", "rollout", "random"]
METHOD_KO = {"ig": "IG", "rollout": "rollout", "random": "무작위(하한)"}
KS = ["1", "3", "5", "10"]


def load(results: Path, naopc: bool, v2: bool = True) -> tuple[dict, bool]:
    """{seed: [기록]}, v2 여부. 파일이 없으면 건너뛴다 — 실험 도중에도 돌릴 수 있다.

    **v2 = 정답 스팬 보호 정의**(2026-08-15). v1 은 정답을 마스킹해 충실도가 타당성의
    복사본이 됐다(COMP@10 0.881 ≈ recall@10 0.944). v2 파일이 있으면 그것을 쓰고,
    없으면 v1 으로 물러난다. **어느 쪽을 썼는지 보고서 머리에 반드시 적는다.**
    """
    for use_v2 in ((True, False) if v2 else (False,)):
        out = {}
        for seed in (1, 2, 3):
            name = (f"seed{seed}_naopc.jsonl" if naopc
                    else (f"seed{seed}_v2.jsonl" if use_v2 else f"seed{seed}.jsonl"))
            p = results / name
            if not p.exists():
                continue
            recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            if recs:
                out[seed] = recs
        if out:
            # **미완성 시드를 버린다.** agg() 는 시드별 평균을 다시 평균하므로,
            # 4건짜리 시드가 299건짜리 시드와 같은 무게를 갖는다. 실행 도중에
            # 돌려 보면 표가 통째로 왜곡된다. 최대 문서 수의 절반 미만이면 뺀다.
            n = {s: len({r["doc_id"] for r in v}) for s, v in out.items()}
            top = max(n.values())
            drop = [s for s, c in n.items() if c < top * 0.5]
            for s in drop:
                del out[s]
            if drop:
                print(f"  미완성 시드 제외: {drop} (문서 수 {[n[s] for s in drop]} < {top}의 절반)")
            return out, use_v2
    return {}, False


def agg(by_seed: dict, keyf, valf) -> dict:
    """{키: (시드평균들의 평균, 표준편차, 시드수, 표본수)}"""
    per_seed: dict = defaultdict(dict)
    for seed, recs in by_seed.items():
        acc: dict = defaultdict(list)
        for r in recs:
            v = valf(r)
            if v is None:
                continue
            acc[keyf(r)].append(v)
        for k, vs in acc.items():
            per_seed[k][seed] = (st.mean(vs), len(vs))
    out = {}
    for k, d in per_seed.items():
        means = [m for m, _ in d.values()]
        n = sum(c for _, c in d.values())
        out[k] = (st.mean(means), st.pstdev(means) if len(means) > 1 else 0.0, len(means), n)
    return out


def fmt(t, digits=3) -> str:
    if t is None:
        return "—"
    m, s, n_seed, _ = t
    return f"{m:.{digits}f} ± {s:.{digits}f}" if n_seed > 1 else f"{m:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=EXP / "results")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out_path = a.out or (a.results / "실험결과.md")

    full, is_v2 = load(a.results, naopc=False)
    sub, _ = load(a.results, naopc=True)
    if not full and not sub:
        print(f"결과 파일이 없다: {a.results}")
        return

    L: list[str] = []
    add = L.append
    add("# 실험 결과 — 표 3·4·5 (자동 생성)")
    add("")
    add("> `_tools\\report_exp.py` 가 만든다. **손으로 고치지 말 것.**")
    add("> 시드 내에서 문서·필드를 평균한 뒤 시드 간 평균±표준편차를 낸다.")
    add("")
    if is_v2:
        add("## ✅ 정답 스팬 보호 정의 (v2)")
        add("")
        add("텍스트 마스킹에서 **정답 스팬을 보호**한다 — 순위에서 빼고 다음 순위로 $k$ 를")
        add("채우며, SUFF 에서는 보호 집합을 남긴다. 모달 교란의 대상은 **근거 영역**이고")
        add("지터는 δ 세 값으로 잰다. 원고에 실을 값은 이쪽이다.")
        add("")
    else:
        add("## ⚠ 구 정의 (v1) — 원고에 그대로 싣지 말 것")
        add("")
        add("텍스트 마스킹이 **정답 스팬까지 지운다.** KIE 는 BIO 시퀀스 라벨링이라 정답을")
        add("지우면 라벨을 못 맞히는 것이 정의상 당연하므로, 충실도가 타당성의 복사본이 된다")
        add("(실측: IG 의 COMP@10 0.881 ≈ recall@10 0.944, 82.9%에서 정답이 통째로 마스킹).")
        add("**`seed{N}_v2.jsonl` 이 생기면 자동으로 그쪽을 쓴다.**")
        add("")
    for label, d in (("전체 300건", full), ("NAOPC 부분표본 50건", sub)):
        if d:
            docs = {s: len({r["doc_id"] for r in v}) for s, v in d.items()}
            add(f"- **{label}** — 시드 {sorted(d)} · 문서 " +
                ", ".join(f"seed{s}={n}" for s, n in sorted(docs.items())) +
                f" · 기록 {sum(len(v) for v in d.values()):,}줄")
    add("")

    # ── 표 3: 집계 수준 × 마스킹 (RQ1)
    add("## 표 3. 집계 수준과 마스킹 조건에 따른 충실도 (RQ1)")
    add("")
    add("형태소 수준에서만 마스킹 조건을 나눈다. 서브워드·어절은 조건이 하나뿐이다.")
    add("**내용과 기능의 합은 전체와 같지 않다** — 어간과 어미가 결합된 혼합 형태소(2.9%)를")
    add("두 조건 모두에서 제외하기 때문이다(2.2절).")
    add("")
    have_naopc = any("naopc" in r["faith"] for v in sub.values() for r in v)
    head = "| 집계 수준 | 마스킹 | 기법 | COMP@10 | SUFF@10 | AOPC"
    add(head + (" | NAOPC |" if have_naopc else " |"))
    add("|---|---|---|---:|---:|---:" + ("|---:|" if have_naopc else "|"))

    def kf(r):
        return (r["level"], r["mask"], r["method"])

    comp = agg(full, kf, lambda r: (r["faith"].get("comp@k") or [None] * 4)[-1])
    suff = agg(full, kf, lambda r: (r["faith"].get("suff@k") or [None] * 4)[-1])
    aopc = agg(full, kf, lambda r: r["faith"].get("aopc_comp"))
    nao = agg(sub, kf, lambda r: r["faith"].get("naopc")) if have_naopc else {}

    for lv in LEVELS:
        for mk in (MASKS if lv == "morpheme" else ["all"]):
            for me in METHODS:
                k = (lv, mk, me)
                row = (f"| {LEVEL_KO[lv]} | {MASK_KO[mk]} | {METHOD_KO[me]} | "
                       f"{fmt(comp.get(k), 4)} | {fmt(suff.get(k), 4)} | {fmt(aopc.get(k), 4)} ")
                add(row + (f"| {fmt(nao.get(k), 4)} |" if have_naopc else "|"))
    add("")
    if not have_naopc:
        add("> **NAOPC 는 산출하지 않는다 (2026-08-16 저자 결정).** 두 가지 근거가 있다.")
        add("> ① **결론을 바꾸지 않는다.** NAOPC 의 상한·하한은 (문서, 필드)에만 의존하므로")
        add("> 귀속 기법·집계 수준·마스킹 조건에 걸쳐 **같은 분모**로 작용한다. 본 연구의 비교는")
        add("> 모두 단일 모델 내부이므로 정규화가 순위를 바꿀 수 없다.")
        add("> ② **문서 단위에서 계산 비용이 과도하다.** 빔 서치가 시퀀스 길이의 제곱에 비례하는데,")
        add("> 본 조건은 후보 어절 중앙값 140·빔 5 로 **1시드에 약 399 GPU-시간**이 든다")
        add("> (순전파 37.8ms 실측 기준). 원 논문이 검증한 문장 단위 과제는 N=20~50 이다.")
        add("")

    # ── 표 4: 귀속 기법별 타당성 (RQ2)
    add("## 표 4. 귀속 기법별 타당성 (RQ2)")
    add("")
    add("근거 정답 $R_k$ 와의 공간적 일치. 어절 수준 집계 기준이며, 다른 안(案)의 어절은")
    add("제외(ignore)한 뒤 계산한다(2.4.1).")
    add("")
    add("| 기법 | IoU@1 | IoU@5 | F1@5 | hit@1 |")
    add("|---|---:|---:|---:|---:|")
    eo = {s: [r for r in v if r["level"] == "eojeol"] for s, v in full.items()}
    for me in METHODS:
        def pick(k, metric):
            return agg({s: [r for r in v if r["method"] == me] for s, v in eo.items()},
                       lambda r: me, lambda r: (r["plaus"].get(k) or {}).get(metric)).get(me)
        add(f"| {METHOD_KO[me]} | {fmt(pick('1','iou'))} | {fmt(pick('5','iou'))} | "
            f"{fmt(pick('5','f1'))} | {fmt(pick('1','hit1'))} |")
    add("")
    add("**무작위 귀속은 하한 기준선이다.** 어떤 기법도 이보다 낮으면 그 기법의 설명은")
    add("근거 영역과 무관하다는 뜻이다.")
    add("")

    # ── 표 5: 모달별 충실도 (RQ3)
    add("## 표 5. 필드별·모달별 교란 민감도 (RQ3)")
    add("")
    add("각 값은 교란 전후의 예측 확률 차이다. 클수록 그 모달에 의존한다는 뜻이다.")
    add("텍스트는 상위 10단위 마스킹(COMP@10), 레이아웃은 좌표 지터와 필드 간 교환,")
    add("이미지는 근거 영역 occlusion 이다.")
    add("")
    # 지터 폭이 여럿이면 열을 그만큼 만든다. δ 하나만 재면 「반응 없음」이 모델의
    # 성질인지 δ 가 자연 변이보다 작아서인지 구분되지 않는다(실측 변이 표준편차 66.8).
    jkeys = sorted({k for v in full.values() for r in v for k in r["faith"]
                    if k.startswith("layout_jitter")},
                   key=lambda k: int(k.split("@")[1]) if "@" in k else 0)
    jhead = " ".join(f"| 지터 δ={k.split('@')[1]}" if "@" in k else "| 지터" for k in jkeys)
    add(f"| 필드 | 텍스트 {jhead} | 레이아웃(교환) | 이미지 |")
    add("|---|---:" + "|---:" * len(jkeys) + "|---:|---:|")
    eo_ig = {s: [r for r in v if r["level"] == "eojeol" and r["method"] == "ig"]
             for s, v in full.items()}
    for f in FIELDS:
        sel = {s: [r for r in v if r["field"] == f] for s, v in eo_ig.items()}
        t = agg(sel, lambda r: f, lambda r: (r["faith"].get("comp@k") or [None] * 4)[-1]).get(f)
        js = [agg(sel, lambda r: f, (lambda kk: lambda r: r["faith"].get(kk))(k)).get(f)
              for k in jkeys]
        w = agg(sel, lambda r: f, lambda r: r["faith"].get("layout_swap")).get(f)
        o = agg(sel, lambda r: f, lambda r: r["faith"].get("image_occlusion")).get(f)
        add(f"| {f} | {fmt(t,4)} " + " ".join(f"| {fmt(x,4)}" for x in js)
            + f" | {fmt(w,4)} | {fmt(o,4)} |")
    add("")
    add("**모달별 값을 단일 지수로 합산하지 않는다**(2.3.3). 가중치 선정의 자의성이 크다.")
    add("")

    # ── 괴리 사례 후보 (그림 4용)
    add("## 타당성-충실도 괴리 사례 후보 (그림 4)")
    add("")
    add("두 축은 독립이므로 어긋나는 사례가 존재한다(2.3.5). 아래는 seed 1·IG·어절 기준")
    add("상위 후보다. **그림 4의 히트맵은 별도 렌더러가 필요하다** — 선택한 문서에 대해")
    add("귀속을 다시 계산해 이미지 위에 얹어야 한다.")
    add("")
    cand = [r for r in full.get(1, [])
            if r["level"] == "eojeol" and r["method"] == "ig"
            and (r["plaus"].get("5") or {}).get("iou") is not None
            and r["faith"].get("aopc_comp") is not None]
    hi_lo = sorted(cand, key=lambda r: -(r["plaus"]["5"]["iou"] - r["faith"]["aopc_comp"] * 10))[:5]
    lo_hi = sorted(cand, key=lambda r: (r["plaus"]["5"]["iou"] - r["faith"]["aopc_comp"] * 10))[:5]
    for title, rows in (("높은 타당성 · 낮은 충실도", hi_lo), ("낮은 타당성 · 높은 충실도", lo_hi)):
        add(f"**{title}**")
        add("")
        add("| doc_id | 필드 | IoU@5 | AOPC |")
        add("|---|---|---:|---:|")
        for r in rows:
            add(f"| {r['doc_id']} | {r['field']} | {r['plaus']['5']['iou']:.3f} | "
                f"{r['faith']['aopc_comp']:.4f} |")
        add("")

    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"생성: {out_path}")

    # ── 그림 3: 집계 수준별 AOPC (RQ1 핵심)
    try:
        make_fig3(aopc, have_naopc, nao, "ko")
        make_fig3(aopc, have_naopc, nao, "en")     # 영문 원고가 투고본이라 함께 낸다
    except Exception as e:                       # 도표는 실패해도 표는 남긴다
        print(f"그림 3 생성 실패: {type(e).__name__}: {e}")


# 그림 3 영문판 문구. 좌표·색은 국문판과 동일하게 두고 텍스트만 갈아 끼운다.
LEVEL_EN = {"subword": "Subword", "morpheme": "Morpheme", "eojeol": "Eojeol"}
MASK_EN = {"all": "all", "content": "content", "functional": "functional"}
# 범례는 「하한→무작위 기준선」 전면 개칭(08-17)을 따른다 — 2차 검수 N16·집필 검수 지적.
# ⚠ 그림 전용이다. 실험결과.md 표의 METHOD_KO 라벨은 원고 쪽에서 개칭 처리하므로 여기서
#   같이 바꾸면 무결성 대조 기준이 흔들린다.
METHOD_EN = {"ig": "IG", "rollout": "rollout", "random": "random baseline"}
METHOD_KO_FIG = {"ig": "IG", "rollout": "rollout", "random": "무작위 기준선"}


def make_fig3(aopc: dict, have_naopc: bool, nao: dict, lang: str = "ko") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for fam in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

    conds = [("subword", "all"), ("morpheme", "all"),
             ("morpheme", "content"), ("morpheme", "functional"), ("eojeol", "all")]
    LV, MK, ME = ((LEVEL_KO, MASK_KO, METHOD_KO_FIG) if lang == "ko"
                  else (LEVEL_EN, MASK_EN, METHOD_EN))
    labels = [f"{LV[l]}\n({MK[m]})" if l == "morpheme" else LV[l] for l, m in conds]
    colors = {"ig": "#0072B2", "rollout": "#D55E00", "random": "#999999"}

    src, ylab = (nao, "NAOPC") if have_naopc else (aopc, "AOPC")
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    w = 0.26
    for i, me in enumerate(METHODS):
        xs = [j + (i - 1) * w for j in range(len(conds))]
        ys = [src.get((l, m, me), (0, 0, 0, 0))[0] for l, m in conds]
        es = [src.get((l, m, me), (0, 0, 0, 0))[1] for l, m in conds]
        # capsize·테두리는 전체 품질 상향(08-18 저자 지시)으로 한 단계씩 키웠다
        ax.bar(xs, ys, w, yerr=es, capsize=4, label=ME[me],
               color=colors[me], edgecolor="white", linewidth=0.8,
               error_kw={"linewidth": 1.3})
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title(
        f"집계 수준·마스킹 조건별 {ylab} (시드 3회 평균±표준편차)" if lang == "ko"
        else f"{ylab} by aggregation level and masking condition "
             f"(mean ± SD over 3 seeds)", fontsize=11.5)
    # 위치 고정 - 자동(best)은 글자를 키우면 Eojeol 막대와 겹친다. functional 그룹
    # 위쪽이 항상 비므로 그 자리에 둔다.
    ax.legend(fontsize=10, frameon=False, loc="center", bbox_to_anchor=(0.66, 0.55))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    sfx = "" if lang == "ko" else "_en"
    # PNG 는 600 dpi — 저자 요청(2026-08-17, 고해상 출력). PDF 는 벡터라 dpi 무관.
    for ext in ("pdf", "png"):
        p = FIG / f"figure3_aggregation_level{sfx}.{ext}"
        fig.savefig(p, dpi=600, facecolor="white")
        print(f"생성: {p}")


if __name__ == "__main__":
    main()
