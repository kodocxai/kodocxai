"""그림 1 — 2층 설명가능성 평가 프레임워크 개념도 (논문 2.3절).

보여야 하는 것
  ① 입력 → 모델 → 귀속 의 흐름, 그리고 평가 대상이 (모델, 귀속기법) **쌍**이라는 것
  ② 근거 정답 $R_k$ 가 KIE 주석에서 **무비용으로** 유도된다는 것 (2.1, RQ2)
  ③ 귀속 단위를 서브워드→형태소→어절로 재집계한다는 것 (2.2, RQ1)
  ④ 정량층 3지표군과 정성층 2차원, 그리고 **이 논문이 실증하는 범위**의 구분

정성층은 점선으로 그린다 — 설계만 제시하고 실증은 후속 연구이기 때문이다(2.3.4).

레이아웃 주의: 상자 안 글줄은 높이에 맞춰 배치한다. 줄 수가 늘면 높이도 같이 키워야
겹치지 않는다(`H1`/`H2` 상수).

벡터(PDF)와 미리보기(PNG)를 함께 낸다. MDPI 는 벡터를 권장한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, ScaledTranslation

OUT = Path(r"D:\AIHub데이터\_manuscript\figures")

for fam in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    try:
        matplotlib.font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

INK, GRAY = "#1a1a1a", "#8a8a8a"
BLUE, ORANGE, GREEN, PINK = "#0072B2", "#D55E00", "#009E73", "#CC79A7"

H1, H2 = 0.155, 0.26           # 글줄 1개 / 2개짜리 상자 높이
HB = 0.26                      # 아래 줄 상자 높이 - **위 줄(H2)과 같게 맞춘다**(08-17 저자
                               # 재재보정 지시: 두 줄 높이가 다르면 어색. 두 줄 제목의
                               # 최소 여백은 rk 상자의 gap 으로 확보)
LINE = 0.052                   # 글줄 간격


COND = 0.90                    # 상자 내용 글줄의 장평(가로 배율). 1.0 = 압축 없음


def ctext(ax, x, y, s, fs, color, sx=COND):
    """가운데 정렬 · 위 기준(va='top') 텍스트를 장평 `sx` 로 그린다.

    matplotlib 의 Text 는 자간·장평을 지원하지 않는다(2026-08-18 저자 지시로 확인).
    글리프를 경로(TextPath)로 바꿔 가로만 `sx` 배 압축한다 - 수식($...$)도 mathtext
    경로로 변환되므로 함께 압축된다. 앵커 이동은 ScaledTranslation 이 그리기 시점에
    데이터 좌표를 변환하므로 savefig dpi 와 무관하게 정확하다.
    """
    fp = FontProperties(family=plt.rcParams["font.family"], size=fs)
    tp = TextPath((0, 0), s, size=fs, prop=fp)
    bb = tp.get_extents()
    tr = (Affine2D().scale(sx, 1.0)
          .translate(-(bb.x0 + bb.x1) / 2 * sx, -bb.y1)   # ha=center · va=top
          .scale(1 / 72.0)                                # 포인트 → 인치
          + ax.figure.dpi_scale_trans                     # 인치 → 픽셀 (dpi 추종)
          + ScaledTranslation(x, y, ax.transData))        # 앵커는 그리기 시점에 환산
    ax.add_patch(PathPatch(tp, facecolor=color, edgecolor="none",
                           transform=tr, zorder=3))


def box(ax, x, y, w, h, title, lines, color, dashed=False, fill=True, fs=11.0, gap=LINE):
    """`gap` 은 글줄 간격. 수식이 든 줄은 세로로 커지므로 넉넉히 준다.

    제목에 `\\n` 을 넣으면 두 줄 제목이 된다 - 긴 영문 제목이 상자 폭을 넘을 때 쓴다.
    글줄 시작 위치가 제목 줄 수만큼 내려간다.
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.8, edgecolor=color,
        facecolor=(color + "14") if fill else "white",
        linestyle=(0, (5, 3)) if dashed else "solid", zorder=2))
    # 제목 아래에 글줄을 쌓되, 전체를 상자 세로 가운데에 맞춘다
    n_title = title.count("\n") + 1
    # 추가 높이는 실제 글줄 전진(fs 11 × 행간 1.2 ≈ 0.029)에 맞춘다. 0.050 이면 제목과
    # 첫 글줄 사이가 한 줄 제목 상자보다 넓어져 어색하다(08-17 저자 재재재보정 지시)
    t_extra = (n_title - 1) * 0.030           # 제목 둘째 줄부터의 추가 높이
    # gap 은 수(균일 간격) 또는 글줄 수와 같은 길이의 목록.
    # 목록이면 각 값을 **제목 꼭대기 기준 절대 오프셋**으로 쓴다 - 수식처럼 세로로
    # 큰 줄이 섞일 때 줄마다 자리를 직접 지정하기 위함(proxy 병기, 08-18).
    if isinstance(gap, (list, tuple)):
        offs = list(gap)
        block = offs[-1] + 0.060
    else:
        offs = [0.060 + t_extra + i * gap for i in range(len(lines))]
        block = 0.055 + t_extra + len(lines) * gap
    top = y + h / 2 + block / 2
    ax.text(x + w / 2, top, title, ha="center", va="top",
            fontsize=fs, fontweight="bold", color=INK, zorder=3,
            linespacing=1.2)
    for i, ln in enumerate(lines):
        # 내용 글줄은 장평 압축(ctext) - 저자 지시(08-18). 제목·각주는 압축하지 않는다
        ctext(ax, x + w / 2, top - offs[i], ln, fs=9.0, color="#3c3c3c")


def arrow(ax, p, q, color=GRAY, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color=color,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}", zorder=1))


# ── 문구 — 국문·영문 (2026-08-17 영문판 추가)
#   영문 원고가 투고본이므로 그림도 영문판이 필요하다. 좌표·색·구조는 그대로 두고
#   텍스트만 갈아 끼운다 — 두 판의 배치가 어긋나면 캡션 참조가 꼬인다.
T = {
    "ko": {
        "in": ("입력", ["문서 이미지 $D$", "OCR 토큰 $T$ · 상자 $B$"]),
        "model": ("모델 $f$", ["BIO 태깅", "예측 $\\hat{y}_k$"]),
        "attr": ("귀속 $A$", ["$\\phi_i$ (서브워드 단위)", "IG · rollout · 무작위"]),
        "ann": ("KIE 주석", ["필드 $f_k$ ↔ 값 스팬 $y_k$", "↔ 경계 상자"]),
        # (위치 기반 proxy) 병기는 3차 검수 P2 잔여(08-18 집필 요청) - 영문과 짝
        "rk": ("근거 정답 $R_k$",
               ["(위치 기반 proxy)", "$\\bigcup_{t_i \\in y_k} b_i$",
                "추가 rationale 주석 불요"]),
        "reagg": ("단위 재집계", ["서브워드 → 형태소 → 어절", "(교착어 특이성)"]),
        "target": "평가 대상 = (모델 $f$, 귀속 기법 $A$) 쌍",
        "rq2": "RQ2  KIE 주석에서 근거 정답 유도",
        "rq1": "RQ1  집계 수준",
        "quant": "정량층 — 본 논문에서 실증",
        "plaus": ("타당성 (plausibility)", ["IoU@$k$ · F1 · Hit@1  vs  $R_k$"]),
        "faith": ("충실도 (faithfulness)", ["COMP · SUFF · AOPC"]),
        "modal": ("모달별 교란 민감도", ["텍스트 · 레이아웃 · 이미지"]),
        "rq3": "RQ3  모달별 분해",
        "qual": "정성층 — 설계만 제시, 실증은 후속 연구",
        "rubric": ("루브릭 (5점 척도)", ["의미적 일관성 · 사용자 검증가능성"]),
        # 「두 층은 상보적이다」 첫 문장은 뺐다(2026-08-17 집필 검수) — 뒤 문장이
        # 층이 아니라 축(타당성·충실도) 이야기라 층/축이 섞여 읽힌다. 국·영 동일 적용.
        "foot": ("타당성과 충실도는 상관이 보장되지 않는 독립 축이므로 "
                 "합산하지 않고 분리 보고한다."),
    },
    "en": {
        "in": ("Input", ["Document image $D$", "OCR tokens $T$ · boxes $B$"]),
        "model": ("Model $f$", ["BIO tagging", "Prediction $\\hat{y}_k$"]),
        "attr": ("Attribution $A$", ["$\\phi_i$ (subword level)", "IG · rollout · random"]),
        "ann": ("KIE annotation", ["Field $f_k$ ↔ value span $y_k$", "↔ bounding boxes"]),
        # 제목이 상자 폭(0.175)을 넘어 테두리에 걸치므로 두 줄로 내린다(2026-08-17 저자 지시)
        # (location-based proxy) 병기는 3차 검수 P2 잔여(08-18 집필 요청)
        "rk": ("Ground-truth\nrationale $R_k$",
               ["(location-based proxy)", "$\\bigcup_{t_i \\in y_k} b_i$",
                "No extra rationale annotation"]),
        # 상자 폭이 좁아 화살표 양옆 공백을 빼야 「eojeol」이 잘리지 않는다
        "reagg": ("Unit re-aggregation", ["subword→morpheme→eojeol",
                                          "(agglutinative specificity)"]),
        "target": "Evaluation target = (model $f$, attribution method $A$) pair",
        # RQ2 라벨이 길면 옆의 RQ1 과 겹친다. 국문과 같은 길이로 줄인다
        "rq2": "RQ2  Rationale from KIE annotation",
        "rq1": "RQ1  Aggregation level",
        "quant": "Quantitative layer: validated in this paper",
        "plaus": ("Plausibility", ["IoU@$k$ · F1 · Hit@1  vs  $R_k$"]),
        "faith": ("Faithfulness", ["COMP · SUFF · AOPC"]),
        "modal": ("Modality-wise perturbation", ["text · layout · image"]),
        "rq3": "RQ3  Modality decomposition",
        "qual": "Qualitative layer: design only, validated in future work",
        "rubric": ("Rubric (5-point scale)",
                   ["semantic consistency · user verifiability"]),
        "foot": ("Plausibility and faithfulness are independent axes with no guaranteed "
                 "correlation, and are therefore reported separately rather than combined."),
    },
}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=("ko", "en"), default="ko")
    a = ap.parse_args()
    t = T[a.lang]
    suffix = "" if a.lang == "ko" else "_en"

    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    W = 0.175
    X1, X2, X3 = 0.020, 0.228, 0.436
    # 위 줄이 0.26 으로 커진 만큼 내린다 - 상단 「평가 대상」 문구(YT+H2+0.055)가
    # 캔버스(1.0)를 넘지 않아야 한다
    YT, YB = 0.65, 0.32                        # 위 줄 · 아래 줄

    # ── 위 줄: 입력 → 모델 → 귀속
    box(ax, X1, YT, W, H2, *t["in"], INK, fill=False)
    box(ax, X2, YT, W, H2, *t["model"], INK, fill=False)
    box(ax, X3, YT, W, H2, *t["attr"], INK, fill=False)

    # ── 아래 줄: 주석 → 근거 정답 · 재집계  (상자 높이 HB - 두 줄 제목 여백 확보)
    box(ax, X1, YB, W, HB, *t["ann"], ORANGE)
    # proxy 병기(짧은 줄)는 제목 바로 아래, 수식 뒤는 넓게 - 절대 오프셋 지정
    box(ax, X2, YB, W, HB, *t["rk"], ORANGE,
        gap=([0.062, 0.094, 0.170] if a.lang == "en" else [0.050, 0.082, 0.158]))
    box(ax, X3, YB, W, HB, *t["reagg"], GREEN)

    ax.text(X2 + W / 2, YT + H2 + 0.055, t["target"],
            ha="center", fontsize=9.6, color=GRAY, style="italic")

    arrow(ax, (X1 + W, YT + H2 / 2), (X2, YT + H2 / 2))
    arrow(ax, (X2 + W, YT + H2 / 2), (X3, YT + H2 / 2))
    arrow(ax, (X1 + W, YB + HB / 2), (X2, YB + HB / 2), color=ORANGE)
    arrow(ax, (X3 + W / 2, YT), (X3 + W / 2, YB + HB), color=GREEN)      # 귀속 → 재집계

    for x, s, c in ((X2 + W / 2, t["rq2"], ORANGE),
                    (X3 + W / 2, t["rq1"], GREEN)):
        ax.text(x, YB - 0.052, s, ha="center", fontsize=9.2, color=c, fontweight="bold")

    # ── 오른쪽: 정량층
    # PW 0.335 는 오른쪽 끝(0.655+0.335+패딩 0.012)이 캔버스(1.0)를 넘어 테두리가
    # 잘려 나갔다. 폭을 줄여 안쪽으로 들인다(2026-08-17 저자 지시).
    PX, PW = 0.655, 0.323
    ax.add_patch(FancyBboxPatch(
        (PX, 0.415), PW, 0.510, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6, edgecolor=BLUE, facecolor=BLUE + "0d", zorder=0))
    ax.text(PX + PW / 2, 0.962, t["quant"], ha="center",
            fontsize=11.0 if a.lang == "ko" else 10.2, fontweight="bold", color=BLUE)

    bx, bw = PX + 0.018, PW - 0.036
    box(ax, bx, 0.760, bw, H1, *t["plaus"], BLUE)
    # ⚠ NAOPC 를 뺐다(2026-08-17). 본 연구는 NAOPC 를 산출하지 않는다 —
    #   그림에 남아 있으면 원고 3.3.2 의 생략 서술과 정면으로 어긋난다
    box(ax, bx, 0.593, bw, H1, *t["faith"], BLUE)
    box(ax, bx, 0.432, bw, H1, *t["modal"], BLUE)
    ax.text(PX + PW / 2, 0.372, t["rq3"], ha="center",
            fontsize=9.2, color=BLUE, fontweight="bold")

    # ── 오른쪽: 정성층
    ax.add_patch(FancyBboxPatch(
        (PX, 0.070), PW, 0.215, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6, edgecolor=PINK, facecolor="white",
        linestyle=(0, (5, 3)), zorder=0))
    ax.text(PX + PW / 2, 0.315, t["qual"], ha="center",
            fontsize=10.6 if a.lang == "ko" else 9.0, fontweight="bold", color=PINK)
    box(ax, bx, 0.088, bw, 0.170, *t["rubric"], PINK, dashed=True, fill=False)

    arrow(ax, (X3 + W, YT + H2 / 2), (bx, 0.838), rad=-0.10)             # 귀속 → 타당성
    arrow(ax, (X3 + W, YB + HB * 0.75), (bx, 0.505), color=GREEN, rad=-0.10)  # 재집계 → 지표
    arrow(ax, (X2 + W, YB + HB * 0.85), (bx, 0.800), color=ORANGE, rad=-0.22)  # R_k → 타당성
    # 정량층 → 정성층 화살표는 넣지 않는다. RQ3 표시와 정성층 제목 사이를 지나며
    # 둘 다 가리는데, 두 층의 관계는 아래 각주 한 줄로 충분하다.

    ax.text(0.5, 0.020, t["foot"], ha="center",
            fontsize=8.8 if a.lang == "ko" else 8.0, color="#6e6e6e")

    OUT.mkdir(parents=True, exist_ok=True)
    # PNG 는 600 dpi — 저자 요청(2026-08-17, 고해상 출력). PDF 는 벡터라 dpi 무관.
    for ext in ("pdf", "png"):
        p = OUT / f"figure1_framework{suffix}.{ext}"
        fig.savefig(p, dpi=600, bbox_inches="tight", facecolor="white")
        print(f"생성: {p}")


if __name__ == "__main__":
    main()
