"""평가 지표 — 타당성과 충실도 (논문 3.3.1, 3.3.2).

두 축은 상관이 보장되지 않는 독립 축이다. 합산하지 않고 분리 보고한다(3.3.5).
"""

from __future__ import annotations

from typing import Sequence


# ---------------------------------------------------------------------------
# 3.3.1 타당성 — 귀속 상위 k가 근거 정답 영역 R_k 를 가리키는가
# ---------------------------------------------------------------------------

def box_union(boxes: Sequence[Sequence[int]]) -> list[int] | None:
    """[x,y,w,h] 목록 → 이 전부를 감싸는 **외접 사각형**.

    ⚠ 이것은 $R_k$ 가 아니다. 원고 2.1의 $R_k=\\bigcup_{t_i \\in y_k} b_i$ 는 상자들의
    합집합, 곧 **영역**이다. 외접 사각형은 상자 사이의 빈 곳까지 삼킨다 — 값이 두 줄에
    걸치면 사이에 낀 결재란이 들어오고, 실측으로 스팬 하나가 페이지 면적의 0.30% 인데
    외접 사각형은 0.47% 였다. 표시나 요약에만 쓰고, IoU 계산에는 region_iou 를 쓴다.
    """
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def iou(a: Sequence[int] | None, b: Sequence[int] | None) -> float:
    """사각형 두 개의 IoU. 단일 상자끼리 비교할 때만 쓴다."""
    if not a or not b:
        return 0.0
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _region_area(boxes: Sequence[Sequence[int]]) -> float:
    """겹침을 한 번만 세는 합집합 면적. 좌표 압축 + 세로 스캔.

    상자 수가 수십 개 수준이라 O(n²)로 충분하다.
    """
    bs = [b for b in boxes if b and b[2] > 0 and b[3] > 0]
    if not bs:
        return 0.0
    xs = sorted({v for b in bs for v in (b[0], b[0] + b[2])})
    total = 0.0
    for x0, x1 in zip(xs, xs[1:]):
        if x1 <= x0:
            continue
        # 이 세로 띠와 겹치는 상자들의 y구간을 합쳐 길이를 잰다
        ys = sorted((b[1], b[1] + b[3]) for b in bs if b[0] < x1 and b[0] + b[2] > x0)
        if not ys:
            continue
        covered, cy0, cy1 = 0.0, ys[0][0], ys[0][1]
        for y0, y1 in ys[1:]:
            if y0 > cy1:
                covered += cy1 - cy0
                cy0, cy1 = y0, y1
            else:
                cy1 = max(cy1, y1)
        covered += cy1 - cy0
        total += (x1 - x0) * covered
    return total


def region_iou(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> float:
    """상자 **집합** 두 개의 IoU. 원고 2.1의 $R_k=\\bigcup b_i$ 정의를 그대로 따른다.

    외접 사각형으로 재면 근거 정답이 실제보다 넓어져 타당성이 과대평가된다.
    한 덩어리 스팬에서도 1.6배, 서로 떨어진 두 덩어리에서는 21배까지 벌어진다.
    """
    ia = _region_area(list(a) + list(b))     # 합집합 면적
    if ia <= 0:
        return 0.0
    aa, ab = _region_area(a), _region_area(b)
    inter = aa + ab - ia
    return max(0.0, inter) / ia


def plausibility(pred_word_idx: Sequence[int],
                 gold_word_idx: Sequence[int],
                 raw_boxes: Sequence[Sequence[int]],
                 ignore_word_idx: Sequence[int] = ()) -> dict:
    """IoU@k · 집합 F1 · hit@1.

    - IoU@k : 상위 k 단위의 영역 vs 근거 정답 영역 $R_k$ 의 IoU (region_iou)
    - F1    : 어절 집합 기반. 공간 겹침이 아니라 "어느 어절을 골랐나"를 본다
    - hit@1 : 최상위 단위 1개가 근거 정답 안에 있는가 (pointing accuracy)

    `ignore_word_idx` 는 **다른 안(案)의 스팬**이다. 공문서는 한 번의 결재로 수신자별
    시행문을 여러 통 만들어(「1건 기안 다수 시행」) 한 장에 헤더 세트가 여러 벌 붙는다
    (파일럿 300건 중 69건 23.0%). 정답은 최상단 안 하나이지만, 모델이 제3안의 수신을
    짚은 것은 **틀린 것이 아니라 다른 시행문을 본 것**이므로 오답으로 세지 않는다.
    분자에도 분모에도 넣지 않고 통째로 뺀다.
    """
    gold = set(gold_word_idx)
    ign = set(ignore_word_idx) - gold          # 정답이 우선한다
    pred = [i for i in pred_word_idx if i not in ign]
    pset = set(pred)

    if not gold:
        return {"iou": None, "precision": None, "recall": None, "f1": None, "hit1": None}

    inter = len(pset & gold)
    prec = inter / len(pset) if pset else 0.0
    rec = inter / len(gold)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "iou": region_iou([raw_boxes[i] for i in pred if i < len(raw_boxes)],
                          [raw_boxes[i] for i in gold if i < len(raw_boxes)]),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "hit1": float(pred[0] in gold) if pred else 0.0,
    }


# ---------------------------------------------------------------------------
# 3.3.2 충실도 — 교란했을 때 예측 확률이 실제로 떨어지는가
# ---------------------------------------------------------------------------

def comp_suff(p_full: float, p_wo_topk: float, p_only_topk: float) -> dict:
    """COMP@k = p(x) - p(x \\ top-k),  SUFF@k = p(x) - p(top-k).

    COMP가 크면 상위 k를 지웠을 때 예측이 무너진다 → 그 근거에 의존했다는 뜻.
    SUFF가 작으면 상위 k만으로도 예측이 유지된다 → 그 근거로 충분하다는 뜻.
    """
    return {"comp": p_full - p_wo_topk, "suff": p_full - p_only_topk}


def aopc(values: Sequence[float]) -> float:
    """k를 훑으며 얻은 곡선의 면적 요약. 표준 정의대로 단순 평균이다."""
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def naopc(aopc_score: float, lower_limit: float, upper_limit: float) -> float | None:
    """스케일 편향을 보정한 AOPC — Edin et al., ACL 2025 (arXiv 2408.08137) 식 8.

        NAOPC = (AOPC - AOPC_lower) / (AOPC_upper - AOPC_lower)

    상한·하한은 그 모델이 그 입력에서 낼 수 있는 최선/최악의 교란 순서에서 나온다.
    같은 AOPC 값이라도 모델마다 이 범위가 달라 원 점수는 교차 모델 비교에
    쓸 수 없다는 것이 그 논문의 핵심 주장이다.

    한계값은 `kodx_attrib.beam_limit`(논문 Algorithm 1의 빔 서치)으로 구한다.

    ※ 논문의 실증 결과: 정규화는 **교차 모델 순위를 바꾸지만 같은 모델 안에서의
      귀속 기법 순위는 유지한다.** 따라서 한 모델 안에서 기법만 비교하는 표에는
      원 AOPC로도 충분하고, 한국어·영어 모델을 나란히 놓는 표에 NAOPC가 필요하다.
    """
    denom = upper_limit - lower_limit
    if abs(denom) < 1e-9:
        return None
    return (aopc_score - lower_limit) / denom
