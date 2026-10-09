"""귀속 기법과 모달별 교란 (논문 3.3.2, 3.3.3).

귀속은 텍스트 임베딩 위에서 계산한다. 타당성이 OCR 박스를 경유해 계산되므로
귀속 단위는 텍스트 토큰이어야 한다(3.3.1).

교란은 세 모달을 각각 건드린다.
  텍스트   : 단위별 [MASK] 대체 (분포내 마스킹)
  레이아웃 : 근거 영역 좌표 지터 ±delta, 필드 간 좌표 교환
  이미지   : 근거 영역 패치 occlusion
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# 예측 대상 스코어
# ---------------------------------------------------------------------------

def field_score(logits: torch.Tensor, token_idx: Sequence[int],
                label_ids: Sequence[int]) -> torch.Tensor:
    """필드 f_k 예측 스팬에 대한 스칼라 점수.

    스팬 토큰들의 해당 라벨 확률을 기하평균으로 묶는다. 합이 아니라 평균을 쓰는
    이유는 스팬 길이가 다른 필드끼리 비교되기 때문이다.
    """
    probs = logits.softmax(-1)
    picked = torch.stack([probs[0, t, l] for t, l in zip(token_idx, label_ids)])
    return picked.clamp_min(1e-12).log().mean().exp()


# ---------------------------------------------------------------------------
# 귀속 기법
# ---------------------------------------------------------------------------

def integrated_gradients(model, inputs: dict, embed_fn: Callable,
                         score_fn: Callable, steps: int = 32,
                         baseline_token_id: Optional[int] = None) -> torch.Tensor:
    """Integrated Gradients — 텍스트 임베딩 경로.

    baseline은 [MASK] 임베딩을 쓴다. 영벡터 baseline은 학습 분포 밖이라
    3.3.2에서 지적한 OOD 함정을 귀속 쪽에서 그대로 재현한다.
    """
    emb = embed_fn(inputs["input_ids"]).detach()          # (1, T, H)
    if baseline_token_id is None:
        base = torch.zeros_like(emb)
    else:
        base_ids = torch.full_like(inputs["input_ids"], baseline_token_id)
        base = embed_fn(base_ids).detach()

    total = torch.zeros_like(emb)
    for i in range(1, steps + 1):
        a = i / steps
        x = (base + a * (emb - base)).clone().requires_grad_(True)
        score = score_fn(model, inputs, inputs_embeds=x)
        grad = torch.autograd.grad(score, x)[0]
        total += grad
    attr = (emb - base) * total / steps
    return attr.sum(-1)[0]                                 # (T,)


def attention_rollout(attentions: Sequence[torch.Tensor],
                      target_token: int, add_residual: bool = True) -> torch.Tensor:
    """Attention rollout — 층별 어텐션을 곱해 누적한다.

    attentions: 층별 (1, heads, T, T)
    """
    result = None
    for A in attentions:
        a = A[0].mean(0)                                   # 헤드 평균 (T, T)
        if add_residual:
            a = a + torch.eye(a.size(0), device=a.device)
            a = a / a.sum(-1, keepdim=True)
        result = a if result is None else a @ result
    return result[target_token]                            # (T,)


def random_attribution(n_tokens: int, generator: torch.Generator) -> torch.Tensor:
    """하한 기준선. 이것과 유의하게 다르지 않은 기법은 쓸모가 없다."""
    return torch.rand(n_tokens, generator=generator)


# ---------------------------------------------------------------------------
# 교란 — 텍스트
# ---------------------------------------------------------------------------

def mask_tokens(input_ids: torch.Tensor, token_idx: Sequence[int],
                mask_token_id: int) -> torch.Tensor:
    """분포내 마스킹. 토큰을 제거하지 않고 [MASK]로 바꾼다.

    제거하면 입력 길이가 변해 학습 분포 밖으로 나간다. 3.3.2에서 채택한 방식이다.
    """
    out = input_ids.clone()
    for t in token_idx:
        out[0, t] = mask_token_id
    return out


def keep_only(input_ids: torch.Tensor, token_idx: Sequence[int],
              mask_token_id: int, special_idx: Sequence[int] = ()) -> torch.Tensor:
    """SUFF용 — 상위 k와 특수토큰만 남기고 나머지를 마스킹한다."""
    keep = set(token_idx) | set(special_idx)
    out = input_ids.clone()
    for t in range(out.size(1)):
        if t not in keep:
            out[0, t] = mask_token_id
    return out


# ---------------------------------------------------------------------------
# 교란 — 레이아웃
# ---------------------------------------------------------------------------

def jitter_boxes(bbox: torch.Tensor, token_idx: Sequence[int], delta: int,
                 generator: torch.Generator) -> torch.Tensor:
    """근거 영역 좌표를 ±delta 만큼 흔든다. 좌표계는 0~1000이다."""
    out = bbox.clone()
    if not token_idx:
        return out
    # 난수는 **생성기가 있는 장치에서** 뽑고 그 뒤에 옮긴다. generator 는 재현을 위해
    # CPU 로 고정하는데, device=cuda 로 바로 뽑으면 장치가 어긋나 RuntimeError 가 난다.
    # 토큰마다 호출하지 않고 한 번에 뽑는다 — 같은 시드에서 같은 값이 나오고 더 빠르다.
    shifts = torch.randint(-delta, delta + 1, (len(token_idx), 2),
                           generator=generator).to(bbox.device)
    for k, t in enumerate(token_idx):
        out[0, t, 0::2] = (out[0, t, 0::2] + shifts[k, 0]).clamp(0, 1000)
        out[0, t, 1::2] = (out[0, t, 1::2] + shifts[k, 1]).clamp(0, 1000)
    return out


def swap_boxes(bbox: torch.Tensor, idx_a: Sequence[int],
               idx_b: Sequence[int]) -> torch.Tensor:
    """필드 간 좌표 교환. 위치 정보에만 의존하는 예측을 드러낸다."""
    out = bbox.clone()
    n = min(len(idx_a), len(idx_b))
    for i in range(n):
        a, b = idx_a[i], idx_b[i]
        out[0, a], out[0, b] = bbox[0, b].clone(), bbox[0, a].clone()
    return out


# ---------------------------------------------------------------------------
# 교란 — 이미지
# ---------------------------------------------------------------------------

def occlude_image(image: torch.Tensor, boxes_norm: Sequence[Sequence[int]],
                  fill: float = 0.0) -> torch.Tensor:
    """근거 영역에 해당하는 이미지 패치를 가린다.

    image      : (1, 3, S, S)  LayoutXLM은 S=224
    boxes_norm : 0~1000 좌표의 [x0,y0,x1,y1] 목록
    """
    out = image.clone()
    S = out.shape[-1]
    for x0, y0, x1, y1 in boxes_norm:
        a, b = int(x0 * S / 1000), int(y0 * S / 1000)
        c, d = math.ceil(x1 * S / 1000), math.ceil(y1 * S / 1000)
        if c > a and d > b:
            out[0, :, b:d, a:c] = fill
    return out


# ---------------------------------------------------------------------------
# NAOPC용 최선/최악 순서 근사
# ---------------------------------------------------------------------------

def beam_limit(eval_masked: Callable[[Sequence[int]], float],
               full_output: float, candidates: Sequence[int],
               beam_size: int = 5, mode: str = "upper",
               max_steps: Optional[int] = None) -> float:
    """AOPC의 상한/하한을 빔 서치로 찾는다 — NAOPC 논문 Algorithm 1.

    Edin et al., *Normalized AOPC*, ACL 2025 (arXiv 2408.08137)의 FindLimit을
    그대로 옮긴 것이다. 부분 순서를 빔에 유지하며 한 단위씩 늘리고, 각 단계의
    누적 점수 기준으로 상위 B개만 남긴다.

        누적점수(ord) = sum_i [ f(x) - f(p(x, ord_1:i)) ]
        AOPC 한계     = 누적점수 / N

    mode="upper"면 누적점수를 최대화(= 최선의 comprehensiveness),
    mode="lower"면 최소화한다. 논문은 beam size 5를 기본으로 쓰되
    AG-News에서는 1000이 필요했다고 보고한다. B=1이면 탐욕 탐색과 같다.

    ⚠ max_steps로 자르면 논문 정의(전체 N단계 평균)에서 벗어난다.
      비용 때문에 자를 경우 논문에 그 사실을 명시해야 한다.
    """
    if mode not in ("upper", "lower"):
        raise ValueError(mode)
    N = len(candidates)
    if N == 0:
        return 0.0
    steps = min(N, max_steps or N)

    beam: list[tuple[list[int], float]] = [([], 0.0)]
    for _ in range(steps):
        cand: list[tuple[list[int], float]] = []
        for order, acc in beam:
            used = set(order)
            for j in candidates:
                if j in used:
                    continue
                new_order = order + [j]
                acc2 = acc + (full_output - eval_masked(new_order))
                cand.append((new_order, acc2))
        if not cand:
            break
        cand.sort(key=lambda t: t[1], reverse=(mode == "upper"))
        beam = cand[:beam_size]

    return beam[0][1] / steps


def aopc_full(eval_masked: Callable[[Sequence[int]], float],
              full_output: float, order: Sequence[int]) -> float:
    """AOPC를 논문 정의(식 1)대로 계산한다 — 전체 순서를 훑으며 평균.

        AOPC(f, x, r) = (1/N) * sum_{i=1..N} [ f(x) - f(p(x, r_1:i)) ]

    상위 k개만 보는 격자 방식과 달리 모든 단계를 평균한다. NAOPC로 정규화하려면
    반드시 이 정의로 계산해야 한다 — 상한·하한이 이 정의 위에서 정의되기 때문이다.
    """
    if not order:
        return 0.0
    total = 0.0
    for i in range(1, len(order) + 1):
        total += full_output - eval_masked(order[:i])
    return total / len(order)
