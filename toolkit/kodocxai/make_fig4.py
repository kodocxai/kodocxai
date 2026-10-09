"""그림 4 — 타당성·충실도 괴리 사례. 귀속 히트맵을 실제 시행문 위에 얹는다.

원고 4.3절이 참조한다. 보여야 하는 것은 하나다 — **두 축은 독립이다.**
표 4(타당성)와 표 3(충실도)이 각각 평균만 말하므로, 평균 뒤에 숨은 어긋남을
사례로 보인다.

  (a) 높은 타당성 · 낮은 충실도 — 귀속이 정답 영역을 잘 맞히는데, 그 영역을
      가려도 예측이 무너지지 않는다. 「사람 눈에 그럴듯함」이 「모델이 실제로
      쓴다」를 함의하지 않는다는 뜻이다.
  (b) 낮은 타당성 · 높은 충실도 — 귀속이 정답 영역을 전혀 못 맞히는데, 그
      영역을 가리면 예측이 무너진다. 그럴듯하지 않은 설명이 충실할 수 있다.

**모델 재실행이 없다.** v2 재실험에서 `--save-phi` 로 저장해 둔 서브워드 귀속
(`_exp\\results\\phi_seed{N}.jsonl`)을 읽어 어절로 재집계할 뿐이다. 토크나이저만
돌리므로 CPU 로 즉시 끝난다. 수치(IoU@5·AOPC)도 손으로 계산하지 않고
`seed{N}_v2.jsonl` 의 해당 레코드에서 그대로 읽는다.

사용
    python make_fig4.py [--seed 1] [--case doc_id:필드 ...] [--out <경로>]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r"D:\AIHub데이터\_exp")

PILOT = Path(r"D:\AIHub데이터\_pilot")
RES = Path(r"D:\AIHub데이터\_exp\results")
MODEL = Path(r"D:\AIHub데이터\_exp\runs\seed1\last")
OUT = Path(r"D:\AIHub데이터\_manuscript\figures")

# 기본 사례 — `실험결과.md` 하단 후보 목록에서 양 극단을 하나씩 골랐다.
# 발신명의는 IoU@5 가 가장 높은 축(0.911), 시행일자는 IoU@5 가 0 인 축이다.
DEFAULT_CASES = [
    ("5350109-1999-0002-1246", "발신명의"),
    ("5350109-2000-0001-0727", "시행일자"),
]
PANEL_TAG = ["(a)", "(b)", "(c)", "(d)"]

HEAT = (202, 0, 32)          # 귀속 히트맵 — 적색 계열
GOLD = (0, 114, 178)         # 근거 정답 R_k — Okabe–Ito 파랑
INK = (20, 20, 20)
GREY = (110, 110, 110)
# 영문 라벨은 원고 표 1 의 필드 표기와 통일한다(2차 검수 N16 — Issuer·Date of issue 는 구표기).
LABEL_EN = {
    "제목": "Title", "수신": "Recipient", "문서번호": "Document number",
    "시행일자": "Issuance date", "발신명의": "Issuing authority",
}
TOP_K = 5                    # IoU@5 와 같은 k 를 쓴다. 그래야 그림이 표와 맞는다

# 오버레이 문구 국·영 (2026-08-17 영문판 추가 — 문서 스캔은 그대로, 텍스트만 교체)
T = {
    "ko": {
        "field": lambda f: f"{f} ({LABEL_EN[f]})",
        "head_hi": "높은 타당성 · 낮은 충실도",
        "head_lo": "낮은 타당성 · 높은 충실도",
        "omit": "본문 생략",
        "lg_heat": "IG 귀속 (어절 단위, 양의 값만 · 문서 내 최댓값으로 정규화)",
        "lg_gold": "근거 정답 R_k = 정답 값 스팬 어절 상자의 합집합",
        # E1(2026-08-17): AOPC 는 표시된 top-k 가 아니라 정답 제외 차순위를 가려 잰다 -
        # 「AOPC 가 세는 대상」이라 쓰면 (a)처럼 정답이 상위인 패널에서 모순으로 읽힌다
        "lg_topk": (f"귀속 상위 {TOP_K}위 (붉은 실선 + 순위) = IoU@{TOP_K} 의 대상 · "
                    "충실도 교란은 정답을 뺀 차순위를 가린다"),
        "lg_aopc": ("AOPC > 0 이면 상위 어절을 가릴 때 예측이 무너진다는 뜻이고, "
                    "AOPC < 0 이면 오히려 예측이 굳어진다는 뜻이다."),
    },
    "en": {
        "field": lambda f: f"{LABEL_EN[f]} ({f})",
        "head_hi": "High plausibility · low faithfulness",
        "head_lo": "Low plausibility · high faithfulness",
        "omit": "body omitted",
        "lg_heat": "IG attribution (eojeol level; positive values only, normalized by the document maximum)",
        "lg_gold": "Ground-truth rationale R_k: union of the eojeol boxes of the gold value span",
        "lg_topk": (f"Top-{TOP_K} attributed eojeols (red outline + rank): what IoU@{TOP_K} scores; "
                    "faithfulness masks the next-ranked non-gold eojeols"),
        "lg_aopc": ("AOPC > 0 means the prediction collapses when the top eojeols are masked; "
                    "AOPC < 0 means it becomes even more confident."),
    },
}

# ── 개인정보 마스킹 (2026-08-16) — 그림 2 와 **같은 기준** ────────────────
#   가린다 — 공무원 실명 · 전화/팩스 · 개인 업무 이메일 · 민원인 법인명과 지번 · 자필 서명
#   남긴다 — 기관명 · 기관 홈페이지 URL · 문서번호 · 시행일자 · 제목 · 발신명의 · 직인
#
# 남기는 이유가 그림 2 보다 하나 더 있다. **(b) 패널의 귀속 상위 5위에 문서번호·
# 시행일자 항목명과 기관 URL 이 들어간다.** 그것이 이 그림의 발견 자체이므로 가리면
# 그림이 성립하지 않는다. 다행히 상위 5위에 개인정보는 하나도 없다 — 확인했다.
#
# 색인은 `Example.words` 기준(읽기 순서)이며 `docs.jsonl` 의 `id` 와 다르다.
MASK_FILL = (58, 58, 58)

PII_WORDS = {
    "5350109-1999-0002-1246": {
        15, 18,              # 전화·전송 번호
        21, 24, 26,          # 과장 안OO · 담당주사 오OO · 담당자 배OO
        54,                  # 기안자 이름
    },
    "5350109-2000-0001-0727": {
        15, 16,              # 전화·FAX
        19, 21, 23,          # 과장 · 공장설립담당주사 · 담당자
        27,                  # 개인 업무 이메일 (가장 직접적인 식별자)
        43, 44,              # 수신 필드의 지번·법인명 (상자는 남고 글자만 가려진다)
        94, 99, 100, 109, 110,   # 승인내용 표의 회사명·소재지 값
    },
}
# 어절이 아니라 그림인 것 — 자필 서명. 원본 픽셀 좌표다.
# (b) 문서에는 결재란이 없어 서명 영역도 없다.
PII_REGIONS = {
    "5350109-1999-0002-1246": [
        (1590, 670, 2150, 860),     # 시장 결재란 자필 — 이름이 판독된다
        (1270, 1060, 1540, 1160),   # 기안자 서명
        (1740, 1055, 2010, 1170),   # 담당주사 서명
    ],
}


def font(size: int):
    for name in ("malgun.ttf", "malgunbd.ttf", "NanumGothic.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


SX = 0.90                       # 머리말·범례 글줄 장평. 그림 1·2와 동일 원칙(08-18 저자 지시)


def ctext(base: Image.Image, xy, s: str, fnt, fill, sx: float = SX):
    """왼쪽 위 기준 텍스트를 장평 `sx` 로 그린다 (make_fig2.py 와 동일 방식)."""
    b = fnt.getbbox(s)
    w, h = b[2], b[3]
    if w <= 0:
        return
    tmp = Image.new("RGBA", (w + 2, h + 2), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((0, 0), s, font=fnt, fill=fill)
    tmp = tmp.resize((max(1, round((w + 2) * sx)), h + 2), Image.LANCZOS)
    base.paste(tmp, (int(xy[0]), int(xy[1])), tmp)


def load_metrics(seed: int, cases: list) -> dict:
    """seed{N}_v2.jsonl 에서 (doc, field) 의 IG·어절 레코드를 찾아 온다.

    수치를 다시 계산하지 않는다 — 표와 그림이 어긋나면 안 되기 때문이다.
    """
    want = {(d, f) for d, f in cases}
    got = {}
    p = RES / f"seed{seed}_v2.jsonl"
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["doc_id"], r["field"])
            if (key in want and r["method"] == "ig"
                    and r["level"] == "eojeol" and r["mask"] == "all"):
                got[key] = r
    missing = want - set(got)
    if missing:
        raise SystemExit(f"{p.name} 에 없는 사례: {sorted(missing)}")
    return got


def load_phi(seed: int, cases: list) -> dict:
    """phi_seed{N}.jsonl 에서 서브워드 귀속 벡터를 찾아 온다."""
    want = {(d, f) for d, f in cases}
    got = {}
    p = RES / f"phi_seed{seed}.jsonl"
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["doc_id"], r["field"])
            if key in want and r["method"] == "ig":
                got[key] = r["phi"]
    missing = want - set(got)
    if missing:
        raise SystemExit(
            f"{p.name} 에 없는 사례: {sorted(missing)}\n"
            f"v2 재실험을 --save-phi 없이 돌린 시드일 수 있다.")
    return got


def fold_page(img: Image.Image, boxes: list, bottom_pad: float = 0.03):
    """아래 여백을 잘라내고, 어절이 하나도 없는 큰 구간을 접는다.

    시행문 서식은 본문과 발신명의 사이가 크게 비어 있어 그대로 두면 그림의 절반이
    여백이 된다. **어절이 없는 구간만** 접으므로 귀속 상자를 잃지 않는다.
    반환: (접은 이미지, 원본 y → 접은 뒤 y 함수)
    """
    W, H = img.size
    bot = min(H, int(max(b[1] + b[3] for b in boxes) + bottom_pad * H))
    img = img.crop((0, 0, W, bot))
    H = bot

    rows = sorted((b[1], b[1] + b[3]) for b in boxes if b[1] < H)
    gap, cur = None, rows[0][1]
    for y0, y1 in rows:
        if y0 - cur > H * 0.12:
            gap = (int(cur + H * 0.015), int(y0 - H * 0.015))
        cur = max(cur, y1)

    band = int(H * 0.035)
    y_shift = 0
    if gap:
        top_part = img.crop((0, 0, W, gap[0]))
        bot_part = img.crop((0, gap[1], W, H))
        y_shift = (gap[1] - gap[0]) - band
        H -= y_shift
        img = Image.new("RGB", (W, H), "white")
        img.paste(top_part, (0, 0))
        img.paste(bot_part, (0, gap[0] + band))

    def shift(y: int) -> int:
        return y - y_shift if (gap and y >= gap[1]) else y

    return img, shift, gap, band


def draw_panel(ex, phi_word: list, gold_widx: set, rec: dict,
               field: str, tag: str, headline: str, t: dict,
               mask: bool = True) -> Image.Image:
    """문서 한 장에 히트맵 + 근거 정답 + 상위 k 순위를 얹은 패널."""
    img = Image.open(ex.image_path).convert("RGB")
    img, shift, gap, band = fold_page(img, ex.raw_boxes)
    W, H = img.size

    head = int(H * 0.075)
    canvas = Image.new("RGB", (W, H + head), "white")
    canvas.paste(img, (0, head))
    dr = ImageDraw.Draw(canvas, "RGBA")
    lw = max(3, W // 400)

    # ── 개인정보 마스킹. **히트맵보다 먼저** 그린다 — 그래야 가린 어절도 자기
    #    귀속 색을 그대로 유지한다. 마스킹이 귀속 정보를 지우면 그림이 왜곡된다.
    pii = PII_WORDS.get(ex.doc_id, set()) if mask else set()
    if mask:
        for i in sorted(pii):
            if i >= len(ex.raw_boxes):
                continue
            x, y, w, h = ex.raw_boxes[i]
            y = shift(y) + head
            dr.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], fill=MASK_FILL)
        for (x0, y0, x1, y1) in PII_REGIONS.get(ex.doc_id, []):
            dr.rectangle([x0, shift(y0) + head, x1, shift(y1) + head], fill=MASK_FILL)

    # ── 히트맵. 양(+)의 귀속만 채색한다. 상위 k 선정이 최댓값 기준이므로
    #    음수는 순위에 들어오지 않고, 칠하면 오히려 읽기 어렵다.
    pos = [v for v in phi_word if v > 0]
    vmax = max(pos) if pos else 1.0
    order = sorted(range(len(phi_word)), key=lambda i: -phi_word[i])
    topk = [i for i in order[:TOP_K]]

    for i, v in enumerate(phi_word):
        if v <= 0 or i >= len(ex.raw_boxes):
            continue
        x, y, w, h = ex.raw_boxes[i]
        y = shift(y) + head
        # 알파를 선형으로 준다. 하한을 두면 귀속이 거의 0 인 어절까지 분홍으로
        # 물들어 문서 전체가 칠해진 것처럼 보인다 — 실제로 강한 어절만 남긴다.
        a = int(215 * min(1.0, v / vmax))
        if a < 4:
            continue
        dr.rectangle([x, y, x + w, y + h], fill=HEAT + (a,))

    # ── 상위 k 상자를 안쪽에 붉은 실선으로 두른다. 순위 뱃지는 모서리에 놓이므로
    #    뱃지만으로는 어느 상자를 가리키는지 헷갈린다 — 특히 (b) 처럼 정답 상자
    #    바로 옆이 상위에 올라온 경우 겹쳐 보인다.
    order_pos = [i for i in sorted(range(len(phi_word)), key=lambda i: -phi_word[i])
                 if phi_word[i] > 0][:TOP_K]
    for i in order_pos:
        if i >= len(ex.raw_boxes):
            continue
        x, y, w, h = ex.raw_boxes[i]
        y = shift(y) + head
        dr.rectangle([x + lw, y + lw, x + w - lw, y + h - lw],
                     outline=HEAT + (255,), width=lw)

    # ── 근거 정답 R_k — 어절 상자의 합집합. 외접 사각형이 아니다(2.4절)
    for i in sorted(gold_widx):
        if i >= len(ex.raw_boxes):
            continue
        x, y, w, h = ex.raw_boxes[i]
        y = shift(y) + head
        dr.rectangle([x, y, x + w, y + h], outline=GOLD + (255,), width=lw)

    # ── 상위 k 순위표. IoU@5 가 무엇을 세는지 눈으로 보이게 한다
    fs_r = max(13, W // 62)
    fr = font(fs_r)
    for rank, i in enumerate(topk, 1):
        if i >= len(ex.raw_boxes) or phi_word[i] <= 0:
            continue
        x, y, w, h = ex.raw_boxes[i]
        y = shift(y) + head
        r = int(fs_r * 0.78)
        dr.ellipse([x - r, y - r, x + r, y + r], fill=HEAT + (255,))
        tw = dr.textlength(str(rank), font=fr)
        dr.text((x - tw / 2, y - fs_r * 0.62), str(rank), font=fr, fill=(255, 255, 255))

    # ── 접은 자리 표시
    if gap:
        gy = gap[0] + band // 2 + head
        fsg = max(14, W // 60)
        for k in range(0, W, fsg * 4):
            dr.line([k, gy, k + fsg * 2, gy], fill=(175, 175, 175), width=max(2, lw // 2))
        dr.rectangle([int(W * 0.41), gy - fsg, int(W * 0.59), gy + fsg], fill=(255, 255, 255))
        dr.text((int(W * 0.435), gy - fsg), t["omit"], font=font(fsg), fill=(150, 150, 150))

    # ── 패널 머리말
    fs = max(17, W // 34)
    fb, fm = font(fs), font(int(fs * 0.80))
    iou = rec["plaus"]["5"]["iou"]
    aopc = rec["faith"]["aopc_comp"]
    ctext(canvas, (int(W * 0.015), int(head * 0.06)),
          f"{tag} {t['field'](field)}", fb, INK)
    ctext(canvas, (int(W * 0.015), int(head * 0.06) + int(fs * 1.18)),
          headline, fm, HEAT if aopc < 0 else GOLD)
    # \uc720\ub2c8\ucf54\ub4dc \ub9c8\uc774\ub108\uc2a4(U+2212)\ub294 \ub9d1\uc740\uace0\ub515\uc5d0 \uc5c6\uc5b4 \ub450\ubd80(\u25a1)\ub85c \ucc0d\ud78c\ub2e4. ASCII \ub97c \uc4f4\ub2e4.
    stat = f"IoU@5 = {iou:.3f}   AOPC = {aopc:.3f}"
    ctext(canvas, (int(W * 0.015), int(head * 0.06) + int(fs * 2.20)), stat, fm, GREY)
    dr.line([0, head - 2, W, head - 2], fill=(215, 215, 215), width=2)
    return canvas


def legend_strip(width: int, unit: int, t: dict) -> Image.Image:
    """패널 아래 공통 범례. 그림 하나에 한 번만 그린다."""
    fs = max(15, unit)
    f1, f2 = font(fs), font(int(fs * 0.92))
    h = int(fs * 8.4)          # 마지막 설명 줄까지 들어가야 한다
    strip = Image.new("RGB", (width, h), "white")
    dr = ImageDraw.Draw(strip, "RGBA")
    dr.line([0, 2, width, 2], fill=(215, 215, 215), width=2)

    x, y = int(width * 0.012), int(fs * 1.0)
    # 히트맵 그라데이션 견본 — 패널과 같은 선형 알파를 쓴다
    gw, gh = int(fs * 7), int(fs * 0.95)
    for k in range(gw):
        a = int(215 * k / max(1, gw - 1))
        dr.line([x + k, y, x + k, y + gh], fill=HEAT + (a,))
    dr.rectangle([x, y, x + gw, y + gh], outline=(150, 150, 150), width=1)
    ctext(strip, (x + gw + int(fs * 0.6), y - int(fs * 0.12)),
          t["lg_heat"], f1, INK)

    y += int(fs * 1.9)
    dr.rectangle([x, y, x + int(fs * 1.6), y + gh], outline=GOLD + (255,), width=max(2, fs // 7))
    ctext(strip, (x + gw + int(fs * 0.6), y - int(fs * 0.12)),
          t["lg_gold"], f1, INK)

    y += int(fs * 1.9)
    dr.rectangle([x, y, x + int(fs * 1.6), y + gh], outline=HEAT + (255,), width=max(2, fs // 7))
    r = int(fs * 0.55)
    cx = x + int(fs * 1.6)
    dr.ellipse([cx - r, y - r, cx + r, y + r], fill=HEAT + (255,))
    tw = dr.textlength("1", font=f2)
    dr.text((cx - tw / 2, y - int(fs * 0.5)), "1", font=f2, fill=(255, 255, 255))
    ctext(strip, (x + gw + int(fs * 0.6), y - int(fs * 0.12)),
          t["lg_topk"], f1, INK)

    y += int(fs * 1.9)
    ctext(strip, (x, y), t["lg_aopc"], f2, GREY)
    return strip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--case", action="append", default=None,
                    help="doc_id:필드 — 여러 번 줄 수 있다")
    ap.add_argument("--lang", choices=("ko", "en"), default="ko",
                    help="오버레이·범례 언어. 문서 스캔 자체는 동일하다")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-mask", dest="mask", action="store_false",
                    help="개인정보 마스킹을 끈다 — 내부 확인용. 논문에 쓰지 말 것")
    a = ap.parse_args()
    t = T[a.lang]
    if a.out is None:
        a.out = OUT / ("figure4_divergence" + ("" if a.lang == "ko" else "_en"))
    if not a.mask:
        print("! 마스킹 없음 — 논문 지면에 쓰지 말 것")

    cases = ([tuple(c.split(":", 1)) for c in a.case] if a.case else DEFAULT_CASES)
    if any(len(c) != 2 for c in cases):
        raise SystemExit("--case 는 doc_id:필드 형식이다")

    from kodx_data import build_examples, load_processor
    from kodx_units import aggregate, subword_char_spans

    metrics = load_metrics(a.seed, cases)
    phis = load_phi(a.seed, cases)

    ex_all = {e.doc_id: e for e in
              build_examples(PILOT / "docs.jsonl", PILOT / "images",
                             annotations=PILOT / "annotations_park.jsonl")}
    proc = load_processor(str(MODEL), apply_ocr=False)
    tok = proc.tokenizer

    panels = []
    for n, (doc, fld) in enumerate(cases):
        e = ex_all.get(doc)
        if e is None:
            raise SystemExit(f"docs.jsonl 에 없는 문서: {doc}")
        rec = metrics[(doc, fld)]

        # 평가와 **똑같이** 인코딩해야 저장된 φ 와 색인이 맞는다
        img = Image.open(e.image_path).convert("RGB")
        enc = proc(img, e.words, boxes=e.boxes, truncation=True,
                   padding="max_length", max_length=512,
                   return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        spans = subword_char_spans(tok, e.words, {"word_ids": enc.word_ids(),
                                                  "offset_mapping": offsets[0]})
        phi_word, units = aggregate(phis[(doc, fld)], spans, e.words, "eojeol")
        assert len(phi_word) == len(e.words), (len(phi_word), len(e.words))

        gold = set(e.gold_spans[fld]["word_idx"])
        aopc = rec["faith"]["aopc_comp"]
        headline = t["head_hi"] if aopc < 0.2 else t["head_lo"]
        panels.append(draw_panel(e, phi_word, gold, rec, fld, PANEL_TAG[n],
                                 headline, t, mask=a.mask))

        # 상위 k 에 마스킹 대상이 끼면 그림의 논지가 가려진다. 조용히 넘기지 않는다.
        pii = PII_WORDS.get(doc, set()) if a.mask else set()
        rank5 = [i for i in sorted(range(len(phi_word)), key=lambda i: -phi_word[i])
                 if phi_word[i] > 0][:TOP_K]
        clash = [i for i in rank5 if i in pii]
        if clash:
            print(f"    ⚠ 상위 {TOP_K}위에 마스킹 대상이 있다: "
                  f"{[(i, e.words[i]) for i in clash]} — 캡션에 밝힐 것")
        gold_clash = sorted(pii & gold)
        if gold_clash:
            print(f"    ⚠ 정답 스팬이 마스킹된다: {[(i, e.words[i]) for i in gold_clash]}")

        print(f"{PANEL_TAG[n]} {doc}  {fld}")
        print(f"    IoU@5 {rec['plaus']['5']['iou']:.4f}   "
              f"AOPC {aopc:.4f}   p_full {rec['p_full']:.4f}")
        print(f"    어절 {len(e.words)}개 · 정답 어절 {len(gold)}개 · "
              f"양의 귀속 {sum(1 for v in phi_word if v > 0)}개")
        print(f"    정답 값  : {' '.join(e.words[i] for i in sorted(gold))!r}")
        rank = [i for i in sorted(range(len(phi_word)), key=lambda i: -phi_word[i])
                if phi_word[i] > 0][:TOP_K]
        # 캡션에 쓸 재료다. 상위 어절이 정답 안인지 밖인지가 IoU 의 내용이다.
        print("    상위 5위 : " + " / ".join(
            f"{r}.{e.words[i]}{'*' if i in gold else ''}" for r, i in enumerate(rank, 1)))
        print("               (* = 근거 정답 R_k 안)")

    # ── 패널을 같은 높이로 맞춰 가로로 잇는다
    Ht = max(p.size[1] for p in panels)
    scaled = []
    for p in panels:
        w, h = p.size
        nw = max(1, int(w * Ht / h))
        scaled.append(p.resize((nw, Ht), Image.LANCZOS))
    sep = max(8, Ht // 150)
    Wt = sum(p.size[0] for p in scaled) + sep * (len(scaled) - 1)

    strip = legend_strip(Wt, max(15, Wt // 95), t)
    fig = Image.new("RGB", (Wt, Ht + strip.size[1]), "white")
    x = 0
    for p in scaled:
        fig.paste(p, (x, 0))
        x += p.size[0] + sep
    fig.paste(strip, (0, Ht))
    d = ImageDraw.Draw(fig)
    for k in range(1, len(scaled)):
        gx = sum(scaled[j].size[0] for j in range(k)) + sep * (k - 1) + sep // 2
        d.line([gx, 0, gx, Ht], fill=(200, 200, 200), width=max(2, sep // 3))

    OUT.mkdir(parents=True, exist_ok=True)
    png = a.out.with_suffix(".png")
    pdf = a.out.with_suffix(".pdf")
    fig.save(png, dpi=(300, 300))
    fig.convert("RGB").save(pdf, "PDF", resolution=300.0)
    print(f"\n생성: {png}  ({fig.size[0]}×{fig.size[1]})")
    print(f"생성: {pdf}")


if __name__ == "__main__":
    main()
