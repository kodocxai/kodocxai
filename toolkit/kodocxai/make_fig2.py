"""그림 2 — 주석 스키마 예시. 실제 시행문 위에 5필드 스팬과 근거 정답 $R_k$를 얹는다.

논문 2.4절이 참조하는 그림이다. 보여야 하는 것은 셋이다.
  ① 필드 타입 ↔ 값 스팬 ↔ 경계 상자가 1:1로 연결된다는 주석 구조
  ② 어절 단위 주석 — 「김 해 시 장」처럼 한 단어가 여러 어절로 쪼개져 있다는 한국어 OCR 특성
  ③ 근거 정답 $R_k$ = 값 스팬 어절들의 경계 상자 **합집합**(영역)이지 외접 사각형이 아니라는 것

출력은 300 dpi 상당의 PNG. MDPI 본문 폭(약 170 mm)에 맞춘다.

사용
    python make_fig2.py [--doc <doc_id>] [--out <경로>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PILOT = Path(r"D:\AIHub데이터\_pilot")
OUT = Path(r"D:\AIHub데이터\_manuscript\figures")
FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]

# 색은 색맹 안전 팔레트에서 골랐다(Okabe–Ito). 흑백 인쇄에서도 명도가 갈린다.
COLORS = {
    "제목":     (0, 114, 178),      # 파랑
    "수신":     (213, 94, 0),       # 주황
    "문서번호": (0, 158, 115),      # 초록
    "시행일자": (204, 121, 167),    # 분홍
    "발신명의": (86, 180, 233),     # 하늘
}
# 영문 라벨은 원고 표 1 의 필드 표기와 통일한다(2차 검수 N16 — Issuer·Date of issue 는 구표기).
LABEL_EN = {
    "제목": "Title", "수신": "Recipient", "문서번호": "Document number",
    "시행일자": "Issuance date", "발신명의": "Issuing authority",
}

# 범례·설명 문구 국·영 (2026-08-17 영문판 추가 — 문서 스캔은 그대로, 오버레이 텍스트만 교체)
T = {
    "ko": {
        "title": "KoDocXAI 주석 스키마",
        "omit": "본문 생략",
        "eojeol": lambda n: f"어절 {n}개",
        "field": lambda f: f"{f}  ({LABEL_EN[f]})",
        "foot": ("실선 = 어절 경계 상자 b_i",
                 "실선의 합집합 = 근거 정답 R_k",
                 "점선 = 외접 사각형 (참고)"),
    },
    "en": {
        "title": "KoDocXAI annotation schema",
        "omit": "body omitted",
        "eojeol": lambda n: f"{n} eojeol{'s' if n != 1 else ''}",
        "field": lambda f: f"{LABEL_EN[f]}  ({f})",
        # 용어는 원고 EN 그림 2 캡션과 통일 — eojeol bounding box · ground-truth
        # rationale · enclosing rectangle
        "foot": ("Solid = eojeol bounding boxes b_i",
                 "Union of solid boxes = ground-truth rationale R_k",
                 "Dashed = enclosing rectangle (for reference)"),
    },
}

# ── 개인정보 마스킹 (2026-08-16) ──────────────────────────────────────
# 논문 지면은 오픈 액세스이고 영구 색인되므로, 원본에 남아 있는 개인 식별정보를
# 가린다. 원고 3.4.1 이 서식 선정 기준으로 「개인정보 통제 가능성」을 들었으므로
# 가리지 않으면 그 서술과 모순된다.
#
#   가린다 — 공무원 실명 · 전화/팩스 · 민원인 법인명과 지번 · 손글씨 서명
#   남긴다 — 기관명(김해시) · 문서번호 · 시행일자 · 제목 · 발신명의 · 접수 직인
#
# 남기는 다섯 필드는 **주석 대상 그 자체**라 가리면 그림이 무의미해진다. 직인은
# 캡션이 「체계적 결손이 발생하는 위치」로 지목하는 대상이므로 역시 남긴다.
# 수신 필드는 값이 민원인 법인이라 딜레마인데, **상자는 남기고 글자만 가려**
# 필드-스팬-상자의 연결 구조는 보이되 값은 읽히지 않게 한다.
MASK_FILL = (58, 58, 58)          # 의도적 가림임이 분명히 보이도록 진회색 단색
MASK_GLYPH = "▨"

PII_WORDS = {
    "5350109-1997-0001-0034": {
        11, 12,                    # 전화·전송 번호 + 담당자 실명(최덕곤)
        30, 31, 33,                # 수신 필드 안의 리·지번·법인명
        51, 52, 53, 54, 55, 56,    # 본문에 반복된 같은 주소·법인명
    },
}
# 어절이 아니라 이미지인 것 — 손글씨 서명. 원본 픽셀 좌표다.
PII_REGIONS = {
    "5350109-1997-0001-0034": [
        (1780, 700, 2150, 895),     # 시장 결재란 서명
        (1435, 1080, 1920, 1180),   # 기안란 서명 2개 (전결 도장은 건드리지 않는다)
    ],
}


def font(size: int):
    for name in ("malgun.ttf", "malgunbd.ttf", "NanumGothic.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


SX = 0.90                       # 범례 글줄 장평(가로 배율). 그림 1과 동일 원칙(08-18 저자 지시)


def ctext(base: Image.Image, xy, s: str, fnt, fill, sx: float = SX):
    """왼쪽 위 기준 텍스트를 장평 `sx` 로 그린다.

    PIL 은 자간·장평을 지원하지 않으므로 투명 레이어에 그린 뒤 가로만 리샘플한다.
    세로는 그대로라 줄 간격 계산이 바뀌지 않는다.
    """
    b = fnt.getbbox(s)
    w, h = b[2], b[3]
    if w <= 0:
        return
    tmp = Image.new("RGBA", (w + 2, h + 2), (0, 0, 0, 0))
    ImageDraw.Draw(tmp).text((0, 0), s, font=fnt, fill=fill)
    tmp = tmp.resize((max(1, round((w + 2) * sx)), h + 2), Image.LANCZOS)
    base.paste(tmp, (int(xy[0]), int(xy[1])), tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="5350109-1997-0001-0034")
    ap.add_argument("--lang", choices=("ko", "en"), default="ko",
                    help="범례·설명 언어. 문서 스캔 자체는 동일하다")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-mask", dest="mask", action="store_false",
                    help="개인정보 마스킹을 끈다 — 내부 확인용. 논문에 쓰지 말 것")
    a = ap.parse_args()
    t = T[a.lang]
    if a.out is None:
        sfx = "" if a.lang == "ko" else "_en"
        a.out = OUT / f"figure2_annotation_schema{sfx}.png"

    docs = {d["doc_id"]: d for d in
            (json.loads(l) for l in
             (PILOT / "docs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    gold = {}
    for line in (PILOT / "annotations_park.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gold[r["doc_id"]] = r

    d = docs[a.doc]
    rec = gold[a.doc]
    by_id = {w["id"]: w for w in d["words"]}

    img = Image.open(PILOT / "images" / d["image"]["file_name"]).convert("RGB")
    W, H = img.size

    # 문서 아래쪽 빈 공간은 잘라낸다. 값이 있는 마지막 어절보다 조금 아래까지만 남긴다.
    used = [by_id[i]["bbox"] for f in FIELDS
            for i in (rec["fields"][f].get("word_ids") or []) if i in by_id]
    bottom = min(H, int(max(b[1] + b[3] for b in used) + 0.04 * H))
    img = img.crop((0, 0, W, bottom))
    H = bottom

    # 본문과 발신명의 사이는 대개 크게 비어 있다(시행문 서식이 그렇다). 그대로 두면
    # 그림의 절반이 여백이 되므로, 어절이 하나도 없는 구간을 접고 생략 표시를 넣는다.
    rows = sorted((w["bbox"][1], w["bbox"][1] + w["bbox"][3]) for w in d["words"]
                  if w["bbox"][1] < H)
    gap = None
    cur = rows[0][1]
    for y0, y1 in rows:
        if y0 - cur > H * 0.12:
            gap = (int(cur + H * 0.015), int(y0 - H * 0.015))
        cur = max(cur, y1)
    y_shift = 0          # 접힌 아래쪽 좌표를 옮길 양
    band = int(H * 0.035)
    if gap:
        top_part = img.crop((0, 0, W, gap[0]))
        bot_part = img.crop((0, gap[1], W, H))
        y_shift = (gap[1] - gap[0]) - band
        H = H - y_shift
        img = Image.new("RGB", (W, H), "white")
        img.paste(top_part, (0, 0))
        img.paste(bot_part, (0, gap[0] + band))

    def shift(y: int) -> int:
        """원본 y → 접은 뒤 y."""
        return y - y_shift if (gap and y >= gap[1]) else y

    # 범례를 얹을 여백을 오른쪽에 만든다. 필드 용어 통일(Issuing authority 등)로
    # 영문 병기가 길어져 0.40 까지 넓혔다가, 글줄 장평 90% 압축(08-18)으로 오른쪽
    # 여백이 남아 0.37 로 되줄였다 - 여백이 쏠리지 않게.
    pad = int(W * 0.37)
    canvas = Image.new("RGB", (W + pad, H), "white")
    canvas.paste(img, (0, 0))
    dr = ImageDraw.Draw(canvas, "RGBA")

    lw = max(3, W // 500)

    # ── 개인정보 마스킹. 주석 상자보다 **먼저** 그려야 상자·지시선이 위에 남는다
    pii = PII_WORDS.get(a.doc, set()) if a.mask else set()
    if a.mask:
        for i in sorted(pii):
            if i not in by_id:
                continue
            x, y, w, h = by_id[i]["bbox"]
            y = shift(y)
            dr.rectangle([x - 2, y - 2, x + w + 2, y + h + 2], fill=MASK_FILL)
        for (x0, y0, x1, y1) in PII_REGIONS.get(a.doc, []):
            dr.rectangle([x0, shift(y0), x1, shift(y1)], fill=MASK_FILL)
        print(f"마스킹: 어절 {len(pii)}개 · 영역 {len(PII_REGIONS.get(a.doc, []))}개")
    else:
        print("! 마스킹 없음 — 논문 지면에 쓰지 말 것")
    for f in FIELDS:
        g = rec["fields"][f]
        ids = [i for i in (g.get("word_ids") or []) if i in by_id]
        if not ids:
            continue
        col = COLORS[f]
        # ① 어절마다 상자를 그린다 — 이것들의 합집합이 R_k 다
        for i in ids:
            x, y, w, h = by_id[i]["bbox"]
            y = shift(y)
            dr.rectangle([x, y, x + w, y + h], fill=col + (46,), outline=col + (255,), width=lw)
        # ② 외접 사각형은 점선으로 — 합집합과 다르다는 것을 보인다
        x0 = min(by_id[i]["bbox"][0] for i in ids)
        y0 = shift(min(by_id[i]["bbox"][1] for i in ids))
        x1 = max(by_id[i]["bbox"][0] + by_id[i]["bbox"][2] for i in ids)
        y1 = shift(max(by_id[i]["bbox"][1] + by_id[i]["bbox"][3] for i in ids))
        for k in range(0, int(x1 - x0), lw * 6):
            dr.line([x0 + k, y0, min(x0 + k + lw * 3, x1), y0], fill=col + (150,), width=lw)
            dr.line([x0 + k, y1, min(x0 + k + lw * 3, x1), y1], fill=col + (150,), width=lw)
        for k in range(0, int(y1 - y0), lw * 6):
            dr.line([x0, y0 + k, x0, min(y0 + k + lw * 3, y1)], fill=col + (150,), width=lw)
            dr.line([x1, y0 + k, x1, min(y0 + k + lw * 3, y1)], fill=col + (150,), width=lw)
        # ③ 지시선
        dr.line([x1, (y0 + y1) // 2, W + int(pad * 0.06), (y0 + y1) // 2],
                fill=col + (200,), width=lw)

    # 생략 표시 — 접은 자리임을 알린다
    if gap:
        gy = gap[0] + band // 2
        fsg = max(14, W // 60)
        for k in range(0, W, fsg * 4):
            dr.line([k, gy, k + fsg * 2, gy], fill=(170, 170, 170), width=max(2, lw // 2))
        dr.rectangle([int(W * 0.42), gy - fsg, int(W * 0.58), gy + fsg], fill=(255, 255, 255))
        dr.text((int(W * 0.44), gy - fsg), t["omit"], font=font(fsg), fill=(140, 140, 140))

    # 범례
    fs = max(16, W // 46)
    fb, fr = font(fs), font(int(fs * 0.86))
    ty = int(H * 0.04)

    def wrap(text: str, fnt, limit: int) -> list:
        """범례 폭에 맞춰 줄바꿈. 값 문자열이 길어 그대로 두면 잘린다."""
        out, cur = [], ""
        for tok in text.split(" "):
            trial = (cur + " " + tok).strip()
            if dr.textlength(trial, font=fnt) <= limit or not cur:
                cur = trial
            else:
                out.append(cur)
                cur = tok
        if cur:
            out.append(cur)
        return out
    dr.text((W + int(pad * 0.08), ty), t["title"], font=fb, fill=(20, 20, 20))
    ty += int(fs * 2.0)
    for f in FIELDS:
        g = rec["fields"][f]
        n = len(g.get("word_ids") or [])
        col = COLORS[f]
        dr.rectangle([W + int(pad * 0.08), ty, W + int(pad * 0.08) + fs, ty + fs],
                     fill=col + (70,), outline=col + (255,), width=max(2, lw // 2))
        ctext(canvas, (W + int(pad * 0.08) + int(fs * 1.5), ty - 2),
              t["field"](f), fb, (20, 20, 20))
        ty += int(fs * 1.35)
        # 값이 길면 범례 폭을 넘어 잘린다. 줄바꿈해 넣고, 넘친 줄만큼 아래로 민다.
        # 장평 압축만큼 실제 폭이 줄므로 줄바꿈 한계는 압축 전 기준으로 넓힌다.
        limit = int((pad - int(pad * 0.16) - int(fs * 1.5)) / SX)
        # ⚠ 범례는 값 문자열을 그대로 인쇄한다. 이미지만 가리고 여기를 두면
        #   가린 값이 오른쪽에 고스란히 남는다. 어절 단위로 같이 가린다.
        ids = [i for i in (g.get("word_ids") or []) if i in by_id]
        if pii and any(i in pii for i in ids):
            shown = " ".join(MASK_GLYPH * max(2, len(by_id[i]["text"]))
                             if i in pii else by_id[i]["text"] for i in ids)
        else:
            shown = g.get("value", "")
        lines = wrap("“" + shown + "”", fr, limit)
        for k, ln in enumerate(lines):
            ctext(canvas, (W + int(pad * 0.08) + int(fs * 1.5), ty + k * int(fs * 1.15)),
                  ln, fr, (70, 70, 70))
        ty += (len(lines) - 1) * int(fs * 1.15)
        ctext(canvas, (W + int(pad * 0.08) + int(fs * 1.5), ty + int(fs * 1.15)),
              t["eojeol"](n), fr, (120, 120, 120))
        ty += int(fs * 3.0)

    ty += int(fs * 0.6)
    dr.line([W + int(pad * 0.08), ty, W + pad - int(pad * 0.08), ty],
            fill=(200, 200, 200), width=2)
    ty += int(fs * 1.1)
    # 영문 설명이 범례 폭을 넘으므로 값 문자열과 같은 방식으로 줄바꿈한다
    for s in t["foot"]:
        for ln in wrap(s, fr, int((pad - int(pad * 0.16)) / SX)):
            ctext(canvas, (W + int(pad * 0.08), ty), ln, fr, (90, 90, 90))
            ty += int(fs * 1.15)
        ty += int(fs * 0.15)

    OUT.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out, dpi=(300, 300))
    print(f"생성: {a.out}  ({canvas.size[0]}×{canvas.size[1]})")
    # 그림 1·3·4 는 PDF 를 함께 내보낸다. 그림 2 만 빠져 있어 맞춘다 —
    # MDPI 투고는 벡터/고해상 PDF 를 선호한다.
    pdf = a.out.with_suffix(".pdf")
    canvas.convert("RGB").save(pdf, "PDF", resolution=300.0)
    print(f"생성: {pdf}")
    print(f"문서: {a.doc}")
    for f in FIELDS:
        g = rec["fields"][f]
        print(f"   {f:5s} {len(g.get('word_ids') or []):2d}어절  {g.get('value')!r}")


if __name__ == "__main__":
    main()
