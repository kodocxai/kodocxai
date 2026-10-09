"""AI-Hub OCR 라벨 어댑터.

세 계열의 라벨 스키마를 공통 스키마로 변환한다.

  계열 A  공공행정문서 OCR 정식 라벨   flat  (images / annotations, bbox=[x,y,w,h])
  계열 B  공공행정문서 OCR 부분 라벨   flat  (A와 동일 스키마, id 비연속)
  계열 C  023.OCR 데이터(공공)         중첩  (Images / Bbox, x[4]/y[4] 꼭짓점)

zip을 풀지 않고 중앙 디렉터리에서 직접 스트리밍한다. 800GB 원본을 건드리지 않기 위한 전제다.

표준 라이브러리만 사용한다.

사용 예
  # 인.허가 라벨 200건을 공통 스키마 JSONL로
  python aihub_adapter.py convert --label-zip "D:\\AIHub데이터\\공공행정문서 OCR\\Training\\[라벨]train.zip" \
      --include "/인.허가/" --limit 200 --out out.jsonl

  # 필드 추출률 통계만
  python aihub_adapter.py stats --label-zip "...\\[라벨]train.zip" --include "/인.허가/" --sample 2000

  # _label_samples 의 샘플 4종으로 자체 검증
  python aihub_adapter.py selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# zip 엔트리명 인코딩
# ---------------------------------------------------------------------------

def entry_name(info: zipfile.ZipInfo) -> str:
    """엔트리명을 CP949로 되살린다.

    AI-Hub zip은 UTF-8 플래그 없이 CP949로 파일명을 기록한다. zipfile은 이 경우
    CP437로 디코딩해 두므로 되돌린 뒤 CP949로 다시 읽는다.
    """
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return info.filename


# ---------------------------------------------------------------------------
# 공통 스키마
# ---------------------------------------------------------------------------

@dataclass
class Word:
    """어절 단위 어노테이션. bbox는 [x, y, w, h] (COCO 동일).

    id는 **원 라벨의 annotation.id가 아니라 배열 내 위치**다.
    원 라벨의 id는 중복되는 경우가 있어(실측: 89어절 중 고유 id 88개) 키로 쓸 수 없다.
    중복 id를 그대로 두면 줄 재구성과 주석 도구의 어절 선택이 함께 망가진다.
    원값은 orig_id로 보존한다.
    """
    id: int
    text: str
    bbox: list[int]
    orig_id: Optional[int] = None


@dataclass
class Line:
    """bbox 기준으로 재구성한 한 줄."""
    id: int
    text: str
    bbox: list[int]
    word_ids: list[int]


@dataclass
class Doc:
    doc_id: str
    schema: str                      # "A" | "B" | "C"
    source: dict
    image: dict
    words: list[Word] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    fields: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# 스키마 감지 + 정규화
# ---------------------------------------------------------------------------

_PARTLY = ("partly", "부분")


def detect_schema(obj: dict, entry: str = "") -> str:
    """A/B는 스키마가 동일하므로 정규화 결과는 같다. 출처 표기용으로만 갈라둔다.

    구분은 엔트리 경로로만 한다. id 연속성으로 추정하면 정식 라벨에서도 결번이
    있어 오판한다(실측 확인).
    """
    if "annotations" in obj and "images" in obj:
        low = entry.lower()
        return "B" if any(m in low for m in _PARTLY) else "A"
    if "Bbox" in obj and "Images" in obj:
        return "C"
    raise ValueError("알 수 없는 라벨 스키마")


def _norm_flat(obj: dict, entry: str, schema: str) -> Doc:
    img = (obj.get("images") or [{}])[0]
    file_name = img.get("image.file.name") or ""
    doc_id = Path(file_name).stem or Path(entry).stem

    words: list[Word] = []
    for a in obj.get("annotations") or []:
        b = a.get("annotation.bbox")
        text = a.get("annotation.text")
        if not b or len(b) != 4 or text is None:
            continue
        words.append(Word(id=len(words), text=str(text),
                          bbox=[int(round(v)) for v in b],
                          orig_id=a.get("id")))

    return Doc(
        doc_id=doc_id,
        schema=schema,
        source={
            "dataset": "공공행정문서 OCR",
            "entry": entry,
            "category": img.get("image.category"),
            "org_code": img.get("image.make.code"),
            "year": img.get("image.make.year"),
        },
        image={
            "file_name": file_name,
            "width": img.get("image.width"),
            "height": img.get("image.height"),
            "dpi": None,
        },
        words=words,
    )


def _norm_nested(obj: dict, entry: str) -> Doc:
    img = obj.get("Images") or {}
    ds = obj.get("Dataset") or {}

    words: list[Word] = []
    for i, b in enumerate(obj.get("Bbox") or []):
        xs, ys, text = b.get("x"), b.get("y"), b.get("data")
        if not xs or not ys or text is None:
            continue
        # x[4]/y[4] 꼭짓점이지만 실측상 축 정렬 사각형이다. 회전은 없다.
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        words.append(Word(id=len(words), text=str(text),
                          bbox=[int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                          orig_id=b.get("id")))

    return Doc(
        doc_id=str(img.get("identifier") or Path(entry).stem),
        schema="C",
        source={
            "dataset": ds.get("name") or "대규모 OCR 데이터(공공)",
            "entry": entry,
            "label_path": ds.get("label_path"),
            # 아래 정수 코드는 코드북이 JSON에 없다. AI-Hub 구축 문서 필요.
            "code_category": ds.get("category"),
            "code_type": ds.get("type"),
            "code_writing_style": img.get("writing_style"),
            "code_group": img.get("group"),
            "code_year": img.get("year"),
        },
        image={
            "file_name": f"{img.get('identifier')}.{img.get('type', 'jpg')}",
            "width": img.get("width"),
            "height": img.get("height"),
            "dpi": img.get("dpi"),
        },
        words=words,
    )


def normalize(obj: dict, entry: str = "") -> Doc:
    schema = detect_schema(obj, entry)
    if schema == "C":
        return _norm_nested(obj, entry)
    return _norm_flat(obj, entry, schema)


# ---------------------------------------------------------------------------
# 줄 재구성
#
# 어노테이션 순서는 판독 순서가 아니다. 컬럼 방향으로 섞여 있어 반드시 bbox로
# 다시 묶어야 한다.
#
# 같은 줄 판정은 세로 겹침 비율 >= 0.5 (겹침 / 두 높이의 작은 쪽). 세로 중심
# 거리를 쓰면 따옴표·기호처럼 높이가 작은 어절이 별도 줄로 떨어져 나간다.
# ---------------------------------------------------------------------------

LINE_TOL = 0.5


def build_lines(words: list[Word], tol_ratio: float = LINE_TOL) -> list[Line]:
    """어절을 줄로 묶는다.

    판정 기준은 **세로 중심 거리 < 문서 어절 높이 중앙값 × 0.5**다.

    개별 어절의 높이나 겹침 비율로 판정하면 안 된다. 결재란 도장 글자처럼
    두 줄 높이에 걸친 어절이 하나라도 있으면(실측: 「전결」 h=100, 본문 h=55)
    그것이 기준이 된 밴드가 아래 줄까지 통째로 흡수한다. 그러면 수신처럼 두 줄에
    걸친 값에서 어절이 뒤섞인다.

    중앙값은 이상치에 흔들리지 않으므로 이 문제를 피한다.
    """
    if not words:
        return []

    hs = sorted(w.bbox[3] for w in words)
    med = hs[len(hs) // 2] or 1
    tol = max(8.0, tol_ratio * med)

    def cy(w: Word) -> float:
        return w.bbox[1] + w.bbox[3] / 2

    bands: list[list] = []          # [중심 합, 개수, [words]]
    for w in sorted(words, key=lambda w: (cy(w), w.bbox[0])):
        c = cy(w)
        if bands and abs(c - bands[-1][0] / bands[-1][1]) <= tol:
            bands[-1][0] += c
            bands[-1][1] += 1
            bands[-1][2].append(w)
        else:
            bands.append([c, 1, [w]])

    lines: list[Line] = []
    for i, band in enumerate(bands):
        bucket = sorted(band[2], key=lambda w: w.bbox[0])
        x0 = min(w.bbox[0] for w in bucket)
        y0 = min(w.bbox[1] for w in bucket)
        x1 = max(w.bbox[0] + w.bbox[2] for w in bucket)
        y1 = max(w.bbox[1] + w.bbox[3] for w in bucket)
        lines.append(Line(
            id=i,
            text=" ".join(w.text for w in bucket),
            bbox=[x0, y0, x1 - x0, y1 - y0],
            word_ids=[w.id for w in bucket],
        ))
    return lines


# ---------------------------------------------------------------------------
# 필드 추출
#
# 「제 목」처럼 라벨이 어절로 쪼개져 있는 경우가 절반에 가깝다. 줄을 먼저 만든 뒤
# 공백을 제거한 문자열에서 라벨을 찾으면 분리/비분리를 한 번에 처리할 수 있다.
# 이 처리 유무로 제목 추출률이 41.2% → 92.2%로 바뀐다.
# ---------------------------------------------------------------------------

FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "제목": ("제목",),
    "수신": ("수신", "수신자"),
    "참조": ("참조",),
    "문서번호": ("문서번호", "문서변호"),   # OCR 오독 변형 포함
    "시행일자": ("시행일자", "시행일"),
}

_STRIP_LEAD = re.compile(r"^[:：\.\s\-]+")


def _compact(s: str) -> str:
    return re.sub(r"\s+", "", s)


    # 결재란 고정 어휘. 이 표는 문서 상단 우측에 세로로 붙어 있다.
_STAMP_WORDS = frozenset(
    "취급 보존 기안 협조 결재 공람 부시장 부군수 선결 지시 접수 담당 계장 과장 국장".split()
)


def stamp_boundary(words: list[Word],
                   page_height: Optional[int]) -> Optional[tuple[int, int]]:
    """결재란(우측 도장 표)의 왼쪽 경계 x와 아래 끝 y를 추정한다.

    시행문은 「수신」·「시행일자」 줄 오른쪽에 결재란이 같은 높이로 붙어 있다.
    가로 간격만으로는 못 끊는다 — 표가 본문에 바짝 붙어 있어 간격이 작다(실측).
    결재란 고정 어휘가 상단에 세로로 여러 개 모여 있다는 점을 이용한다.

    y 끝을 함께 돌려주는 이유: 제목은 보통 결재란보다 아래에 온다. 높이 구간을
    따지지 않고 x로만 자르면 긴 제목이 통째로 잘린다(실측: 「… ( 제 1」).
    """
    H = page_height or 3500
    hits = [w for w in words
            if w.bbox[1] < H * 0.45 and _compact(w.text) in _STAMP_WORDS]
    if len(hits) < 2:
        return None
    # 어휘가 표의 첫 칸(라벨 열)에 있으므로 그 왼쪽 끝이 표의 경계다
    return min(w.bbox[0] for w in hits), max(w.bbox[1] + w.bbox[3] for w in hits)


    # 서식에 인쇄된 빈 괄호·구두점. 기입된 내용이 아니므로 값에서 뺀다.
_FORM_TAIL = re.compile(r"^[()（）년\.·,'\"\s\-]+$")


def _strip_form_tail(words: list[Word]) -> list[Word]:
    """값 끝에 붙은 서식 인쇄 문자를 떼어낸다.

    시행일자가 「97. 11. 26 ( 년)」처럼 나오는데 「( 년)」은 서식에 인쇄된 빈칸이지
    기입된 내용이 아니다. 실측상 값의 36%가 이런 꼬리를 달고 있다.

    숫자를 포함한 어절을 만나면 멈춘다. 「오전」처럼 서식 문자가 아닌 어절도 남긴다.
    """
    out = list(words)
    while out:
        t = out[-1].text
        if any(ch.isdigit() for ch in t) or not _FORM_TAIL.fullmatch(t):
            break
        out.pop()
    return out


def _cut_at_gap(words: list[Word], page_width: Optional[int],
                boundary: Optional[int] = None) -> list[Word]:
    """가로로 크게 벌어지는 지점, 또는 결재란 경계에서 값을 끊는다."""
    if not words:
        return words
    W = page_width or 2500
    out = []
    for i, cur in enumerate(words):
        if boundary is not None and cur.bbox[0] >= boundary:
            break
        if i:
            prev = words[i - 1]
            gap = cur.bbox[0] - (prev.bbox[0] + prev.bbox[2])
            if gap > max(2.5 * prev.bbox[3], 0.045 * W):
                break
        out.append(cur)
    return out


def extract_fields(doc: Doc) -> dict:
    """줄 단위로 라벨을 찾아 값을 떼어낸다.

    라벨 어절과 값 어절의 경계는 어절 텍스트를 순서대로 이어붙이며 찾는다.
    이렇게 하면 '제목' 한 어절과 '제'+'목' 두 어절이 같은 코드로 처리된다.
    """
    out: dict = {}
    by_id = {w.id: w for w in doc.words}
    boundary = stamp_boundary(doc.words, doc.image.get("height"))

    for line in doc.lines:
        line_words = [by_id[i] for i in line.word_ids if i in by_id]
        if not line_words:
            continue

        acc = ""
        for k, w in enumerate(line_words):
            acc += _compact(w.text)
            acc_clean = _STRIP_LEAD.sub("", acc.replace(":", "").replace("：", ""))
            for name, variants in FIELD_LABELS.items():
                if name in out:
                    continue
                if acc_clean in variants or acc_clean.rstrip(".") in variants:
                    # 결재란 경계는 결재란과 같은 높이 구간의 줄에만 적용한다
                    bx = boundary[0] if (boundary and line.bbox[1] < boundary[1]) else None
                    rest = _cut_at_gap(line_words[k + 1:], doc.image.get("width"), bx)
                    if name == "시행일자":
                        rest = _strip_form_tail(rest)
                    value = _STRIP_LEAD.sub("", " ".join(w2.text for w2 in rest)).strip()
                    if not value:
                        break
                    x0 = min(w2.bbox[0] for w2 in rest)
                    y0 = min(w2.bbox[1] for w2 in rest)
                    x1 = max(w2.bbox[0] + w2.bbox[2] for w2 in rest)
                    y1 = max(w2.bbox[1] + w2.bbox[3] for w2 in rest)
                    out[name] = {
                        "value": value,
                        # 채점용. 예측값에도 같은 정규화를 적용해야 한다 — 한국어 OCR
                        # 라벨은 「제 목」·「김 해 시」처럼 어절이 쪼개져 있어, 정규화
                        # 없이 문자열을 비교하면 맞힌 예측이 틀린 것으로 채점된다.
                        "value_norm": _compact(value),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "line_id": line.id,
                        "word_ids": [w2.id for w2 in rest],
                        "label_split": k > 0,      # 라벨이 어절로 쪼개져 있었는지
                    }
                    break
            # 라벨 후보 길이를 넘어가면 이 줄에서 더 볼 것이 없다
            if len(acc_clean) > 6:
                break

    # 수신·참조는 값이 두 줄에 걸치는 경우가 잦다(실측 19.6%).
    #   수 신  김해시 상동면 매리 227-1번지
    #          은성전자(주) 대표
    # 둘 다 수신 값이므로 이어붙인다.
    for name in ("수신", "참조"):
        if name in out:
            _append_continuation(doc, out[name], by_id, boundary)

    발신 = _find_issuer(doc, used_lines={v["line_id"] for v in out.values()})
    if 발신:
        out["발신명의"] = 발신
    return out


_ANY_LABEL = re.compile(r"^(수신|수 신|참조|제목|경유|발신|문서번호|시행일자|붙임|첨부)")


def _append_continuation(doc: Doc, info: dict, by_id: dict,
                         boundary: Optional[tuple]) -> None:
    """값이 다음 줄로 이어지면 붙인다. 제자리에서 info를 갱신한다.

    이어지는 줄로 보려면 셋을 모두 만족해야 한다.
      - 다른 필드 라벨로 시작하지 않는다
      - 값 컬럼 안에서 시작한다 (라벨 컬럼이 아니라)
      - 바로 아래 줄이다 (줄 간격이 한 줄 높이의 1.2배 이내)
    """
    lid = info.get("line_id")
    if lid is None or lid + 1 >= len(doc.lines):
        return
    cur, nxt = doc.lines[lid], doc.lines[lid + 1]

    if _ANY_LABEL.match(_compact(nxt.text)):
        return

    vx0, vx1 = info["bbox"][0], info["bbox"][0] + info["bbox"][2]
    if not (vx0 - 50 <= nxt.bbox[0] <= vx1):
        return
    if nxt.bbox[1] - (cur.bbox[1] + cur.bbox[3]) >= cur.bbox[3] * 1.2:
        return

    words = [by_id[i] for i in nxt.word_ids if i in by_id]
    bx = boundary[0] if (boundary and nxt.bbox[1] < boundary[1]) else None
    words = _cut_at_gap(sorted(words, key=lambda w: w.bbox[0]),
                        doc.image.get("width"), bx)
    if not words:
        return

    extra = " ".join(w.text for w in words).strip()
    info["value"] = (info["value"] + " " + extra).strip()
    info["value_norm"] = _compact(info["value"])
    info["word_ids"] = list(info["word_ids"]) + [w.id for w in words]
    x0 = min(info["bbox"][0], min(w.bbox[0] for w in words))
    y0 = min(info["bbox"][1], min(w.bbox[1] for w in words))
    x1 = max(vx1, max(w.bbox[0] + w.bbox[2] for w in words))
    y1 = max(info["bbox"][1] + info["bbox"][3],
             max(w.bbox[1] + w.bbox[3] for w in words))
    info["bbox"] = [x0, y0, x1 - x0, y1 - y0]
    info["multiline"] = True


# 발신명의에는 라벨이 붙지 않는다. 기관장 직위 어미로 잡는다.
#
# 과장·국장·서장 등 실무 직위는 제외한다. 수신처와 결재란에 흔히 등장해
# 오탐을 만든다(실측: '태화수지공업사국장'이 발신명의로 잡혔다).
_ISSUER = re.compile(
    r"^[가-힣]{2,8}(시장|군수|구청장|도지사|지사|청장|교육감|장관|차관"
    r"|위원장|이사장|총장)$"
)
# 도장란·결재란 어휘로 시작하는 줄은 발신명의가 아니다
_FIELD_PREFIX = re.compile(
    r"^(수신|참조|제목|경유|발신|취급|선결|전결|결재|담당|접수|보존|받음|시간|기간)"
)

# 크기·위치는 조건으로 쓰지 않는다. 실측 결과 둘 다 틀렸다.
#   - 크기: 발신명의는 자간만 넓고 글자 높이는 본문과 비슷하다 (배율 0.95~1.14)
#   - 위치: 2면 이후 문서에서는 페이지 중상단(21% 지점)에 온다
# 판별은 "줄 전체가 기관장 직위로 끝나는 이름 하나뿐"이라는 조건에 건다.


def _find_issuer(doc: Doc, used_lines: set) -> Optional[dict]:
    """기관장 직위로 끝나는 줄 중 글자가 가장 큰 것을 발신명의로 본다.

    - 이미 다른 필드로 소비된 줄은 제외한다 (수신 「김해시장」 오인 방지)
    - 줄 전체가 기관명이어야 한다. 라벨·도장란이 붙은 줄은 제외한다
    - 후보가 여럿이면 글자가 가장 큰 줄을 택한다
    """
    cands = []
    for ln in doc.lines:
        if ln.id in used_lines:
            continue
        compact = _compact(ln.text)
        if _FIELD_PREFIX.match(compact) or not _ISSUER.match(compact):
            continue
        cands.append(ln)
    if not cands:
        return None

    best = max(cands, key=lambda ln: ln.bbox[3])
    return {
        # value는 다른 필드와 같은 규칙(어절을 공백으로 이음)으로 둔다. BIO 예측을
        # 문자열로 되돌리면 이 형태가 나오므로 정본은 이쪽이어야 한다.
        "value": best.text.strip(),
        "value_norm": _compact(best.text),
        "bbox": best.bbox,
        "line_id": best.id,
        "word_ids": best.word_ids,
        "confidence": "pattern",
    }


# ---------------------------------------------------------------------------
# zip 순회
# ---------------------------------------------------------------------------

def iter_docs(label_zip: Path, include: Optional[str] = None,
              limit: Optional[int] = None, step: int = 1,
              with_fields: bool = True) -> Iterator[Doc]:
    with zipfile.ZipFile(label_zip) as z:
        entries = []
        for info in z.infolist():
            if info.is_dir():
                continue
            name = entry_name(info)
            if not name.lower().endswith(".json"):
                continue
            if include and include not in name:
                continue
            entries.append((name, info))

        n = 0
        for i in range(0, len(entries), step):
            name, info = entries[i]
            try:
                obj = json.loads(z.read(info).decode("utf-8"))
                doc = normalize(obj, name)
            except (ValueError, UnicodeDecodeError) as e:
                print(f"  ! 건너뜀 {name}: {e}", file=sys.stderr)
                continue
            doc.lines = build_lines(doc.words)
            if with_fields:
                doc.fields = extract_fields(doc)
            yield doc
            n += 1
            if limit and n >= limit:
                return


def count_entries(label_zip: Path, include: Optional[str] = None) -> int:
    with zipfile.ZipFile(label_zip) as z:
        return sum(
            1 for info in z.infolist()
            if not info.is_dir()
            and entry_name(info).lower().endswith(".json")
            and (not include or include in entry_name(info))
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_convert(a: argparse.Namespace) -> int:
    zp = Path(a.label_zip)
    total = count_entries(zp, a.include)
    step = max(1, total // a.limit) if (a.limit and a.sampled) else 1
    print(f"대상 {total}건" + (f" (step={step} 층화 샘플링)" if step > 1 else ""), file=sys.stderr)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for doc in iter_docs(zp, a.include, a.limit, step, with_fields=not a.no_fields):
            f.write(doc.to_json() + "\n")
            n += 1
    print(f"{n}건 → {out}", file=sys.stderr)
    return 0


def cmd_stats(a: argparse.Namespace) -> int:
    zp = Path(a.label_zip)
    total = count_entries(zp, a.include)
    step = max(1, total // a.sample)
    print(f"전체 {total}건 중 step={step} 샘플링\n")

    n = 0
    hit = {k: 0 for k in list(FIELD_LABELS) + ["발신명의"]}
    split_hit = 0
    words_sum = lines_sum = 0
    for doc in iter_docs(zp, a.include, a.sample, step):
        n += 1
        words_sum += len(doc.words)
        lines_sum += len(doc.lines)
        for k, v in doc.fields.items():
            hit[k] = hit.get(k, 0) + 1
            if k == "제목" and v.get("label_split"):
                split_hit += 1

    if not n:
        print("샘플 0건")
        return 1

    print(f"샘플 {n}건 / 문서당 어절 {words_sum/n:.1f}개 / 줄 {lines_sum/n:.1f}개\n")
    print("--- 필드 추출률 ---")
    for k in list(FIELD_LABELS) + ["발신명의"]:
        c = hit.get(k, 0)
        print(f"  {k:<8} {c:>6} / {n}  ({100*c/n:.1f}%)")
    if hit.get("제목"):
        print(f"\n  제목 중 라벨 분리('제'+'목') 케이스: {split_hit} "
              f"({100*split_hit/hit['제목']:.1f}%)")
    return 0


def cmd_selftest(a: argparse.Namespace) -> int:
    d = Path(a.samples)
    files = sorted(d.glob("*.json"))
    if not files:
        print(f"샘플이 없다: {d}")
        return 1

    ok = True
    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        doc = normalize(obj, p.name)
        doc.lines = build_lines(doc.words)
        doc.fields = extract_fields(doc)

        print(f"\n=== {p.name}")
        print(f"  스키마 {doc.schema} / {doc.image['width']}x{doc.image['height']} "
              f"/ 어절 {len(doc.words)} / 줄 {len(doc.lines)}")
        if not doc.words:
            print("  ! 어절 0개 — 정규화 실패")
            ok = False
        for k, v in doc.fields.items():
            mark = " (분리라벨)" if v.get("label_split") else ""
            mark += " (추정)" if v.get("confidence") == "heuristic" else ""
            print(f"    {k:<6} {v['value'][:60]}{mark}")
        if not doc.fields:
            print("    (추출된 필드 없음)")
    print()
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="AI-Hub OCR 라벨 어댑터")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="라벨 zip → 공통 스키마 JSONL")
    c.add_argument("--label-zip", required=True)
    c.add_argument("--include", default=None, help="엔트리 경로 부분일치 필터 (예: /인.허가/)")
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--sampled", action="store_true", help="앞에서 자르지 않고 균등 샘플링")
    c.add_argument("--no-fields", action="store_true", help="필드 추출 생략")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_convert)

    s = sub.add_parser("stats", help="필드 추출률 통계")
    s.add_argument("--label-zip", required=True)
    s.add_argument("--include", default=None)
    s.add_argument("--sample", type=int, default=1000)
    s.set_defaults(func=cmd_stats)

    t = sub.add_parser("selftest", help="_label_samples 샘플로 자체 검증")
    t.add_argument("--samples", default=r"D:\AIHub데이터\_label_samples")
    t.set_defaults(func=cmd_selftest)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
