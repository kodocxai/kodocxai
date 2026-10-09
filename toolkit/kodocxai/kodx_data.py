"""KoDocXAI 데이터 계층.  (파일명 `kodx_*.py` 는 동결물이라 바꾸지 않는다)

어댑터 출력(공통 스키마 JSONL) + 원천 이미지 → LayoutXLM 입력과 BIO 라벨.

두 종류의 분할을 만든다.
  train : 어댑터 자동 라벨 (약지도). 파일럿 문서는 반드시 제외한다
  eval  : 사람이 검수한 파일럿 주석 (정답)

**약지도 학습임을 논문에 명시해야 한다.** 학습 라벨은 규칙 기반 자동 추출이며
제목 기준 92% 커버리지·나머지는 미검출이다. 평가 라벨만 사람이 검수했다.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]
LABELS = ["O"] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


@dataclass
class Example:
    doc_id: str
    words: list[str]
    boxes: list[list[int]]          # 0~1000 정규화 (LayoutXLM 규약)
    labels: list[int]               # 어절 단위 BIO
    image_path: Optional[Path]
    raw_boxes: list[list[int]]      # 원본 픽셀 좌표 — 타당성 계산에 쓴다
    size: tuple[int, int]           # (width, height)
    gold_spans: dict                # {필드: {"word_idx": [...], "value": str}}
    # 다른 안(案)의 어절 — 평가에서 제외(ignore)한다. 오답이 아니다. 가이드 1-1
    ignore_word_idx: list[int] = field(default_factory=list)


def load_processor(name_or_path, **kw):
    """LayoutXLM 프로세서를 연다.

    **`AutoProcessor` 를 쓰면 안 된다.** `microsoft/layoutxlm-base` 에 대해
    `LayoutLMv2Processor` + `LayoutLMv2TokenizerFast`(WordPiece)를 잡아 오는데,
    LayoutXLM 은 XLM-R 계열 SentencePiece 어휘를 쓴다. 경고만 찍고 넘어간 뒤
    첫 인코딩에서 `ValueError: Id not recognized` 로 죽는다 (실측 2026-08-14).

    미세조정 산출물을 다시 열 때도 같은 함수를 쓴다.
    """
    from transformers import LayoutXLMProcessor
    return LayoutXLMProcessor.from_pretrained(name_or_path, **kw)


def normalize_box(b: list[int], w: int, h: int) -> list[int]:
    """[x,y,w,h] 픽셀 → [x0,y0,x1,y1] 0~1000. LayoutXLM은 이 형식만 받는다."""
    x0, y0, bw, bh = b
    f = lambda v, d: max(0, min(1000, int(1000 * v / max(1, d))))
    return [f(x0, w), f(y0, h), f(x0 + bw, w), f(y0 + bh, h)]


def _bio(n_words: int, spans: dict, word_id_to_idx: dict) -> list[int]:
    """어절 단위 BIO 라벨.

    **필드 값이 판독 순서에서 끊길 수 있다.** 수신이 두 줄에 걸치면 그 사이에
    결재란 어절(「국장」 등)이 끼어든다. 실측상 문서의 18.3%가 그렇다.

        [29] 김해시 <수신>  [30] 상동면 <수신>  [31] 매리 <수신>
        [32] 227-1번지 <수신>  [33] 국장 (결재란)  [34] 은성전자(주) <수신>

    이때 끊긴 뒤를 I- 로 이으면 BIO 규칙 위반이다(I- 앞에 O가 온다).
    **연속 구간마다 B- 로 다시 시작**한다. 불연속 개체의 표준 처리 방식이며,
    스팬 자체(word_ids·bbox = 근거 정답 R_k)는 온전히 남는다.
    """
    labels = [LABEL2ID["O"]] * n_words
    for field, sp in spans.items():
        if field not in FIELDS:
            continue
        idxs = sorted(word_id_to_idx[i] for i in sp if i in word_id_to_idx)
        prev = None
        for idx in idxs:
            tag = "I" if (prev is not None and idx == prev + 1) else "B"
            labels[idx] = LABEL2ID[f"{tag}-{field}"]
            prev = idx
    return labels


def _reading_order(words_raw: list[dict], lines: Optional[list[dict]]) -> list[dict]:
    """어절을 판독 순서로 세운다.

    원 라벨의 어노테이션 순서는 판독 순서가 아니라 컬럼 방향으로 섞여 있다.
    원시 y로 정렬하면 **같은 줄 안에서 순서가 깨진다** — 한 줄 안에서도 어절마다
    y가 몇 픽셀씩 다르기 때문이다(실측: 시행일자가 '26' → '97.' 순으로 뒤집혔다).
    BIO 스팬이 끊기므로 치명적이다.

    어댑터가 만들어 둔 `lines`(줄은 y순, 줄 안은 x순)를 그대로 쓴다.
    """
    by_id = {w["id"]: w for w in words_raw}
    if lines:
        out, seen = [], set()
        for ln in lines:
            for wid in ln.get("word_ids", []):
                w = by_id.get(wid)
                if w is not None and wid not in seen:
                    seen.add(wid)
                    out.append(w)
        # 줄에 속하지 못한 어절이 있으면 뒤에 붙인다
        out.extend(w for w in words_raw if w["id"] not in seen)
        if len(out) == len(words_raw):
            return out
    return sorted(words_raw, key=lambda w: (w["bbox"][1], w["bbox"][0]))


def load_docs(jsonl: Path) -> Iterator[dict]:
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_examples(
    docs_jsonl: Path,
    images_dir: Optional[Path] = None,
    annotations: Optional[Path] = None,
    exclude_ids: Optional[set] = None,
    require_image: bool = True,
) -> list[Example]:
    """공통 스키마 JSONL → Example 목록.

    annotations 가 주어지면 그 검수 결과를 정답으로 쓴다(평가 분할).
    없으면 어댑터 자동 필드를 라벨로 쓴다(학습 분할, 약지도).
    """
    gold: dict[str, dict] = {}
    if annotations and annotations.exists():
        for line in annotations.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            gold[r["doc_id"]] = r                # append-only이므로 나중 것이 이긴다

    out: list[Example] = []
    for d in load_docs(docs_jsonl):
        doc_id = d["doc_id"]
        if exclude_ids and doc_id in exclude_ids:
            continue

        words_raw = d["words"]
        if not words_raw:
            continue
        W, H = d["image"].get("width"), d["image"].get("height")
        if not W or not H:
            continue

        img = None
        if images_dir is not None:
            img = images_dir / d["image"]["file_name"]
            if require_image and not img.exists():
                continue

        ws = _reading_order(words_raw, d.get("lines"))
        word_id_to_idx = {w["id"]: i for i, w in enumerate(ws)}

        # status 의미
        #   ok / corrected : 값과 영역 모두 정답
        #   string_na      : **영역은 정답, 문자열은 못 씀.** 원 라벨이 두 줄을 한
        #                    어절로 묶어 값 순서가 깨진 경우다(필드 인스턴스의 약 7.2%).
        #                    근거 정답 R_k 계산에는 그대로 쓰고, 문자열 정확일치
        #                    채점에서만 제외해야 한다
        #   absent         : 이 문서에 그 필드가 없음
        USABLE = ("ok", "corrected", "string_na")
        ignore_ids: list[int] = []
        if doc_id in gold:
            fields = gold[doc_id]["fields"]
            spans = {f: v["word_ids"] for f, v in fields.items()
                     if v.get("status") in USABLE and v.get("word_ids")}
            values = {f: v.get("value", "") for f, v in fields.items()
                      if v.get("status") in USABLE}
            str_ok = {f: v.get("status") != "string_na"
                      for f, v in fields.items() if v.get("status") in USABLE}
            # 다른 안(案)의 스팬. 정답도 오답도 아니다 — 평가에서 통째로 뺀다.
            # 공문서는 한 번의 결재로 수신자별 시행문을 여러 통 만들어(「1건 기안 다수 시행」)
            # 한 장에 헤더 세트가 여러 벌 붙는다(파일럿 300건 중 69건 23.0%). 정답은 최상단
            # 안 하나이지만, 모델이 제3안의 수신을 짚은 것은 다른 시행문을 본 것이지 틀린
            # 것이 아니다. 주석 가이드 1-1 참조.
            ignore_ids = [i for sp in gold[doc_id].get("secondary_spans", [])
                          for i in sp.get("word_ids", [])]
        else:
            spans = {f: v["word_ids"] for f, v in (d.get("fields") or {}).items()
                     if f in FIELDS and v.get("word_ids")}
            values = {f: v.get("value", "") for f, v in (d.get("fields") or {}).items()
                      if f in FIELDS}
            str_ok = {f: True for f in spans}      # 자동 라벨은 구분하지 않는다

        out.append(Example(
            doc_id=doc_id,
            words=[w["text"] for w in ws],
            boxes=[normalize_box(w["bbox"], W, H) for w in ws],
            labels=_bio(len(ws), spans, word_id_to_idx),
            image_path=img,
            raw_boxes=[w["bbox"] for w in ws],
            size=(W, H),
            gold_spans={f: {"word_idx": sorted(word_id_to_idx[i] for i in ids
                                               if i in word_id_to_idx),
                            "value": values.get(f, ""),
                            # False면 문자열 채점에서 제외한다. 영역 채점은 그대로
                            "string_usable": str_ok.get(f, True)}
                        for f, ids in spans.items()},
            ignore_word_idx=sorted(word_id_to_idx[i] for i in set(ignore_ids)
                                   if i in word_id_to_idx),
        ))
    return out


def export_train_from_zip(label_zip: Path, out_jsonl: Path,
                          exclude_ids: set, limit: Optional[int] = None,
                          include: str = "/인.허가/") -> int:
    """학습 분할용 JSONL을 라벨 zip에서 직접 뽑는다(어댑터 자동 라벨).

    파일럿 문서는 exclude_ids로 반드시 제외한다. 평가셋 누출을 막는다.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_tools"))
    from aihub_adapter import iter_docs        # noqa: E402

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_jsonl.open("w", encoding="utf-8") as f:
        for doc in iter_docs(label_zip, include, None, 1):
            if doc.doc_id in exclude_ids:
                continue
            if not doc.fields:
                continue                        # 필드가 하나도 없는 면은 학습에서 뺀다
            f.write(doc.to_json() + "\n")
            n += 1
            if limit and n >= limit:
                break
    return n


def export_images(src_zip: Path, names: set, out_dir: Path,
                  resize_to: Optional[int] = 224) -> int:
    """원천 zip에서 이미지를 꺼낸다.

    resize_to를 주면 그 크기로 줄여 저장한다. LayoutXLM은 입력 이미지를 224×224로
    리사이즈해 쓰므로 원본 해상도가 모델에 전달되지 않는다. 미리 줄이면 Colab
    업로드가 수백 MB로 끝난다(원본은 21.83GB).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_tools"))
    from aihub_adapter import entry_name        # noqa: E402

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(src_zip) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = Path(entry_name(info)).name
            if name not in names:
                continue
            data = z.read(info)
            if resize_to:
                from PIL import Image
                import io
                im = Image.open(io.BytesIO(data)).convert("RGB")
                im = im.resize((resize_to, resize_to), Image.BILINEAR)
                im.save(out_dir / name, "JPEG", quality=90)
            else:
                (out_dir / name).write_bytes(data)
            n += 1
    return n
