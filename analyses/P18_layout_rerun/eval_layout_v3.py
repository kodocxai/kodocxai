"""레이아웃 교란만 재계산 (2026-10-26, 외부 독립 검토 R1 ②).

논문 5.3절 서술대로 (1) 지터를 어절 단위로, (2) 좌표 교환 상대를 다른 필드의 주석
스팬 토큰으로 바꿔 표 7의 레이아웃 열만 다시 잰다. 텍스트·이미지·귀속 점수는 손대지
않는다. 훈련된 가중치를 그대로 불러 순전파만 한다.

검증용으로 같은 입력에서 p_full 과 구(舊) 규칙 교환값을 다시 계산해 기존 결과
(seed{N}_v2.jsonl)와 대조한다. p_full 이 같으면 전처리·모델·대상 토큰이 같다는 뜻이고,
구 규칙 교환값이 같으면 교환 함수 자체가 같다는 뜻이다(결정적이라 정확히 일치해야 한다).

난수: 지터 이동량은 (훈련 시드, 문서, 필드, δ)로 결정되는 CPU 생성기에서 한 번 뽑는다.
실행 순서·중단 재개와 무관하게 재현된다.

사용 예
  D:\kodx-venv\Scripts\python.exe eval_layout_v3.py --model runs\seed1\last ^
      --docs ..\_pilot\docs.jsonl --images data\pilot_images224 ^
      --annotations ..\_pilot\annotations_park.jsonl ^
      --ref results\seed1_v2.jsonl --out results\layout_v3_seed1.jsonl --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kodx_data import build_examples                                  # noqa: E402
from kodx_units import subword_char_spans                             # noqa: E402
import kodx_attrib as A                                               # noqa: E402


def token_idx_for_words(spans, word_idx: set) -> list:
    return [t for t, s in enumerate(spans) if s and s[0] in word_idx]


def load_ref(path: Path) -> dict:
    ref = {}
    if not path or not path.exists():
        return ref
    for line in path.open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["method"] == "ig" and r["level"] == "eojeol" and r["mask"] == "all":
            ref[(r["doc_id"], r["field"])] = r
    return ref


def main() -> int:
    ap = argparse.ArgumentParser(description="레이아웃 교란 재계산 (5.3절 서술대로)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--annotations", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default=None, help="기존 결과 seed{N}_v2.jsonl (대조용)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jitter", type=int, nargs="+", default=[25, 67, 134])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification
    from kodx_data import load_processor
    from PIL import Image

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = load_processor(a.model, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(dev).eval()
    tok = proc.tokenizer

    ex = build_examples(Path(a.docs), Path(a.images),
                        Path(a.annotations) if a.annotations else None)
    if a.limit:
        ex = ex[: a.limit]
    ref = load_ref(Path(a.ref)) if a.ref else {}
    print(f"평가 문서 {len(ex)}건 / 장치 {dev} / 대조 레코드 {len(ref)}건", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.open(encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["doc_id"], r["field"]))
            except ValueError:
                continue
        print(f"이미 계산된 인스턴스 {len(done)}개 — 건너뛴다", flush=True)
    f_out = out.open("a", encoding="utf-8")

    def gen_for(doc_id: str, field: str, cond: str) -> torch.Generator:
        mix = (a.seed << 32) ^ zlib.crc32(f"{doc_id}|{field}|{cond}".encode("utf-8"))
        return torch.Generator(device="cpu").manual_seed(mix)

    n_inst = 0
    for n_doc, e in enumerate(ex):
        if not e.gold_spans:
            continue
        img = Image.open(e.image_path).convert("RGB")
        enc = proc(img, e.words, boxes=e.boxes, truncation=True,
                   padding="max_length", max_length=512,
                   return_offsets_mapping=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping")
        spans = subword_char_spans(tok, e.words, {"word_ids": enc.word_ids(),
                                                  "offset_mapping": offsets[0]})
        word_of_token = [s[0] if s else None for s in spans]
        inputs = {k: v.to(dev) for k, v in enc.items()}

        with torch.no_grad():
            base_logits = model(**inputs).logits
        pred_ids = base_logits.argmax(-1)[0]

        for field, gold in e.gold_spans.items():
            if (e.doc_id, field) in done:
                continue
            gold_widx = set(gold["word_idx"])
            if not gold_widx:
                continue
            tgt_tokens = token_idx_for_words(spans, gold_widx)
            if not tgt_tokens:
                continue
            label_ids = [int(pred_ids[t]) for t in tgt_tokens]

            def sc(ov=None):
                inp = dict(inputs) if ov is None else {**inputs, **ov}
                return float(A.field_score(model(**inp).logits, tgt_tokens, label_ids))

            gold_tk = tgt_tokens
            other_words = set()
            partner_fields = []
            for f2, g2 in e.gold_spans.items():
                if f2 == field:
                    continue
                ws = set(g2["word_idx"]) - gold_widx
                if ws:
                    other_words |= ws
            partner = A.swap_partner_tokens(word_of_token, gold_tk, other_words)
            if partner:
                pw = {word_of_token[t] for t in partner}
                partner_fields = [f2 for f2, g2 in e.gold_spans.items()
                                  if f2 != field and pw & set(g2["word_idx"])]
            # 구 규칙(검증용): 정답 밖 토큰을 문서 순서대로 앞에서 N개
            other_old = [t for t in range(len(spans))
                         if spans[t] and t not in gold_tk][: len(gold_tk)]

            rec = {"doc_id": e.doc_id, "field": field, "n_gold_tk": len(gold_tk),
                   "n_gold_words": len(gold_widx), "faith": {}, "check": {}}
            with torch.no_grad():
                p_full = sc()
                rec["p_full"] = p_full
                for d in a.jitter:
                    p_j = sc(ov={"bbox": A.jitter_boxes_by_word(
                        inputs["bbox"], gold_tk, word_of_token, d,
                        gen_for(e.doc_id, field, f"jitter@{d}"))})
                    rec["faith"][f"layout_jitter@{d}"] = p_full - p_j
                if partner:
                    p_swap = sc(ov={"bbox": A.swap_boxes(inputs["bbox"], gold_tk, partner)})
                    rec["faith"]["layout_swap"] = p_full - p_swap
                else:
                    rec["faith"]["layout_swap"] = None
                rec["swap_partner_n"] = len(partner)
                rec["swap_partner_fields"] = partner_fields
                p_old = sc(ov={"bbox": A.swap_boxes(inputs["bbox"], gold_tk, other_old)}) \
                    if other_old else None
                rec["check"]["layout_swap_oldrule"] = (p_full - p_old) if p_old is not None else None

            r0 = ref.get((e.doc_id, field))
            if r0:
                rec["check"]["p_full_ref"] = r0.get("p_full")
                rec["check"]["layout_swap_ref"] = r0["faith"].get("layout_swap")
                for d in a.jitter:
                    rec["check"][f"layout_jitter@{d}_ref"] = r0["faith"].get(f"layout_jitter@{d}")
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f_out.flush()
            n_inst += 1

        if (n_doc + 1) % 25 == 0:
            print(f"  {n_doc+1}/{len(ex)} 문서, {n_inst} 인스턴스", flush=True)

    f_out.close()
    print(f"\n완료 → {out} ({n_inst} 인스턴스 추가)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
