"""이미지 모달 기여가 0인 것이 실제 현상인지 교란 실패인지 판별한다 (원고 4.3절 근거).

표 5의 이미지 열이 0으로 나오면 세 가지를 의심해야 한다. 이 스크립트가 셋을 모두 배제한다.

  ① 가린 면적이 작아서?      → 이미지 **전체**를 0·평균·무작위로 대체해 본다
  ② 소프트맥스 포화 때문?     → 확률이 아니라 **로짓**에서 변화량을 잰다
  ③ 입력이 모델에 안 닿아서?  → 변화량이 정확히 0인지, 미세하게라도 0이 아닌지 본다

대조군으로 텍스트 마스킹과 좌표 제거를 같은 척도로 함께 잰다. 세 모달의 변화량을
같은 단위(로짓)로 비교해야 「이미지가 작다」가 아니라 「얼마나 작다」를 말할 수 있다.

    python check_image_modality.py --model runs\seed1\last [--n 5]
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kodx_data import build_examples, load_processor          # noqa: E402

PILOT = Path(r"D:\AIHub데이터\_pilot")
EXP = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(EXP / "runs/seed1/last"))
    ap.add_argument("--images", default=str(EXP / "data/pilot_images224"))
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification

    proc = load_processor(a.model, apply_ocr=False)
    m = AutoModelForTokenClassification.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = m.to(dev).eval()

    ex = [e for e in build_examples(PILOT / "docs.jsonl", Path(a.images),
                                    annotations=PILOT / "annotations_park.jsonl")
          if e.gold_spans][: a.n]

    acc: dict[str, list] = {k: [] for k in
                            ("이미지 전체 0", "이미지 무작위", "좌표 제거", "텍스트 마스킹")}
    zero_exact = 0

    for e in ex:
        enc = proc(Image.open(e.image_path).convert("RGB"), e.words, boxes=e.boxes,
                   truncation=True, padding="max_length", max_length=512,
                   return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            base = m(**inputs).logits[0]
        tgt = [t for t in range(base.shape[0]) if int(base[t].argmax()) != 0][:20]
        if not tgt:
            continue

        def d_logit(ov) -> float:
            with torch.no_grad():
                lg = m(**{**inputs, **ov}).logits[0]
            return (lg[tgt] - base[tgt]).abs().mean().item()

        img = inputs["image"]
        ids = inputs["input_ids"].clone()
        ids[0, tgt] = proc.tokenizer.mask_token_id
        bb = inputs["bbox"].clone()
        bb[0, tgt] = 0

        vals = {
            "이미지 전체 0": d_logit({"image": torch.zeros_like(img)}),
            "이미지 무작위": d_logit({"image": torch.randint(
                0, 256, img.shape, dtype=img.dtype, device=img.device)}),
            "좌표 제거": d_logit({"bbox": bb}),
            "텍스트 마스킹": d_logit({"input_ids": ids}),
        }
        for k, v in vals.items():
            acc[k].append(v)
        if vals["이미지 전체 0"] == 0.0:
            zero_exact += 1

    if not acc["텍스트 마스킹"]:
        print("대상 토큰이 있는 문서가 없다")
        return

    text = st.mean(acc["텍스트 마스킹"])
    print(f"문서 {len(acc['텍스트 마스킹'])}건 · 모델 {a.model}\n")
    print(f"{'교란':16s} {'|Δ로짓| 평균':>12s} {'텍스트 대비':>12s}")
    for k in ("이미지 전체 0", "이미지 무작위", "좌표 제거", "텍스트 마스킹"):
        mv = st.mean(acc[k])
        ratio = f"1/{text/mv:,.0f}" if mv > 0 and mv < text else ("1" if mv == text else "—")
        print(f"  {k:14s} {mv:12.5f} {ratio:>12s}")

    print()
    if zero_exact == len(acc["이미지 전체 0"]):
        print("⚠ 이미지 변화량이 **정확히 0** 이다 — 시각 경로가 모델에 닿지 않는 배선 오류를")
        print("  의심할 것. 교란이 아니라 구현을 먼저 확인해야 한다.")
    else:
        print("이미지 변화량이 0이 아니므로 시각 경로는 연결돼 있다.")
        print("전체를 대체해도 텍스트·레이아웃보다 자릿수가 작다면, 이는 교란 실패가 아니라")
        print("**모델이 이미지를 쓰지 않는다**는 뜻이다 (원고 4.3절).")


if __name__ == "__main__":
    main()
