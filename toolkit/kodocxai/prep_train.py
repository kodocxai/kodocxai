"""학습 분할 준비.

인.허가 33,823건에서 **파일럿 300건을 제외하고** 어댑터 자동 라벨로 학습셋을 만든다.
이미지는 224×224로 줄여 저장한다 — LayoutXLM이 그 크기로 리사이즈해 쓰므로
원본 해상도가 모델에 전달되지 않는다. 21.83GB가 수백 MB가 된다.

사용 예
  python prep_train.py --n 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kodx_data import export_images, export_train_from_zip, load_docs   # noqa: E402

BASE = Path(r"D:\AIHub데이터\공공행정문서 OCR")
PILOT = Path(r"D:\AIHub데이터\_pilot")
OUT = Path(r"D:\AIHub데이터\_exp\data")


def main() -> int:
    ap = argparse.ArgumentParser(description="학습 분할 준비")
    ap.add_argument("--n", type=int, default=5000, help="학습 문서 수")
    ap.add_argument("--size", type=int, default=224, help="이미지 저장 크기. 0이면 원본")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # 파일럿 문서는 평가셋이다. 학습에 절대 들어가면 안 된다.
    pilot_ids = {d["doc_id"] for d in load_docs(PILOT / "docs.jsonl")}
    print(f"파일럿 {len(pilot_ids)}건 제외")

    train_jsonl = OUT / "train.jsonl"
    n = export_train_from_zip(
        BASE / "Training" / "[라벨]train.zip", train_jsonl, pilot_ids, a.n)
    print(f"학습 문서 {n}건 → {train_jsonl}")

    names = {d["image"]["file_name"] for d in load_docs(train_jsonl)}
    img_dir = OUT / (f"images{a.size}" if a.size else "images")
    got = export_images(BASE / "Training" / "[원천]train1.zip", names, img_dir,
                        a.size or None)
    print(f"이미지 {got}건 → {img_dir}")

    # 평가셋 이미지도 같은 크기로 준비한다(모델 입력 일관성)
    if a.size:
        ev_names = {d["image"]["file_name"] for d in load_docs(PILOT / "docs.jsonl")}
        ev_dir = OUT / f"pilot_images{a.size}"
        got_ev = export_images(BASE / "Training" / "[원천]train1.zip", ev_names,
                               ev_dir, a.size)
        print(f"평가 이미지 {got_ev}건 → {ev_dir}")

    size_mb = sum(p.stat().st_size for p in img_dir.glob("*")) / 1024 / 1024
    (OUT / "manifest.json").write_text(json.dumps({
        "n_train": n, "n_images": got, "image_size": a.size,
        "excluded_pilot": len(pilot_ids),
        "images_mb": round(size_mb, 1),
        "label_source": "어댑터 자동 추출 (약지도). 사람 검수 아님 — 논문에 명시할 것",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n이미지 총 {size_mb:.1f} MB — Colab 업로드 가능한 크기다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
