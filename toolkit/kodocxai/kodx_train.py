"""LayoutXLM 미세조정 (논문 3.5).

RTX 3060 8GB에서도 돌아가도록 기본값을 잡았다.
  bf16 AMP + gradient checkpointing + (가능하면) 8-bit Adam + batch 1 + 누적 16

Colab에서 돌릴 때는 --batch 8 --accum 2 정도로 올리면 된다.

**중단 복구가 들어 있다.** 매 에폭 끝에 체크포인트를 쓰고, 같은 --out 으로 다시
실행하면 이어서 학습한다. Colab 세션이 끊겨도 처음부터 다시 돌릴 필요가 없다.

사용 예
  python kodx_train.py --train _exp/data/train.jsonl --images _exp/data/images224 \
      --out _exp/runs/layoutxlm_seed1 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kodx_data import FIELDS, LABELS, LABEL2ID, build_examples   # noqa: E402

MODEL_NAME = "microsoft/layoutxlm-base"


def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class KodxDataset(Dataset):
    def __init__(self, examples, processor, max_len: int = 512):
        self.ex = examples
        self.proc = processor
        self.max_len = max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        from PIL import Image
        e = self.ex[i]
        img = Image.open(e.image_path).convert("RGB")
        enc = self.proc(
            img, e.words, boxes=e.boxes, word_labels=e.labels,
            truncation=True, padding="max_length", max_length=self.max_len,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def build_optimizer(model, lr: float, use_8bit: bool):
    if use_8bit:
        try:
            import bitsandbytes as bnb
            print("  8-bit Adam 사용")
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr)
        except Exception as e:
            print(f"  bitsandbytes 사용 불가({e}) → 표준 AdamW")
    return torch.optim.AdamW(model.parameters(), lr=lr)


def main() -> int:
    ap = argparse.ArgumentParser(description="LayoutXLM 미세조정")
    ap.add_argument("--train", required=True, help="학습 분할 JSONL (어댑터 출력)")
    ap.add_argument("--images", required=True, help="이미지 디렉터리")
    ap.add_argument("--out", required=True, help="체크포인트 디렉터리")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--no-8bit", action="store_true")
    ap.add_argument("--no-checkpointing", action="store_true")
    # 스모크 테스트용. 본 학습에 몇 시간을 걸기 전에 파이프라인과 VRAM을 먼저 본다.
    ap.add_argument("--limit", type=int, default=0,
                    help="학습 문서를 앞에서 N건만 쓴다 (0=전체). 스모크 테스트용")
    ap.add_argument("--max-steps", type=int, default=0,
                    help="에폭당 최대 스텝 (0=제한 없음). 스모크 테스트용")
    a = ap.parse_args()

    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForTokenClassification
    from kodx_data import load_processor

    print(f"모델 로드: {a.model}")
    proc = load_processor(a.model, apply_ocr=False)
    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(LABELS),
        id2label={i: l for l, i in LABEL2ID.items()}, label2id=LABEL2ID,
    )

    # 중단 복구
    start_epoch = 0
    state = out / "state.json"
    if state.exists():
        st = json.loads(state.read_text(encoding="utf-8"))
        start_epoch = st.get("epoch", 0)
        if start_epoch >= a.epochs:
            print(f"이미 {start_epoch}에폭 완료. 할 일이 없다.")
            return 0
        print(f"체크포인트에서 재개: {start_epoch}에폭부터")
        model = AutoModelForTokenClassification.from_pretrained(str(out / "last"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    # ⚠ LayoutXLM(=LayoutLMv2 구조)은 transformers 에서 gradient checkpointing 을
    # 지원하지 않는다(4.57 기준 ValueError). 계획 단계의 "bf16 + checkpointing +
    # 8-bit Adam ≈ 6GB" 추정은 이 기능을 전제했으므로 실측으로 다시 잡아야 한다.
    # 없어도 학습은 되지만 활성값이 그대로 남아 VRAM 이 늘어난다.
    if not a.no_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("gradient checkpointing 사용")
        except ValueError as e:
            print(f"gradient checkpointing 사용 불가 — {e}")
            print("  → 활성값이 그대로 남는다. VRAM 이 모자라면 --max-len 을 줄일 것")

    ex = build_examples(Path(a.train), Path(a.images))
    if a.limit:
        ex = ex[:a.limit]
        print(f"[스모크] 학습 문서를 {a.limit}건으로 제한한다")
    print(f"학습 문서 {len(ex)}건 / 라벨 {len(LABELS)}종")
    dl = DataLoader(KodxDataset(ex, proc, a.max_len), batch_size=a.batch,
                    shuffle=True, num_workers=0)

    opt = build_optimizer(model, a.lr, not a.no_8bit)

    # 정밀도 선택.
    # torch.cuda.is_bf16_supported()는 **에뮬레이션까지 True로 답한다.** T4(Turing,
    # compute 7.5)에서 True가 나오지만 텐서코어 bf16이 없어 실제로는 느리다.
    # 네이티브 bf16은 Ampere(8.0) 이상이므로 compute capability로 직접 판정한다.
    amp_dtype, scaler = None, None
    if dev == "cuda":
        major = torch.cuda.get_device_capability()[0]
        if major >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")   # fp16은 언더플로 방지가 필요하다
    prec = {torch.bfloat16: "bf16", torch.float16: "fp16(+GradScaler)"}.get(amp_dtype, "fp32")
    print(f"장치 {dev} / 정밀도 {prec} / 유효 배치 {a.batch * a.accum}")

    for epoch in range(start_epoch, a.epochs):
        model.train()
        running, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(dl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast(dev, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**batch).loss / a.accum
            (scaler.scale(loss) if scaler else loss).backward()
            running += loss.item() * a.accum
            n += 1
            if (step + 1) % a.accum == 0:
                if scaler:
                    scaler.unscale_(opt)                # 클리핑 전에 스케일을 되돌린다
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            if (step + 1) % 200 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(dl)} loss {running/n:.4f}",
                      flush=True)
            if a.max_steps and (step + 1) >= a.max_steps:
                peak = (torch.cuda.max_memory_allocated() / 1024**3) if dev == "cuda" else 0
                print(f"  [스모크] {step+1}스텝에서 중단 · loss {running/n:.4f} · "
                      f"최대 VRAM {peak:.2f} GB", flush=True)
                break

        model.save_pretrained(out / "last")
        proc.save_pretrained(out / "last")
        state.write_text(json.dumps({"epoch": epoch + 1, "seed": a.seed,
                                     "loss": running / max(1, n)}, ensure_ascii=False),
                         encoding="utf-8")
        print(f"epoch {epoch+1} 완료 · loss {running/max(1,n):.4f} · 저장됨")

    print(f"\n학습 완료 → {out/'last'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
