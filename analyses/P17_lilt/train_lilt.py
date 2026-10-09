# -*- coding: utf-8 -*-
"""LiLT-InfoXLM 미세조정 (P17, 1시드 - R5 대안 모델 시도).

kodx_train.py 의 구조·하이퍼파라미터를 그대로 따르되(동일 데이터·에폭·배치·
누적·lr·시드 절차), 입력만 lilt_shim 경유(이미지 없음). kodx_train 자체는
건드리지 않는다(LayoutXLM 정본 경로 보존).

사용: python train_lilt.py --out runs_lilt\seed1 --seed 1 [--limit N --max-steps N]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

EXP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kodx_data import LABELS, LABEL2ID, build_examples        # noqa: E402
from lilt_shim import load_lilt_processor                      # noqa: E402

MODEL_NAME = "SCUT-DLVCLab/lilt-infoxlm-base"


class LiltDataset(Dataset):
    def __init__(self, examples, proc, max_len=512):
        self.ex, self.proc, self.max_len = examples, proc, max_len

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, i):
        e = self.ex[i]
        enc = self.proc(None, e.words, boxes=e.boxes, word_labels=e.labels,
                        truncation=True, padding="max_length",
                        max_length=self.max_len, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(EXP / "data" / "train.jsonl"))
    ap.add_argument("--images", default=str(EXP / "data" / "images224"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--no-checkpointing", action="store_true")
    ap.add_argument("--no-8bit", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="AMP 끄기(진단용)")
    # 실측(실행노트 10-07): SCUT 체크포인트의 InfoXLM 텍스트 흐름 가중치로는
    # 이 과제가 수렴하지 않는다(20건 과적합도 불가). LiLT 설계상 텍스트 흐름은
    # 임의 XLM-R 계열과 결합 가능하므로 xlm-roberta-base 로 치환 초기화한다
    # (레이아웃 흐름은 LiLT 사전학습 유지). 치환 후 수렴 정상(300스텝 loss 0.03).
    ap.add_argument("--xlmr-init", action="store_true",
                    help="텍스트 흐름을 xlm-roberta-base 가중치로 초기화")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForTokenClassification

    print(f"모델 로드: {a.model}")
    proc = load_lilt_processor(a.model)
    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(LABELS),
        id2label={i: l for l, i in LABEL2ID.items()}, label2id=LABEL2ID)
    if a.xlmr_init:
        from transformers import AutoModel
        xsd = AutoModel.from_pretrained("xlm-roberta-base").state_dict()
        tgt = model.lilt.state_dict()
        new = {k: xsd[k] for k in tgt if k in xsd and xsd[k].shape == tgt[k].shape}
        model.lilt.load_state_dict(new, strict=False)
        print(f"텍스트 흐름 xlm-roberta-base 초기화: {len(new)}/{len(tgt)}개 키 치환")

    start_epoch = 0
    state = out / "state.json"
    if state.exists():
        st = json.loads(state.read_text(encoding="utf-8"))
        start_epoch = st.get("epoch", 0)
        if start_epoch >= a.epochs:
            print(f"이미 {start_epoch}에폭 완료.")
            return 0
        print(f"재개: {start_epoch}에폭부터")
        model = AutoModelForTokenClassification.from_pretrained(str(out / "last"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    if not a.no_checkpointing:
        try:
            model.gradient_checkpointing_enable()
            print("gradient checkpointing 사용")
        except ValueError as e:
            print(f"gradient checkpointing 사용 불가 - {e}")

    # 학습 집합 구성은 kodx_train 과 동일(이미지 존재 문서로 한정해 같은 5,000건)
    ex = build_examples(Path(a.train), Path(a.images))
    if a.limit:
        ex = ex[:a.limit]
        print(f"[스모크] {a.limit}건 제한")
    print(f"학습 문서 {len(ex)}건 / 라벨 {len(LABELS)}종")
    dl = DataLoader(LiltDataset(ex, proc, a.max_len), batch_size=a.batch,
                    shuffle=True, num_workers=0)
    opt = None
    if not a.no_8bit:
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=a.lr)
            print("  8-bit Adam 사용")
        except Exception:
            pass
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)

    amp_dtype, scaler = None, None
    if dev == "cuda" and not a.fp32:
        if torch.cuda.get_device_capability()[0] >= 8:
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")
    print(f"장치 {dev} / 유효 배치 {a.batch * a.accum}")

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
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
            if (step + 1) % 200 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(dl)} "
                      f"loss {running/n:.4f}", flush=True)
            if a.max_steps and (step + 1) >= a.max_steps:
                peak = (torch.cuda.max_memory_allocated() / 1024**3) if dev == "cuda" else 0
                print(f"  [스모크] {step+1}스텝 중단 · loss {running/n:.4f} · "
                      f"최대 VRAM {peak:.2f} GB", flush=True)
                break

        model.save_pretrained(out / "last")
        proc.save_pretrained(out / "last")
        state.write_text(json.dumps({"epoch": epoch + 1, "seed": a.seed,
                                     "loss": running / max(1, n)},
                                    ensure_ascii=False), encoding="utf-8")
        print(f"epoch {epoch+1} 완료 · loss {running/max(1,n):.4f} · 저장됨", flush=True)

    print(f"\n학습 완료 → {out/'last'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
