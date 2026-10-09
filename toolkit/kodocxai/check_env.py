"""실행 환경 진단 — LayoutXLM을 돌릴 수 있는가.

추측하지 말고 이걸 돌려서 판정한다. 단계별로 무엇이 막혔는지 정확히 알려준다.

  python check_env.py              # 설치 상태만 (다운로드 없음, 몇 초)
  python check_env.py --model      # 모델 실제 로드까지 (약 1.4GB 다운로드)
  python check_env.py --model --forward   # 더미 입력 순전파까지 + VRAM 측정
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys

OK, NO, WARN = "  [OK]  ", "  [실패]", "  [주의]"


def probe(name: str, attr: str = "__version__") -> tuple[bool, str]:
    try:
        m = importlib.import_module(name)
        return True, str(getattr(m, attr, "설치됨"))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description="LayoutXLM 실행 환경 진단")
    ap.add_argument("--model", action="store_true", help="모델을 실제로 로드해 본다")
    ap.add_argument("--forward", action="store_true", help="더미 순전파까지 실행")
    ap.add_argument("--model-name", default="microsoft/layoutxlm-base")
    a = ap.parse_args()

    print(f"Python {platform.python_version()} / {platform.system()} {platform.release()}\n")

    print("=== 필수 패키지 ===")
    results = {}
    for pkg in ("torch", "transformers", "PIL", "numpy", "sentencepiece"):
        ok, info = probe(pkg)
        results[pkg] = ok
        print(f"{OK if ok else NO} {pkg:<14} {info[:70]}")

    print("\n=== LayoutXLM 시각 백본 ===")
    d_ok, d_info = probe("detectron2")
    results["detectron2"] = d_ok
    print(f"{OK if d_ok else NO} detectron2     {d_info[:70]}")
    if not d_ok:
        print("       └ LayoutXLM(=LayoutLMv2 구조)은 시각 백본에 detectron2를 쓴다.")
        print("         Windows용 공식 휠이 없어 MSVC 빌드가 필요하다.")

    print("\n=== 선택 패키지 ===")
    for pkg, why in (("bitsandbytes", "8-bit Adam (8GB VRAM에 권장)"),
                     ("mecab", "형태소 집계 — RQ1 표 3에 필요"),
                     ("konlpy", "mecab 대체 경로")):
        ok, info = probe(pkg)
        print(f"{OK if ok else WARN} {pkg:<14} {info[:40]:<40} {why}")

    print("\n=== GPU ===")
    if results.get("torch"):
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            vram = p.total_memory / 1024**3
            print(f"{OK} {p.name}  VRAM {vram:.1f} GB")
            print(f"{OK if torch.cuda.is_bf16_supported() else WARN} bf16 지원: "
                  f"{torch.cuda.is_bf16_supported()}")
            if vram < 10:
                print(f"{WARN} 10GB 미만 — kodx_train.py 기본값(batch 1, 누적 16, "
                      f"gradient checkpointing, 8-bit Adam)을 그대로 쓸 것")
        else:
            print(f"{NO} CUDA 사용 불가 — CPU 학습은 현실적이지 않다")
    else:
        print(f"{NO} torch 없음 — 판정 불가")

    if not a.model:
        print("\n설치 상태만 확인했다. 모델이 실제로 로드되는지 보려면 --model 을 붙일 것.")
        return 0 if all(results.values()) else 1

    print(f"\n=== 모델 로드: {a.model_name} ===")
    try:
        from transformers import LayoutXLMProcessor, AutoModelForTokenClassification
        print("  프로세서 로드 중…", flush=True)
        # AutoProcessor 를 쓰면 LayoutLMv2 계열(WordPiece)을 잡아 와 첫 인코딩에서
        # ValueError: Id not recognized 로 죽는다. 전용 클래스를 명시한다.
        proc = LayoutXLMProcessor.from_pretrained(a.model_name, apply_ocr=False)
        print(f"{OK} 프로세서 OK  ({type(proc.tokenizer).__name__}, "
              f"mask_token_id={proc.tokenizer.mask_token_id})")

        print("  모델 로드 중… (최초 1회 약 1.4GB 다운로드)", flush=True)
        model = AutoModelForTokenClassification.from_pretrained(a.model_name, num_labels=11)
        n = sum(p.numel() for p in model.parameters())
        print(f"{OK} 모델 OK  파라미터 {n/1e6:.1f}M")
    except Exception as e:
        print(f"{NO} 모델 로드 실패\n       {type(e).__name__}: {e}")
        if "detectron2" in str(e).lower():
            print("\n       → detectron2 문제다. README의 대응 선택지 3개를 볼 것.")
        return 1

    if not a.forward:
        print("\n모델 로드까지 성공. 순전파와 VRAM을 보려면 --forward 를 붙일 것.")
        return 0

    print("\n=== 더미 순전파 ===")
    try:
        import torch
        from PIL import Image
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev).eval()
        img = Image.new("RGB", (224, 224), "white")
        words = ["제목", "공장설립", "승인", "통보"]
        boxes = [[100, 100, 200, 130], [220, 100, 400, 130],
                 [410, 100, 500, 130], [510, 100, 600, 130]]
        enc = proc(img, words, boxes=boxes, return_tensors="pt",
                   padding="max_length", truncation=True, max_length=512)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        print(f"{OK} 순전파 OK  logits {tuple(out.logits.shape)}")
        if dev == "cuda":
            print(f"{OK} 추론 시 VRAM 사용 "
                  f"{torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
            print("       └ 학습은 여기에 그래디언트·옵티마이저 상태가 더 붙는다")
    except Exception as e:
        print(f"{NO} 순전파 실패\n       {type(e).__name__}: {e}")
        return 1

    print("\n전부 통과. kodx_train.py 를 돌릴 수 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
