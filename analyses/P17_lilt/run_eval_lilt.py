# -*- coding: utf-8 -*-
"""LiLT 평가 래퍼 (P17) - 정본 eval_kie·kodx_eval 을 무수정 재사용.

원리: 두 정본 스크립트는 kodx_data.load_processor 로 프로세서를 얻는다.
여기서 그 참조를 LiLT 심으로 바꿔치운 뒤 정본 main() 을 그대로 부른다 -
채점·집계·프로토콜 코드 경로가 LayoutXLM 평가와 완전히 동일해진다.
(이미지 인자는 문서 집합 구성에만 쓰이고 심이 무시한다.)

사용:
  python run_eval_lilt.py kie    --model runs_lilt\seed1\last --out kie_lilt_seed1.json
  python run_eval_lilt.py attrib --model runs_lilt\seed1\last --out lilt_seed1_v2.jsonl \
      --save-phi phi_lilt_seed1.jsonl
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(HERE))

import kodx_data                                   # noqa: E402
from lilt_shim import load_lilt_processor          # noqa: E402

kodx_data.load_processor = load_lilt_processor


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in ("kie", "attrib"):
        print("첫 인자: kie | attrib")
        return 2
    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "kie":
        import eval_kie
        eval_kie.load_processor = load_lilt_processor   # 모듈 top-level 바인딩 치환
        sys.argv = ["eval_kie.py"] + rest
        eval_kie.main()
        return 0
    import kodx_eval                                    # main 내부에서 kodx_data 참조
    sys.argv = ["kodx_eval.py",
                "--docs", r"D:\AIHub데이터\_pilot\docs.jsonl",
                "--images", str(EXP / "data" / "pilot_images224"),
                "--annotations", r"D:\AIHub데이터\_pilot\annotations_park.jsonl",
                "--skip-naopc"] + rest
    return kodx_eval.main()


if __name__ == "__main__":
    sys.exit(main())
