# -*- coding: utf-8 -*-
"""LiLT용 프로세서 심(shim) - LayoutXLMProcessor 호출면 호환 (P17, 2026-10-07).

LiLT(lilt-infoxlm-base)는 텍스트+레이아웃만 받는다(시각 백본 없음). 기존
파이프라인(kodx_train 구조·eval_kie·kodx_eval)은 LayoutXLMProcessor의
`proc(img, words, boxes=..., word_labels=...)` 호출면에 묶여 있으므로, 같은
서명을 받되 이미지를 무시하고 input_ids·attention_mask·bbox(·labels)를
돌려주는 심을 둔다.

어휘 주의: LiLT-InfoXLM과 LayoutXLM은 같은 XLM-R SentencePiece 계열이라
서브워드 분절이 동일하다 - subword_char_spans·어절 다수결 디코딩이 그대로
맞는다. 라벨은 LayoutXLM 프로세서 기본과 같게 **첫 서브워드에만** 붙이고
나머지는 -100 (only_label_first_subword 동치).

bbox 규약: 어절 박스(0~1000 정규화)를 그 어절의 모든 서브워드에 복제,
특수·패딩 토큰은 [0,0,0,0] (LayoutLM 계열 표준 - 토크나이저가 처리).
"""
from __future__ import annotations


class LiltShimProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, img, words, boxes=None, word_labels=None,
                 truncation=True, padding="max_length", max_length=512,
                 return_offsets_mapping=False, return_tensors="pt"):
        # img 는 서명 호환용 - LiLT 에는 시각 입력이 없다.
        # 실측(2026-10-07): lilt-infoxlm-base 리포는 LayoutXLMTokenizerFast 를
        # 싣고 있어 boxes·word_labels 를 직접 받는다(bbox 정렬·첫 서브워드
        # 라벨링 포함 - LayoutXLM 프로세서와 동일 코드 경로). 그대로 위임한다.
        return self.tokenizer(
            words, boxes=boxes, word_labels=word_labels,
            truncation=truncation, padding=padding, max_length=max_length,
            return_offsets_mapping=return_offsets_mapping,
            return_tensors=return_tensors)

    def save_pretrained(self, path):
        self.tokenizer.save_pretrained(path)


def load_lilt_processor(name_or_path, **kw):
    """kodx_data.load_processor 대체재 (kw 는 apply_ocr 등 서명 호환용)."""
    from transformers import AutoTokenizer
    return LiltShimProcessor(AutoTokenizer.from_pretrained(name_or_path))
