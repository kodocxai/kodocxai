"""집계 수준 — 서브워드 / 형태소 / 어절 (논문 3.2절).

RQ1의 조작 변인이다. 동일한 귀속 벡터를 세 수준에서 비교할 수 있어야 한다.

형태소 수준 점수는 중첩 비율 가중 합으로 정의된다.

    phi(m) = sum_tau ( |overlap(tau, m)| / |tau| ) * phi(tau)
    phi(w) = sum_{m in w} phi(m)

MeCab-ko가 없으면 형태소 수준은 비활성화되고 서브워드·어절 두 수준만 쓴다.
논문 표에 형태소 열이 필요하므로 결국 설치해야 한다 — 설치 안내는 README 참조.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# 조사(J*)·어미(E*)는 기능 형태소, 나머지는 내용 형태소로 본다(2.2절).
FUNCTIONAL_PREFIX = ("J", "E")


def _is_func_tag(tag: str) -> bool:
    return tag.startswith(FUNCTIONAL_PREFIX)


@dataclass
class Morph:
    surface: str
    tag: str
    word_idx: int
    start: int          # 어절 내 문자 오프셋
    end: int

    @property
    def parts(self) -> list[str]:
        """복합 태그를 성분으로 쪼갠다. `VV+EC` → ['VV', 'EC']."""
        return self.tag.split("+")

    @property
    def is_functional(self) -> bool:
        """성분이 **전부** 조사·어미일 때만 기능 형태소다."""
        return all(_is_func_tag(t) for t in self.parts)

    @property
    def is_content(self) -> bool:
        """성분에 조사·어미가 **하나도 없을** 때만 내용 형태소다."""
        return not any(_is_func_tag(t) for t in self.parts)

    @property
    def is_mixed(self) -> bool:
        """어간과 어미가 한 형태소로 묶인 경우. `한다`=XSV+EC, `되어`=VV+EC 등.

        mecab-ko-dic 이 활용형을 하나로 내보내는데, 표면형을 더 쪼갤 수 없다 —
        「한다」의 어미 'ㄴ다'는 음절 안에 들어 있어 문자 오프셋으로 가를 수 없다.
        **내용에도 기능에도 넣지 않는다.** 2.2절의 마스킹 3조건에서 (ii) 내용 형태소와
        (iii) 기능 형태소를 서로 배타적이고 순수하게 유지하기 위함이며, 그 대가로
        (ii)+(iii)가 (i) 전체와 같지 않다. 비율은 논문에 명시한다(실측 3.3%).
        """
        return not self.is_functional and not self.is_content


class Tagger:
    """MeCab-ko 래퍼. 없으면 사용 불가 상태로 남는다."""

    def __init__(self):
        self.impl = None
        self.name = None
        try:
            import mecab                      # python-mecab-ko
            self.impl = mecab.MeCab()
            self.name = "python-mecab-ko"
        except Exception:
            try:
                from konlpy.tag import Mecab   # konlpy
                self.impl = Mecab()
                self.name = "konlpy.Mecab"
            except Exception:
                self.impl = None

    @property
    def available(self) -> bool:
        return self.impl is not None

    def morphs(self, word: str, word_idx: int) -> list[Morph]:
        """어절 하나를 형태소로 분해하고 어절 내 문자 구간을 붙인다.

        형태소 분석기는 오프셋을 주지 않으므로 표면형을 앞에서부터 되찾아 맞춘다.
        표면형이 원문과 달라지는 경우(불규칙 활용 등)에는 되찾기가 실패하는데,
        그때는 남은 구간 전체를 그 형태소에 할당한다.
        """
        if not self.available:
            return [Morph(word, "UNK", word_idx, 0, len(word))]

        pairs = self.impl.pos(word)
        out: list[Morph] = []
        cur = 0
        for surface, tag in pairs:
            pos = word.find(surface, cur)
            if pos < 0:
                # 표면형 복원 실패. 남은 구간을 통째로 준다.
                out.append(Morph(surface, tag, word_idx, cur, len(word)))
                cur = len(word)
                continue
            out.append(Morph(surface, tag, word_idx, pos, pos + len(surface)))
            cur = pos + len(surface)
        return out or [Morph(word, "UNK", word_idx, 0, len(word))]


def subword_char_spans(tokenizer, words: list[str], encoding) -> list[Optional[tuple]]:
    """각 서브워드가 어느 어절의 어느 문자 구간인지 돌려준다.

    반환: 토큰 인덱스별 (word_idx, char_start, char_end) 또는 None(특수토큰).

    HuggingFace fast tokenizer의 offset_mapping은 `words` 리스트를 입력으로 준
    경우 **어절 내부 오프셋**을 준다. word_ids()와 짝지으면 (어절, 문자구간)이 나온다.
    """
    word_ids = encoding.word_ids() if hasattr(encoding, "word_ids") else encoding["word_ids"]
    offsets = encoding["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()

    out: list[Optional[tuple]] = []
    for t, wid in enumerate(word_ids):
        if wid is None:
            out.append(None)
            continue
        s, e = offsets[t]
        if e <= s:
            out.append(None)          # 빈 구간(특수 처리된 토큰)
            continue
        out.append((wid, int(s), int(e)))
    return out


def aggregate(phi_sub: list[float],
              spans: list[Optional[tuple]],
              words: list[str],
              level: str,
              tagger: Optional[Tagger] = None,
              mask: str = "all") -> tuple[list[float], list[dict]]:
    """서브워드 귀속 점수를 요청한 수준으로 재집계한다.

    level : "subword" | "morpheme" | "eojeol"
    mask  : "all" | "content" | "functional"  — 형태소 수준에서만 의미가 있다.
            3.2절의 마스킹 조건 3종에 대응한다.

    반환: (수준별 점수, 수준별 단위 메타)
          메타에는 word_idx가 들어 있어 어절 bbox로 되돌릴 수 있다.

    **⚠ LayoutXLM 의 시퀀스는 텍스트 토큰만이 아니다.** LayoutLMv2 계열은 시각 백본이
    낸 7×7=49 개 패치 토큰을 뒤에 이어 붙이므로 출력 길이가 512+49=561 이 된다.
    귀속 벡터도 561 개로 나오는데 뒤 49 개는 텍스트 단위에 대응하지 않는다.
    여기서는 **앞쪽 텍스트 구간만** 쓴다. 시각 토큰의 기여는 2.3.3 의 이미지 모달
    교란(occlusion)으로 따로 측정한다 — 텍스트 집계에 섞으면 두 모달이 뒤엉킨다.
    """
    phi_sub = list(phi_sub)[:len(spans)]

    if level == "subword":
        units = [{"word_idx": s[0] if s else None, "kind": "subword", "index": i}
                 for i, s in enumerate(spans)]
        return list(phi_sub), units

    if level == "eojeol":
        n = len(words)
        acc = [0.0] * n
        for phi, s in zip(phi_sub, spans):
            if s is None:
                continue
            acc[s[0]] += phi
        units = [{"word_idx": i, "kind": "eojeol", "index": i} for i in range(n)]
        return acc, units

    if level != "morpheme":
        raise ValueError(f"알 수 없는 집계 수준: {level}")

    if tagger is None or not tagger.available:
        raise RuntimeError(
            "형태소 수준에는 MeCab-ko가 필요하다. README의 설치 안내를 볼 것.")

    # 어절별 형태소 목록
    morphs: list[Morph] = []
    for i, w in enumerate(words):
        morphs.extend(tagger.morphs(w, i))

    by_word: dict[int, list[int]] = {}
    for j, m in enumerate(morphs):
        by_word.setdefault(m.word_idx, []).append(j)

    acc = [0.0] * len(morphs)
    for phi, s in zip(phi_sub, spans):
        if s is None:
            continue
        wid, ts, te = s
        tlen = max(1, te - ts)
        for j in by_word.get(wid, []):
            m = morphs[j]
            ov = min(te, m.end) - max(ts, m.start)
            if ov > 0:
                acc[j] += phi * (ov / tlen)      # 중첩 비율 가중

    # 어간과 어미가 한 형태소로 묶인 것(VV+EC 등, 실측 3.3%)은 어느 조건에도 넣지
    # 않는다. 그래야 내용 조건과 기능 조건이 서로 배타적이고 순수해진다.
    keep = range(len(morphs))
    if mask == "content":
        keep = [j for j in keep if morphs[j].is_content]
    elif mask == "functional":
        keep = [j for j in keep if morphs[j].is_functional]

    scores = [acc[j] for j in keep]
    units = [{"word_idx": morphs[j].word_idx, "kind": "morpheme", "index": j,
              "surface": morphs[j].surface, "tag": morphs[j].tag,
              "functional": morphs[j].is_functional,
              "mixed": morphs[j].is_mixed} for j in keep]
    return scores, units


def units_to_word_idx(units: list[dict], scores: list[float], top_k: int) -> list[int]:
    """상위 k개 단위가 가리키는 어절 인덱스 집합.

    타당성은 bbox 위에서 재므로, 어떤 수준에서 집계했든 어절로 되돌려야 한다.
    """
    if len(scores) != len(units):
        raise ValueError(
            f"scores({len(scores)})와 units({len(units)})의 길이가 다르다 — "
            f"aggregate() 가 짝이 맞지 않는 두 목록을 돌려줬다는 뜻이다")
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
    seen, out = set(), []
    for i in ranked:
        wi = units[i].get("word_idx")
        if wi is None or wi in seen:
            continue
        seen.add(wi)
        out.append(wi)
    return out
