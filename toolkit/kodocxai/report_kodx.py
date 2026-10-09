# -*- coding: utf-8 -*-
"""KoDocXAI 벤치마크 v1.0 — 논문 보고용 통계 생성기.
   (파일명과 출력 경로 `KoDX-Eval_통계.md` 는 동결물이라 바꾸지 않는다)

주석 결과와 파일럿 세트에서 3.4절·표 2에 넣을 수치를 뽑아 마크다운 한 장으로 낸다.
언제든 다시 돌릴 수 있고, 돌릴 때마다 같은 입력에 같은 출력이 나온다.

    python report_kodx.py                 # _pilot\KoDX-Eval_통계.md 생성
    python report_kodx.py --out other.md

Python 3.10 표준 라이브러리만 쓴다 (프로젝트 규약).

설계 메모
---------
* 어댑터 정확도는 `status`/`source` 플래그가 아니라 **값 비교**로 낸다.
  어절을 확인차 클릭만 해도 source 가 human 이 되고 되돌아가지 않기 때문이다.
  기준선은 `docs.jsonl` 의 자동 초안이고, 이 파일은 주석 중 절대 바뀌지 않아야 한다.
* 채점 정규화(`norm`)는 gold 와 예측 양쪽에 똑같이 적용한다. 3.4절에 그대로 적는다.
* string_na(직인 겹침·두 줄 묶임)는 **영역은 유효하고 문자열만 무효**다.
  문자열 지표에서는 빼고, 영역 지표와 부재율에서는 살린다.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PILOT = Path(r"D:\AIHub데이터\_pilot")
FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]
PRESENT = ("ok", "corrected", "string_na")   # 필드가 문서에 있다고 판정된 상태
STR_USABLE = ("ok", "corrected")             # 문자열까지 신뢰할 수 있는 상태


def STATUS3(s: str | None) -> str:
    """저장 상태를 보고 단위 3값으로 접는다. `corrected` 는 `ok` 와 같은 뜻이다."""
    return "absent" if s == "absent" else ("string_na" if s == "string_na" else "present")

# 「1건 기안 다수 시행」 탐지용 (4-1절). 표기가 「제 2 안」·「(제'3안)」·「[ 2 안]」처럼
# 흔들려 공백과 따옴표를 허용한다. 헤더 세트를 셀 때 하단 「수신처 :」 명단은 제외한다 —
# 그것은 또 하나의 수신 필드가 아니라 「수신처 참조」가 가리키는 명단이다.
AN_MARK = re.compile(r"[\[(「]?\s*제?\s*['\"]?\s*[1-9]\s*안")
HDR_SU = re.compile(r"^\s*수\s*신(?!\s*처)")
HDR_JE = re.compile(r"^\s*제\s*목")


# ---------------------------------------------------------------- 정규화
def norm(s: str | None) -> str:
    """채점용 정규화. **gold 와 예측에 동일하게 적용한다.**

    1) 공백 제거 — 한국어 OCR 라벨은 「김 해 시 장」처럼 한 단어가 쪼개져 있다
    2) 앞뒤 구두점 제거 — 「수 신 : 김해시…」의 콜론이 독립 어절로 잡히는 일이 잦다
       (수신 자동추출의 21%). 라벨 기호이지 값이 아니다
    """
    if not s:
        return ""
    t = "".join(s.split())
    while t and unicodedata.category(t[0]).startswith("P"):
        t = t[1:]
    while t and unicodedata.category(t[-1]).startswith("P"):
        t = t[:-1]
    return t


def stratum(rec: dict) -> str:
    year = int(rec["source"]["year"])
    band = "1994-96" if year <= 1996 else ("1997-99" if year <= 1999 else "2000+")
    title = norm((rec.get("fields", {}).get("제목") or {}).get("value_norm"))
    if not title:
        return f"{band} / 제목없음"
    for s in ("건축", "공장", "농지", "창업"):
        if s in title:
            return f"{band} / {s}"
    return f"{band} / 기타"


def pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "—"


# ---------------------------------------------------------------- 적재
def load_docs() -> list[dict]:
    return [json.loads(l) for l in (PILOT / "docs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def load_annotations(suffix: str = "") -> dict[str, dict[str, dict]]:
    """{주석자: {doc_id: 기록}}. append-only 이므로 같은 doc_id 는 나중 것이 이긴다.

    **값은 나중 것이 이기지만 소요 시간은 그러면 안 된다.** 화살표로 앞뒤를 오가면
    그때마다 재저장되어(실측 문서당 평균 4.3회) 마지막 기록의 `dwell_ms` 는 0에
    가깝다. 그대로 중앙값을 내면 33초짜리 작업이 0.6초로 보고된다.

    그래서 그 문서에 들인 **모든 방문의 합**을 `dwell_total_ms` 로, 처음 판단에 걸린
    시간을 `dwell_first_ms` 로 따로 새겨 둔다. 원 기록이 다 남아 있어 사후 복원이 된다.
    """
    out: dict[str, dict[str, dict]] = {}
    root = PILOT / "_backup" if suffix else PILOT
    for path in sorted(root.glob(f"annotations_*{suffix}.jsonl")):
        who = path.stem.replace("annotations_", "").replace(suffix, "")
        recs: dict[str, dict] = {}
        dwell: dict[str, list[int]] = {}
        seen_ts: dict[str, list] = {}
        seen_gl: dict[str, set] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["doc_id"]] = r
                if isinstance(r.get("dwell_ms"), int):
                    dwell.setdefault(r["doc_id"], []).append(r["dwell_ms"])
                if r.get("ts"):
                    seen_ts.setdefault(r["doc_id"], []).append(r["ts"])
                if r.get("guideline"):
                    seen_gl.setdefault(r["doc_id"], set()).add(r["guideline"])
        for doc_id, ds in dwell.items():
            recs[doc_id]["dwell_first_ms"] = ds[0]
            recs[doc_id]["dwell_total_ms"] = sum(ds)
            recs[doc_id]["visits"] = len(ds)
        # 작업 시각과 기준 판도 마지막 기록만 보면 안 된다. 재검토로 다시 저장하면
        # 그 시점의 값으로 덮여, 「8시간 작업」이 「14분 작업」으로 보고된다.
        for doc_id, r in recs.items():
            r["ts_first"] = seen_ts.get(doc_id, [r.get("ts")])[0]
            r["guidelines_all"] = sorted(seen_gl.get(doc_id, set()))
        if recs:
            out[who] = recs
    return out


# ---------------------------------------------------------------- 지표
def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """2x2 (있음/없음) Cohen's κ."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ---------------------------------------------------------------- 본문
def build(out_path: Path) -> str:
    docs = load_docs()
    by_id = {d["doc_id"]: d for d in docs}
    anns = load_annotations()
    # IAA 는 **조정 전(pre-adjudication)** 주석으로 낸다. 조정 후 골드로 계산하면
    # 주석자 간 일치도가 아니라 조정 작업의 흔적을 재게 된다. `_backup\` 에 스냅숏이
    # 있으면 6절만 그것으로 계산하고, 없으면 현재 파일을 쓴다(그 경우 둘은 같다).
    anns_iaa = load_annotations(suffix=".pre_adjudication")
    adjudicated = bool(anns_iaa)
    if not anns_iaa:
        anns_iaa = anns
    # 주 주석자 = 기록이 가장 많은 사람. 파일명 사전순으로 잡으면 안 된다 —
    # 두 번째 주석자는 `--only-double` 로 60건만 달기 때문에, 이름이 앞서면
    # 표 1과 어댑터 기준선이 300건이 아니라 60건으로 계산된다(실측: lee < park).
    anns = dict(sorted(anns.items(), key=lambda kv: -len(kv[1])))
    manifest = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))

    L: list[str] = []
    add = L.append

    # 표기는 KoDocXAI 로 통일한다(명명결정_KoDocXAI_202608.v2.md).
    # **파일 경로 `KoDX-Eval_통계.md` 와 시드 문자열은 동결물이라 바꾸지 않는다.**
    add("# KoDocXAI 벤치마크 v1.0 — 통계")
    add("")
    add(f"생성 {datetime.now().isoformat(timespec='seconds')} · `_tools\\report_kodx.py`")
    add("")
    add("이 파일은 **자동 생성물이다.** 손으로 고치지 말고 스크립트를 다시 돌린다.")
    add("논문 3.4절과 표 2의 수치는 모두 여기서 가져온다.")
    add("")

    # ── 1. 출처와 표집
    src = manifest.get("source", {})
    order = manifest.get("order", {})
    add("## 1. 출처와 표집")
    add("")
    add("| 항목 | 값 |")
    add("|---|---|")
    add(f"| 원 데이터셋 | {src.get('dataset', '공공행정문서 OCR')} (AI-Hub) |")
    add(f"| 서식 | {src.get('category', '인.허가')} 시행문 |")
    add(f"| 모집단 | {src.get('population', '—'):,}건 |" if isinstance(src.get("population"), int)
        else f"| 모집단 | {src.get('population', '—')} |")
    add(f"| 후보 풀 | {src.get('pool', '—')}건 (step {src.get('pool_step', '—')}) |")
    add(f"| 표본 | {len(docs)}건 |")
    add(f"| 층화 | 연도 3구간 × 제목 계열 → {len(manifest.get('strata', {}))}개 층, 비례 배분 |")
    # 표 셀 안의 파이프는 escape 해야 열이 깨지지 않는다 (method 에 sha256(SEED|doc_id) 가 들어간다).
    # f-string 표현식 안에는 백슬래시를 못 쓰므로(3.10) 밖에서 만든다.
    method = order.get("method", "—").replace("|", "\\|")
    add(f"| 제시 순서 | {method} |")
    add(f"| 순서 시드 | `{order.get('seed', '—')}` |")
    add("")
    add("제시 순서를 무작위화한 이유는 주석자의 숙련도 변화가 특정 층에 몰리지 않게 하기")
    add("위해서다. 시드가 고정되어 있어 같은 순서를 언제든 복원할 수 있다.")
    add("")

    # ── 2. 층별 분포
    strata = Counter(stratum(d) for d in docs)
    add("## 2. 층별 분포")
    add("")
    add("| 층 | 문서 수 | 비율 |")
    add("|---|---:|---:|")
    for k in sorted(strata):
        add(f"| {k} | {strata[k]} | {pct(strata[k], len(docs))} |")
    add(f"| **합계** | **{len(docs)}** | 100.0% |")
    add("")

    # ── 3. 주석 진행
    add("## 3. 주석 진행")
    add("")
    if not anns:
        add("아직 주석 기록이 없다.")
        add("")
    else:
        add("| 주석자 | 완료 | 진행률 | 기준 판 | 작업 기간 | 첫 판단(중앙값) | 총 소요(중앙값) |")
        add("|---|---:|---:|---|---|---:|---:|")
        for who, recs in anns.items():
            # 시작은 **최초** 기록, 끝은 **마지막** 기록. 재검토 재저장에 덮이지 않게 한다
            starts = sorted(r["ts_first"] for r in recs.values() if r.get("ts_first"))
            ends = sorted(r["ts"] for r in recs.values() if r.get("ts"))
            span = f"{starts[0][:16]} ~ {ends[-1][:16]}" if starts and ends else "—"

            def med_of(key: str) -> str:
                v = [r[key] for r in recs.values() if isinstance(r.get(key), int)]
                return f"{statistics.median(v) / 1000:.0f}초" if v else "—"

            # 거쳐 온 기준 판 전부. 마지막 것만 보면 규칙 개정 이력이 사라진다
            gv = ", ".join(sorted({g for r in recs.values()
                                   for g in (r.get("guidelines_all") or [r.get("guideline", "?")])}))
            add(f"| {who} | {len(recs)} | {pct(len(recs), len(docs))} | {gv} | {span} | "
                f"{med_of('dwell_first_ms')} | {med_of('dwell_total_ms')} |")
        add("")
        add("소요 시간은 문서 체류 시간이다. 자리를 비우면 부풀 수 있어 평균이 아니라")
        add("중앙값으로 보고한다.")
        add("")
        add("**두 열을 나눈 이유.** 화살표로 앞뒤를 오가면 그때마다 재저장되므로, 마지막")
        add("기록의 체류 시간만 보면 0초에 가깝다. 「첫 판단」은 그 문서를 처음 확정할 때")
        add("걸린 시간이고, 「총 소요」는 재방문까지 더한 값이다. 논문에는 **첫 판단**을")
        add("쓴다 — 재방문은 규칙 개정 때문에 생긴 것이라 정상 작업 속도가 아니다.")
        add("")

    primary = next(iter(anns.values()), {})

    # ── 4. 필드별 통계 (표 2 본체)
    add("## 4. 필드별 통계 — 표 2")
    add("")
    if not primary:
        add("주석 기록이 없어 산출할 수 없다.")
        add("")
    else:
        add("| 필드 | 있음 | 없음 | 문자열 불가 | 부재율 | 평균 길이(자) | 평균 어절 |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for f in FIELDS:
            vals = [v for v in (r["fields"].get(f) for r in primary.values()) if v]
            present = [v for v in vals if v["status"] in PRESENT]
            absent = [v for v in vals if v["status"] == "absent"]
            na = [v for v in vals if v["status"] == "string_na"]
            usable = [v for v in present if v["status"] in STR_USABLE]
            chars = [len(norm(v.get("value_norm") or v.get("value"))) for v in usable]
            words = [len(v.get("word_ids") or []) for v in present]
            c = f"{statistics.mean(chars):.1f}" if chars else "—"
            w = f"{statistics.mean(words):.1f}" if words else "—"
            add(f"| {f} | {len(present)} | {len(absent)} | {len(na)} | "
                f"{pct(len(absent), len(vals))} | {c} | {w} |")
        add("")
        add("「문자열 불가」는 **영역은 정답이고 문자열만 복원 불가**인 경우다 — 직인 겹침(가이드 5-3),")
        add("두 줄이 한 어절로 묶인 라벨(5-2). 근거 타당성(RQ2) 표본에는 남고 문자열 채점에서만 빠진다.")
        add("")

    # ── 4-1. 다중 시행문 (제N안)
    add("### 4-1. 한 장에 시행문이 여러 벌인 문서 — 「1건 기안 다수 시행」")
    add("")
    an_mark = sum(1 for d in docs if AN_MARK.search(" ".join(w["text"] for w in d["words"])))
    multi = 0
    for d in docs:
        lines = [ln["text"] for ln in d.get("lines", [])]
        su = sum(1 for t in lines if HDR_SU.match(t))
        je = sum(1 for t in lines if HDR_JE.match(t))
        multi += max(su, je) >= 2
    add(f"| 항목 | 문서 수 | 비율 |")
    add("|---|---:|---:|")
    add(f"| 「제N안」 표식이 있는 문서 | {an_mark} | {pct(an_mark, len(docs))} |")
    add(f"| 한 장에 헤더 세트 2벌 이상 | {multi} | {pct(multi, len(docs))} |")
    if primary:
        withsec = [r for r in primary.values() if r.get("secondary_spans")]
        spans = [s for r in withsec for s in r["secondary_spans"]]
        by_f = Counter(s.get("field") for s in spans)
        add(f"| 주석 완료분 중 분리가 일어난 문서 | {len(withsec)} | {pct(len(withsec), len(primary))} |")
        add("")
        add(f"분리된 스팬 {len(spans)}개 — "
            + (" · ".join(f"{f} {n}개" for f, n in by_f.most_common()) or "없음"))
    add("")
    add("공문서는 한 번의 결재로 수신자별 시행문을 여러 통 만든다. 각 안은 서로 다른")
    add("수신자에게 나가는 **별개의 시행문**이므로, 정답은 페이지 최상단 안 하나로 한다")
    add("(가이드 1-1). 아래쪽 안의 스팬은 버리지 않고 `secondary_spans` 로 옮겨,")
    add("평가에서 오답이 아니라 **제외(ignore)** 로 다룬다 — 모델이 제3안의 수신을 짚은 것은")
    add("틀린 것이 아니라 다른 시행문을 본 것이기 때문이다.")
    add("")

    # ── 5. 어댑터 기준선
    add("## 5. 자동 추출(어댑터) 기준선")
    add("")
    add("사람 정답 대비 어댑터 초안의 정확도다. **상태 플래그가 아니라 값 비교로 낸다** —")
    add("주석자가 확인차 어절을 클릭하기만 해도 `source` 가 `human` 으로 바뀌고 되돌아가지")
    add("않기 때문에, 플래그로 세면 어댑터 오류가 부풀려진다.")
    add("")
    if not primary:
        add("주석 기록이 없어 산출할 수 없다.")
        add("")
    else:
        add("| 필드 | 탐지 P | 탐지 R | 값 일치(탐지분) | 종단 정확도 |")
        add("|---|---:|---:|---:|---:|")
        for f in FIELDS:
            tp = fp = fn = 0
            hit = tot = 0
            for doc_id, r in primary.items():
                g = r["fields"].get(f)
                if not g:
                    continue
                a = ((by_id.get(doc_id) or {}).get("fields") or {}).get(f)
                gold_present = g["status"] in PRESENT
                proposed = a is not None
                if proposed and gold_present:
                    tp += 1
                elif proposed and not gold_present:
                    fp += 1
                elif not proposed and gold_present:
                    fn += 1
                if gold_present and g["status"] in STR_USABLE:
                    tot += 1
                    if proposed and norm(a.get("value_norm")) == norm(g.get("value_norm") or g.get("value")):
                        hit += 1
            p = pct(tp, tp + fp)
            rc = pct(tp, tp + fn)
            add(f"| {f} | {p} | {rc} | {pct(hit, tp)} | {pct(hit, tot)} |")
        add("")
        add("탐지 = 어댑터가 무언가를 제안했는가. 값 일치 = 제안한 것 중 정답과 같은가.")
        add("종단 = 정답이 있는 문서 중 어댑터가 제안까지 맞힌 비율. 문자열 불가는 분모에서 뺀다.")
        add("")

    # ── 6. IAA
    add("## 6. 주석자 간 일치도 (IAA)")
    add("")
    double_ids = {d["doc_id"] for i, d in enumerate(docs) if i % 5 == 0}
    add(f"이중 주석 대상은 제시 순서 5의 배수 **{len(double_ids)}건(20%)** 이다.")
    add("")
    def iaa_table(a_recs: dict, b_recs: dict, shared: list) -> None:
        add("| 필드 | 있음/없음 κ | 상태 3값 일치 | 스팬 완전일치 | 스팬 Jaccard | 문자열 일치 |")
        add("|---|---:|---:|---:|---:|---:|")
        for f in FIELDS:
            pairs, st_eq, st_n = [], 0, 0
            span_eq, span_n, jac, str_eq, str_n = 0, 0, [], 0, 0
            for doc_id in shared:
                x = a_recs[doc_id]["fields"].get(f)
                y = b_recs[doc_id]["fields"].get(f)
                if not x or not y:
                    continue
                px, py = x["status"] in PRESENT, y["status"] in PRESENT
                pairs.append((px, py))
                # 있음/없음 κ 는 「문자열 불가」와 「확인」을 구분하지 못한다. string_na 판정은
                # 문자열 채점 표본을 가르는 기준이므로 3값 일치율을 따로 낸다.
                st_n += 1
                st_eq += (STATUS3(x["status"]) == STATUS3(y["status"]))
                if px and py:
                    sx, sy = set(x.get("word_ids") or []), set(y.get("word_ids") or [])
                    span_n += 1
                    span_eq += (sx == sy)
                    jac.append(jaccard(sx, sy))
                    if x["status"] in STR_USABLE and y["status"] in STR_USABLE:
                        str_n += 1
                        str_eq += norm(x.get("value_norm")) == norm(y.get("value_norm"))
            k = cohen_kappa(pairs)
            ks = f"{k:.3f}" if k is not None else "—"
            js = f"{statistics.mean(jac):.3f}" if jac else "—"
            add(f"| {f} | {ks} | {pct(st_eq, st_n)} | {pct(span_eq, span_n)} | {js} | "
                f"{pct(str_eq, str_n)} |")
        add("")

    if len(anns_iaa) < 2:
        who = ", ".join(anns_iaa) or "(없음)"
        add(f"현재 주석자 **{len(anns_iaa)}명** ({who}) — 두 번째 주석자가 `--only-double` 로 작업하면")
        add("이 절이 자동으로 채워진다.")
        add("")
        add("```powershell")
        add("python annotate.py --annotator <이름2> --only-double")
        add("```")
        add("")
    else:
        names = sorted(anns_iaa, key=lambda k: -len(anns_iaa[k]))
        shared = sorted(set(anns_iaa[names[0]]) & set(anns_iaa[names[1]]) & double_ids)
        add(f"주석자 **{names[0]}** vs **{names[1]}** · 공통 {len(shared)}건")
        add("")

        if adjudicated:
            add("### 6-1. 발신명의 정의 명확화 **전** (가이드 v1.3)")
            add("")
            add("두 주석자가 **공통으로** 상단 레터헤드를 발신명의로 판정한 시기의 값이다.")
            add("가이드가 「하단에 명의가 없는 경우」를 규정하지 않았던 탓이며, 부주의가 아니다.")
            add("**같은 오판을 공유하면 일치도는 높아진다** — 이 표가 그 상한을 보여 준다.")
            add("")
            iaa_table(anns_iaa[names[0]], anns_iaa[names[1]], shared)
            add("### 6-2. 발신명의 정의 명확화 **후** (가이드 v1.4)")
            add("")
            add("규칙 1-2를 세운 뒤 두 주석자가 각각 해당 문서를 재판정한 결과다.")
            add("**논문에 싣는 값은 이쪽이다.** 6-1은 필드 정의의 모호성이 일치도를 얼마나")
            add("부풀리는지를 보이는 대조군으로만 쓴다.")
            add("")
            post = sorted(set(anns[names[0]]) & set(anns[names[1]]) & double_ids)
            iaa_table(anns[names[0]], anns[names[1]], post)
        else:
            iaa_table(anns_iaa[names[0]], anns_iaa[names[1]], shared)

        add("κ 는 필드 존재 여부(있음/없음)에 대한 Cohen's κ 다. **상태 3값 일치**는")
        add("있음·없음·문자열 불가 세 값에 대한 단순 일치율로, κ 가 구분하지 못하는")
        add("`string_na` 판정의 재현성을 본다. 스팬 Jaccard 는 두 주석자가 고른 어절 집합의")
        add("교집합/합집합이며, 근거 정답 $R_k$ 의 안정성을 나타낸다.")
        add("")

    # ── 7. 재현 절차
    add("## 7. 재현 절차")
    add("")
    add("```")
    add("1) 층화 추출     _tools\\prep_pilot.py       → _pilot\\docs.jsonl, images/, manifest.json")
    add(f"2) 순서 무작위화  sha256(\"{order.get('seed','—')}\" | doc_id) 오름차순")
    add("3) 주석          _tools\\annotate.py         → _pilot\\annotations_<주석자>.jsonl")
    add("4) 통계          _tools\\report_kodx.py      → 이 파일")
    add("```")
    add("")
    add("주석 기준은 `_pilot\\주석가이드.md`(논문 3.4절 공개 대상)이며, 각 기록에 적용 판이")
    add("`guideline` 키로 새겨진다. 기준이 바뀌면 판을 올리고 영향받는 기록을 가려낼 수 있다.")
    add("")
    add("**어댑터(`aihub_adapter.py`)와 `docs.jsonl` 은 주석 기간 중 고정한다.** 자동 초안이")
    add("기준선이므로 도중에 바뀌면 앞뒤 문서의 정확도를 하나의 수치로 낼 수 없다.")
    add("")

    text = "\n".join(L)
    out_path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PILOT / "KoDX-Eval_통계.md"))
    a = ap.parse_args()
    out = Path(a.out)
    build(out)
    print(f"생성: {out}")


if __name__ == "__main__":
    main()
