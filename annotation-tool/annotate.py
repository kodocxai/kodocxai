"""KoDocXAI 주석 도구.

어댑터가 자동 추출한 필드를 사람이 검수해 정답으로 승격시킨다.
전부 손으로 치는 것이 아니라 O/X만 찍는 구조라 건당 수 초면 끝난다.

설계 원칙
  - **매 건 즉시 디스크에 기록한다.** 저장 버튼이 없다. 데스크탑이 꺼져도 직전 건까지 남는다
  - 재시작하면 마지막으로 주석한 다음 문서부터 자동으로 이어간다
  - 표준 라이브러리만 쓴다. 설치할 것이 없다

사용 예
  python annotate.py --annotator kim
  python annotate.py --annotator lee --only-double   # 이중 주석 대상만

브라우저가 자동으로 열린다. 열리지 않으면 http://127.0.0.1:8765 로 접속한다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PILOT = Path(r"D:\AIHub데이터\_pilot")
FIELDS = ["제목", "수신", "문서번호", "시행일자", "발신명의"]

# 저장 가능한 확정 상태. 어절 클릭 직후의 'corrected' 는 중간 상태이며, O 로 확정해야 한다.
#   ok         값이 맞다
#   absent     이 문서에 그 필드가 없다
#   string_na  영역은 맞으나 문자열을 복원할 수 없다 (직인 겹침·두 줄 병합 라벨)
FINAL = ("ok", "absent", "string_na")

# 이중 주석 비율 — 2.md 스코프 컷: 이중 주석은 부분(10~20%)만, IAA는 그 부분에서 산출
DOUBLE_EVERY = 5     # 5건마다 1건 = 20%

# doc_id → 300건 제시 순서에서의 위치. `--only-double`·`--only-ids` 로 목록을 걸러내도
# 기록에는 원래 위치가 남아야 한다. main() 에서 채운다.
TRUE_INDEX: dict = {}

# 적용 중인 주석 기준의 판(版). 기록마다 새겨 두면 규칙이 바뀌었을 때 어느 기록이 어느
# 기준으로 만들어졌는지 사후에 가려낼 수 있다. 가이드를 고치면 여기도 올린다.
#   v1.0  최초
#   v1.1  직인 겹침 M(5-3), 상단 서식란 우선을 전 헤더 필드로 확대(1), 구두점 어절 제외(5-4)
#   v1.2  5-4를 「앞뒤 구두점 어절만 제외, 중간은 유지」로 한정하고 도구가 자동 처리
#         시행일자 말미의 마침표는 날짜 표기의 일부이므로 남긴다 (저자 결정 2026-08-14)
#   v1.3  제N안 다중 등장(1-1) — 정답은 최상단 안, 나머지는 secondary_spans 로 분리
#   v1.4  상단 레터헤드는 발신명의가 아니다(1-2). 하단에 명의가 없으면 없음(N)
GUIDELINE = "v1.4"


# 값의 앞뒤에 붙은, 구두점만으로 된 어절. 「수 신 : 김해시…」의 콜론처럼 라벨 기호이지
# 값이 아니다(가이드 5-4). 자동 초안의 20~23%에 들어 있다.
#
# **안쪽 것은 절대 빼지 않는다.** 「공영16000 - 135」의 하이픈, 「1997. 8 . 13」의 점은
# 값의 일부라, 빼면 정규화 후 「공영16000135」로 붙어 채점에서 틀린다(모집단 51건).
# report_kodx.py 의 norm() 도 앞뒤 구두점만 제거하므로 규칙이 서로 맞는다.
_EDGE_PUNCT = re.compile(r"^[\s:：.,·'\"()（）\[\]/\-~]+$")

# 말미 마침표를 남기는 필드. 「96. 8. 16 .」의 끝 점은 라벨 기호가 아니라 한국어 날짜
# 표기의 일부다. 마침표만 해당하며 콜론·괄호는 시행일자에서도 뗀다 — 그것들은 서식
# 기호다. 채점 norm() 이 앞뒤 구두점을 지우므로 수치에는 영향이 없고, 값 문자열과
# 근거 정답 $R_k$ 의 영역이 원문에 충실해진다. (저자 결정 2026-08-14)
_KEEP_TAIL_DOT = ("시행일자",)
_DOT_ONLY = re.compile(r"^[.\s]+$")


def trim_ids(name: str, ids: list, text: dict) -> list:
    """스팬에서 앞뒤 구두점 어절을 뗀 id 목록. 안쪽은 손대지 않는다."""
    keep = list(ids)
    while keep and _EDGE_PUNCT.fullmatch(text.get(keep[0], "\0") or "\0"):
        keep.pop(0)
    while keep and _EDGE_PUNCT.fullmatch(text.get(keep[-1], "\0") or "\0"):
        if name in _KEEP_TAIL_DOT and _DOT_ONLY.fullmatch(text.get(keep[-1], "\0") or "\0"):
            break
        keep.pop()
    return keep


def trim_edge_punct(fields: dict, words: list) -> dict:
    """자동 초안 스팬에서 앞뒤 구두점 어절을 떼어낸다.

    어댑터는 콜론을 **문자열에서만** 빼고 `word_ids`·`bbox` 에는 남겨 둔다
    (`aihub_adapter.py` 의 `_STRIP_LEAD` 가 값 문자열에만 걸린다). 그래서 주석자
    화면에서는 콜론을 빼든 두든 똑같아 보이는데, 실제로는 근거 정답 $R_k$ 의 영역과
    IAA 스팬 일치율이 달라진다. 사람이 20% 빈도로 오는 판단을 300건 내내 일관되게
    유지하기 어려우므로 도구가 대신한다.

    `docs.jsonl` 은 건드리지 않는다 — 어댑터 기준선의 기준점이기 때문이다.
    여기서 손보는 것은 주석자에게 보여 줄 초안뿐이다.
    """
    text = {w["id"]: w.get("text", "") for w in words}
    box = {w["id"]: w["bbox"] for w in words}
    out = {}
    for name, a in (fields or {}).items():
        ids = list(a.get("word_ids") or [])
        keep = trim_ids(name, ids, text)
        if keep == ids:
            out[name] = a
            continue
        if not keep:            # 구두점뿐인 스팬은 값이 아니다. 초안에서 뺀다
            continue
        out[name] = {**a, **span_of(keep, text, box)}
    return out


# ---------------------------------------------------------------- 제N안 분리
#
# 「1건 기안 다수 시행」 — 한 번의 결재로 수신자별 시행문을 여러 통 만드는 공문서 관행이다.
# 그래서 한 장에 헤더 세트가 여러 벌 붙는다(파일럿 300건 중 69건 23.0%, 제N안 표식은
# 156건 52.0%). 각 안은 서로 다른 수신자에게 나가는 **별개의 시행문**이다.
#
#   「제1안」 수 신 : 내부결재            제 목 : … "건의"      ← 내부결재용 기안문
#   (제 2 안) 수 신 : 제1안 건축주 주소 성명 이기               ← 민원인 앞 통보문
#   (제 3 안) 수 신 : 수신처 참조  → 하단 「수신처 : 세무과장…」  ← 관계부서 앞 통보문
#
# 이것들을 한 필드로 이으면 어느 안의 정답도 아닌 문자열이 되고(「세무과장 내외 동장」),
# 근거 정답 $R_k$ 는 두 곳을 감싸는 페이지 15% 크기의 사각형이 된다.
#
# **정답은 페이지 최상단 안 하나로 한다** (가이드 1 「상단 서식란 우선」과 같은 규칙이며
# 어댑터의 위→아래 첫 매칭과도 일치한다). 아래쪽 안의 스팬은 버리지 않고 `secondary_spans`
# 로 옮겨, 평가에서 오답이 아니라 **제외(ignore)** 로 다룬다. 모델이 제3안의 수신을 짚은 것은
# 틀린 것이 아니라 다른 시행문을 본 것이기 때문이다.
#
# 세로 간격 4배로 가른다. 실측 분포가 여기서 깨끗하게 갈린다 — 스팬 내부 최대 세로 간격이
# 90%까지 1.14배 이하인데 92%에서 10.89배로 건너뛴다. 2~8배 구간에는 사례가 하나뿐이다.
# 정상적인 2줄 수신 값은 1.7배 남짓이라 잘리지 않는다.
VGAP = 4.0


def split_clusters(ids: list, box: dict, med_h: float) -> list:
    """세로로 크게 떨어진 덩어리로 나눈다. 위에서 아래 순서로 돌려준다."""
    ws = sorted((i for i in ids if i in box), key=lambda i: box[i][1])
    if not ws:
        return []
    out, cur = [], [ws[0]]
    for a, b in zip(ws, ws[1:]):
        if box[b][1] - box[a][1] > med_h * VGAP:
            out.append(cur)
            cur = [b]
        else:
            cur.append(b)
    out.append(cur)
    return out


def median_word_height(words: list) -> float:
    hs = sorted(w["bbox"][3] for w in words)
    return hs[len(hs) // 2] if hs else 60.0


def split_secondary(rec: dict, doc: dict) -> list:
    """기록을 제자리에서 고친다. 정답은 최상단 덩어리, 나머지는 secondary_spans 로.

    돌려주는 것은 옮긴 내역(보고용)이다. 옮길 것이 없으면 빈 목록이다.
    """
    text = {w["id"]: w.get("text", "") for w in doc["words"]}
    box = {w["id"]: w["bbox"] for w in doc["words"]}
    med = median_word_height(doc["words"])
    moved = []
    for name in FIELDS:
        g = (rec.get("fields") or {}).get(name)
        if not g or not g.get("word_ids"):
            continue
        cl = split_clusters(g["word_ids"], box, med)
        if len(cl) < 2:
            continue
        # 원래 순서(판독 순서)를 유지한 채 최상단 덩어리만 남긴다
        head = set(cl[0])
        keep = [i for i in g["word_ids"] if i in head]
        for extra in cl[1:]:
            sp = span_of([i for i in g["word_ids"] if i in set(extra)], text, box)
            sp["field"] = name
            rec.setdefault("secondary_spans", []).append(sp)
            moved.append((name, sp["value"]))
        g.update(span_of(keep, text, box))
    return moved


def span_of(ids: list, text: dict, box: dict) -> dict:
    """어절 id 목록에서 값·정규화값·bbox 를 다시 만든다. 클라이언트 toggle() 과 같은 규칙."""
    xs = [box[i] for i in ids]
    x0 = min(b[0] for b in xs)
    y0 = min(b[1] for b in xs)
    x1 = max(b[0] + b[2] for b in xs)
    y1 = max(b[1] + b[3] for b in xs)
    value = " ".join(text.get(i, "") for i in ids)
    return {"word_ids": ids, "value": value,
            "value_norm": re.sub(r"\s+", "", value),
            "bbox": [x0, y0, x1 - x0, y1 - y0]}


class Store:
    """주석 저장소. append-only JSONL이라 중간에 죽어도 앞부분이 살아남는다."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.records: dict[str, dict] = {}
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue          # 중단 중 잘린 마지막 줄
                    self.records[r["doc_id"]] = r      # 나중 것이 이긴다

    def save(self, rec: dict) -> None:
        with self.lock:
            self.records[rec["doc_id"]] = rec
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())      # 전원이 꺼져도 남도록 강제로 내린다


class Handler(BaseHTTPRequestHandler):
    docs: list[dict] = []
    store: Store
    annotator: str = ""

    def log_message(self, *args):        # 콘솔을 조용히
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # UI가 코드에 인라인돼 있어 캐시되면 서버를 다시 켜도 예전 JS가 돈다.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        u = urlparse(self.path)

        if u.path == "/":
            self._send(200, HTML.encode("utf-8"), "text/html; charset=utf-8")

        elif u.path == "/api/state":
            done = set(self.store.records)
            # 재검토 모드에서는 목록이 전부 이미 완료 상태다. 「다음 미완료 문서」 규칙을
            # 그대로 쓰면 마지막 문서로 보내 버리므로, 처음부터 훑게 한다.
            first = 0 if getattr(self, "review_mode", False) else next(
                (i for i, d in enumerate(self.docs) if d["doc_id"] not in done),
                len(self.docs) - 1)
            self._json({
                "n": len(self.docs), "done": len(done), "start": first,
                "annotator": self.annotator, "fields": FIELDS,
                "focus": getattr(self, "focus", None),
            })

        elif u.path.startswith("/api/doc/"):
            i = int(u.path.rsplit("/", 1)[1])
            if not 0 <= i < len(self.docs):
                self._json({"error": "범위 밖"})
                return
            d = self.docs[i]
            self._json({
                "index": i, "doc_id": d["doc_id"],
                "image": d["image"], "words": d["words"],
                # 줄 정보는 값 스팬을 판독 순서로 잇는 데 쓴다. 없으면 두 줄에
                # 걸친 값(수신 등)에서 어절이 섞인다.
                "lines": d.get("lines", []),
                # 앞뒤 구두점 어절은 도구가 미리 떼어낸다 (가이드 5-4, v1.2)
                "auto": trim_edge_punct(d.get("fields", {}), d["words"]),
                "saved": self.store.records.get(d["doc_id"]),
                "double": getattr(self, "all_double", False) or (i % DOUBLE_EVERY == 0),
            })

        elif u.path.startswith("/img/"):
            name = Path(u.path[5:]).name
            p = PILOT / "images" / name
            if not p.exists():
                self._send(404, b"no image", "text/plain")
                return
            self._send(200, p.read_bytes(), "image/jpeg")

        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            self._send(404, b"not found", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        rec = json.loads(self.rfile.read(n).decode("utf-8"))
        rec["annotator"] = self.annotator
        # 확정되지 않은 필드가 있는 기록은 받지 않는다. 저장되면 그 문서가 '완료'로 잡혀
        # 다시 나오지 않으므로, 빠진 필드가 영영 정답 없이 남는다. 브라우저에도 같은 검사가
        # 있으나 여기서 한 번 더 막는다.
        got = rec.get("fields") or {}
        pending = [f for f in FIELDS if (got.get(f) or {}).get("status") not in FINAL]
        if pending:
            self._json({"ok": False, "reason": "incomplete", "pending": pending,
                        "done": len(self.store.records)})
            return
        # 논문 보고용 계측. 사후 복원이 불가능하므로 저장 시점에 새긴다.
        #   ts        작업 시각 — 소요 시간·주석자 드리프트 분석
        #   guideline 적용 기준 판 — 규칙 개정 전후 기록을 가려내기 위함
        # dwell_ms(문서 체류 시간)는 클라이언트가 실어 보낸다. 자리를 비우면 부풀 수 있어
        # 보고할 때는 평균이 아니라 중앙값을 쓴다.
        rec["ts"] = datetime.now().isoformat(timespec="seconds")
        rec["guideline"] = GUIDELINE
        # index 는 **300건 전체 제시 순서**에서의 위치다. 클라이언트가 보내오는 값은
        # 걸러낸 목록 안의 위치라, `--only-double`·`--only-ids` 모드에서는 어긋난다.
        # 주석자 숙련도 드리프트를 순서로 분석하려면 원래 위치라야 하므로 여기서 바로잡는다.
        rec["index"] = TRUE_INDEX.get(rec["doc_id"], rec.get("index"))
        # 다른 안(案)의 스팬이 섞여 있으면 정답에서 떼어 secondary_spans 로 옮긴다.
        # 주석자는 보이는 대로 다 고르면 되고, 가르는 일은 도구가 한다.
        doc = next((d for d in self.docs if d["doc_id"] == rec["doc_id"]), None)
        moved = split_secondary(rec, doc) if doc else []
        # 클라이언트는 필드 5종만 돌려보낸다. 재검토로 다시 저장할 때 이전에 갈라 둔
        # secondary_spans 가 통째로 사라지므로 여기서 이어 붙인다. (실측: 발신명의
        # 전면 재확인 한 번에 145개 중 144개가 날아갔다.)
        prev = (self.store.records.get(rec["doc_id"]) or {}).get("secondary_spans") or []
        if prev:
            seen = {(s.get("field"), tuple(s.get("word_ids") or []))
                    for s in rec.get("secondary_spans", [])}
            gold = {i for f in FIELDS
                    for i in ((rec.get("fields") or {}).get(f) or {}).get("word_ids") or []}
            for s in prev:
                key = (s.get("field"), tuple(s.get("word_ids") or []))
                # 정답으로 승격된 스팬은 되살리지 않는다
                if key in seen or (set(s.get("word_ids") or []) & gold):
                    continue
                rec.setdefault("secondary_spans", []).append(s)
        self.store.save(rec)
        self._json({"ok": True, "done": len(self.store.records),
                    "moved": [f"{f}: {v[:24]}" for f, v in moved]})


HTML = r"""<!doctype html>
<meta charset="utf-8">
<title>KoDocXAI 주석</title>
<style>
  :root { --bg:#12141a; --panel:#1b1f28; --line:#2c323f; --fg:#e6e9ef; --dim:#9aa3b2;
          --ok:#3fb950; --no:#f85149; --sel:#d29922; --acc:#58a6ff; }
  * { box-sizing:border-box }
  body { margin:0; height:100vh; display:flex; background:var(--bg); color:var(--fg);
         font:14px/1.5 "Malgun Gothic",system-ui,sans-serif; overflow:hidden }
  #left { flex:1; overflow:auto; position:relative; background:#0b0d11 }
  #wrap { position:relative; display:inline-block; transform-origin:0 0 }
  #wrap img { display:block }
  .w { position:absolute; border:1px solid rgba(88,166,255,.30); cursor:pointer }
  .w:hover { background:rgba(88,166,255,.22) }
  .w.sel { background:rgba(210,153,34,.42); border-color:var(--sel) }
  .w.auto { background:rgba(63,185,80,.20); border-color:rgba(63,185,80,.6) }
  /* 원 라벨이 두 줄을 한 상자로 묶은 어절. 쪼갤 수 없다 */
  .w.tall { border:2px dashed var(--no) }
  #right { width:400px; background:var(--panel); border-left:1px solid var(--line);
           display:flex; flex-direction:column }
  header { padding:12px 14px; border-bottom:1px solid var(--line) }
  #prog { height:4px; background:var(--line); border-radius:2px; margin-top:8px }
  #prog i { display:block; height:100%; background:var(--acc); border-radius:2px; width:0 }
  #fields { flex:1; overflow:auto; padding:8px }
  .f { border:1px solid var(--line); border-radius:8px; padding:10px; margin-bottom:8px;
       background:#161a22 }
  .f.active { border-color:var(--sel); box-shadow:0 0 0 1px var(--sel) }
  .f .hd { display:flex; justify-content:space-between; align-items:center; gap:8px }
  .f .nm { font-weight:700 }
  .f .st { font-size:12px; padding:2px 8px; border-radius:99px; background:#222836; color:var(--dim) }
  .f .st.ok { background:rgba(63,185,80,.18); color:var(--ok) }
  .f .st.no { background:rgba(248,81,73,.18); color:var(--no) }
  .f .st.absent { background:#2a2f3a; color:var(--dim) }
  .f .st.na { background:rgba(210,153,34,.18); color:var(--sel) }
  .f .hint { margin-top:6px; font-size:12px; color:var(--sel) }
  .f .val { margin-top:6px; font-size:13px; word-break:break-all; color:var(--fg) }
  .f .val.empty { color:var(--dim); font-style:italic }
  .f .btns { margin-top:8px; display:flex; gap:6px }
  button { background:#232a36; color:var(--fg); border:1px solid var(--line);
           border-radius:6px; padding:5px 10px; cursor:pointer; font-size:12px }
  button:hover { border-color:var(--acc) }
  button.ok { border-color:rgba(63,185,80,.5) } button.no { border-color:rgba(248,81,73,.5) }
  footer { padding:10px 14px; border-top:1px solid var(--line); font-size:12px; color:var(--dim) }
  kbd { background:#232a36; border:1px solid var(--line); border-radius:4px;
        padding:1px 5px; font-size:11px }
  #msg { padding:6px 14px; font-size:12px; color:var(--ok); min-height:26px }
  .badge { font-size:11px; padding:1px 7px; border-radius:99px; background:rgba(210,153,34,.2);
           color:var(--sel); margin-left:6px }
</style>
<div id="left"><div id="wrap"><img id="img"><div id="boxes"></div></div></div>
<div id="right">
  <header>
    <div style="display:flex;justify-content:space-between">
      <b id="docid">—</b><span id="cnt" style="color:var(--dim)"></span>
    </div>
    <div id="prog"><i></i></div>
  </header>
  <div id="msg"></div>
  <div id="fields"></div>
  <footer>
    <kbd>1</kbd>~<kbd>5</kbd> 필드 선택 · <kbd>O</kbd> 맞음 · <kbd>N</kbd> 이 문서에 없음 ·
    <kbd>M</kbd> 영역은 맞고 문자열 불가 — <b>셋 다 누르면 다음 필드로 넘어간다</b><br>
    값이 틀리면 <b>어절을 클릭</b>해 스팬을 고친 뒤 <kbd>O</kbd>,
    또는 <b>마지막 어절을 더블클릭</b>하면 바로 확정되고 다음 필드로 간다<br>
    <kbd>Enter</kbd> 저장 후 다음 · <kbd>←</kbd><kbd>→</kbd> 이동 · <kbd>+</kbd><kbd>-</kbd> 확대<br>
    <span style="color:var(--no)">붉은 점선</span> = 두 줄 묶임 <b>의심</b> — 값 문자열이 실제로
    뒤섞였으면 <kbd>M</kbd>, 멀쩡하면 <kbd>O</kbd><br>
    <b>다섯 필드가 모두 확정되어야 저장된다</b>
  </footer>
</div>
<script>
const FIELDS = ["제목","수신","문서번호","시행일자","발신명의"];
const FINAL  = ["ok","absent","string_na"];   // 저장 가능한 확정 상태. corrected 는 중간 상태다
let S = { i:0, n:0, doc:null, cur:0, ann:{}, zoom:1, words:[] };

const $ = s => document.querySelector(s);

async function boot(){
  const st = await (await fetch("/api/state")).json();
  S.n = st.n;
  // 한 필드만 다시 볼 때는 문서를 열자마자 그 필드가 잡혀 있게 한다
  S.focus = st.focus ? Math.max(0, FIELDS.indexOf(st.focus)) : null;
  if (st.focus) document.title = "재검토: " + st.focus;
  load(st.start);
}

async function load(i){
  if (i < 0 || i >= S.n) return;
  const d = await (await fetch("/api/doc/"+i)).json();
  S.i = i; S.doc = d; S.cur = (S.focus ?? 0); S.words = d.words;
  S.t0 = performance.now();          // 문서 체류 시간 측정 시작 (논문 보고용)

  // 어절 → 줄 번호. 줄은 y순, 줄 안은 x순으로 이미 정렬돼 있다.
  S.lineOf = {};
  (d.lines || []).forEach((ln, li) => (ln.word_ids || []).forEach(wid => { S.lineOf[wid] = li; }));

  // 두 줄이 한 어절로 묶인 라벨을 가려낸다.
  //
  // 문서 전체 높이 중앙값과 비교하면 안 된다. 발신명의(「김 해 시 장」)나 기관명처럼
  // **줄 전체가 큰 글씨**인 경우가 있어 그 줄 어절이 통째로 오탐된다.
  // 실측: 앞 60개 문서에서 그 방식이 표시한 156개 중 59개가 이 오탐이었고, 하필
  // 발신명의에 몰려 있어 M(문자열 불가)을 잘못 찍게 만든다.
  //
  // 두 줄에 걸친 어절은 **자기 줄 동료보다** 크다. 그래서 같은 줄 어절의 중앙 높이와
  // 비교한다. 실측에서 진짜 묶임은 동료 대비 1.7~3.0배, 큰 글씨는 0.95~1.05배로 갈렸다.
  //
  // 줄에 혼자인 어절은 비교 대상이 없어 표시하지 않는다. 이 표시는 "확인해 보라"는
  // 안내일 뿐이므로, 놓치는 것보다 잘못 표시하는 쪽이 해롭다.
  const byLine = {};
  S.words.forEach(w => { const l = S.lineOf[w.id]; if (l === undefined) return;
                         (byLine[l] = byLine[l] || []).push(w); });
  S.tall = new Set();
  S.words.forEach(w => {
    const mates = (byLine[S.lineOf[w.id]] || []).filter(m => m.id !== w.id);
    if (!mates.length) return;
    const hs = mates.map(m => m.bbox[3]).sort((a,b) => a-b);
    const med = hs[Math.floor(hs.length/2)] || 1;
    if (w.bbox[3] > med * 1.5) S.tall.add(w.id);
  });

  // 저장된 주석이 있으면 그것을, 없으면 자동 추출값을 초안으로 올린다
  S.ann = {};
  for (const f of FIELDS){
    const saved = d.saved && d.saved.fields && d.saved.fields[f];
    if (saved) { S.ann[f] = {...saved}; continue; }
    const a = d.auto[f];
    S.ann[f] = a ? { status:null, value:a.value, value_norm:a.value_norm||"",
                     word_ids:a.word_ids||[], bbox:a.bbox, source:"auto" }
                 : { status:null, value:"", value_norm:"", word_ids:[], bbox:null, source:"none" };
  }

  $("#img").src = "/img/" + d.image.file_name;
  $("#img").onload = () => { fit(); draw(); };
  $("#docid").innerHTML = esc(d.doc_id) + (d.double ? '<span class="badge">이중 주석</span>' : '');
  $("#cnt").textContent = (i+1) + " / " + S.n;
  $("#prog i").style.width = (100*(i+1)/S.n) + "%";
  $("#msg").textContent = d.saved ? "저장된 주석을 불러왔습니다" : "";
  render();
}

function fit(){
  const img = $("#img");
  S.zoom = Math.min(1, ($("#left").clientWidth - 24) / img.naturalWidth);
  applyZoom();
}
function applyZoom(){ $("#wrap").style.transform = "scale("+S.zoom+")"; }

function draw(){
  const cur = FIELDS[S.cur];
  const sel = new Set(S.ann[cur].word_ids || []);
  const autoIds = new Set();
  for (const f of FIELDS){ const a = S.doc.auto[f]; if (a) (a.word_ids||[]).forEach(x=>autoIds.add(x)); }
  // 본문 높이의 1.5배를 넘는 어절은 원 라벨이 두 줄을 한 상자로 묶은 것일 수 있다.
  // 어절 내부는 쪼갤 수 없으므로 주석자가 시간을 낭비하지 않도록 표시해 둔다.
  $("#boxes").innerHTML = S.words.map(w => {
    const [x,y,ww,hh] = w.bbox;
    let cls = sel.has(w.id) ? "w sel" : (autoIds.has(w.id) ? "w auto" : "w");
    if (S.tall.has(w.id)) cls += " tall";
    const tip = S.tall.has(w.id) ? `${w.text}  (두 줄이 묶인 것으로 보임 — 확인 요)` : w.text;
    return `<div class="${cls}" data-id="${w.id}" title="${esc(tip)}"
             style="left:${x}px;top:${y}px;width:${ww}px;height:${hh}px"></div>`;
  }).join("");
  $("#boxes").querySelectorAll(".w").forEach(el => {
    el.onclick = () => toggle(parseInt(el.dataset.id));

    // 마지막 어절에서 더블클릭 = "이 필드 끝" → 확정하고 다음 필드로.
    // 손을 마우스에서 키보드로 옮기지 않고 수정을 끝낼 수 있다.
    //
    // 주의: dblclick 앞에 click 이 두 번 발생하고 toggle 은 말 그대로 토글이라 상쇄된다.
    // 상쇄 결과가 처음 상태에 따라 달라지므로(선택 전이면 '선택 안 됨', 선택 후면 '선택됨'),
    // 여기서는 **그 어절이 반드시 들어 있도록** 맞춘다. 두 경우 모두 같은 결과가 된다.
    el.ondblclick = () => {
      const id = parseInt(el.dataset.id);
      const a = S.ann[FIELDS[S.cur]];
      if (!(a.word_ids || []).includes(id)) toggle(id);
      if ((a.word_ids || []).length) mark("ok");   // 빈 스팬이면 확정하지 않는다
    };
  });
}

function esc(s){ return String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

function toggle(id){
  const a = S.ann[FIELDS[S.cur]];
  const ids = new Set(a.word_ids || []);
  ids.has(id) ? ids.delete(id) : ids.add(id);
  // 판독 순서로 잇는다 — 줄 먼저, 같은 줄 안에서는 x. x만으로 정렬하면
  // 두 줄에 걸친 값(수신이 흔하다)에서 어절이 뒤섞인다.
  // 줄 안에서 y를 먼저 보면 안 된다 — 한 줄 안에서도 어절마다 y가 몇 픽셀씩
  // 달라 x 순서가 깨진다(실측 확인).
  const chosen = S.words.filter(w => ids.has(w.id)).sort((p,q) =>
    ((S.lineOf[p.id] ?? 9e9) - (S.lineOf[q.id] ?? 9e9)) || (p.bbox[0] - q.bbox[0]));
  a.word_ids = chosen.map(w=>w.id);
  a.value = chosen.map(w=>w.text).join(" ");
  a.value_norm = a.value.replace(/\s+/g, "");   // 채점용. 어댑터와 같은 규칙
  a.bbox = union(chosen);
  // 클릭만으로 확정하지 않는다. 수정이 몇 어절로 끝날지 알 수 없어 클릭에 "다음 필드로"를
  // 붙일 수 없기 때문이다. 그래서 O 를 눌러 확정하게 하고, mark() 가 다음 필드로 넘겨준다.
  // 즉 O 는 여분의 키가 아니라 "이 필드 끝냈다"는 신호다.
  // (한때 클릭 즉시 ok 로 바꿔 봤으나, 다음 필드로 넘기려면 결국 1~5 를 눌러야 해서
  //  키 횟수가 같았고 오히려 번거로웠다. 되돌렸다.)
  a.status = a.word_ids.length ? "corrected" : null;
  a.source = "human";
  render(); draw();
}

function union(ws){
  if (!ws.length) return null;
  const x0=Math.min(...ws.map(w=>w.bbox[0])), y0=Math.min(...ws.map(w=>w.bbox[1]));
  const x1=Math.max(...ws.map(w=>w.bbox[0]+w.bbox[2])), y1=Math.max(...ws.map(w=>w.bbox[1]+w.bbox[3]));
  return [x0,y0,x1-x0,y1-y0];
}

function render(){
  $("#fields").innerHTML = FIELDS.map((f,k) => {
    const a = S.ann[f];
    // 저장되는 상태는 확인 / 없음 / 문자열 불가 셋뿐이다.
    // 「수정함」은 어절을 클릭한 직후의 중간 상태이며, O 를 눌러야 확정되고 다음 필드로 넘어간다.
    const st = a.status==="ok" ? '<span class="st ok">확인</span>'
             : a.status==="corrected" ? '<span class="st no">수정함 — <kbd>O</kbd>로 확정</span>'
             : a.status==="absent" ? '<span class="st absent">없음</span>'
             : a.status==="string_na" ? '<span class="st na">문자열 불가</span>'
             : '<span class="st">미검토</span>';
    const nz = a.value_norm || (a.value || "").replace(/\s+/g, "");
    const v = a.value
      ? esc(a.value) + (nz && nz !== a.value ? ` <span style="color:var(--dim)">→ ${esc(nz)}</span>` : "")
      : "(자동 추출 실패)";
    // 값 스팬에 두 줄 묶임 의심 어절이 들어 있으면 알려준다.
    // 어디까지나 **추정**이다. 값 문자열이 실제로 뒤섞여 보이는지 눈으로 확인하고 판단한다.
    const merged = (a.word_ids || []).some(i => S.tall.has(i));
    const hint = merged && a.status !== "string_na"
      ? '<div class="hint">두 줄이 한 어절로 묶인 것으로 <b>보임</b> — 값이 실제로 뒤섞였으면 <b>M</b></div>' : "";
    return `<div class="f ${k===S.cur?"active":""}" data-k="${k}">
      <div class="hd"><span class="nm">${k+1}. ${f}</span>${st}</div>
      <div class="val ${a.value?"":"empty"}">${v}</div>${hint}
      <div class="btns">
        <button class="ok" data-a="ok">O 맞음</button>
        <button data-a="string_na">M 문자열불가</button>
        <button class="no" data-a="absent">N 없음</button>
      </div></div>`;
  }).join("");
  $("#fields").querySelectorAll(".f").forEach(el => {
    el.onclick = e => {
      S.cur = parseInt(el.dataset.k);
      const act = e.target.dataset && e.target.dataset.a;
      if (act === "ok" || act === "absent" || act === "string_na") mark(act);
      else { render(); draw(); }
    };
  });
}

function mark(status){
  const a = S.ann[FIELDS[S.cur]];
  if (status === "absent"){ a.status="absent"; a.value=""; a.value_norm=""; a.word_ids=[]; a.bbox=null; }
  else { a.status = status; }
  if (S.cur < FIELDS.length-1) S.cur++;
  render(); draw();
}

async function save(next){
  const rec = { doc_id:S.doc.doc_id, index:S.i, double:S.doc.double,
                dwell_ms: Math.round(performance.now() - S.t0), fields:{} };
  for (const f of FIELDS){
    const a = S.ann[f];
    if (!a.status) continue;                 // 미검토는 기록하지 않는다
    rec.fields[f] = { status:a.status, value:a.value, value_norm:a.value_norm || a.value.replace(/\s+/g,""),
                      word_ids:a.word_ids, bbox:a.bbox, source:a.source };
  }
  // 확정되지 않은 필드가 있으면 저장하지 않는다. 저장되면 그 문서는 '완료'로 잡혀 다시
  // 나오지 않으므로, 빠진 필드가 영영 정답 없이 남는다.
  // 실측: 재시작 전 29건 중 1건이 제목 없이 저장돼 있었다.
  // 「수정함」도 확정이 아니다 — 클릭은 토글이라 값이 의도치 않게 바뀔 수 있어, 고친 뒤
  // O 로 한 번 확인하게 한다.
  const pending = FIELDS.filter(f => !FINAL.includes(S.ann[f].status));
  if (pending.length){
    $("#msg").textContent = "확정 안 됨: " + pending.join(", ")
      + " — O(맞음) / N(없음) / M(문자열 불가) 중 하나로 확정한다";
    return;
  }
  const r = await (await fetch("/api/save",{method:"POST",body:JSON.stringify(rec)})).json();
  // 다른 안(案)의 스팬은 서버가 정답에서 떼어 secondary_spans 로 옮긴다. 버리는 것이
  // 아니라 평가에서 제외할 영역으로 따로 두는 것이므로, 무엇이 옮겨졌는지만 알린다.
  $("#msg").textContent = "저장됨 · 완료 " + r.done + "건"
    + ((r.moved && r.moved.length) ? "  |  다른 안으로 분리: " + r.moved.join(" / ") : "");
  if (next) load(Math.min(S.i+1, S.n-1));
}

addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (k >= "1" && k <= "5"){ S.cur = +k-1; render(); draw(); }
  else if (k === "o"){ mark("ok"); }
  else if (k === "x"){ $("#msg").textContent = "어절을 클릭해 올바른 값을 지정하세요"; }
  else if (k === "n"){ mark("absent"); }
  else if (k === "m"){ mark("string_na"); }
  else if (e.key === "Enter"){ save(true); }
  else if (e.key === "ArrowRight"){ save(false); load(S.i+1); }
  else if (e.key === "ArrowLeft"){ save(false); load(S.i-1); }
  else if (e.key === "+" || e.key === "="){ S.zoom *= 1.2; applyZoom(); }
  else if (e.key === "-"){ S.zoom /= 1.2; applyZoom(); }
});

boot();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="KoDocXAI 주석 도구")
    ap.add_argument("--annotator", required=True, help="주석자 식별자 (IAA 산출에 쓰인다)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--only-double", action="store_true",
                    help="이중 주석 대상만 (index %% %d == 0)" % DOUBLE_EVERY)
    ap.add_argument("--only-ids", type=Path,
                    help="doc_id 목록 파일(한 줄에 하나)에 있는 문서만. 재검토용")
    ap.add_argument("--focus", choices=FIELDS,
                    help="한 필드만 다시 볼 때. 문서를 열면 그 필드가 선택된 채로 시작하고 "
                         "첫 문서부터 훑는다")
    a = ap.parse_args()

    docs_path = PILOT / "docs.jsonl"
    if not docs_path.exists():
        print(f"파일럿 세트가 없다: {docs_path}\n먼저 prep_pilot.py 를 실행할 것.")
        return 1

    docs = [json.loads(l) for l in docs_path.open(encoding="utf-8") if l.strip()]
    TRUE_INDEX.update({d["doc_id"]: i for i, d in enumerate(docs)})   # 걸러내기 **전**의 위치
    if a.only_double:
        docs = [d for i, d in enumerate(docs) if i % DOUBLE_EVERY == 0]
    if a.only_ids:
        # 재검토 모드. 이미 단 주석이 초안으로 뜨므로 고칠 곳만 보면 된다.
        want = {s.strip() for s in a.only_ids.read_text(encoding="utf-8").splitlines() if s.strip()}
        docs = [d for d in docs if d["doc_id"] in want]
        if not docs:
            print(f"목록에 해당하는 문서가 없다: {a.only_ids}")
            return 1

    out = PILOT / f"annotations_{a.annotator}.jsonl"
    Handler.docs = docs
    # 이중 주석 배지. 걸러낸 목록에서는 위치가 다시 매겨지므로 i % 5 로 다시 재면
    # 60건 중 12건에만 붙는다 — 실제로는 전부 이중 주석 대상이다.
    Handler.all_double = a.only_double
    Handler.review_mode = bool(a.only_ids) or bool(a.focus)
    Handler.focus = a.focus
    Handler.store = Store(out)
    Handler.annotator = a.annotator

    url = f"http://127.0.0.1:{a.port}"
    print(f"주석자   {a.annotator}")
    print(f"문서     {len(docs)}건 (완료 {len(Handler.store.records)}건)")
    print(f"저장     {out}  — 매 건 즉시 기록, 재시작 시 이어서 진행")
    print(f"\n{url}  (Ctrl+C 로 종료)")

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n종료. 진행분은 모두 저장되어 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
