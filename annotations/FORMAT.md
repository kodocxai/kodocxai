# KoDocXAI benchmark v1.0 - public annotation layer format

Files: `kodocxai_v1.0.jsonl` (300 documents, annotator A1) and
`kodocxai_v1.0_double.jsonl` (60 documents, independent second annotator A2,
used for inter-annotator agreement). One JSON object per line, one line per
document. Integrity hashes for both files are recorded in
`manifest_public.json`.

The layer contains **no AI-Hub source content**: no images, no transcribed
text, no coordinates. Spans are expressed only as integer indices
(`annotation_ids`) into the label entries of the AI-Hub source dataset, which
every user must download individually (see `../scripts/reassemble.py`).

## Record schema

```json
{
  "doc_id": "5350093-2001-0001-0001",
  "source": {
    "dataset": "공공행정문서 OCR (AI-Hub)",
    "label_entry": "01.라벨링데이터(Json)/인.허가/.../<doc_id>.json",
    "label_sha256": "<sha256 of the source label file>"
  },
  "guideline": "v1.4",
  "annotator": "A1",
  "fields": {
    "제목":     {"status": "ok", "annotation_ids": [67, 68, 69]},
    "수신":     {"status": "ok", "annotation_ids": [48, 49, 50, 51, 54, 55]},
    "문서번호": {"status": "ok", "annotation_ids": [29, 30, 31]},
    "시행일자": {"status": "ok", "annotation_ids": [36, 37, 38]},
    "발신명의": {"status": "ok", "annotation_ids": [212, 213, 214, 215],
                 "annotation_occ": [0, 0, 0, 0]}
  },
  "secondary_spans": [
    {"field": "제목", "annotation_ids": [49, 50, 51, 52]}
  ]
}
```

| Key | Meaning |
|---|---|
| `doc_id` | Document identifier of the AI-Hub source dataset |
| `source.label_entry` | Path of the source label file inside `Training/[라벨]train.zip` |
| `source.label_sha256` | SHA-256 of that label file - reassembly refuses a mismatching source |
| `guideline` | Annotation guideline version (full text in `../guidelines/`) |
| `annotator` | `A1` (primary) or `A2` (independent double annotation) |
| `fields` | The five KIE fields: 제목 (title), 수신 (recipient), 문서번호 (document number), 시행일자 (issuance date), 발신명의 (issuing authority) |
| `fields.*.status` | `ok` (value present and recoverable), `absent` (field does not exist in the document), `string_na` (region identified but the string is irrecoverable; stays in the plausibility sample, excluded from string matching - see the paper, Section 3.4.1) |
| `fields.*.annotation_ids` | Integer `annotation.id` values of the source label entry whose word regions constitute the value span |
| `fields.*.annotation_occ` | Present only where needed (87 of 360 records): the source labels reuse `annotation.id` values within a document (135 of 300 documents), so an id alone can be ambiguous. `annotation_occ[i]` is the occurrence ordinal (0-based, in label-file order) of `annotation_ids[i]` among entries sharing that id. Absent means ordinal 0 everywhere |
| `secondary_spans` | Additional same-form occurrences of a field value elsewhere in the document (e.g. the title repeated in a lower dispatch copy); ignored eojeols for plausibility scoring - see the paper, Section 3.4.1. Entries may also carry `annotation_occ` |

## Manifest

`manifest_public.json` records: benchmark name, source dataset, the required
AI-Hub download, per-file SHA-256 and record counts, stratum allocation of the
300-document sample, and the fixed randomization seed strings
(`kodx-eval-v1`, `kodx-naopc-v1`) retained verbatim for reproducibility.
