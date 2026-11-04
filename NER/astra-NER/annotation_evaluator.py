"""Reproducible comparison of scientific annotations (Python >= 3.10; optional pymorphy3).

Public API: evaluate(reference, prediction, ...), fit_tfidf(*datasets).
Optional lemmatization: evaluate(..., lemmatizer="pymorphy3"). Raw scores
and raw-source offsets are never replaced by lemma-based string scores.
Inputs: CSV/JSON/JSONL paths or in-memory documents. Categories are METHOD,
OBJECT, RESULT, DATA, TASK, METRIC. IDs of graph entities are NOT required.

New in 1.2: source-unit Cohen kappa (automatic when offsets/text exist) and
optional one-to-one embedding-threshold phrase agreement. See evaluate().

Default output measures AGREEMENT, not correctness. `reference` becomes a gold
standard only after independent adjudication. Span F1 is unavailable when
coordinates are missing; textual equality is never mislabeled as span F1.
See README.md for the schema, empty-set policy and reproducible TF-IDF fitting.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Mapping, Sequence

__version__ = "1.2.0"
CATEGORIES = ("METHOD", "OBJECT", "RESULT", "DATA", "TASK", "METRIC")
TOKEN_PATTERN = r"(?u)\b\w+\b"  # Include one-character names, digits and negations.
_TOKENS = re.compile(TOKEN_PATTERN)
_TRANSLATION = str.maketrans({"ё": "е", "Ё": "Е", "–": "-", "—": "-",
    "‐": "-", "‑": "-", "«": '"', "»": '"', "“": '"', "”": '"',
    "‘": "'", "’": "'"})


def normalize_text(text: str, mode: str = "light") -> str:
    """For STRING metrics only. Never applied to source offsets.

    strict: preserve characters; light: NFC, casefold, ё/е, typographic quotes
    and dashes, whitespace. No stemming, OCR repair, punctuation removal,
    transliteration or deletion of numbers/negation. Latin A != Cyrillic А.
    """
    if not isinstance(text, str):
        raise TypeError("Annotation text must be a string")
    if mode == "strict":
        return text
    if mode != "light":
        raise ValueError("normalization must be 'light' or 'strict'")
    return " ".join(unicodedata.normalize("NFC", text).translate(_TRANSLATION).casefold().split())


def _read(data: Any) -> Any:
    if not isinstance(data, (str, Path)):
        return data
    path = Path(data)
    if not path.is_file():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    if ext in {".jsonl", ".ndjson"}:
        rows = []
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {i} of {path}") from e
        return rows
    if ext == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    raise ValueError("Expected .csv, .json or .jsonl, not " + ext)


def _document_key(d: Mapping[str, Any], fallback: str | None = None) -> str:
    key = d.get("article_link") or d.get("Link") or d.get("article_id") or d.get("document_id") or fallback
    if key is None or not str(key).strip():
        raise ValueError("Each document needs article_link or article_id")
    # Do not silently combine different versions or arbitrary URLs.
    return str(key).strip()


def _category(a: Mapping[str, Any]) -> str:
    cat = a.get("category", a.get("label", a.get("entity_group")))
    if not isinstance(cat, str) or cat.upper().strip() not in CATEGORIES:
        raise ValueError(f"Unknown category: {cat!r}; use {CATEGORIES}")
    return cat.upper().strip()


def _intervals(item: Mapping[str, Any]) -> tuple[tuple[int, int], ...] | None:
    if "spans" in item:
        raw = item["spans"]
    elif "start" in item or "end" in item:
        if "start" not in item or "end" not in item:
            raise ValueError("Both start and end are required")
        raw = [(item["start"], item["end"])]
    else:
        return None
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("spans must be a non-empty list of [start, end]")
    spans = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each span must have exactly two coordinates")
        s, e = pair
        if isinstance(s, bool) or isinstance(e, bool) or not isinstance(s, int) or not isinstance(e, int):
            raise ValueError("Coordinates must be integer Unicode code-point offsets")
        if s < 0 or e <= s:
            raise ValueError(f"Invalid half-open interval {(s, e)}")
        spans.append((s, e))
    spans.sort()
    if any(spans[i][0] < spans[i - 1][1] for i in range(1, len(spans))):
        raise ValueError("Segments of a discontinuous mention must not overlap")
    return tuple(spans)


def _load(data: Any) -> dict[str, dict[str, Any]]:
    raw = _read(data)
    if isinstance(raw, Mapping) and "documents" in raw:
        raw = raw["documents"]
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise TypeError("Expected a document or a list of document dictionaries")
    # Flat mention-level records are also accepted.
    if raw and all(isinstance(x, Mapping) and ("category" in x or "label" in x or "entity_group" in x) for x in raw):
        grouped: dict[str, dict[str, Any]] = {}
        for x in raw:
            key = _document_key(x, "__single__")
            grouped.setdefault(key, {"article_id": key, "annotations": []})["annotations"].append(x)
        raw = list(grouped.values())
    out: dict[str, dict[str, Any]] = {}
    for d in raw:
        if not isinstance(d, Mapping):
            raise TypeError("Every document must be a mapping")
        key = _document_key(d, "__single__" if len(raw) == 1 else None)
        if key in out:
            raise ValueError(f"Duplicate document key {key!r}; merge records explicitly")
        src = d.get("source", {})
        source_text = src.get("text") if isinstance(src, Mapping) else None
        if source_text is None:
            source_text = d.get("source_text")
        if source_text is not None and not isinstance(source_text, str):
            raise TypeError("source_text must be a string")
        source_hash = src.get("sha256_utf8") if isinstance(src, Mapping) else None
        if source_hash and source_text is not None:
            if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != source_hash:
                raise ValueError(f"Source hash mismatch in {key}")
        doc: dict[str, Any] = {"source_text": source_text, "source_hash": source_hash,
            "annotations": {c: [] for c in CATEGORIES}}
        entries = d.get("annotations")
        if entries is None:
            entries = []
            lower = {str(k).lower(): v for k, v in d.items()}
            for cat in CATEGORIES:
                values = lower.get(cat.lower(), [])
                if values is None:
                    values = []
                if isinstance(values, str):
                    values = [] if not values.strip() else [v.strip() for v in values.split("|")]
                if not isinstance(values, (list, tuple)):
                    raise TypeError(f"{key}/{cat} must contain a list or a pipe-delimited string")
                for v in values:
                    if isinstance(v, str):
                        entries.append({"category": cat, "surface": v})
                    elif isinstance(v, Mapping):
                        entries.append({**v, "category": cat})
                    else:
                        raise TypeError("An annotation must be a string or a mapping")
        if not isinstance(entries, (list, tuple)):
            raise TypeError("annotations must be a list")
        for idx, a in enumerate(entries):
            if not isinstance(a, Mapping):
                raise TypeError("Each annotation must be a mapping")
            cat = _category(a)
            surface = a.get("surface", a.get("text", a.get("word")))
            if surface is None:
                coords = _intervals(a)
                if coords and source_text is not None:
                    surface = " ".join(source_text[s:e] for s, e in coords)
                else:
                    raise ValueError("Each annotation needs surface/text, or offsets with source_text")
            if not isinstance(surface, str) or not surface.strip():
                raise ValueError("Empty annotation texts are not allowed; use an empty list")
            occurrences = []
            raw_mentions = a.get("mentions")
            if raw_mentions is not None:
                if not isinstance(raw_mentions, list):
                    raise TypeError("mentions must be a list")
                for m in raw_mentions:
                    if not isinstance(m, Mapping):
                        raise TypeError("mentions must contain mappings")
                    sp = _intervals(m)
                    if sp is None:
                        raise ValueError("Every provided mention needs offsets")
                    occurrences.append({"spans": sp, "text": m.get("text", m.get("surface"))})
            else:
                sp = _intervals(a)
                if sp is not None:
                    occurrences.append({"spans": sp, "text": surface})
            doc["annotations"][cat].append({"surface": surface,
                "annotation_id": a.get("annotation_id", f"{key}:{idx}"),
                "occurrences": occurrences})
        out[key] = doc
    return out


def _surfaces(entries: Sequence[Mapping[str, Any]], normalization: str) -> set[str]:
    return {normalize_text(a["surface"], normalization) for a in entries}


def _features(surfaces: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in sorted(surfaces):
        tokens = _TOKENS.findall(text)
        # Never create artificial bigrams across entity boundaries.
        counts.update("1:" + t for t in tokens)
        counts.update("2:" + a + "\x1f" + b for a, b in zip(tokens, tokens[1:]))
    return counts


def _fit(feature_docs: Sequence[Counter[str]], normalization: str) -> dict[str, Any]:
    df: Counter[str] = Counter()
    for doc in feature_docs:
        df.update(doc.keys())
    n = len(feature_docs)
    return {"schema_version": "annotation_tfidf_v1", "normalization": normalization,
        "token_pattern": TOKEN_PATTERN, "ngram_range": [1, 2],
        "cross_entity_bigrams": False, "tf": "raw_count", "smooth_idf": True,
        "n_documents": n, "idf": {t: math.log((1 + n) / (1 + k)) + 1 for t, k in sorted(df.items())}}


def _fit_tfidf_raw(*datasets: Any, normalization: str = "light") -> dict[str, Any]:
    """Fit a frozen IDF vocabulary on fixed datasets, independent of later predictions."""
    normalize_text("", normalization)
    features = []
    for dataset in datasets:
        for doc in _load(dataset).values():
            features.extend(_features(_surfaces(doc["annotations"][c], normalization)) for c in CATEGORIES)
    if not features:
        raise ValueError("No documents to fit")
    return _fit(features, normalization)


def _cosine(a: Counter[str], b: Counter[str], state: Mapping[str, Any]) -> tuple[float | None, dict[str, float | None]]:
    idf = state["idf"]
    va = {t: n * idf[t] for t, n in a.items() if t in idf}
    vb = {t: n * idf[t] for t, n in b.items() if t in idf}
    na = math.sqrt(sum(x * x for x in va.values()))
    nb = math.sqrt(sum(x * x for x in vb.values()))
    if not na and not nb:
        score = None
    elif not na or not nb:
        score = 0.0
    else:
        score = max(0.0, min(1.0, sum(v * vb.get(t, 0.0) for t, v in va.items()) / (na * nb)))
    coverage = {"reference": sum(n for t, n in a.items() if t in idf) / sum(a.values()) if a else None,
                "prediction": sum(n for t, n in b.items() if t in idf) / sum(b.values()) if b else None}
    return score, coverage


def _counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    union = tp + fp + fn
    return {"tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if union else None,
        "jaccard": tp / union if union else None}


def _set_score(a: set[Any], b: set[Any]) -> dict[str, Any]:
    return _counts(len(a & b), len(b - a), len(a - b))


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _concept_map(data: Any, normalization: str) -> dict[tuple[str, str, str], set[str]]:
    if data is None:
        return {}
    raw = _read(data)
    if isinstance(raw, Mapping):
        raw = raw.get("mappings", [])
    if not isinstance(raw, list):
        raise TypeError("concept_map must be a list, or a file containing mappings")
    out = {}
    for r in raw:
        key = (_document_key(r), _category(r), normalize_text(r.get("surface", r.get("text")), normalization))
        ids = r.get("concept_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids):
            raise ValueError("Every reviewed mapping needs non-empty concept_ids")
        if key in out and out[key] != set(ids):
            raise ValueError(f"Conflicting reviewed mappings for {key}")
        out[key] = set(ids)
    return out


def _concepts(key: str, cat: str, surfaces: set[str], mapping: Mapping) -> tuple[set[tuple[str, str]], int]:
    out = set()
    found = 0
    for s in surfaces:
        ids = mapping.get((key, cat, s))
        if ids is None:
            # Unreviewed strings remain string-specific; NEVER guessed from local entity_id.
            out.add(("unmapped_surface", s))
        else:
            out.update(("reviewed_concept", i) for i in ids)
            found += 1
    return out, found


def _span_set(entries: Sequence[Mapping[str, Any]], source: str | None) -> tuple[set[tuple], int]:
    spans = set()
    missing = 0
    for a in entries:
        if not a["occurrences"]:
            missing += 1
        for m in a["occurrences"]:
            sp = m["spans"]
            if source is not None:
                if sp[-1][1] > len(source):
                    raise ValueError(f"Offset outside source: {sp}")
                if m["text"] is not None:
                    expected = " ".join(source[s:e] for s, e in sp)
                    if expected != m["text"]:
                        raise ValueError(f"Span text mismatch for {a['annotation_id']!r}: {m['text']!r} != {expected!r}")
            spans.add(sp)
    return spans, missing


def _span_iou(a: tuple, b: tuple) -> float:
    inter = sum(max(0, min(y, v) - max(x, u)) for x, y in a for u, v in b)
    total = sum(e - s for s, e in a) + sum(e - s for s, e in b) - inter
    return inter / total if total else 0.0


def _maximum_matching(a: set[tuple], b: set[tuple], threshold: float) -> int:
    """Maximum-cardinality bipartite matching. Not greedy; each span used once."""
    left, right = sorted(a), sorted(b)
    edges = [[j for j, y in enumerate(right) if _span_iou(x, y) >= threshold] for x in left]
    match_left: dict[int, int] = {}
    match_right: dict[int, int] = {}
    for root in range(len(left)):
        queue = deque([root])
        seen_left = {root}
        parent_right: dict[int, int] = {}
        endpoint = None
        while queue and endpoint is None:
            i = queue.popleft()
            for j in edges[i]:
                if j in parent_right:
                    continue
                parent_right[j] = i
                if j not in match_right:
                    endpoint = j
                    break
                nxt = match_right[j]
                if nxt not in seen_left:
                    seen_left.add(nxt)
                    queue.append(nxt)
        if endpoint is not None:
            j = endpoint
            while True:
                i = parent_right[j]
                previous = match_left.get(i)
                match_left[i], match_right[j] = j, i
                if previous is None:
                    break
                j = previous
    return len(match_left)


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"groups": len(rows)}
    for name in ("string_exact", "concept_reviewed", "span_exact", "span_overlap"):
        available = [r[name] for r in rows if r.get(name) is not None]
        if not available:
            out[name] = None
            continue
        out[name] = {"micro": _counts(*(sum(v[k] for v in available) for k in ("tp", "fp", "fn"))),
            "macro_jaccard": _mean([v["jaccard"] for v in available]),
            "macro_f1": _mean([v["f1"] for v in available]),
            "evaluated_groups": len(available),
            "nonempty_groups": sum(v["jaccard"] is not None for v in available)}
    out["cosine_tfidf_macro"] = _mean([r["cosine_tfidf"] for r in rows])
    return out


def _evaluate_raw(reference: Any, prediction: Any, *, normalization: str = "light",
             sources: Any = None, concept_map: Any = None, tfidf_state: Any = None,
             overlap_threshold: float = 0.5, require_same_documents: bool = False) -> dict[str, Any]:
    """Compare all six categories, document by document.

    reference/prediction: CSV/JSON/JSONL paths or document dictionaries/lists.
    sources: optional {article_link_or_id: raw_text}, or a JSON file with it.
    concept_map: optional REVIEWED crosswalk; not mandatory prediction IDs.
    tfidf_state: frozen fit_tfidf state or JSON path; default fits the current pair.
    overlap_threshold: character-span IoU threshold for diagnostic partial F1.
    require_same_documents: default False; missing/extra docs count as omissions/
        additions, never silently drop by an inner join. True raises on mismatch.

    Exact span metrics require offsets for every entry in that document/category
    on both sides. Missing offsets => None, not an invented span score. Local
    entity/concept IDs are deliberately ignored in ordinary NER metrics.
    Both-empty groups => None; excluded from macro means. One-empty => F1/J=0.
    """
    normalize_text("", normalization)
    if not 0 < overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in (0, 1]")
    ref, pred = _load(reference), _load(prediction)
    keys = sorted(ref.keys() | pred.keys())
    if not keys:
        raise ValueError("Both datasets contain no documents")
    if require_same_documents and ref.keys() != pred.keys():
        raise ValueError("Document sets differ")
    raw_sources = {} if sources is None else _read(sources)
    if not isinstance(raw_sources, Mapping):
        raise TypeError("sources must map document keys to raw strings")
    mapping = _concept_map(concept_map, normalization)
    empty = {"annotations": {c: [] for c in CATEGORIES}, "source_text": None, "source_hash": None}
    prepared = []
    for key in keys:
        r, p = ref.get(key, empty), pred.get(key, empty)
        versions = [x for x in [r["source_text"], p["source_text"], raw_sources.get(key)] if x is not None]
        if any(not isinstance(t, str) for t in versions):
            raise TypeError("Source values must be strings")
        if versions and any(t != versions[0] for t in versions[1:]):
            raise ValueError(f"Different source versions for {key}; realign offsets first")
        declared = {h for h in [r["source_hash"], p["source_hash"]] if h}
        if len(declared) > 1:
            raise ValueError(f"Different source hashes for {key}")
        if declared and versions and hashlib.sha256(versions[0].encode("utf-8")).hexdigest() not in declared:
            raise ValueError(f"Source content differs from declared hash for {key}")
        source = versions[0] if versions else None
        for cat in CATEGORIES:
            re_, pe = r["annotations"][cat], p["annotations"][cat]
            rs, ps = _surfaces(re_, normalization), _surfaces(pe, normalization)
            prepared.append((key, cat, re_, pe, rs, ps, _features(rs), _features(ps), source))
    if tfidf_state is None:
        state = _fit([features for x in prepared for features in x[6:8]], normalization)
        fitting = "current_pair_all_document_category_groups"
    else:
        state = _read(tfidf_state)
        if not isinstance(state, Mapping) or state.get("schema_version") != "annotation_tfidf_v1":
            raise ValueError("Invalid tfidf_state; use fit_tfidf or this module's saved state")
        if (state.get("normalization") != normalization or state.get("token_pattern") != TOKEN_PATTERN
                or state.get("ngram_range") != [1, 2] or state.get("cross_entity_bigrams") is not False
                or state.get("tf") != "raw_count" or state.get("smooth_idf") is not True):
            raise ValueError("TF-IDF configuration mismatch")
        if not isinstance(state.get("idf"), Mapping) or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in state["idf"].values()):
            raise ValueError("IDF weights must be finite positive numbers")
        fitting = "frozen_state"
    rows = []
    for key, cat, re_, pe, rs, ps, rf, pf, source in prepared:
        cosine, vocab_coverage = _cosine(rf, pf, state)
        row: dict[str, Any] = {"article_id": key, "category": cat,
            "reference_entries": len(re_), "prediction_entries": len(pe),
            "reference_unique_strings": len(rs), "prediction_unique_strings": len(ps),
            "reference_duplicate_strings": len(re_) - len(rs), "prediction_duplicate_strings": len(pe) - len(ps),
            "string_exact": _set_score(rs, ps), "cosine_tfidf": cosine,
            "tfidf_feature_coverage": vocab_coverage,
            "matched_strings": sorted(rs & ps), "reference_only_strings": sorted(rs - ps),
            "prediction_only_strings": sorted(ps - rs), "concept_reviewed": None}
        if concept_map is not None:
            rc, nr = _concepts(key, cat, rs, mapping)
            pc, np_ = _concepts(key, cat, ps, mapping)
            row["concept_reviewed"] = _set_score(rc, pc)
            row["concept_mapping_coverage"] = {"reference": nr / len(rs) if rs else None,
                "prediction": np_ / len(ps) if ps else None,
                "complete": nr == len(rs) and np_ == len(ps)}
            row["matched_concepts"] = sorted(rc & pc)
            row["reference_only_concepts"] = sorted(rc - pc)
            row["prediction_only_concepts"] = sorted(pc - rc)
        rspan, missing_r = _span_set(re_, source)
        pspan, missing_p = _span_set(pe, source)
        row["span_coverage"] = {"reference_entries_missing_offsets": missing_r,
            "prediction_entries_missing_offsets": missing_p, "reference_unique_spans": len(rspan),
            "prediction_unique_spans": len(pspan), "source_text_validated": source is not None}
        if missing_r or missing_p:
            row["span_exact"] = row["span_overlap"] = None
            row["span_unavailable_reason"] = "Some entries lack offsets; string agreement is not span F1."
        else:
            row["span_exact"] = _set_score(rspan, pspan)
            tp = _maximum_matching(rspan, pspan, overlap_threshold)
            row["span_overlap"] = _counts(tp, len(pspan) - tp, len(rspan) - tp)
        rows.append(row)
    serialized = json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {"module_version": __version__, "interpretation": "pairwise_agreement_not_gold_accuracy",
        "config": {"categories": list(CATEGORIES), "normalization": normalization,
            "empty_both": "null_excluded_from_macro", "single_empty": "jaccard_and_f1_zero",
            "offsets": "Unicode code points; zero-based; end exclusive; raw source",
            "entity_ids_required": False, "tfidf_fitting": fitting,
            "tfidf_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "span_overlap_char_iou_threshold": overlap_threshold,
            "concept_map_is_expert_input": concept_map is not None},
        "documents": {"reference": len(ref), "prediction": len(pred),
            "reference_only": sorted(ref.keys() - pred.keys()), "prediction_only": sorted(pred.keys() - ref.keys())},
        "rows": rows, "by_article": {k: _aggregate([r for r in rows if r["article_id"] == k]) for k in keys},
        "by_category": {c: _aggregate([r for r in rows if r["category"] == c]) for c in CATEGORIES},
        "overall": _aggregate(rows), "tfidf_state": state}


# ---------------- Optional lemma-based STRING comparison ----------------
# This layer never rewrites input text, offsets, raw scores, or entity labels.

class Pymorphy3Lemmatizer:
    """Conservative Russian word lemmatization; not entity linking.

    Only dictionary-known all-Cyrillic words longer than one character are
    lemmatized. Latin names, mixed-alphabet tokens, digits, punctuation and
    explicitly protected abbreviations are kept. Unknown words are not guessed.
    Selects the first probability-ranked parse: lexical ambiguity is NOT resolved
    using the sentence. The same instance is applied to both compared datasets.

    Pin packages with requirements-lemma.txt. No downloads occur at runtime.
    This adapter is optional; the original evaluator remains dependency-free.
    """
    def __init__(self, protected_tokens: Sequence[str] = (
        "ТНУ", "ГКНС", "СВ", "БПК", "ЗМУ", "ПТК", "РФ", "СНГ", "СССР",
        "Вт", "кВт", "МВт", "кДж", "МДж", "Гкал", "Мкал", "км", "мг",
        "кг", "мм", "га", "млн", "тыс", "сут", "гг"
    )) -> None:
        import importlib.metadata
        from functools import lru_cache
        try:
            import pymorphy3
            import pymorphy3_dicts_ru
        except ImportError as exc:
            raise ImportError(
                "Optional Russian lemmatizer unavailable. Run: "
                "python -m pip install -r requirements-lemma.txt; "
                "or pass a callable and an explicit lemmatizer_id. "
                "No fallback lemmatizer is silently substituted."
            ) from exc
        if isinstance(protected_tokens, str) or any(not isinstance(t, str) or not t for t in protected_tokens):
            raise TypeError("protected_tokens must be a sequence of nonempty strings")
        self._protected = frozenset(normalize_text(t) for t in protected_tokens)
        dictionary_path = Path(pymorphy3_dicts_ru.get_path())
        # Use the declared dictionary explicitly: ignore environment overrides.
        self._morph = pymorphy3.MorphAnalyzer(path=str(dictionary_path), lang="ru")
        hasher = hashlib.sha256()
        for file in sorted(p for p in dictionary_path.rglob("*") if p.is_file()):
            hasher.update(file.relative_to(dictionary_path).as_posix().encode("utf-8"))
            hasher.update(b"\0")
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            hasher.update(b"\0")
        self._metadata = {
            "kind": "pymorphy3", "policy": "ru_dictionary_surface_v1",
            "package_version": importlib.metadata.version("pymorphy3"),
            "dictionary_package": "pymorphy3-dicts-ru",
            "dictionary_version": importlib.metadata.version("pymorphy3-dicts-ru"),
            "dictionary_sha256": hasher.hexdigest(),
            "token_pattern": TOKEN_PATTERN,
            "parse_selection": "first_probability_ranked_parse",
            "unknown_words": "unchanged", "single_letters": "unchanged",
            "non_cyrillic_and_mixed_tokens": "unchanged",
            "punctuation_numbers_negation": "not_deleted",
            "protected_tokens": sorted(self._protected),
        }
        self._cached_token = lru_cache(maxsize=100000)(self._token)

    def _token(self, word: str) -> str:
        lower = unicodedata.normalize("NFC", word).casefold()
        if (len(lower) <= 1 or re.fullmatch(r"[а-яё]+", lower) is None
                or normalize_text(lower) in self._protected):
            return word
        if not self._morph.word_is_known(lower):
            return word
        parses = self._morph.parse(lower)
        if not parses:
            return word
        chosen = parses[0]
        if "Abbr" in chosen.tag:
            return word
        return chosen.normal_form

    def __call__(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("The lemmatizer expects a string")
        return _TOKENS.sub(lambda m: self._cached_token(m.group()),
                           unicodedata.normalize("NFC", text))

    def metadata(self) -> dict[str, Any]:
        # A defensive copy prevents accidental modification of a frozen IDF signature.
        return json.loads(json.dumps(self._metadata, ensure_ascii=False))


def _resolve_lemmatizer(lemmatizer: Any, lemmatizer_id: str | None = None) -> tuple[Any, dict[str, Any]]:
    if isinstance(lemmatizer, str):
        if lemmatizer != "pymorphy3":
            raise ValueError("Supported built-in lemmatizer: 'pymorphy3'; or pass a callable")
        lemmatizer = Pymorphy3Lemmatizer()
    if not callable(lemmatizer):
        raise TypeError("lemmatizer must be 'pymorphy3' or a callable str -> str")
    if hasattr(lemmatizer, "metadata") and callable(lemmatizer.metadata):
        meta = lemmatizer.metadata()
        if not isinstance(meta, Mapping) or not meta:
            raise ValueError("lemmatizer.metadata() must return a nonempty JSON mapping")
        meta = dict(meta)
        if lemmatizer_id is not None:
            if not isinstance(lemmatizer_id, str) or not lemmatizer_id.strip():
                raise ValueError("lemmatizer_id must be nonempty")
            meta["explicit_id"] = lemmatizer_id
    else:
        if not isinstance(lemmatizer_id, str) or not lemmatizer_id.strip():
            raise ValueError("Custom lemmatizers need a stable lemmatizer_id including versions/rules")
        meta = {"kind": "custom", "id": lemmatizer_id,
                "reproducibility": "caller_responsible_for_pinning_model_and_rules"}
    try:
        meta = json.loads(json.dumps(meta, ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Lemmatizer metadata must be finite JSON data") from exc
    return lemmatizer, meta


def _lemma_documents(data: Any, lemmatizer: Any, normalization: str) -> tuple[list[dict], dict]:
    docs, changes, merges = [], [], []
    cache: dict[str, str] = {}
    for key, d in _load(data).items():
        entries = []
        for cat in CATEGORIES:
            groups: dict[str, set[str]] = defaultdict(set)
            for a in d["annotations"][cat]:
                surface = a["surface"]
                if surface not in cache:
                    value = lemmatizer(surface)
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("A lemmatizer must return a nonempty string for every nonempty annotation")
                    cache[surface] = normalize_text(value, normalization)
                value = cache[surface]
                original = normalize_text(surface, normalization)
                entries.append({"category": cat, "surface": value})
                groups[value].add(original)
                if original != value:
                    changes.append({"article_id": key, "category": cat,
                                    "original_surface": surface, "lemmatized_surface": value})
            for lemma, variants in groups.items():
                if len(variants) > 1:
                    merges.append({"article_id": key, "category": cat,
                                   "lemmatized_surface": lemma, "source_variants": sorted(variants)})
        # Omitting source/offsets here is intentional. They remain in the raw report.
        docs.append({"article_id": key, "annotations": entries})
    # Repeated mentions are still counted by the raw span score, not by string sets.
    unique_changes = list({(x["article_id"], x["category"], x["original_surface"]): x for x in changes}.values())
    return docs, {"changed_unique_surfaces": unique_changes, "merged_variants": merges}


def _wrap_lemma_state(state: dict, metadata: dict, normalization: str) -> dict:
    return {"schema_version": "annotation_lemma_tfidf_v1", "lemmatizer": metadata,
            "normalization": normalization, "tfidf_state": state}


def fit_tfidf(*datasets: Any, normalization: str = "light", lemmatizer: Any = None,
              lemmatizer_id: str | None = None) -> dict[str, Any]:
    """Freeze an IDF vocabulary; raw by default, lemma-specific when requested.

    Raw state -> evaluate(tfidf_state=...). Lemma state ->
    evaluate(lemmatizer=..., lemma_tfidf_state=...). The states are incompatible
    by design. Do not reuse a raw IDF vocabulary for lemmatized surfaces.
    """
    if lemmatizer is None:
        if lemmatizer_id is not None:
            raise ValueError("lemmatizer_id requires a lemmatizer")
        return _fit_tfidf_raw(*datasets, normalization=normalization)
    lemma, meta = _resolve_lemmatizer(lemmatizer, lemmatizer_id)
    transformed = [_lemma_documents(d, lemma, normalization)[0] for d in datasets]
    return _wrap_lemma_state(_fit_tfidf_raw(*transformed, normalization=normalization), meta, normalization)


def _evaluate_lexical(reference: Any, prediction: Any, *, normalization: str = "light",
             sources: Any = None, concept_map: Any = None, tfidf_state: Any = None,
             overlap_threshold: float = 0.5, require_same_documents: bool = False,
             lemmatizer: Any = None, lemmatizer_id: str | None = None,
             lemma_tfidf_state: Any = None) -> dict[str, Any]:
    """Compare original NER annotations and optionally lemmatized string lists.

    Backward-compatible: with lemmatizer=None the previous raw metrics are used.
    With lemmatizer='pymorphy3' or a versioned callable, report['lemmatized'] adds
    per-article/per-category Jaccard, precision/recall/F1 and TF-IDF cosine after
    applying ONE lemmatizer to BOTH lists. Main span metrics are unchanged.

    Model inputs need not be lemmatized. Pass original surfaces and raw offsets.
    If a model runs on a rewritten source, first map its offsets back to the
    original source. This function never invents or realigns those offsets.
    A lemma match is lexical agreement, NOT proof of correct meaning or category.

    concept_map, when supplied, affects only the original report; it is not
    automatically combined with lemmatization. No graph relations are required.
    """
    if lemmatizer is None and (lemma_tfidf_state is not None or lemmatizer_id is not None):
        raise ValueError("lemma_tfidf_state/lemmatizer_id requires a lemmatizer")
    result = _evaluate_raw(reference, prediction, normalization=normalization,
        sources=sources, concept_map=concept_map, tfidf_state=tfidf_state,
        overlap_threshold=overlap_threshold, require_same_documents=require_same_documents)
    if lemmatizer is None:
        return result
    lemma, meta = _resolve_lemmatizer(lemmatizer, lemmatizer_id)
    rr, ra = _lemma_documents(reference, lemma, normalization)
    pp, pa = _lemma_documents(prediction, lemma, normalization)
    lemma_state = None
    if lemma_tfidf_state is not None:
        wrapped = _read(lemma_tfidf_state)
        if (not isinstance(wrapped, Mapping)
                or wrapped.get("schema_version") != "annotation_lemma_tfidf_v1"):
            raise ValueError("Use a lemma-specific state from fit_tfidf(..., lemmatizer=...)")
        if wrapped.get("lemmatizer") != meta or wrapped.get("normalization") != normalization:
            raise ValueError("Lemmatizer/version/dictionary/rules mismatch with frozen lemma TF-IDF state")
        lemma_state = wrapped.get("tfidf_state")
        if not isinstance(lemma_state, Mapping):
            raise ValueError("Invalid nested lemma TF-IDF state")
    secondary = _evaluate_raw(rr, pp, normalization=normalization,
        tfidf_state=lemma_state, overlap_threshold=overlap_threshold,
        require_same_documents=require_same_documents)
    secondary["interpretation"] = "lemmatized_string_agreement_not_strict_ner_accuracy"
    secondary["lemmatizer"] = meta
    secondary["audit"] = {"reference": ra, "prediction": pa}
    secondary["config"]["offsets"] = "not_applicable; see main raw report for validated spans"
    secondary["config"]["lemma_input"] = "original_annotation_surface_not_whole_source"
    secondary["config"]["same_lemmatizer_both_sides"] = True
    for row in secondary["rows"]:
        row["span_exact"] = row["span_overlap"] = None
        row["span_unavailable_reason"] = "Lemmatized strings are a secondary score; use raw report for spans."
    keys = list(secondary["by_article"])
    secondary["by_article"] = {k: _aggregate([r for r in secondary["rows"] if r["article_id"] == k]) for k in keys}
    secondary["by_category"] = {c: _aggregate([r for r in secondary["rows"] if r["category"] == c]) for c in CATEGORIES}
    secondary["overall"] = _aggregate(secondary["rows"])
    secondary["tfidf_state"] = _wrap_lemma_state(secondary["tfidf_state"], meta, normalization)
    result["lemmatized"] = secondary
    return result


# ---------------- Agreement on source positions: Cohen's kappa ----------------
KAPPA_TOKEN_PATTERN = r"(?u)\w+|[^\w\s]"


def _kappa_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, Any]:
    """Unweighted binary Cohen kappa; undefined cases are JSON null, never 1."""
    n = tn + fp + fn + tp
    out = {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "n_units": n,
           "kappa": None, "observed_agreement": None, "expected_agreement": None,
           "reference_positive_rate": None, "prediction_positive_rate": None}
    if not n:
        out["undefined_reason"] = "no_evaluation_units"
        return out
    # Integer numerators avoid cancellation when almost all units are negative.
    expected_numerator = (tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)
    denominator = n * n - expected_numerator
    out.update(observed_agreement=(tn + tp) / n,
               expected_agreement=expected_numerator / (n * n),
               reference_positive_rate=(tp + fn) / n,
               prediction_positive_rate=(tp + fp) / n)
    if denominator == 0:
        out["undefined_reason"] = "expected_agreement_is_one_constant_identical_labels"
    else:
        out["kappa"] = ((tn + tp) * n - expected_numerator) / denominator
        out["undefined_reason"] = None
    return out


def _kappa_units(source: str, unit: str) -> list[tuple[int, int]]:
    if unit == "token":
        return [m.span() for m in re.finditer(KAPPA_TOKEN_PATTERN, source)]
    return [(i, i + 1) for i, ch in enumerate(source) if not ch.isspace()]


def _unit_labels(spans: set[tuple], units: list[tuple[int, int]]) -> bytearray:
    from bisect import bisect_right
    labels = bytearray(len(units))
    ends = [end for _, end in units]
    for mention in spans:
        for start, end in mention:
            i = bisect_right(ends, start)
            while i < len(units) and units[i][0] < end:
                labels[i] = 1
                i += 1
    return labels


def _kappa_pool(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r["available"]]
    pooled = _kappa_from_counts(*(sum(r[k] for r in usable) for k in ("tn", "fp", "fn", "tp")))
    complete = len(usable) == len(rows)
    return {"kappa": pooled["kappa"] if complete else None,
            "complete": complete, "evaluated_groups": len(usable), "total_groups": len(rows),
            "missing_groups": [{"article_id": r["article_id"], "category": r["category"],
                                "reason": r["unavailable_reason"]} for r in rows if not r["available"]],
            "pooled_counts_available_subset": pooled,
            "macro_document_kappa_available_subset": _mean([r["kappa"] for r in usable])}


def _kappa_report(reference: Any, prediction: Any, sources: Any, unit: str) -> dict[str, Any]:
    if unit not in {"token", "character"}:
        raise ValueError("kappa_unit must be 'token' or 'character'")
    ref, pred = _load(reference), _load(prediction)
    supplied = {} if sources is None else _read(sources)
    empty = {"annotations": {c: [] for c in CATEGORIES}, "source_text": None}
    rows = []
    for key in sorted(ref.keys() | pred.keys()):
        r, p = ref.get(key, empty), pred.get(key, empty)
        # _evaluate_raw has already validated source versions and source hashes.
        source = next((s for s in (r["source_text"], p["source_text"], supplied.get(key))
                       if s is not None), None)
        units = _kappa_units(source, unit) if source is not None else []
        for cat in CATEGORIES:
            rs, mr = _span_set(r["annotations"][cat], source)
            ps, mp = _span_set(p["annotations"][cat], source)
            row = {"article_id": key, "category": cat, "available": False,
                   "kappa": None, "n_units": None, "tn": None, "fp": None, "fn": None, "tp": None}
            if source is None:
                row["unavailable_reason"] = "original_source_text_required"
            elif mr or mp:
                row["unavailable_reason"] = "complete_offsets_required_on_both_sides"
            else:
                rl, pl = _unit_labels(rs, units), _unit_labels(ps, units)
                counts = [0, 0, 0, 0]  # 00=TN, 01=FP, 10=FN, 11=TP
                for a, b in zip(rl, pl):
                    counts[2 * a + b] += 1
                row.update(_kappa_from_counts(*counts))
                row.update(available=True, unavailable_reason=None,
                           source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest())
            rows.append(row)
    by_category = {c: _kappa_pool([r for r in rows if r["category"] == c]) for c in CATEGORIES}
    keys = sorted(ref.keys() | pred.keys())
    by_article = {}
    for key in keys:
        group = [r for r in rows if r["article_id"] == key]
        available = [r for r in group if r["available"]]
        complete = len(available) == len(group)
        vals = [r["kappa"] for r in available]
        by_article[key] = {"macro_kappa": _mean(vals) if complete else None,
            "macro_kappa_available_subset": _mean(vals), "complete": complete,
            "evaluated_categories": len(available),
            "defined_categories": sum(v is not None for v in vals)}
    complete = all(v["complete"] for v in by_category.values())
    cat_values = [v["kappa"] for v in by_category.values()]
    return {"config": {"unit": unit, "token_pattern": KAPPA_TOKEN_PATTERN if unit == "token" else None,
        "whitespace": "excluded", "labeling": "binary_membership_per_category; any_character_overlap",
        "background": "all_evaluation_units_in_the_same_original_source_including_O",
        "overlapping_categories": "independent_binary_kappa_for_each_category",
        "undefined": "null; never coerced to zero or one", "lemmatization": "not_applied",
        "scope_warning": "Use identical preagreed source scope; changing background changes kappa.",
        "interpretation": "position_membership_agreement; not exact_entity_boundary_accuracy"},
        "rows": rows, "by_article": by_article, "by_category": by_category,
        "overall": {"macro_kappa": _mean(cat_values) if complete else None,
                    "macro_kappa_complete_categories": _mean(cat_values),
                    "complete": complete, "defined_categories": sum(v is not None for v in cat_values),
                    "total_groups": len(rows), "evaluated_groups": sum(r["available"] for r in rows)}}


# ---------------- Optional embedding-based phrase agreement ----------------
class SentenceTransformerEmbedder:
    """Optional, explicitly selected encoder. No model downloaded by default.

    Remote IDs require an immutable 40-hex commit revision. Local directories
    are fingerprinted. Set local_files_only=False explicitly for first download.
    Text exceeding the encoder's context raises instead of being silently cut.
    This adapter is symmetric: identical encoding policy for both annotators.
    """
    def __init__(self, model_name_or_path: str | Path, *, revision: str | None = None,
                 device: str = "cpu", batch_size: int = 32, local_files_only: bool = True) -> None:
        import importlib.metadata
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        name = str(model_name_or_path)
        path = Path(name)
        local = path.is_dir()
        if not local and (not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None):
            raise ValueError("Pin remote embedding model with embedding_revision=<40-hex commit SHA>; "
                             "or use a fixed local model directory")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Optional sentence-transformers is not installed. Install requirements-embeddings.txt "
                              "or pass precomputed embeddings/a versioned encoder. No TF-IDF fallback is used.") from exc
        self.model = SentenceTransformer(name, revision=revision, device=device,
            local_files_only=local_files_only, trust_remote_code=False)
        self.model.eval()
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}
        self._meta = {"kind": "sentence_transformers", "model": name, "revision": revision,
            "device": device, "batch_size": batch_size, "prompt": "",
            "max_seq_length": int(self.model.max_seq_length), "long_text_policy": "error_no_truncation",
            "normalize_embeddings": True, "local_files_only": local_files_only,
            "versions": {p: importlib.metadata.version(p) for p in
                         ("sentence-transformers", "transformers", "torch", "numpy")}}
        if local:
            hasher = hashlib.sha256()
            count = 0
            for file in sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts):
                hasher.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
                with file.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        hasher.update(chunk)
                hasher.update(b"\0")
                count += 1
            self._meta.update(local_directory_sha256=hasher.hexdigest(), local_file_count=count)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, str):
            raise TypeError("encode expects a sequence of strings, not a single string")
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        tokenizer = self.model.tokenizer
        limit = int(self.model.max_seq_length)
        for start in range(0, len(missing), self.batch_size):
            group = missing[start:start + self.batch_size]
            tokens = tokenizer(group, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
            for text, ids in zip(group, tokens):
                if len(ids) > limit:
                    raise ValueError(f"Embedding input has {len(ids)} tokens, limit {limit}. "
                        "Select a longer-context encoder; do not silently cut RESULT/TASK. "
                        f"Text starts: {text[:80]!r}")
            values = self.model.encode(group, batch_size=self.batch_size, show_progress_bar=False,
                normalize_embeddings=True, convert_to_numpy=True, prompt="")
            for text, vector in zip(group, values):
                self._cache[text] = [float(x) for x in vector]
        return [self._cache[t] for t in texts]

    def metadata(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._meta, ensure_ascii=False))


def _embedding_thresholds(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        if any(k not in CATEGORIES and k != "*" for k in value):
            raise ValueError("Unknown category in semantic_threshold")
        if any(c not in value and "*" not in value for c in CATEGORIES):
            raise ValueError("Specify a threshold for all six categories, or a '*' default")
        values = {c: value[c] if c in value else value["*"] for c in CATEGORIES}
    else:
        values = {c: value for c in CATEGORIES}
    result = {}
    for c, v in values.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not -1 <= v <= 1:
            raise ValueError("semantic_threshold must be an explicit finite number in [-1, 1], "
                             "or a per-category mapping; calibrate on a held-out development set")
        result[c] = float(v)
    return result


def _vectors_for_texts(texts: Sequence[str], encoder: Any, encoder_id: str | None,
                      revision: str | None, local_files_only: bool, normalization: str
                      ) -> tuple[dict[str, list[float]], dict[str, Any]]:
    if isinstance(encoder, (str, Path)):
        encoder = SentenceTransformerEmbedder(encoder, revision=revision, local_files_only=local_files_only)
    elif revision is not None:
        raise ValueError("embedding_revision only applies to a string/path embedding_model")
    if isinstance(encoder, Mapping):
        if encoder.get("schema_version") == "annotation_embeddings_v1":
            if encoder.get("normalization", "strict") != normalization:
                raise ValueError("Cached embedding normalization differs from semantic_normalization")
            values = encoder.get("vectors")
            metadata = encoder.get("metadata")
            if not isinstance(values, Mapping) or not isinstance(metadata, Mapping) or not metadata:
                raise ValueError("Embedding cache needs vectors and nonempty versioned metadata")
        else:
            values, metadata = encoder, None
        missing = [t for t in texts if t not in values]
        if missing:
            raise ValueError(f"Embedding cache missing {len(missing)} texts; first: {missing[0]!r}")
        raw = [values[t] for t in texts]
    else:
        provider = getattr(encoder, "encode", None)
        if not callable(provider):
            provider = encoder
        if not callable(provider):
            raise TypeError("embedding_model must be a model name, vector mapping, callable, or encode() object")
        metadata_fn = getattr(encoder, "metadata", None)
        metadata = metadata_fn() if callable(metadata_fn) else None
        raw = provider(list(texts)) if texts else []
    if metadata is None:
        if not isinstance(encoder_id, str) or not encoder_id.strip():
            raise ValueError("Custom/cached embeddings require embedding_model_id with model/revision/pooling rules")
        metadata = {"kind": "custom_or_cached", "id": encoder_id,
                    "reproducibility": "caller_responsible_for_pinning_weights_and_encoding"}
    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError("Embedding metadata must be a nonempty JSON mapping")
    metadata = json.loads(json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False))
    if encoder_id is not None:
        metadata["user_supplied_id"] = encoder_id
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)) or len(raw) != len(texts):
        raise ValueError("Encoder must return one vector per input text, in input order")
    vectors: dict[str, list[float]] = {}
    dimension = None
    digest = hashlib.sha256()
    for text, vector in zip(texts, raw):
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError("Each embedding must be a nonempty numeric vector")
        try:
            if any(isinstance(x, (bool, str, bytes)) for x in vector):
                raise ValueError("Nonnumeric embedding component")
            vals = [float(x) for x in vector]
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Embeddings must contain finite real numbers") from exc
        if any(not math.isfinite(x) for x in vals):
            raise ValueError("Embeddings contain NaN/Inf")
        if dimension is None:
            dimension = len(vals)
        elif dimension != len(vals):
            raise ValueError("All embeddings must have the same dimension")
        norm = math.hypot(*vals)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("Zero/overflowing embedding vector is not a valid cosine input")
        vectors[text] = [x / norm for x in vals]
        digest.update(json.dumps([text, vals], ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        digest.update(b"\n")
    metadata.update(dimension=dimension, encoded_unique_texts=len(texts), supplied_vectors_sha256=digest.hexdigest())
    return vectors, metadata


def _semantic_assignment(scores: list[list[float]], threshold: float) -> list[tuple[int, int]]:
    """Exact 1:1 matching: maximum cardinality, then maximum summed cosine.

    Rectangular Hungarian assignment with one private dummy column per row.
    The cardinality bonus dominates any possible difference in total cosine.
    Deterministic for the same sorted text order. No greedy duplicate matches.
    """
    n = len(scores)
    m = len(scores[0]) if n else 0
    if not n or not m:
        return []
    if any(len(row) != m for row in scores):
        raise ValueError("Ragged similarity matrix")
    if any(not math.isfinite(s) or not -1 <= s <= 1 for row in scores for s in row):
        raise ValueError("Similarity matrix must contain finite cosines in [-1,1]")
    # Keep the smaller side in rows for performance; restore orientation later.
    if n > m:
        swapped = _semantic_assignment([list(x) for x in zip(*scores)], threshold)
        return sorted((j, i) for i, j in swapped)
    bonus = 2 * min(n, m) + 1
    forbidden = (n + m + 1) * bonus
    width = m + n
    def cost(i: int, j: int) -> float:
        if j >= m:
            return 0.0
        score = scores[i][j]
        return -bonus - score if score >= threshold else float(forbidden)
    # Row/column potentials and augmenting paths; indices here are 1-based.
    u, v = [0.0] * (n + 1), [0.0] * (width + 1)
    p, way = [0] * (width + 1), [0] * (width + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = [math.inf] * (width + 1), [False] * (width + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, width + 1):
                if not used[j]:
                    cur = cost(i0 - 1, j - 1) - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(width + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1)
             if p[j] and scores[p[j] - 1][j - 1] >= threshold]
    return sorted(pairs)


def _pair_audit_flags(left: str, right: str) -> list[str]:
    """Warnings only, not a factuality checker and not a second acceptance rule."""
    number_pattern = r"(?<!\w)[+\-−]?\d+(?:[.,]\d+)?(?:[eE][+\-]?\d+)?"
    numbers = lambda s: sorted(t.replace(",", ".").replace("−", "-") for t in re.findall(number_pattern, s))
    negations = {"не", "нет", "без", "not", "no", "without", "neither", "nor"}
    neg = lambda s: Counter(t for t in _TOKENS.findall(s.casefold()) if t in negations)
    flags = []
    if numbers(left) != numbers(right):
        flags.append("different_numeric_literals_review_required")
    if neg(left) != neg(right):
        flags.append("different_negation_markers_review_required")
    return flags


def _semantic_counts(matched: int, nr: int, np_: int) -> dict[str, Any]:
    result = _counts(matched, np_ - matched, nr - matched)
    result.update(matched=matched, reference_count=nr, prediction_count=np_,
                  match_rate=matched / max(nr, np_) if max(nr, np_) else None)
    return result


def _semantic_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    micro = _semantic_counts(sum(r["matched"] for r in rows),
        sum(r["reference_count"] for r in rows), sum(r["prediction_count"] for r in rows))
    return {"micro": micro, "macro_match_rate": _mean([r["match_rate"] for r in rows]),
            "macro_precision": _mean([r["precision"] for r in rows]),
            "macro_recall": _mean([r["recall"] for r in rows]),
            "macro_f1": _mean([r["f1"] for r in rows]),
            "macro_jaccard": _mean([r["jaccard"] for r in rows]),
            "groups": len(rows), "nonempty_groups": sum(r["match_rate"] is not None for r in rows)}


def _semantic_report(reference: Any, prediction: Any, *, encoder: Any, encoder_id: str | None,
                     revision: str | None, local_files_only: bool, thresholds: dict[str, float],
                     normalization: str, max_pairs: int) -> dict[str, Any]:
    normalize_text("", normalization)
    if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs <= 0:
        raise ValueError("semantic_max_pairs must be a positive integer")
    ref, pred = _load(reference), _load(prediction)
    empty = {"annotations": {c: [] for c in CATEGORIES}}
    prepared, unique = [], set()
    for key in sorted(ref.keys() | pred.keys()):
        for cat in CATEGORIES:
            re_ = ref.get(key, empty)["annotations"][cat]
            pe = pred.get(key, empty)["annotations"][cat]
            left, right = sorted(_surfaces(re_, normalization)), sorted(_surfaces(pe, normalization))
            if len(left) * len(right) > max_pairs:
                raise ValueError(f"Semantic matrix for {key}/{cat} exceeds semantic_max_pairs={max_pairs}; "
                                 "raise the explicit limit or evaluate smaller document groups")
            prepared.append((key, cat, left, right, len(re_) - len(left), len(pe) - len(right)))
            unique.update(left)
            unique.update(right)
    vectors, metadata = _vectors_for_texts(sorted(unique), encoder, encoder_id, revision,
                                          local_files_only, normalization)
    rows = []
    for key, cat, left, right, rd, pd in prepared:
        scores = [[1.0 if a == b else max(-1.0, min(1.0, math.fsum(x * y for x, y in zip(vectors[a], vectors[b]))))
                   for b in right] for a in left]
        pairs = _semantic_assignment(scores, thresholds[cat])
        matched_left, matched_right = {i for i, _ in pairs}, {j for _, j in pairs}
        row = {"article_id": key, "category": cat, "threshold": thresholds[cat],
               **_semantic_counts(len(pairs), len(left), len(right)),
               "reference_collapsed_duplicate_entries": rd, "prediction_collapsed_duplicate_entries": pd,
               "mean_matched_cosine": _mean([scores[i][j] for i, j in pairs]),
               "matched_pairs": [{"reference_text": left[i], "prediction_text": right[j],
                   "cosine": scores[i][j], "audit_flags": _pair_audit_flags(left[i], right[j])} for i, j in pairs],
               "unmatched_reference": [x for i, x in enumerate(left) if i not in matched_left],
               "unmatched_prediction": [x for j, x in enumerate(right) if j not in matched_right]}
        rows.append(row)
    by_category = {c: _semantic_aggregate([r for r in rows if r["category"] == c]) for c in CATEGORIES}
    overall = _semantic_aggregate(rows)
    overall["category_macro_match_rate"] = _mean([v["micro"]["match_rate"] for v in by_category.values()])
    return {"interpretation": "embedding_threshold_phrase_agreement_not_factual_correctness_or_span_accuracy",
        "config": {"normalization": normalization, "lemmatization": "not_applied",
            "unit": "unique_phrase_per_article_and_category", "cosine_thresholds": thresholds,
            "threshold_rule": "cosine >= threshold", "matching": "one_to_one_maximum_cardinality_then_total_cosine",
            "match_rate_formula": "matched / max(reference_count, prediction_count)",
            "both_empty": "null_excluded_from_macro", "one_empty": "match_rate_zero",
            "cross_document_and_cross_category_matches": False,
            "audit_flags_affect_matching": False,
            "warning": "Embedding similarity does not establish logical equivalence, factuality, or correct numbers/negation."},
        "embedding_model": metadata, "rows": rows, "by_category": by_category,
        "by_article": {k: _semantic_aggregate([r for r in rows if r["article_id"] == k])
                       for k in sorted(ref.keys() | pred.keys())}, "overall": overall}


def evaluate(reference: Any, prediction: Any, *, normalization: str = "light",
             sources: Any = None, concept_map: Any = None, tfidf_state: Any = None,
             overlap_threshold: float = 0.5, require_same_documents: bool = False,
             lemmatizer: Any = None, lemmatizer_id: str | None = None,
             lemma_tfidf_state: Any = None, compute_kappa: bool = True,
             kappa_unit: str = "token", embedding_model: Any = None,
             embedding_model_id: str | None = None, embedding_revision: str | None = None,
             embedding_local_files_only: bool = True, semantic_threshold: Any = None,
             semantic_normalization: str = "strict", semantic_max_pairs: int = 2_000_000) -> dict[str, Any]:
    """Evaluate NER: original, optional lemma, positional kappa, optional semantic.

    All pre-1.2 parameters and scores keep their meaning. New top-level reports:
      cohen_kappa: binary membership per category on common source tokens (default)
          or non-whitespace characters; requires original source AND all offsets.
          Does not infer offsets from strings or choose an arbitrary first match.
      semantic: optional unique-phrase agreement from cosine >= semantic_threshold,
          matched 1:1 within each article/category. Accepts an explicit encoder,
          .encode() object, named SentenceTransformer or precomputed vector map.

    Threshold is REQUIRED when embeddings are enabled: no universally valid
    default. A scalar or per-category mapping (optional '*' default) is accepted.
    match_rate = matched / max(n_reference, n_prediction). Directional precision,
    recall, F1, Jaccard and auditable pairs are returned too. Semantics never
    replace exact spans or raw scores and are not automatically lemmatized.
    Named remote models require embedding_revision=<40-hex commit SHA>.
    Custom encoders require metadata() or a pinned embedding_model_id.
    No graph links or concept IDs are required from NER model predictions.
    """
    if not isinstance(compute_kappa, bool):
        raise TypeError("compute_kappa must be a bool")
    if kappa_unit not in {"token", "character"}:
        raise ValueError("kappa_unit must be 'token' or 'character'")
    if embedding_model is None and any(v is not None for v in (embedding_model_id, embedding_revision, semantic_threshold)):
        raise ValueError("Semantic parameters require embedding_model; no vectors are guessed")
    thresholds = _embedding_thresholds(semantic_threshold) if embedding_model is not None else None
    result = _evaluate_lexical(reference, prediction, normalization=normalization, sources=sources,
        concept_map=concept_map, tfidf_state=tfidf_state, overlap_threshold=overlap_threshold,
        require_same_documents=require_same_documents, lemmatizer=lemmatizer,
        lemmatizer_id=lemmatizer_id, lemma_tfidf_state=lemma_tfidf_state)
    if compute_kappa:
        result["cohen_kappa"] = _kappa_report(reference, prediction, sources, kappa_unit)
    if embedding_model is not None:
        result["semantic"] = _semantic_report(reference, prediction, encoder=embedding_model,
            encoder_id=embedding_model_id, revision=embedding_revision,
            local_files_only=embedding_local_files_only, thresholds=thresholds,
            normalization=semantic_normalization, max_pairs=semantic_max_pairs)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("prediction")
    parser.add_argument("--output", default="evaluation.json")
    parser.add_argument("--sources")
    parser.add_argument("--concept-map")
    parser.add_argument("--tfidf-state")
    parser.add_argument("--normalization", choices=["strict", "light"], default="light")
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--lemmatizer", choices=["pymorphy3"])
    parser.add_argument("--lemma-tfidf-state")
    parser.add_argument("--no-kappa", action="store_true")
    parser.add_argument("--kappa-unit", choices=["token", "character"], default="token")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-model-id")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--embedding-cache", help="JSON of vectors + metadata (no model download)")
    parser.add_argument("--semantic-threshold", type=float)
    parser.add_argument("--semantic-thresholds", help="JSON mapping of per-category thresholds")
    parser.add_argument("--semantic-normalization", choices=["strict", "light"], default="strict")
    args = parser.parse_args()
    if args.embedding_model and args.embedding_cache:
        parser.error("Choose --embedding-model OR --embedding-cache")
    if args.semantic_threshold is not None and args.semantic_thresholds:
        parser.error("Choose --semantic-threshold OR --semantic-thresholds")
    embedder = _read(args.embedding_cache) if args.embedding_cache else args.embedding_model
    thresholds = _read(args.semantic_thresholds) if args.semantic_thresholds else args.semantic_threshold
    result = evaluate(args.reference, args.prediction, sources=args.sources, concept_map=args.concept_map,
        tfidf_state=args.tfidf_state, normalization=args.normalization, overlap_threshold=args.overlap_threshold,
        lemmatizer=args.lemmatizer, lemma_tfidf_state=args.lemma_tfidf_state,
        compute_kappa=not args.no_kappa, kappa_unit=args.kappa_unit,
        embedding_model=embedder, embedding_model_id=args.embedding_model_id,
        embedding_revision=args.embedding_revision, embedding_local_files_only=not args.allow_model_download,
        semantic_threshold=thresholds, semantic_normalization=args.semantic_normalization)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
