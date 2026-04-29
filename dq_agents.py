"""
Multi-Agent Data Quality System for NoiPA datasets
====================================================

This module implements the core logic of the six agents that compose the
data-quality pipeline. The agents are designed to be invoked from a
LangGraph orchestrator (see main.ipynb) but they can also be called
independently for unit testing and from the Streamlit UI (see app.py).

The pipeline:
    1. Schema Validation       -> column names, types, allowed values
    2. Completeness Analysis   -> nulls, placeholders, sparse columns
    3. Consistency Validation  -> formats, cross-column logic, duplicates
    4. Anomaly Detection       -> numerical outliers, rare categories
    5. Remediation             -> reliability score + LLM-driven suggestions
    6. Auto-Fix (Cleaner)      -> applies safe fixes and emits a fix log

Author: Eleonora Cappetti, Daniela Chiezzi, Chiara De Sio
Course:  Machine Learning 2025/26 - Reply x LUISS Project
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Strings that look like missing data even if pandas does not treat them as NaN.
PLACEHOLDER_VALUES = {
    "n/a", "na", "n.a.", "n.d.", "nd", "nan", "none", "null",
    "-", "--", "---", "/", "//", "///", "", " ", "?", "??", "???",
    "00", "000", "0000",
    "unknown", "sconosciuto", "missing", "vuoto",
    "dato mancante", "non disponibile", "non specificato",
    "tbd", "da verificare", "da definire",
}

# Italian province codes (ISO 3166-2 :IT). Used to validate provincia_sede.
ITALIAN_PROVINCES = {
    "AG", "AL", "AN", "AO", "AR", "AP", "AT", "AV", "BA", "BT", "BL", "BN",
    "BG", "BI", "BO", "BZ", "BS", "BR", "CA", "CL", "CB", "CE", "CT", "CZ",
    "CH", "CO", "CS", "CR", "KR", "CN", "EN", "FM", "FE", "FI", "FG", "FC",
    "FR", "GE", "GO", "GR", "IM", "IS", "SP", "AQ", "LT", "LE", "LC", "LI",
    "LO", "LU", "MC", "MN", "MS", "MT", "ME", "MI", "MO", "MB", "NA", "NO",
    "NU", "OR", "PD", "PA", "PR", "PV", "PG", "PU", "PE", "PC", "PI", "PT",
    "PN", "PZ", "PO", "RG", "RA", "RC", "RE", "RI", "RN", "RM", "RO", "SA",
    "SS", "SV", "SI", "SR", "SO", "SU", "TA", "TE", "TR", "TO", "TP", "TN",
    "TV", "TS", "UD", "VA", "VE", "VB", "VC", "VR", "VV", "VI", "VT",
}

# NoiPA payroll taxonomy reference. We learn cross-column rules from the
# data itself in `check_cross_column_consistency()`; this map is used
# only as a fallback when the dataset is too small for rule mining.
# It is intentionally permissive because real NoiPA data has a long tail
# of legitimate combinations.
NOIPA_TAXONOMY_REFERENCE: dict[str, set[str]] = {
    "Erariali":      {"IRPEF", "IRAP", "Addizionali Comunali", "Addizionali Regionali"},
    "Previdenziali": {"INPS", "INAIL", "Previdenziali a carico del datore di lavoro",
                      "Previdenziali a carico del lavoratore"},
    "Varie":         {"Ritenute Sindacali", "Altre voci", "Addizionali Comunali",
                      "Addizionali Regionali", "IRPEF", "IRAP", "Prestiti",
                      "Riscatti"},
}

# Reliability score weights -- chosen so that completeness and schema
# (the two pillars of "is the data even usable") dominate.
SCORE_WEIGHTS = {
    "completeness": 0.40,
    "schema":       0.30,
    "consistency":  0.20,
    "anomaly":      0.10,
}


# ---------------------------------------------------------------------------
# Schema Agent
# ---------------------------------------------------------------------------

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_VALID_IDENT_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")


def to_snake_case(name: str) -> str:
    """
    Convert an arbitrary column name to a clean snake_case identifier.

    Examples:
        "Tipo Imposta"     -> "tipo_imposta"
        "2cod_imposta"     -> "col_2cod_imposta"
        "ente%code"        -> "ente_code"
        "att ivazioni"     -> "att_ivazioni"
    """
    # Normalize unicode (e.g. accented letters -> ascii equivalents).
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    # Replace any non-alphanumeric character with underscore.
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    # Insert underscore between camelCase boundaries.
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Collapse repeated underscores.
    name = re.sub(r"_+", "_", name).strip("_")
    name = name.lower()
    # Identifiers cannot start with a digit -> prepend "col_".
    if name and name[0].isdigit():
        name = f"col_{name}"
    return name or "unnamed_column"


def diagnose_column_name(col: str) -> dict[str, Any]:
    """
    Inspect a single column name and return a structured diagnosis.

    The diagnosis distinguishes between "minor" cosmetic problems
    (casing) and "major" issues (spaces, special characters, leading
    digit) so that the schema score can weight them differently.
    """
    issues: list[str] = []
    severity = "ok"

    if " " in col:
        issues.append("contains_spaces")
        severity = "major"
    if not _VALID_IDENT_RE.match(col):
        issues.append("special_characters")
        severity = "major"
    if col and col[0].isdigit():
        issues.append("starts_with_digit")
        severity = "major"
    if _VALID_IDENT_RE.match(col) and not _SNAKE_CASE_RE.match(col):
        issues.append("not_snake_case")
        if severity == "ok":
            severity = "minor"

    return {
        "column":     col,
        "issues":     issues,
        "severity":   severity,
        "suggestion": to_snake_case(col) if issues else col,
    }


def _is_mostly_numeric(series: pd.Series, threshold: float = 0.8) -> bool:
    """True if at least `threshold` of non-null values can be parsed as numbers."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    parsed = pd.to_numeric(non_null, errors="coerce")
    return parsed.notna().mean() >= threshold


def _looks_like_date(series: pd.Series) -> bool:
    """Heuristic: True if most values match a YYYYMM / YYYY-MM / ISO-date pattern."""
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return False
    patterns = [
        r"^\d{6}$",                      # YYYYMM
        r"^\d{4}-\d{2}$",                # YYYY-MM
        r"^\d{4}/\d{2}$",                # YYYY/MM
        r"^\d{4}-\d{2}-\d{2}",           # YYYY-MM-DD...
        r"^\d{2}/\d{2}/\d{4}$",          # DD/MM/YYYY
    ]
    matches = sum(non_null.str.match(p).sum() for p in patterns)
    return matches / len(non_null) > 0.7


def infer_expected_type(series: pd.Series) -> str:
    """
    Guess the semantic type of a column from its values.

    Falls back gracefully so that a column with a few stray strings in
    a numeric field is still classified as numeric.
    """
    if _looks_like_date(series):
        return "date"
    if _is_mostly_numeric(series):
        non_null = pd.to_numeric(series.dropna(), errors="coerce").dropna()
        if (non_null % 1 == 0).all():
            return "integer"
        return "float"
    n_unique = series.dropna().nunique()
    if 0 < n_unique <= max(20, int(0.05 * len(series))):
        return "categorical"
    return "string"


def schema_agent(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run the Schema Validation agent on a DataFrame.

    Output structure:
        {
            "columns":          {col: {...per-column analysis...}},
            "duplicate_columns": [(col_a, col_b, similarity), ...],
            "summary": {
                "n_columns":       int,
                "n_naming_issues": int,
                "n_type_issues":   int,
                "n_duplicates":    int,
            }
        }
    """
    cols_info: dict[str, dict[str, Any]] = {}
    n_naming = 0
    n_types = 0

    for col in df.columns:
        diag = diagnose_column_name(col)
        if diag["severity"] != "ok":
            n_naming += 1

        inferred = infer_expected_type(df[col])
        actual = str(df[col].dtype)

        # Detect coercion failures: e.g. expected integer but contains free text.
        type_issues: list[str] = []
        bad_examples: list[str] = []
        if inferred in ("integer", "float"):
            parsed = pd.to_numeric(df[col], errors="coerce")
            mismatches = df[col].notna() & parsed.isna()
            if mismatches.any():
                n_types += 1
                bad = df.loc[mismatches, col].astype(str).head(3).tolist()
                type_issues.append(
                    f"expected {inferred} but {int(mismatches.sum())} value(s) are not numeric"
                )
                bad_examples.extend(bad)

        cols_info[col] = {
            "naming":        diag,
            "expected_type": inferred,
            "actual_dtype":  actual,
            "type_issues":   type_issues,
            "bad_examples":  bad_examples,
            "status":        "FAIL" if (diag["severity"] != "ok" or type_issues) else "OK",
        }

    duplicates = _detect_duplicate_columns(df)

    return {
        "columns": cols_info,
        "duplicate_columns": duplicates,
        "summary": {
            "n_columns":       len(df.columns),
            "n_naming_issues": n_naming,
            "n_type_issues":   n_types,
            "n_duplicates":    len(duplicates),
        },
    }


def _detect_duplicate_columns(df: pd.DataFrame) -> list[tuple[str, str, float]]:
    """
    Find pairs of columns whose non-null values are (nearly) identical.

    This catches the very common NoiPA pattern where the same logical
    field is exported under multiple names (e.g. `tipo_imposta` and
    `Tipo Imposta`, `cod_imposta` and `2cod_imposta`).

    Two similarity scores are computed:
      - raw    : exact string match after stripping
      - normal : case-insensitive match, ignoring placeholder values
    Pairs are reported when EITHER score reaches the threshold; the
    cleaner uses the normalized score so that "AQ" / "Aq" / "?" don't
    mask a duplicate. The reported similarity is the maximum of the two.
    """
    pairs: list[tuple[str, str, float]] = []
    cols = list(df.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            common = df[[a, b]].dropna()
            if len(common) < 5:  # not enough evidence
                continue

            sa = common[a].astype(str).str.strip()
            sb = common[b].astype(str).str.strip()
            sim_raw = (sa == sb).mean()

            # Normalized: same letter, ignore case + placeholder.
            sa_n = sa.str.upper()
            sb_n = sb.str.upper()
            ph = {x.upper() for x in PLACEHOLDER_VALUES}
            mask = ~sa_n.isin(ph) & ~sb_n.isin(ph)
            sim_norm = (sa_n[mask] == sb_n[mask]).mean() if mask.any() else 0.0

            best = max(float(sim_raw), float(sim_norm))
            if best >= 0.95:
                pairs.append((a, b, round(best, 3)))
    return pairs


# ---------------------------------------------------------------------------
# Completeness Agent
# ---------------------------------------------------------------------------

def completeness_agent(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute per-column and overall completeness statistics.

    A cell counts as "missing" if it is either a real null or one of the
    placeholder strings listed in PLACEHOLDER_VALUES. Columns that are
    more than 70% missing are flagged as "sparse" and become candidates
    for removal.
    """
    per_col: dict[str, dict[str, Any]] = {}
    n_rows = len(df)

    for col in df.columns:
        s = df[col]
        null_mask = s.isna()
        ph_mask = (
            ~null_mask
            & s.astype(str).str.strip().str.lower().isin(PLACEHOLDER_VALUES)
        )
        total_missing = int(null_mask.sum() + ph_mask.sum())

        completeness_pct = round(
            100.0 * (n_rows - total_missing) / n_rows, 2
        ) if n_rows else 100.0

        per_col[col] = {
            "null_count":         int(null_mask.sum()),
            "placeholder_count":  int(ph_mask.sum()),
            "placeholder_values": sorted(
                df.loc[ph_mask, col].astype(str).str.strip().unique().tolist()
            ),
            "total_missing":      total_missing,
            "completeness_pct":   completeness_pct,
            "is_sparse":          completeness_pct < 30.0,
            "missing_rows_sample": df.index[null_mask | ph_mask][:10].tolist(),
        }

    total_cells = n_rows * len(df.columns) if n_rows else 0
    total_missing = sum(c["total_missing"] for c in per_col.values())
    sparse_cols = [c for c, info in per_col.items() if info["is_sparse"]]

    overall_pct = (
        round(100.0 * (total_cells - total_missing) / total_cells, 2)
        if total_cells else 100.0
    )

    return {
        "per_column":              per_col,
        "overall_completeness_pct": overall_pct,
        "total_cells":             total_cells,
        "total_missing":           total_missing,
        "sparse_columns":          sparse_cols,
        "summary": {
            "overall_pct":      overall_pct,
            "n_sparse_columns": len(sparse_cols),
            "n_missing_cells":  total_missing,
        },
    }


# ---------------------------------------------------------------------------
# Consistency Agent
# ---------------------------------------------------------------------------

_FORMAT_PATTERNS = [
    ("YYYYMM",     re.compile(r"^\d{6}$")),
    ("YYYY-MM",    re.compile(r"^\d{4}-\d{2}$")),
    ("YYYY/MM",    re.compile(r"^\d{4}/\d{2}$")),
    ("ISO_DATE",   re.compile(r"^\d{4}-\d{2}-\d{2}")),
    ("DD/MM/YYYY", re.compile(r"^\d{2}/\d{2}/\d{4}$")),
    ("DECIMAL",    re.compile(r"^-?\d+\.\d+$")),
    ("INTEGER",    re.compile(r"^-?\d+$")),
]


def _classify_format(value: str) -> str:
    for label, regex in _FORMAT_PATTERNS:
        if regex.match(value):
            return label
    if value.lower() in PLACEHOLDER_VALUES:
        return "placeholder"
    return "text"


def check_format_consistency(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Flag columns where values follow more than one format simultaneously."""
    issues: list[dict[str, Any]] = []
    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip()
        if values.empty:
            continue
        counts = Counter(_classify_format(v) for v in values)
        # Ignore the placeholder "format" -- that's a completeness issue.
        counts.pop("placeholder", None)
        if len(counts) > 1:
            dominant, dominant_n = counts.most_common(1)[0]
            total = sum(counts.values())
            issues.append({
                "column":         col,
                "formats":        dict(counts),
                "dominant":       dominant,
                "dominant_pct":   round(100.0 * dominant_n / total, 1),
                "minority_count": total - dominant_n,
                "suggestion":     f"standardize '{col}' to '{dominant}' "
                                  f"({dominant_n}/{total} values)",
            })
    return issues


def check_cross_column_consistency(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Validate logical relationships between columns.

    Two complementary checks are run:
      1. Tipo_imposta x Imposta -- using rules MINED from the data itself.
         For each Tipo, the set of "valid" Imposta values is taken to
         be the values that appear in the dominant 95% of rows; the
         long-tail combinations are flagged. This avoids hard-coding
         a taxonomy that becomes wrong as soon as Reply ships a new
         dataset.
      2. Provincia validity -- against a static list of Italian
         province codes (this is a closed, well-known set).
    """
    violations: list[dict[str, Any]] = []

    # ---- 1. Tipo x Imposta: data-driven ---------------------------------
    tipo_col = _resolve_column(df, ["tipo_imposta", "Tipo Imposta"])
    imp_col  = _resolve_column(df, ["imposta", "Imposta"])
    if tipo_col and imp_col and tipo_col != imp_col:
        # Build the dominant-combination map.
        sub = df[[tipo_col, imp_col]].dropna().copy()
        sub[tipo_col] = sub[tipo_col].astype(str).str.strip()
        sub[imp_col]  = sub[imp_col].astype(str).str.strip()
        # Normalize tipo-side casing so "Erariali" / "ERARIALI" / "erariali"
        # share the same rule bucket.
        sub["__tipo_norm"] = sub[tipo_col].str.lower()

        # Mine rules only when we have enough evidence: each tipo bucket
        # must have at least MIN_BUCKET rows, otherwise we can't tell
        # "rare" from "normal" with statistical confidence.
        MIN_BUCKET = 30
        rule_set: dict[str, set[str]] = {}
        for tipo, group in sub.groupby("__tipo_norm"):
            if len(group) < MIN_BUCKET:
                continue
            counts = group[imp_col].value_counts()
            cumulative = counts.cumsum() / counts.sum()
            # Keep the smallest set of values that covers >=95% of the rows.
            dominant = counts[cumulative <= 0.95].index.tolist()
            if not dominant:
                dominant = [counts.index[0]]
            rule_set[tipo] = set(dominant)

        # On small datasets the data-driven map is too thin -- fall back
        # to the reference taxonomy. This keeps the agent useful even on
        # tiny test files (e.g. the 50-row toy datasets used during dev).
        if not rule_set:
            rule_set = {
                k.lower(): v for k, v in NOIPA_TAXONOMY_REFERENCE.items()
            }
            rule_source = "reference_taxonomy"
        else:
            rule_source = "data_driven"

        # Flag rows whose Imposta is outside the dominant set for its Tipo.
        for idx, row in sub.iterrows():
            t = row["__tipo_norm"]
            i = row[imp_col]
            if i.lower() in PLACEHOLDER_VALUES:
                continue
            allowed = rule_set.get(t)
            if allowed and i not in allowed:
                violations.append({
                    "type":     "cross_column",
                    "row":      int(idx),
                    "detail":   f"row {idx}: {tipo_col}='{row[tipo_col]}' "
                                f"but {imp_col}='{i}' (rare combination)",
                    "expected": sorted(allowed)[:5],
                })
                if len(violations) >= 100:  # cap to keep report readable
                    break

    # ---- 2. Provincia validity ------------------------------------------
    prov_col = _resolve_column(df, ["provincia_sede", "Provincia Sede", "provincia"])
    if prov_col:
        prov_values = (
            df[prov_col].dropna().astype(str).str.strip().str.upper()
        )
        bad_mask = ~prov_values.isin(ITALIAN_PROVINCES) & ~prov_values.isin(
            {x.upper() for x in PLACEHOLDER_VALUES}
        )
        for idx in prov_values.index[bad_mask][:50]:
            violations.append({
                "type":   "invalid_value",
                "row":    int(idx),
                "detail": f"row {idx}: {prov_col}='{df.at[idx, prov_col]}' "
                          f"is not a valid Italian province code",
            })

    return violations


def _resolve_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """
    Return the first candidate that matches a column in df, case- and
    whitespace-insensitive. Returns None if no candidate matches.
    """
    norm = {to_snake_case(c): c for c in df.columns}
    for cand in candidates:
        key = to_snake_case(cand)
        if key in norm:
            return norm[key]
    return None


def detect_duplicates(df: pd.DataFrame, sample_size: int = 500) -> dict[str, Any]:
    """
    Find exact and near-duplicate rows.

    Exact-duplicate detection is O(n log n) thanks to pandas' hash-based
    grouping. Near-duplicate detection is intrinsically O(n^2) so we
    cap the comparison to `sample_size` rows on big datasets.
    """
    # Exclude pure id columns from the comparison: two rows that differ
    # only in `_id` are still semantic duplicates.
    cmp_cols = [c for c in df.columns if c.lower() not in {"_id", "id"}]
    work = df[cmp_cols] if cmp_cols else df

    exact_mask = work.duplicated(keep=False)
    exact_groups: list[dict[str, Any]] = []
    if exact_mask.any():
        for _, grp in work[exact_mask].groupby(list(work.columns), dropna=False):
            if len(grp) > 1:
                exact_groups.append({
                    "rows":   grp.index.tolist(),
                    "n":      len(grp),
                    "sample": grp.iloc[0].to_dict(),
                })

    # Near-duplicates: pick a sample, look for pairs differing in <= 1 col.
    near: list[dict[str, Any]] = []
    sample = work.sample(min(sample_size, len(work)), random_state=42) if len(work) > sample_size else work
    sample_idx = sample.index.tolist()
    sample_str = sample.astype(str).values

    for i in range(len(sample_idx)):
        for j in range(i + 1, len(sample_idx)):
            diff = sample_str[i] != sample_str[j]
            n_diff = int(diff.sum())
            if n_diff == 1:
                col_diff = work.columns[diff.argmax()]
                near.append({
                    "row_a":   int(sample_idx[i]),
                    "row_b":   int(sample_idx[j]),
                    "column":  col_diff,
                    "value_a": str(sample_str[i][diff.argmax()]),
                    "value_b": str(sample_str[j][diff.argmax()]),
                })
                if len(near) >= 50:  # cap the report
                    break
        if len(near) >= 50:
            break

    return {
        "exact":          {"n_groups": len(exact_groups), "groups": exact_groups[:20]},
        "near":           {"n_pairs": len(near), "pairs": near[:20]},
        "sampling_note":  None if len(work) <= sample_size
                          else f"Near-duplicate scan limited to {sample_size} sampled rows.",
    }


def consistency_agent(df: pd.DataFrame) -> dict[str, Any]:
    """Run all three consistency checks and aggregate."""
    fmt = check_format_consistency(df)
    cross = check_cross_column_consistency(df)
    dups = detect_duplicates(df)

    return {
        "format_issues":          fmt,
        "cross_column_violations": cross,
        "duplicates":             dups,
        "summary": {
            "n_format_issues":   len(fmt),
            "n_cross_column":    len(cross),
            "n_exact_dup_groups": dups["exact"]["n_groups"],
            "n_near_dup_pairs":  dups["near"]["n_pairs"],
        },
    }


# ---------------------------------------------------------------------------
# Anomaly Agent
# ---------------------------------------------------------------------------

def detect_numerical_outliers(series: pd.Series, col_name: str) -> list[dict[str, Any]]:
    """
    Find numerical outliers using both Z-score and IQR.

    A point is flagged only if BOTH methods agree, which keeps the
    false-positive rate low on small datasets.
    """
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 5:
        return []

    mean, std = float(numeric.mean()), float(numeric.std(ddof=0))
    q1, q3 = float(numeric.quantile(0.25)), float(numeric.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    outliers: list[dict[str, Any]] = []
    for idx, value in numeric.items():
        z = abs(value - mean) / std if std > 0 else 0.0
        out_iqr = value < lower or value > upper
        if z > 3 and out_iqr:
            outliers.append({
                "row":     int(idx),
                "column":  col_name,
                "value":   float(value),
                "z_score": round(z, 2),
                "bounds":  {"iqr_low": round(lower, 2), "iqr_high": round(upper, 2)},
            })
    return outliers


def detect_rare_categories(
    series: pd.Series,
    col_name: str,
    min_freq_ratio: float = 0.005,
    min_dataset_size: int = 200,
) -> list[dict[str, Any]]:
    """
    Flag categorical values that appear less often than `min_freq_ratio`
    of the total. The threshold scales with dataset size so we don't
    drown the report in false positives on big datasets, and we skip
    detection entirely on very small datasets where rarity is not
    statistically meaningful.
    """
    values = series.dropna().astype(str).str.strip()
    if values.empty or len(values) < min_dataset_size:
        return []
    counts = values.value_counts()
    n = len(values)
    threshold = max(2, int(n * min_freq_ratio))

    rare: list[dict[str, Any]] = []
    for value, freq in counts.items():
        if freq < threshold and value.lower() not in PLACEHOLDER_VALUES:
            rare.append({
                "column":    col_name,
                "value":     value,
                "frequency": int(freq),
                "total":     int(n),
                "share_pct": round(100.0 * freq / n, 3),
            })
    return rare


def anomaly_agent(df: pd.DataFrame) -> dict[str, Any]:
    """Run univariate outlier detection + rare-category detection."""
    all_outliers: list[dict[str, Any]] = []
    all_rare: list[dict[str, Any]] = []
    column_stats: dict[str, dict[str, Any]] = {}

    for col in df.columns:
        col_lower = col.lower()
        # Identifiers are uninformative; period/date columns look numeric
        # but their "outliers" are just unusual months/years -- not what
        # the analyst means by anomaly. Skip them.
        if col_lower in {"_id", "id"}:
            continue
        if col_lower in {"rata", "anno", "mese", "year", "month", "date",
                         "data", "periodo", "aggregation-time", "aggregation_time"}:
            continue
        if _looks_like_date(df[col]):
            continue

        s = df[col]
        non_null = s.dropna()
        if non_null.empty:
            continue

        numeric_share = pd.to_numeric(non_null, errors="coerce").notna().mean()

        if numeric_share > 0.7:
            outs = detect_numerical_outliers(s, col)
            all_outliers.extend(outs)
            num = pd.to_numeric(non_null, errors="coerce").dropna()
            column_stats[col] = {
                "kind":          "numerical",
                "mean":          round(float(num.mean()), 2),
                "std":           round(float(num.std(ddof=0)), 2),
                "min":           round(float(num.min()), 2),
                "max":           round(float(num.max()), 2),
                "n_outliers":    len(outs),
            }
        else:
            rare = detect_rare_categories(s, col)
            all_rare.extend(rare)
            vc = non_null.astype(str).value_counts()
            column_stats[col] = {
                "kind":         "categorical",
                "n_unique":     int(vc.size),
                "most_common":  vc.index[0] if len(vc) else None,
                "n_rare":       len(rare),
            }

    return {
        "outliers":        all_outliers,
        "rare_categories": all_rare,
        "column_stats":    column_stats,
        "summary": {
            "n_outliers":         len(all_outliers),
            "n_rare_categories":  len(all_rare),
        },
    }


# ---------------------------------------------------------------------------
# Reliability Score
# ---------------------------------------------------------------------------

def compute_schema_score(schema_results: dict[str, Any]) -> float:
    """
    Schema score that distinguishes minor (cosmetic) from major issues.

    Naming-only problems are weighted lightly because the data is still
    usable -- the score should not collapse to zero just because the
    column headers are not snake_case (this was the main feedback at
    the mid-check).
    """
    cols = schema_results["columns"]
    if not cols:
        return 100.0

    # Each column starts at 100; deductions accumulate.
    per_col_scores = []
    for col, info in cols.items():
        score = 100.0
        sev = info["naming"]["severity"]
        if sev == "minor":
            score -= 5      # cosmetic only
        elif sev == "major":
            score -= 20     # spaces / special chars
        if info["type_issues"]:
            score -= 30     # genuine type mismatch is worse
        per_col_scores.append(max(score, 0.0))

    base = float(np.mean(per_col_scores))

    # Duplicate columns are a structural problem -> further penalty.
    n_dup = schema_results["summary"]["n_duplicates"]
    base -= min(20.0, 5.0 * n_dup)

    return round(max(base, 0.0), 2)


def compute_consistency_score(consistency_results: dict[str, Any], n_rows: int) -> float:
    """
    Consistency score. Each violation costs proportionally to its impact:
    format issues affect a whole column, cross-column issues affect a
    single row.
    """
    if not n_rows:
        return 100.0

    fmt = consistency_results["summary"]["n_format_issues"]
    cross = consistency_results["summary"]["n_cross_column"]
    dup_g = consistency_results["summary"]["n_exact_dup_groups"]
    dup_n = consistency_results["summary"]["n_near_dup_pairs"]

    # Format issues: -3 each (small dataset) up to -15 cap.
    score = 100.0
    score -= min(15.0, 3.0 * fmt)
    # Cross-column issues: proportional to their share of the dataset.
    score -= min(40.0, 100.0 * cross / n_rows)
    # Duplicates: each exact group costs 5, near-duplicates 1.
    score -= min(20.0, 5.0 * dup_g + 1.0 * dup_n)

    return round(max(score, 0.0), 2)


def compute_anomaly_score(anomaly_results: dict[str, Any], n_rows: int) -> float:
    """
    Anomaly score scales with dataset size.

    Numerical outliers (statistically defined) carry more weight than
    rare categories (which can simply reflect natural variety in small
    samples). Each anomaly is mapped to a penalty proportional to its
    share of the dataset, then capped.
    """
    if not n_rows:
        return 100.0

    n_out = anomaly_results["summary"]["n_outliers"]
    n_rare = anomaly_results["summary"]["n_rare_categories"]

    # Outliers: -2 each, capped at -50.
    outlier_penalty = min(50.0, 2.0 * n_out)
    # Rare categories: -0.5 each, capped at -20 (they're informational).
    rare_penalty = min(20.0, 0.5 * n_rare)

    return round(max(0.0, 100.0 - outlier_penalty - rare_penalty), 2)


def compute_reliability_score(state: dict[str, Any]) -> dict[str, float]:
    """Aggregate the four sub-scores into a weighted overall score."""
    n_rows = len(state["dataframe"])
    scores = {
        "completeness": state["completeness"]["overall_completeness_pct"],
        "schema":       compute_schema_score(state["schema"]),
        "consistency":  compute_consistency_score(state["consistency"], n_rows),
        "anomaly":      compute_anomaly_score(state["anomaly"], n_rows),
    }
    overall = sum(scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS)
    scores["overall"] = round(overall, 2)
    return scores


# ---------------------------------------------------------------------------
# Auto-Fix Agent
# ---------------------------------------------------------------------------

@dataclass
class FixLogEntry:
    """A single, reversible action taken by the cleaner."""
    action: str
    target: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "target": self.target, **self.details}


def autofix(
    df: pd.DataFrame,
    schema_results: dict[str, Any],
    completeness_results: dict[str, Any],
    consistency_results: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Apply only the *safe* fixes:
      - rename columns to snake_case
      - drop columns that are byte-for-byte duplicates of another column
      - replace placeholder strings with proper NaN
      - drop sparse columns (>70% missing)
      - strip and standardize whitespace inside string values
      - drop exact-duplicate rows
      - canonicalize known categorical values (e.g. trailing spaces)

    Returns the cleaned DataFrame and a structured fix log.
    """
    log: list[FixLogEntry] = []
    out = df.copy()

    # ---- 1. Drop near-identical duplicate columns ----------------------
    # Threshold tuning: 0.97 catches obvious clones (ente vs ente%code)
    # while leaving genuinely different columns alone (e.g. when one
    # column is a normalized version of the other and they disagree on
    # 5%+ of the rows). Pairs in the 0.95-0.97 band are reported by the
    # schema agent but not auto-dropped.
    duplicates = schema_results.get("duplicate_columns", [])
    dropped_cols: set[str] = set()
    for a, b, sim in duplicates:
        if sim < 0.97:
            continue
        # Decide which to keep using name + value quality.
        qa = _column_quality(a, out[a] if a in out.columns else None)
        qb = _column_quality(b, out[b] if b in out.columns else None)
        keep, drop = (a, b) if qa >= qb else (b, a)
        if drop in out.columns and drop not in dropped_cols:
            out = out.drop(columns=drop)
            dropped_cols.add(drop)
            log.append(FixLogEntry(
                action="drop_duplicate_column",
                target=drop,
                details={"kept": keep, "similarity": sim},
            ))

    # ---- 2. Drop sparse columns ----------------------------------------
    for col in completeness_results.get("sparse_columns", []):
        if col in out.columns:
            out = out.drop(columns=col)
            log.append(FixLogEntry(
                action="drop_sparse_column",
                target=col,
                details={"completeness_pct":
                         completeness_results["per_column"][col]["completeness_pct"]},
            ))

    # ---- 3. Rename remaining columns to snake_case ---------------------
    rename_map: dict[str, str] = {}
    used: set[str] = set()
    for col in out.columns:
        new = to_snake_case(col)
        # Avoid collisions after normalization.
        candidate = new
        suffix = 2
        while candidate in used:
            candidate = f"{new}_{suffix}"
            suffix += 1
        used.add(candidate)
        if candidate != col:
            rename_map[col] = candidate
    if rename_map:
        out = out.rename(columns=rename_map)
        log.append(FixLogEntry(
            action="rename_columns",
            target="schema",
            details={"renames": rename_map},
        ))

    # ---- 4. Strip whitespace and replace placeholders with NaN ---------
    for col in out.columns:
        if not pd.api.types.is_string_dtype(out[col]):
            continue
        out[col] = out[col].astype(str).str.strip()
        ph_mask = out[col].str.lower().isin(PLACEHOLDER_VALUES)
        n_ph = int(ph_mask.sum())
        if n_ph:
            out.loc[ph_mask, col] = np.nan
            log.append(FixLogEntry(
                action="replace_placeholders_with_nan",
                target=col,
                details={"n_replaced": n_ph},
            ))

    # ---- 5. Canonicalize categorical typos (case + trailing space) -----
    for col in out.columns:
        if not pd.api.types.is_string_dtype(out[col]):
            continue
        non_null = out[col].dropna()
        if non_null.empty:
            continue
        # Group values by their lowercase-stripped form and pick the
        # most frequent representation as canonical.
        norm = non_null.astype(str).str.strip().str.lower()
        groups = defaultdict(list)
        for raw, k in zip(non_null, norm):
            groups[k].append(raw)
        canonical_map: dict[str, str] = {}
        for k, values in groups.items():
            if len(set(values)) > 1:
                most_common = Counter(values).most_common(1)[0][0]
                for v in set(values):
                    if v != most_common:
                        canonical_map[v] = most_common
        if canonical_map:
            out[col] = out[col].replace(canonical_map)
            log.append(FixLogEntry(
                action="canonicalize_categorical",
                target=col,
                details={"map": canonical_map},
            ))

    # ---- 6. Standardize date / period formats --------------------------
    for issue in consistency_results.get("format_issues", []):
        col = issue["column"]
        if col not in out.columns:
            # Could have been renamed; map through rename_map.
            col = rename_map.get(col, col)
            if col not in out.columns:
                continue
        dominant = issue["dominant"]
        if dominant in {"YYYYMM", "YYYY-MM", "YYYY/MM"}:
            new_values, n_changed = _normalize_period(out[col], dominant)
            if n_changed:
                out[col] = new_values
                log.append(FixLogEntry(
                    action="standardize_format",
                    target=col,
                    details={"to_format": dominant, "n_changed": n_changed},
                ))

    # ---- 7. Drop exact-duplicate rows ----------------------------------
    before = len(out)
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) < before:
        log.append(FixLogEntry(
            action="drop_duplicate_rows",
            target="rows",
            details={"n_dropped": before - len(out)},
        ))

    return out, [e.to_dict() for e in log]


def _column_quality(name: str, series: pd.Series | None = None) -> float:
    """
    Combined name + value quality score.

    Higher = better candidate to keep when two columns are duplicates.
    Name quality alone would always prefer 'provincia_sede' over
    'Provincia Sede' even when the latter holds clean canonical values
    -- so we add a small bonus for value cleanliness (no placeholders,
    consistent casing).
    """
    score = float(_name_quality(name))

    if series is not None and len(series) > 0:
        non_null = series.dropna().astype(str).str.strip()
        if len(non_null) > 0:
            ph = non_null.str.lower().isin(PLACEHOLDER_VALUES).mean()
            score -= 5.0 * ph        # heavy penalty for placeholder-rich
            # bonus for consistent casing
            consistent = (
                (non_null.str.upper() == non_null).mean()
                + (non_null.str.lower() == non_null).mean()
            )
            score += 2.0 * consistent
    return score


def _name_quality(name: str) -> int:
    """Pure-name quality (independent of values). Used as fallback."""
    score = 0
    if " " not in name:               score += 2
    if not re.search(r"[^A-Za-z0-9_]", name): score += 2
    if not name[:1].isdigit():        score += 1
    if name == name.lower():          score += 1
    return score


def _normalize_period(series: pd.Series, target: str) -> tuple[pd.Series, int]:
    """Convert period-like strings to a single target format. Returns (new_series, n_changed)."""
    out = series.copy()
    n = 0
    for idx, raw in series.items():
        if pd.isna(raw):
            continue
        v = str(raw).strip()
        # Try to extract year and month regardless of input format.
        m = re.match(r"^(\d{4})[-/](\d{2})$", v) or re.match(r"^(\d{4})(\d{2})$", v)
        if not m:
            continue
        year, month = m.group(1), m.group(2)
        if target == "YYYYMM":
            new = f"{year}{month}"
        elif target == "YYYY-MM":
            new = f"{year}-{month}"
        elif target == "YYYY/MM":
            new = f"{year}/{month}"
        else:
            continue
        if new != v:
            out.at[idx] = new
            n += 1
    return out, n


# ---------------------------------------------------------------------------
# Public pipeline runner (no LangGraph dependency)
# ---------------------------------------------------------------------------

def run_pipeline_local(df: pd.DataFrame) -> dict[str, Any]:
    """
    Run the full pipeline without any LLM call. Useful for unit tests
    and for the Streamlit UI when the user has no API key.
    """
    state: dict[str, Any] = {"dataframe": df}
    state["schema"]       = schema_agent(df)
    state["completeness"] = completeness_agent(df)
    state["consistency"]  = consistency_agent(df)
    state["anomaly"]      = anomaly_agent(df)
    state["scores"]       = compute_reliability_score(state)
    cleaned, fix_log      = autofix(df, state["schema"], state["completeness"], state["consistency"])
    state["cleaned_df"]   = cleaned
    state["fix_log"]      = fix_log
    return state


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def format_report(state: dict[str, Any]) -> str:
    """Return a human-readable text report from a pipeline state."""
    n_rows = len(state["dataframe"])
    n_cols = len(state["dataframe"].columns)
    sch    = state["schema"]
    comp   = state["completeness"]
    cons   = state["consistency"]
    anom   = state["anomaly"]
    scores = state["scores"]

    lines = [
        "=" * 70,
        "DATA QUALITY REPORT",
        "=" * 70,
        f"Dataset: {n_rows} rows x {n_cols} columns",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"--- Reliability Score: {scores['overall']:.1f}/100 ---",
        f"  Completeness ({SCORE_WEIGHTS['completeness']:.0%}): {scores['completeness']:.1f}",
        f"  Schema       ({SCORE_WEIGHTS['schema']:.0%}): {scores['schema']:.1f}",
        f"  Consistency  ({SCORE_WEIGHTS['consistency']:.0%}): {scores['consistency']:.1f}",
        f"  Anomaly      ({SCORE_WEIGHTS['anomaly']:.0%}): {scores['anomaly']:.1f}",
        "",
        "--- Schema Validation ---",
        f"  Naming issues:   {sch['summary']['n_naming_issues']}/{sch['summary']['n_columns']} columns",
        f"  Type issues:     {sch['summary']['n_type_issues']} columns",
        f"  Duplicate cols:  {sch['summary']['n_duplicates']} pair(s)",
    ]
    for a, b, sim in sch["duplicate_columns"][:5]:
        lines.append(f"     '{a}' ~ '{b}'  (similarity {sim:.0%})")

    lines += [
        "",
        "--- Completeness ---",
        f"  Overall:        {comp['overall_completeness_pct']:.2f}%",
        f"  Missing cells:  {comp['total_missing']}/{comp['total_cells']}",
        f"  Sparse columns: {comp['sparse_columns'] or 'none'}",
        "",
        "--- Consistency ---",
        f"  Format issues:        {cons['summary']['n_format_issues']}",
        f"  Cross-column issues:  {cons['summary']['n_cross_column']}",
        f"  Exact dup groups:     {cons['summary']['n_exact_dup_groups']}",
        f"  Near-dup pairs:       {cons['summary']['n_near_dup_pairs']}",
        "",
        "--- Anomalies ---",
        f"  Numerical outliers:   {anom['summary']['n_outliers']}",
        f"  Rare categories:      {anom['summary']['n_rare_categories']}",
        "",
        "--- Auto-Fix Actions ---",
        f"  {len(state.get('fix_log', []))} fix(es) applied",
    ]
    for entry in state.get("fix_log", [])[:10]:
        lines.append(f"  - {entry['action']}: {entry['target']}")

    lines.append("=" * 70)
    return "\n".join(lines)
