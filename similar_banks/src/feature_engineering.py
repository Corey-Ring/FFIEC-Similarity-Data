from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class FeatureStatus:
    feature_name: str
    status: str
    source_columns: List[str]
    reason: str
    used_in_model: bool = False


@dataclass
class FeatureSet:
    frame: pd.DataFrame
    numeric_features: List[str]
    categorical_features: List[str]
    missing_critical: pd.DataFrame
    data_gaps: List[FeatureStatus]
    feature_status: List[FeatureStatus]


def _first_existing_column(df: pd.DataFrame, candidates: Any) -> Optional[str]:
    if isinstance(candidates, str):
        if candidates in df.columns:
            return candidates
        for suffix in ("_land", "_inst"):
            candidate = f"{candidates}{suffix}"
            if candidate in df.columns:
                return candidate
        return None
    if isinstance(candidates, list):
        for col in candidates:
            resolved = _first_existing_column(df, col)
            if resolved is not None:
                return resolved
    return None


def _candidate_columns(candidates: Any) -> List[str]:
    if isinstance(candidates, str):
        return [candidates]
    if isinstance(candidates, Sequence):
        return [str(col) for col in candidates]
    return []


def _dedupe_columns(columns: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(str(col) for col in columns if col))


def _upsert_feature_status(
    status_map: Dict[str, FeatureStatus],
    feature_name: str,
    status: str,
    source_columns: Sequence[str],
    reason: str,
    used_in_model: Optional[bool] = None,
) -> None:
    existing = status_map.get(feature_name)
    if existing is not None and used_in_model is None:
        used_in_model = existing.used_in_model
    status_map[feature_name] = FeatureStatus(
        feature_name=feature_name,
        status=status,
        source_columns=_dedupe_columns(
            list(source_columns) or (existing.source_columns if existing is not None else [])
        ),
        reason=reason,
        used_in_model=bool(used_in_model),
    )


def _mark_feature_usage(status_map: Dict[str, FeatureStatus], feature_names: Sequence[str]) -> None:
    for feature_name in feature_names:
        existing = status_map.get(feature_name)
        if existing is None:
            status_map[feature_name] = FeatureStatus(
                feature_name=feature_name,
                status="active",
                source_columns=[],
                reason="Feature is available for scoring or similarity drivers.",
                used_in_model=True,
            )
            continue
        status_map[feature_name] = FeatureStatus(
            feature_name=existing.feature_name,
            status=existing.status,
            source_columns=existing.source_columns,
            reason=existing.reason,
            used_in_model=True,
        )


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_mixed_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    yyyymmdd_mask = numeric.notna() & numeric.between(19000101, 21991231)
    if yyyymmdd_mask.any():
        as_text = numeric[yyyymmdd_mask].round().astype("Int64").astype(str)
        parsed.loc[yyyymmdd_mask] = pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")
    return parsed


def _resolve_column(
    df: pd.DataFrame, column: Optional[str], prefer_suffixes: tuple[str, ...] = ()
) -> Optional[str]:
    if not column:
        return None
    if column in df.columns:
        return column
    for suffix in prefer_suffixes:
        candidate = f"{column}_{suffix}"
        if candidate in df.columns:
            return candidate
    return None


def _series_or_default(
    df: pd.DataFrame,
    column: Optional[str],
    default: Any,
    prefer_suffixes: tuple[str, ...] = (),
) -> pd.Series:
    resolved = _resolve_column(df, column, prefer_suffixes=prefer_suffixes)
    if resolved:
        return df[resolved]
    return pd.Series(default, index=df.index)


def compute_growth_features(
    land_history: pd.DataFrame, mappings: Dict[str, Any], logger: logging.Logger
) -> pd.DataFrame:
    ids = mappings["identifiers"]
    dates = mappings["dates"]
    financial = mappings["financial"]

    bank_id_col = ids["bank_id"]
    report_date_col = dates["land_report_date"]
    growth_metrics = {
        "assets": financial["total_assets"],
        "deposits": financial["total_deposits"],
        "loans": financial["total_loans"],
    }

    needed_cols = [bank_id_col, report_date_col] + list(growth_metrics.values())
    hist = land_history[needed_cols].copy()
    hist[report_date_col] = _parse_mixed_date(hist[report_date_col])
    hist[bank_id_col] = pd.to_numeric(hist[bank_id_col], errors="coerce").astype("Int64")

    for col in growth_metrics.values():
        hist[col] = _to_numeric(hist[col])

    hist = hist.dropna(subset=[bank_id_col, report_date_col]).sort_values(
        [bank_id_col, report_date_col]
    )
    for label, source_col in growth_metrics.items():
        lag = hist.groupby(bank_id_col)[source_col].shift(4)
        hist[f"yoy_{label}"] = np.where(
            (lag > 0) & hist[source_col].notna(),
            (hist[source_col] / lag) - 1.0,
            np.nan,
        )

    latest_date = hist[report_date_col].max()
    if pd.isna(latest_date):
        logger.warning("Growth feature computation skipped because land report date is missing.")
        return pd.DataFrame(columns=[bank_id_col])

    window_start = latest_date - pd.DateOffset(years=3)
    rolling_window = hist.loc[
        (hist[report_date_col] > window_start) & (hist[report_date_col] <= latest_date)
    ]
    growth_cols = ["yoy_assets", "yoy_deposits", "yoy_loans"]
    growth_agg = rolling_window.groupby(bank_id_col, as_index=False)[growth_cols].mean()
    growth_agg = growth_agg.rename(
        columns={
            "yoy_assets": "avg_3y_assets_yoy_growth",
            "yoy_deposits": "avg_3y_deposits_yoy_growth",
            "yoy_loans": "avg_3y_loans_yoy_growth",
        }
    )
    return growth_agg


def validate_feature_contract(
    feature_set: FeatureSet,
    numeric_feature_weights: Dict[str, float],
    driver_features: Sequence[str],
    logger: logging.Logger,
) -> FeatureSet:
    status_map = {status.feature_name: status for status in feature_set.feature_status}

    weighted_numeric_features: List[str] = []
    unweighted_numeric_features: List[str] = []
    for feature_name in feature_set.numeric_features:
        existing = status_map.get(feature_name)
        if feature_name in numeric_feature_weights:
            weighted_numeric_features.append(feature_name)
            continue

        unweighted_numeric_features.append(feature_name)
        if existing is not None:
            reason = existing.reason
            if "Not enabled in numeric_feature_weights." not in reason:
                reason = f"{reason} Not enabled in numeric_feature_weights."
            _upsert_feature_status(
                status_map,
                feature_name=feature_name,
                status=existing.status,
                source_columns=existing.source_columns,
                reason=reason,
                used_in_model=False,
            )

    missing_weighted_features = [
        feature_name
        for feature_name in numeric_feature_weights
        if feature_name not in feature_set.numeric_features
    ]
    for feature_name in missing_weighted_features:
        existing = status_map.get(feature_name)
        reason = "Configured numeric weight present but feature is unavailable."
        if existing is not None and existing.reason != reason:
            reason = f"{existing.reason} {reason}"
        _upsert_feature_status(
            status_map,
            feature_name=feature_name,
            status="dropped",
            source_columns=existing.source_columns if existing is not None else [],
            reason=reason,
            used_in_model=False,
        )

    available_features = {
        feature_name
        for feature_name, status in status_map.items()
        if status.status != "dropped"
    }
    missing_driver_features = [
        feature_name for feature_name in driver_features if feature_name not in available_features
    ]
    for feature_name in missing_driver_features:
        existing = status_map.get(feature_name)
        reason = "Feature required for similarity drivers is unavailable."
        if existing is not None and existing.reason != reason:
            reason = f"{existing.reason} {reason}"
        _upsert_feature_status(
            status_map,
            feature_name=feature_name,
            status="dropped",
            source_columns=existing.source_columns if existing is not None else [],
            reason=reason,
            used_in_model=False,
        )

    model_features = (
        list(weighted_numeric_features)
        + list(feature_set.categorical_features)
        + [feature_name for feature_name in driver_features if feature_name in available_features]
    )
    _mark_feature_usage(status_map, model_features)

    if unweighted_numeric_features:
        logger.warning(
            "Dropping %d engineered numeric feature(s) without configured weights: %s",
            len(unweighted_numeric_features),
            ", ".join(unweighted_numeric_features),
        )
    if missing_weighted_features:
        logger.warning(
            "Configured numeric feature(s) unavailable and dropped from scoring: %s",
            ", ".join(missing_weighted_features),
        )
    if missing_driver_features:
        logger.warning(
            "Similarity driver feature(s) unavailable: %s",
            ", ".join(missing_driver_features),
        )

    feature_status = sorted(status_map.values(), key=lambda status: status.feature_name)
    data_gaps = [status for status in feature_status if status.status == "dropped"]
    return FeatureSet(
        frame=feature_set.frame,
        numeric_features=weighted_numeric_features,
        categorical_features=feature_set.categorical_features,
        missing_critical=feature_set.missing_critical,
        data_gaps=data_gaps,
        feature_status=feature_status,
    )


def engineer_bank_features(
    bank_base: pd.DataFrame,
    land_history: pd.DataFrame,
    mappings: Dict[str, Any],
    logger: logging.Logger,
) -> FeatureSet:
    ids = mappings["identifiers"]
    names = mappings["names"]
    financial = mappings["financial"]
    lending = mappings["lending_profile"]
    deposit_structure = mappings["deposit_structure"]
    institutional = mappings["institutional"]

    bank_id_col = ids["bank_id"]
    cert_col = ids["cert"]

    land_name_col = names["land_name"]
    institution_name_col = names["institution_name"]

    status_map: Dict[str, FeatureStatus] = {}

    out = pd.DataFrame()
    out["bank_id"] = pd.to_numeric(bank_base[bank_id_col], errors="coerce").astype("Int64")
    institution_name = _series_or_default(
        bank_base, institution_name_col, np.nan, prefer_suffixes=("inst", "land")
    )
    land_name = _series_or_default(
        bank_base, land_name_col, np.nan, prefer_suffixes=("land", "inst")
    )
    out["subject_name"] = (
        institution_name.where(institution_name.notna(), land_name).fillna("UNKNOWN").astype(str)
    )
    out["cert"] = pd.to_numeric(
        _series_or_default(bank_base, cert_col, np.nan, prefer_suffixes=("inst", "land")),
        errors="coerce",
    ).astype("Int64")

    total_assets_col = financial["total_assets"]
    total_deposits_col = financial["total_deposits"]
    total_loans_col = financial["total_loans"]
    loan_to_deposit_col = financial["loan_to_deposit_ratio"]
    asset_growth_cagr_col = financial["asset_growth_cagr_5y"]
    asset_size_bucket_col = financial["asset_size_bucket"]

    out["total_assets"] = _to_numeric(
        _series_or_default(bank_base, total_assets_col, np.nan, prefer_suffixes=("land", "inst"))
    )
    out["total_deposits"] = _to_numeric(
        _series_or_default(bank_base, total_deposits_col, np.nan, prefer_suffixes=("land", "inst"))
    )
    out["total_loans"] = _to_numeric(
        _series_or_default(bank_base, total_loans_col, np.nan, prefer_suffixes=("land", "inst"))
    )

    for feature_name, series, source_column in [
        ("total_assets", out["total_assets"], total_assets_col),
        ("total_deposits", out["total_deposits"], total_deposits_col),
        ("total_loans", out["total_loans"], total_loans_col),
    ]:
        if series.notna().any():
            _upsert_feature_status(
                status_map,
                feature_name=feature_name,
                status="active",
                source_columns=[source_column],
                reason="Loaded from latest-quarter financial source data.",
            )
        else:
            _upsert_feature_status(
                status_map,
                feature_name=feature_name,
                status="dropped",
                source_columns=[source_column],
                reason="Source column is missing or empty in the latest-quarter snapshot.",
            )

    resolved_loan_to_deposit_col = _resolve_column(
        bank_base, loan_to_deposit_col, prefer_suffixes=("land", "inst")
    )
    loan_to_deposit_series = _to_numeric(
        _series_or_default(bank_base, loan_to_deposit_col, np.nan, prefer_suffixes=("land", "inst"))
    )
    ratio_p95 = (
        float(loan_to_deposit_series.quantile(0.95))
        if loan_to_deposit_series.notna().any()
        else np.nan
    )
    if pd.notna(ratio_p95) and ratio_p95 > 2.0:
        loan_to_deposit_series = loan_to_deposit_series / 100.0
    derived_ratio_series = pd.Series(
        np.where(
            out["total_deposits"] > 0,
            out["total_loans"] / out["total_deposits"],
            np.nan,
        ),
        index=bank_base.index,
    )
    out["loan_to_deposit_ratio"] = loan_to_deposit_series.fillna(derived_ratio_series)
    if resolved_loan_to_deposit_col and loan_to_deposit_series.notna().any():
        _upsert_feature_status(
            status_map,
            feature_name="loan_to_deposit_ratio",
            status="active",
            source_columns=[resolved_loan_to_deposit_col],
            reason="Loaded from the source loan-to-deposit ratio column.",
        )
    elif derived_ratio_series.notna().any():
        _upsert_feature_status(
            status_map,
            feature_name="loan_to_deposit_ratio",
            status="derived",
            source_columns=[total_loans_col, total_deposits_col],
            reason="Derived from total_loans / total_deposits because no usable source ratio column was available.",
        )
    else:
        _upsert_feature_status(
            status_map,
            feature_name="loan_to_deposit_ratio",
            status="dropped",
            source_columns=[loan_to_deposit_col, total_loans_col, total_deposits_col],
            reason="No usable source or derived loan-to-deposit ratio could be computed.",
        )

    resolved_asset_growth_col = _resolve_column(
        bank_base, asset_growth_cagr_col, prefer_suffixes=("land", "inst")
    )
    out["asset_growth_cagr_5y"] = _to_numeric(
        _series_or_default(bank_base, asset_growth_cagr_col, np.nan, prefer_suffixes=("land", "inst"))
    )
    if resolved_asset_growth_col and out["asset_growth_cagr_5y"].notna().any():
        _upsert_feature_status(
            status_map,
            feature_name="asset_growth_cagr_5y",
            status="active",
            source_columns=[resolved_asset_growth_col],
            reason="Loaded from the configured 5-year asset growth source column.",
        )
    else:
        _upsert_feature_status(
            status_map,
            feature_name="asset_growth_cagr_5y",
            status="dropped",
            source_columns=[asset_growth_cagr_col],
            reason="Configured 5-year asset growth column is unavailable in the current dataset.",
        )

    resolved_bucket_col = _resolve_column(
        bank_base, asset_size_bucket_col, prefer_suffixes=("land", "inst")
    )
    if resolved_bucket_col in bank_base.columns:
        out["asset_size_bucket"] = bank_base[resolved_bucket_col].fillna("UNKNOWN").astype(str)
    else:
        bins = [-np.inf, 100_000_000, 1_000_000_000, 10_000_000_000, 100_000_000_000, np.inf]
        labels = ["<=100M", "100M-1B", "1B-10B", "10B-100B", "100B+"]
        out["asset_size_bucket"] = pd.cut(out["total_assets"], bins=bins, labels=labels).astype(str)
        out["asset_size_bucket"] = out["asset_size_bucket"].replace("nan", "UNKNOWN")

    out["log_total_assets"] = np.log1p(out["total_assets"].clip(lower=0))
    out["log_total_deposits"] = np.log1p(out["total_deposits"].clip(lower=0))
    out["log_total_loans"] = np.log1p(out["total_loans"].clip(lower=0))
    out["loans_to_assets"] = np.where(
        out["total_assets"] > 0, out["total_loans"] / out["total_assets"], np.nan
    )
    out["deposits_to_assets"] = np.where(
        out["total_assets"] > 0, out["total_deposits"] / out["total_assets"], np.nan
    )

    for feature_name, source_columns, series, reason in [
        (
            "log_total_assets",
            [total_assets_col],
            out["log_total_assets"],
            "Derived with log1p(total_assets).",
        ),
        (
            "log_total_deposits",
            [total_deposits_col],
            out["log_total_deposits"],
            "Derived with log1p(total_deposits).",
        ),
        (
            "log_total_loans",
            [total_loans_col],
            out["log_total_loans"],
            "Derived with log1p(total_loans).",
        ),
        (
            "loans_to_assets",
            [total_loans_col, total_assets_col],
            pd.Series(out["loans_to_assets"], index=out.index),
            "Derived from total_loans / total_assets.",
        ),
        (
            "deposits_to_assets",
            [total_deposits_col, total_assets_col],
            pd.Series(out["deposits_to_assets"], index=out.index),
            "Derived from total_deposits / total_assets.",
        ),
    ]:
        if pd.Series(series).notna().any():
            _upsert_feature_status(
                status_map,
                feature_name=feature_name,
                status="derived",
                source_columns=source_columns,
                reason=reason,
            )
        else:
            _upsert_feature_status(
                status_map,
                feature_name=feature_name,
                status="dropped",
                source_columns=source_columns,
                reason=f"{reason} Required source values were unavailable.",
            )

    for label, candidate_cols in lending.items():
        feature_name = f"loan_mix_{label}_pct"
        source_candidates = _candidate_columns(candidate_cols)
        source_col = _first_existing_column(bank_base, candidate_cols)
        if source_col is None:
            _upsert_feature_status(
                status_map,
                feature_name=feature_name,
                status="dropped",
                source_columns=source_candidates,
                reason="Configured lending source columns were not found in the dataset.",
            )
            out[feature_name] = np.nan
            continue

        source = _to_numeric(bank_base[source_col])
        source_lower = source_col.lower()
        if "pct" in source_lower or "percent" in source_lower:
            source_p95 = float(source.quantile(0.95)) if source.notna().any() else np.nan
            if pd.notna(source_p95) and source_p95 > 1.5:
                source = source / 100.0
            out[feature_name] = source
            status = "active" if source.notna().any() else "dropped"
            reason = "Loaded from source lending percentage column."
        else:
            out[feature_name] = np.where(out["total_loans"] > 0, source / out["total_loans"], np.nan)
            status = "derived" if pd.Series(out[feature_name], index=out.index).notna().any() else "dropped"
            reason = "Derived from lending amount / total_loans."

        _upsert_feature_status(
            status_map,
            feature_name=feature_name,
            status=status,
            source_columns=source_candidates,
            reason=reason if status != "dropped" else f"{reason} Required values were unavailable.",
        )

    for label, candidate_cols in deposit_structure.items():
        feature_name = f"{label}_pct"
        source_candidates = _candidate_columns(candidate_cols)
        source_col = _first_existing_column(bank_base, candidate_cols)
        if source_col is None:
            _upsert_feature_status(
                status_map,
                feature_name=feature_name,
                status="dropped",
                source_columns=source_candidates + [total_deposits_col],
                reason="Configured deposit-structure source columns were not found in the dataset.",
            )
            out[feature_name] = np.nan
            continue

        source = _to_numeric(bank_base[source_col])
        source_lower = source_col.lower()
        if "pct" in source_lower or "percent" in source_lower:
            source_p95 = float(source.quantile(0.95)) if source.notna().any() else np.nan
            if pd.notna(source_p95) and source_p95 > 1.5:
                source = source / 100.0
            out[feature_name] = source
            status = "active" if source.notna().any() else "dropped"
            reason = "Loaded from source deposit-structure percentage column."
        else:
            out[feature_name] = np.where(
                out["total_deposits"] > 0, source / out["total_deposits"], np.nan
            )
            status = "derived" if pd.Series(out[feature_name], index=out.index).notna().any() else "dropped"
            reason = "Derived from deposit amount / total_deposits."

        _upsert_feature_status(
            status_map,
            feature_name=feature_name,
            status=status,
            source_columns=source_candidates + [total_deposits_col],
            reason=reason if status != "dropped" else f"{reason} Required values were unavailable.",
        )

    charter_col = institutional["charter_type"]
    holding_col = institutional["holding_company_rssd"]
    specialty_col = institutional["specialty_group"]

    out["charter_type"] = _series_or_default(
        bank_base, charter_col, "UNKNOWN", prefer_suffixes=("inst", "land")
    ).fillna("UNKNOWN").astype(str)
    out["holding_company_rssd"] = pd.to_numeric(
        _series_or_default(bank_base, holding_col, np.nan, prefer_suffixes=("inst", "land")),
        errors="coerce",
    )
    out["has_holding_company"] = (
        out["holding_company_rssd"].notna() & (out["holding_company_rssd"] > 0)
    ).astype(int)
    out["specialty_group"] = _series_or_default(
        bank_base, specialty_col, "UNKNOWN", prefer_suffixes=("inst", "land")
    ).fillna("UNKNOWN").astype(str)

    _upsert_feature_status(
        status_map,
        feature_name="charter_type",
        status="active",
        source_columns=[charter_col],
        reason="Loaded from institution metadata.",
    )
    _upsert_feature_status(
        status_map,
        feature_name="has_holding_company",
        status="derived",
        source_columns=[holding_col],
        reason="Derived from holding_company_rssd > 0.",
    )
    _upsert_feature_status(
        status_map,
        feature_name="specialty_group",
        status="active",
        source_columns=[specialty_col],
        reason="Loaded from institution metadata.",
    )

    growth = compute_growth_features(land_history=land_history, mappings=mappings, logger=logger)
    growth = growth.rename(columns={ids["bank_id"]: "bank_id"})
    out = out.merge(growth, how="left", on="bank_id")

    for feature_name in [
        "avg_3y_assets_yoy_growth",
        "avg_3y_deposits_yoy_growth",
        "avg_3y_loans_yoy_growth",
    ]:
        if out[feature_name].notna().any():
            _upsert_feature_status(
                status_map,
                feature_name=feature_name,
                status="derived",
                source_columns=[financial["total_assets"], financial["total_deposits"], financial["total_loans"], mappings["dates"]["land_report_date"]],
                reason="Derived from 3-year trailing average YoY growth on quarterly history.",
            )
        else:
            _upsert_feature_status(
                status_map,
                feature_name=feature_name,
                status="dropped",
                source_columns=[financial["total_assets"], financial["total_deposits"], financial["total_loans"], mappings["dates"]["land_report_date"]],
                reason="Historical data was insufficient to compute a 3-year average YoY growth feature.",
            )

    critical_raw = ["total_assets", "total_deposits", "total_loans"]
    missing_critical = out.loc[
        out[critical_raw].isna().any(axis=1),
        ["bank_id", "subject_name"] + critical_raw,
    ]

    candidate_numeric = [
        "log_total_assets",
        "log_total_deposits",
        "log_total_loans",
        "loans_to_assets",
        "deposits_to_assets",
        "loan_to_deposit_ratio",
        "asset_growth_cagr_5y",
        "avg_3y_assets_yoy_growth",
        "avg_3y_deposits_yoy_growth",
        "avg_3y_loans_yoy_growth",
        "core_deposits_pct",
        "non_interest_deposits_pct",
    ] + [c for c in out.columns if c.startswith("loan_mix_")]

    numeric_features = [
        col for col in candidate_numeric if col in out.columns and not out[col].isna().all()
    ]
    for col in numeric_features:
        if out[col].isna().any():
            group_med = out.groupby("asset_size_bucket")[col].transform("median")
            out[col] = out[col].fillna(group_med)
            out[col] = out[col].fillna(out[col].median())

    out = out.dropna(subset=["bank_id"]).drop_duplicates(subset=["bank_id"], keep="last").copy()
    out["bank_id"] = out["bank_id"].astype("int64")
    out["cert"] = out["cert"].astype("Int64")

    categorical_features = ["charter_type", "has_holding_company", "specialty_group"]
    feature_status = sorted(status_map.values(), key=lambda status: status.feature_name)
    data_gaps = [status for status in feature_status if status.status == "dropped"]

    logger.info(
        "Engineered %d numeric features and %d categorical features",
        len(numeric_features),
        len(categorical_features),
    )
    if data_gaps:
        logger.warning(
            "Detected %d dropped or unavailable feature(s): %s",
            len(data_gaps),
            "; ".join(f"{gap.feature_name}: {gap.reason}" for gap in data_gaps),
        )

    return FeatureSet(
        frame=out,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        missing_critical=missing_critical,
        data_gaps=data_gaps,
        feature_status=feature_status,
    )
