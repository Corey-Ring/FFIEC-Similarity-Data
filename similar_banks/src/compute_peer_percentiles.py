from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    slug: str
    label: str
    direction: str
    required_cols: tuple[str, ...]
    compute: Callable[[pd.DataFrame], pd.Series]
    full_dollar: bool = False


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    numer_num = pd.to_numeric(numer, errors="coerce")
    denom_num = pd.to_numeric(denom, errors="coerce")
    return pd.Series(
        np.where(denom_num > 0, numer_num / denom_num, np.nan),
        index=numer.index,
    )


def _series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def _normalize_percent(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not values.notna().any():
        return values
    p95 = float(values.quantile(0.95))
    if pd.notna(p95) and p95 > 2.0:
        return values / 100.0
    return values


def _build_metric_specs() -> List[MetricSpec]:
    return [
        MetricSpec(
            slug="efficiency_ratio",
            label="Efficiency Ratio",
            direction="lower_better",
            required_cols=("EEFFR",),
            compute=lambda df: _normalize_percent(_series(df, "EEFFR")),
        ),
        MetricSpec(
            slug="net_interest_margin",
            label="Net Interest Margin",
            direction="higher_better",
            required_cols=("NIMR",),
            compute=lambda df: _series(df, "NIMR"),
        ),
        MetricSpec(
            slug="loan_to_deposit_ratio",
            label="Loan to Deposit Ratio",
            direction="target_range",
            required_cols=("LNLSNET", "DEP"),
            compute=lambda df: _safe_div(_series(df, "LNLSNET"), _series(df, "DEP")),
        ),
        MetricSpec(
            slug="loan_loss_provision",
            label="Loan Loss Provision",
            direction="lower_better",
            required_cols=("ELNLOS",),
            compute=lambda df: _series(df, "ELNLOS"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="net_operating_revenue",
            label="Net Operating Revenue",
            direction="higher_better",
            required_cols=("NIM", "NONII"),
            compute=lambda df: _series(df, "NIM") + _series(df, "NONII"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="return_on_assets",
            label="Return on Assets",
            direction="higher_better",
            required_cols=("ROA",),
            compute=lambda df: _series(df, "ROA"),
        ),
        MetricSpec(
            slug="return_on_equity",
            label="Return on Equity",
            direction="higher_better",
            required_cols=("ROE",),
            compute=lambda df: _series(df, "ROE"),
        ),
        MetricSpec(
            slug="total_deposits",
            label="Total Deposits",
            direction="higher_better",
            required_cols=("DEP",),
            compute=lambda df: _series(df, "DEP"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="total_loans_leases",
            label="Total Loans and Leases",
            direction="higher_better",
            required_cols=("LNLSNET",),
            compute=lambda df: _series(df, "LNLSNET"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="deposits_per_fte",
            label="Deposits per FTE",
            direction="higher_better",
            required_cols=("DEP", "NUMEMP"),
            compute=lambda df: _safe_div(_series(df, "DEP"), _series(df, "NUMEMP")),
            full_dollar=True,
        ),
        MetricSpec(
            slug="revenue_per_fte",
            label="Revenue per FTE",
            direction="higher_better",
            required_cols=("NIM", "NONII", "NUMEMP"),
            compute=lambda df: _safe_div(_series(df, "NIM") + _series(df, "NONII"), _series(df, "NUMEMP")),
            full_dollar=True,
        ),
        MetricSpec(
            slug="assets",
            label="Assets",
            direction="higher_better",
            required_cols=("ASSET",),
            compute=lambda df: _series(df, "ASSET"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="liabilities",
            label="Liabilities",
            direction="lower_better",
            required_cols=("LIAB",),
            compute=lambda df: _series(df, "LIAB"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="earning_assets",
            label="Earning Assets",
            direction="higher_better",
            required_cols=("ERNAST",),
            compute=lambda df: _series(df, "ERNAST"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="net_income",
            label="Net Income",
            direction="higher_better",
            required_cols=("NETINC",),
            compute=lambda df: _series(df, "NETINC"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="net_interest_income",
            label="Net Interest Income",
            direction="higher_better",
            required_cols=("NIM",),
            compute=lambda df: _series(df, "NIM"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="net_interest_expense",
            label="Net Interest Expense",
            direction="lower_better",
            required_cols=("EINTEXP",),
            compute=lambda df: _series(df, "EINTEXP"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="premises_fixed_assets_to_assets",
            label="Premises Fixed Assets to Assets",
            direction="lower_better",
            required_cols=("BKPREM", "ASSET"),
            compute=lambda df: _safe_div(_series(df, "BKPREM"), _series(df, "ASSET")),
        ),
        MetricSpec(
            slug="non_interest_expense",
            label="Non-Interest Expense",
            direction="lower_better",
            required_cols=("NONIX",),
            compute=lambda df: _series(df, "NONIX"),
            full_dollar=True,
        ),
        MetricSpec(
            slug="non_interest_expense_pct_total_expenses",
            label="Non-Interest Expense as Percentage of Total Expenses",
            direction="lower_better",
            required_cols=("NONIX", "EINTEXP"),
            compute=lambda df: _safe_div(_series(df, "NONIX"), _series(df, "NONIX") + _series(df, "EINTEXP")),
        ),
    ]


def _parse_mixed_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    yyyymmdd_mask = numeric.notna() & numeric.between(19000101, 21991231)
    if yyyymmdd_mask.any():
        as_text = numeric[yyyymmdd_mask].round().astype("Int64").astype(str)
        parsed.loc[yyyymmdd_mask] = pd.to_datetime(as_text, format="%Y%m%d", errors="coerce")
    return parsed


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file extension: {path}")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output extension: {path}")


def _classify_status(subject_value: float, p25: float, p75: float, direction: str) -> tuple[str | pd.NA, int | pd.NA]:
    if pd.isna(subject_value) or pd.isna(p25) or pd.isna(p75):
        return pd.NA, pd.NA

    if subject_value < p25:
        status = "below_p25"
    elif subject_value > p75:
        status = "above_p75"
    else:
        status = "within_iqr"

    if direction == "higher_better":
        lag_flag = int(status == "below_p25")
    elif direction == "lower_better":
        lag_flag = int(status == "above_p75")
    elif direction == "target_range":
        lag_flag = int(status != "within_iqr")
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    return status, lag_flag


def _peer_percentile_rank_pct(
    subject_value: float,
    peer_values: np.ndarray,
    direction: str,
    median_value: float,
) -> float | pd.NA:
    if pd.isna(subject_value) or peer_values.size == 0:
        return pd.NA

    peer_arr = np.asarray(peer_values, dtype=float)
    subject = float(subject_value)

    if direction == "higher_better":
        better_count = np.sum(peer_arr < subject)
        equal_count = np.sum(np.isclose(peer_arr, subject))
    elif direction == "lower_better":
        better_count = np.sum(peer_arr > subject)
        equal_count = np.sum(np.isclose(peer_arr, subject))
    elif direction == "target_range":
        if pd.isna(median_value):
            return pd.NA
        peer_dist = np.abs(peer_arr - float(median_value))
        subject_dist = abs(subject - float(median_value))
        better_count = np.sum(peer_dist > subject_dist)
        equal_count = np.sum(np.isclose(peer_dist, subject_dist))
    else:
        raise ValueError(f"Unsupported direction: {direction}")

    rank_pct = ((better_count + (0.5 * equal_count)) / peer_arr.size) * 100.0
    return float(rank_pct)


def _required_columns(metric_specs: Iterable[MetricSpec]) -> list[str]:
    required = set()
    for spec in metric_specs:
        required.update(spec.required_cols)
    return sorted(required)


def _default_output_path(base_file: Path) -> Path:
    stem = base_file.stem
    suffix = base_file.suffix
    return base_file.with_name(f"{stem}_PeerBenchmarks{suffix}")


def _metric_output_columns(metric_specs: Iterable[MetricSpec]) -> list[str]:
    cols: list[str] = []
    for spec in metric_specs:
        prefix = f"pb_{spec.slug}"
        cols.extend(
            [
                f"{prefix}_value",
                f"{prefix}_p25",
                f"{prefix}_median",
                f"{prefix}_p75",
                f"{prefix}_delta_to_median",
                f"{prefix}_status",
                f"{prefix}_lag_flag",
                f"{prefix}_peer_percentile_rank_pct",
                f"{prefix}_effective_peer_count",
            ]
        )
    return cols


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach peer p25/median/p75 percentile benchmarks to the 3-year bank file."
    )
    parser.add_argument(
        "--base-file",
        type=Path,
        default=Path("Banksuite_3y_Full_Percentiles.csv"),
        help="Bank-level financial file to augment (CSV or parquet).",
    )
    parser.add_argument(
        "--similarity-file",
        type=Path,
        default=Path("similar_banks/output/similar_banks.parquet"),
        help="Similarity output parquet produced by compute_similar_banks.py.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output file path (CSV or parquet). Defaults to <base>_PeerBenchmarks.<ext>.",
    )
    parser.add_argument(
        "--min-peer-count",
        type=int,
        default=4,
        help="Minimum non-null peer values required to compute percentiles for a metric.",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="If set, apply the latest peer benchmarks to all rows. Default only populates latest-quarter rows.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("peer_percentiles")

    output_file = args.output_file or _default_output_path(args.base_file)

    if not args.base_file.exists():
        raise FileNotFoundError(f"Base file not found: {args.base_file}")
    if not args.similarity_file.exists():
        raise FileNotFoundError(f"Similarity file not found: {args.similarity_file}")

    logger.info("Loading base file: %s", args.base_file)
    base = _read_table(args.base_file)
    if "RSSDID" not in base.columns or "REPDTE" not in base.columns:
        raise KeyError("Base file must contain RSSDID and REPDTE columns.")

    base["RSSDID"] = pd.to_numeric(base["RSSDID"], errors="coerce").astype("Int64")
    base["_pb_repdte"] = _parse_mixed_date(base["REPDTE"])
    latest_date = base["_pb_repdte"].max()
    if pd.isna(latest_date):
        raise ValueError("Unable to resolve latest REPDTE date from base file.")

    latest = (
        base.loc[base["_pb_repdte"] == latest_date]
        .dropna(subset=["RSSDID"])
        .sort_values("_pb_repdte")
        .drop_duplicates(subset=["RSSDID"], keep="last")
        .copy()
    )
    latest["RSSDID"] = latest["RSSDID"].astype("int64")
    logger.info("Latest benchmark date: %s (%d banks)", latest_date.date(), len(latest))

    metric_specs = _build_metric_specs()
    required_cols = _required_columns(metric_specs)
    missing_inputs = [c for c in required_cols if c not in latest.columns]
    if missing_inputs:
        logger.warning(
            "Missing source columns in base file (affected metrics will be blank): %s",
            ", ".join(missing_inputs),
        )

    metric_frame = pd.DataFrame({"RSSDID": latest["RSSDID"]})
    available_specs: list[MetricSpec] = []
    for spec in metric_specs:
        metric_col = f"pb_{spec.slug}_value"
        metric_missing = [c for c in spec.required_cols if c not in latest.columns]
        if metric_missing:
            metric_frame[metric_col] = np.nan
            logger.warning(
                "Metric %s unavailable; missing columns: %s",
                spec.slug,
                ", ".join(metric_missing),
            )
            continue
        values = pd.to_numeric(spec.compute(latest), errors="coerce")
        if spec.full_dollar:
            values = values * 1000.0
        metric_frame[metric_col] = values
        available_specs.append(spec)

    metric_lookup = metric_frame.set_index("RSSDID")

    logger.info("Loading similarity file: %s", args.similarity_file)
    sim = pd.read_parquet(
        args.similarity_file,
        columns=["subject_idrssd", "similar_idrssd"],
    )
    sim["subject_idrssd"] = pd.to_numeric(sim["subject_idrssd"], errors="coerce").astype("Int64")
    sim["similar_idrssd"] = pd.to_numeric(sim["similar_idrssd"], errors="coerce").astype("Int64")
    sim = sim.dropna(subset=["subject_idrssd", "similar_idrssd"]).copy()
    sim["subject_idrssd"] = sim["subject_idrssd"].astype("int64")
    sim["similar_idrssd"] = sim["similar_idrssd"].astype("int64")

    peer_map = (
        sim.groupby("subject_idrssd", sort=False)["similar_idrssd"]
        .apply(list)
        .to_dict()
    )

    logger.info("Computing peer benchmark columns for %d subject banks", len(peer_map))
    benchmark_rows: list[dict[str, object]] = []
    for subject_id, peer_ids in peer_map.items():
        row: dict[str, object] = {
            "RSSDID": int(subject_id),
            "pb_peer_count": int(len(peer_ids)),
            "pb_peer_benchmark_date": pd.Timestamp(latest_date),
        }
        subject_known = subject_id in metric_lookup.index
        row["pb_subject_in_latest_snapshot"] = int(subject_known)

        peer_metrics = metric_lookup.reindex(peer_ids)
        for spec in metric_specs:
            value_col = f"pb_{spec.slug}_value"
            p25_col = f"pb_{spec.slug}_p25"
            med_col = f"pb_{spec.slug}_median"
            p75_col = f"pb_{spec.slug}_p75"
            delta_col = f"pb_{spec.slug}_delta_to_median"
            status_col = f"pb_{spec.slug}_status"
            lag_col = f"pb_{spec.slug}_lag_flag"
            rank_col = f"pb_{spec.slug}_peer_percentile_rank_pct"
            effective_peer_count_col = f"pb_{spec.slug}_effective_peer_count"

            subject_value = metric_lookup.at[subject_id, value_col] if subject_known else np.nan
            row[value_col] = subject_value

            peer_values = pd.to_numeric(peer_metrics[value_col], errors="coerce").dropna().to_numpy(dtype=float)
            row[effective_peer_count_col] = int(peer_values.size)
            if peer_values.size < args.min_peer_count:
                row[p25_col] = np.nan
                row[med_col] = np.nan
                row[p75_col] = np.nan
                row[delta_col] = np.nan
                row[status_col] = pd.NA
                row[lag_col] = pd.NA
                row[rank_col] = np.nan
                continue

            p25, median, p75 = np.percentile(peer_values, [25, 50, 75])
            row[p25_col] = float(p25)
            row[med_col] = float(median)
            row[p75_col] = float(p75)
            row[delta_col] = float(subject_value - median) if pd.notna(subject_value) else np.nan
            status, lag_flag = _classify_status(
                float(subject_value) if pd.notna(subject_value) else np.nan,
                float(p25),
                float(p75),
                spec.direction,
            )
            row[status_col] = status
            row[lag_col] = lag_flag
            row[rank_col] = _peer_percentile_rank_pct(
                float(subject_value) if pd.notna(subject_value) else np.nan,
                peer_values,
                spec.direction,
                float(median),
            )

        benchmark_rows.append(row)

    benchmark_cols_template = [
        "pb_peer_count",
        "pb_peer_benchmark_date",
        "pb_subject_in_latest_snapshot",
        *_metric_output_columns(metric_specs),
    ]
    if benchmark_rows:
        benchmarks = pd.DataFrame(benchmark_rows)
    else:
        benchmarks = pd.DataFrame(columns=["RSSDID", *benchmark_cols_template])

    int_cols = [
        c
        for c in benchmarks.columns
        if c.endswith("_lag_flag") or c.endswith("_effective_peer_count")
    ]
    int_cols.extend([c for c in ("pb_peer_count", "pb_subject_in_latest_snapshot") if c in benchmarks.columns])
    for col in int_cols:
        benchmarks[col] = benchmarks[col].astype("Int64")

    benchmark_cols = [c for c in benchmarks.columns if c != "RSSDID"]

    logger.info("Merging benchmark columns onto base data")
    out = base.merge(benchmarks, how="left", on="RSSDID")

    if not args.all_dates:
        non_latest_mask = out["_pb_repdte"] != latest_date
        out.loc[non_latest_mask, benchmark_cols] = pd.NA

    out = out.drop(columns=["_pb_repdte"])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing output: %s", output_file)
    _write_table(out, output_file)

    logger.info("Completed. Output rows=%d, cols=%d", len(out), len(out.columns))
    logger.info("Added benchmark columns: %d", len(benchmark_cols))
    if available_specs:
        logger.info("Metrics computed: %s", ", ".join(spec.slug for spec in available_specs))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
