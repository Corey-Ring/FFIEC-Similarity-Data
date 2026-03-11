from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_loader import load_source_data, prepare_snapshots
from feature_engineering import engineer_bank_features, validate_feature_contract
from similarity_model import REQUIRED_DRIVER_FEATURES


def _test_mappings() -> dict:
    return {
        "identifiers": {"bank_id": "RSSDID", "cert": "CERT"},
        "names": {"land_name": "NAMEFULL", "institution_name": "NAME_inst"},
        "dates": {"land_report_date": "REPDTE"},
        "financial": {
            "total_assets": "ASSET",
            "total_deposits": "DEP",
            "total_loans": "LNLSNET",
            "loan_to_deposit_ratio": "LOAN_TO_DEPOSIT_RATIO",
            "asset_growth_cagr_5y": "ASSET_GROWTH_CAGR_5Y",
            "asset_size_bucket": "ASSET_SIZE_BUCKET",
        },
        "lending_profile": {
            "cre": ["Lending_Profile_CRE_Pct"],
            "ci": ["Lending_Profile_Industrial_Pct"],
            "residential_mortgage": ["LNRERES"],
            "consumer": ["Lending_Profile_Other_Pct"],
            "agricultural": ["LNAG"],
            "construction": ["LNRECON2"],
        },
        "deposit_structure": {
            "core_deposits": ["COREDEP"],
            "non_interest_deposits": ["DEPNIDOM", "DEPNI"],
        },
        "institutional": {
            "charter_type": "CHARTER",
            "holding_company_rssd": "RSSDHCR",
            "specialty_group": "SPECGRPN",
        },
    }


def _bank_base() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "RSSDID": 1,
                "CERT": 101,
                "NAME_inst": "Alpha Bank",
                "NAMEFULL": "Alpha Bank Full",
                "ASSET": 200.0,
                "DEP": 100.0,
                "LNLSNET": 60.0,
                "COREDEP": 80.0,
                "DEPNIDOM": 10.0,
                "CHARTER": "NAT",
                "RSSDHCR": 5001.0,
                "SPECGRPN": "COMMERCIAL",
                "Lending_Profile_CRE_Pct": 0.25,
                "Lending_Profile_Industrial_Pct": 0.15,
                "LNRERES": 0.20,
                "Lending_Profile_Other_Pct": 0.10,
                "LNAG": 0.05,
                "LNRECON2": 0.08,
            },
            {
                "RSSDID": 2,
                "CERT": 202,
                "NAME_inst": "Beta Bank",
                "NAMEFULL": "Beta Bank Full",
                "ASSET": 300.0,
                "DEP": 100.0,
                "LNLSNET": 70.0,
                "COREDEP": 50.0,
                "DEPNIDOM": 60.0,
                "CHARTER": "NAT",
                "RSSDHCR": 0.0,
                "SPECGRPN": "COMMERCIAL",
                "Lending_Profile_CRE_Pct": 0.20,
                "Lending_Profile_Industrial_Pct": 0.20,
                "LNRERES": 0.15,
                "Lending_Profile_Other_Pct": 0.12,
                "LNAG": 0.04,
                "LNRECON2": 0.07,
            },
        ]
    )


def _land_history() -> pd.DataFrame:
    rows = []
    for bank_id, base_assets, base_dep, base_loans in [(1, 160.0, 90.0, 50.0), (2, 240.0, 85.0, 55.0)]:
        for quarter, multiplier in [
            ("2024-03-31", 1.00),
            ("2024-06-30", 1.05),
            ("2024-09-30", 1.10),
            ("2024-12-31", 1.15),
            ("2025-03-31", 1.20),
        ]:
            rows.append(
                {
                    "RSSDID": bank_id,
                    "REPDTE": quarter,
                    "ASSET": base_assets * multiplier,
                    "DEP": base_dep * multiplier,
                    "LNLSNET": base_loans * multiplier,
                }
            )
    return pd.DataFrame(rows)


def test_deposit_structure_features_are_truthfully_named_and_not_complement_filled() -> None:
    feature_set = engineer_bank_features(
        bank_base=_bank_base(),
        land_history=_land_history(),
        mappings=_test_mappings(),
        logger=logging.getLogger("test"),
    )

    frame = feature_set.frame.sort_values("bank_id").reset_index(drop=True)
    assert "core_deposits_pct" in frame.columns
    assert "non_interest_deposits_pct" in frame.columns
    assert "deposit_mix_retail_pct" not in frame.columns
    assert "deposit_mix_business_pct" not in frame.columns
    assert "core_deposits_per_fte_ttm" not in frame.columns

    assert frame.loc[0, "core_deposits_pct"] == pytest.approx(0.80)
    assert frame.loc[0, "non_interest_deposits_pct"] == pytest.approx(0.10)
    assert frame.loc[1, "core_deposits_pct"] == pytest.approx(0.50)
    assert frame.loc[1, "non_interest_deposits_pct"] == pytest.approx(0.60)
    assert (
        frame.loc[1, "core_deposits_pct"] + frame.loc[1, "non_interest_deposits_pct"]
    ) == pytest.approx(1.10)


def test_validate_feature_contract_reports_missing_weighted_features() -> None:
    feature_set = engineer_bank_features(
        bank_base=_bank_base(),
        land_history=_land_history(),
        mappings=_test_mappings(),
        logger=logging.getLogger("test"),
    )

    validated = validate_feature_contract(
        feature_set=feature_set,
        numeric_feature_weights={
            "core_deposits_pct": 1.0,
            "non_interest_deposits_pct": 1.0,
            "asset_growth_cagr_5y": 0.5,
        },
        driver_features=REQUIRED_DRIVER_FEATURES,
        logger=logging.getLogger("test"),
    )

    assert validated.numeric_features == [
        "core_deposits_pct",
        "non_interest_deposits_pct",
    ]

    status_by_name = {status.feature_name: status for status in validated.feature_status}
    assert status_by_name["core_deposits_pct"].used_in_model
    assert not status_by_name["log_total_assets"].used_in_model
    assert status_by_name["asset_growth_cagr_5y"].status == "dropped"
    assert "Configured numeric weight present but feature is unavailable." in status_by_name[
        "asset_growth_cagr_5y"
    ].reason


def test_local_data_feature_contract_smoke_if_sources_present() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "similar_banks" / "config" / "column_mappings.yaml"
    weights_path = repo_root / "similar_banks" / "config" / "feature_weights.yaml"
    required_files = [
        repo_root / "banksuite_financials_last_3y_full.parquet",
        repo_root / "Locations__Bank_Suite_.csv",
        repo_root / "Institutions__Bank_Suite_.csv",
    ]
    if not all(path.exists() for path in required_files):
        pytest.skip("Local source data not present.")

    mappings = yaml.safe_load(config_path.read_text())
    weights = yaml.safe_load(weights_path.read_text())
    raw = load_source_data(repo_root, mappings)
    bundle = prepare_snapshots(raw, mappings, logging.getLogger("test"))
    feature_set = engineer_bank_features(
        bank_base=bundle.bank_base,
        land_history=bundle.land_history,
        mappings=mappings,
        logger=logging.getLogger("test"),
    )
    validated = validate_feature_contract(
        feature_set=feature_set,
        numeric_feature_weights=weights["numeric_feature_weights"],
        driver_features=REQUIRED_DRIVER_FEATURES,
        logger=logging.getLogger("test"),
    )

    assert validated.numeric_features
    status_by_name = {status.feature_name: status for status in validated.feature_status}
    assert not {
        feature_name
        for feature_name in weights["numeric_feature_weights"]
        if status_by_name[feature_name].status == "dropped"
    }
