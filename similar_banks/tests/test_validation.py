from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from feature_engineering import FeatureSet
from geographic_similarity import GeographicSimilarityEngine
from similarity_model import (
    SimilarBankRecommender,
    gut_check_wells,
    spot_check,
    validate_similarity_output,
)


def test_spot_check_by_id_and_name() -> None:
    df = pd.DataFrame(
        [
            {
                "subject_idrssd": 1,
                "subject_name": "Wells Fargo Bank, National Association",
                "similar_rank": 1,
                "similar_idrssd": 2,
                "similar_name": "JPMorgan Chase Bank, National Association",
                "similarity_score": 0.9,
            },
            {
                "subject_idrssd": 3,
                "subject_name": "Community National Bank",
                "similar_rank": 1,
                "similar_idrssd": 4,
                "similar_name": "Regional Bank",
                "similarity_score": 0.8,
            },
        ]
    )
    by_id = spot_check(df, 1)
    assert len(by_id) == 1
    assert by_id.iloc[0]["subject_name"] == "Wells Fargo Bank, National Association"

    by_name = spot_check(df, "community")
    assert len(by_name) == 1
    assert by_name.iloc[0]["subject_idrssd"] == 3


def test_gut_check_helper_with_synthetic_sample() -> None:
    df = pd.DataFrame(
        [
            {
                "subject_idrssd": 451965,
                "subject_name": "Wells Fargo Bank, National Association",
                "similar_rank": 1,
                "similar_idrssd": 852218,
                "similar_name": "JPMorgan Chase Bank, National Association",
                "similarity_score": 0.95,
            },
            {
                "subject_idrssd": 451965,
                "subject_name": "Wells Fargo Bank, National Association",
                "similar_rank": 2,
                "similar_idrssd": 480228,
                "similar_name": "Bank of America, National Association",
                "similarity_score": 0.93,
            },
        ]
    )
    result = gut_check_wells(df)
    assert result.passed


def test_similarity_output_validation_on_synthetic_recommender_run() -> None:
    frame = pd.DataFrame(
        [
            {
                "bank_id": 1,
                "subject_name": "Alpha Bank",
                "cert": 101,
                "total_assets": 100.0,
                "log_total_assets": 4.615,
                "log_total_deposits": 4.510,
                "log_total_loans": 4.248,
                "loan_to_deposit_ratio": 0.70,
                "core_deposits_pct": 0.80,
                "non_interest_deposits_pct": 0.10,
                "loans_to_assets": 0.60,
                "deposits_to_assets": 0.80,
                "avg_3y_assets_yoy_growth": 0.05,
                "avg_3y_loans_yoy_growth": 0.04,
                "loan_mix_cre_pct": 0.25,
                "loan_mix_ci_pct": 0.15,
                "loan_mix_residential_mortgage_pct": 0.20,
                "loan_mix_consumer_pct": 0.10,
                "loan_mix_agricultural_pct": 0.05,
                "loan_mix_construction_pct": 0.08,
                "charter_type": "NAT",
                "has_holding_company": 1,
                "specialty_group": "COMMERCIAL",
            },
            {
                "bank_id": 2,
                "subject_name": "Beta Bank",
                "cert": 202,
                "total_assets": 110.0,
                "log_total_assets": 4.709,
                "log_total_deposits": 4.585,
                "log_total_loans": 4.317,
                "loan_to_deposit_ratio": 0.72,
                "core_deposits_pct": 0.78,
                "non_interest_deposits_pct": 0.12,
                "loans_to_assets": 0.62,
                "deposits_to_assets": 0.79,
                "avg_3y_assets_yoy_growth": 0.04,
                "avg_3y_loans_yoy_growth": 0.03,
                "loan_mix_cre_pct": 0.24,
                "loan_mix_ci_pct": 0.16,
                "loan_mix_residential_mortgage_pct": 0.21,
                "loan_mix_consumer_pct": 0.11,
                "loan_mix_agricultural_pct": 0.05,
                "loan_mix_construction_pct": 0.07,
                "charter_type": "NAT",
                "has_holding_company": 1,
                "specialty_group": "COMMERCIAL",
            },
            {
                "bank_id": 3,
                "subject_name": "Gamma Bank",
                "cert": 303,
                "total_assets": 400.0,
                "log_total_assets": 5.994,
                "log_total_deposits": 5.857,
                "log_total_loans": 5.704,
                "loan_to_deposit_ratio": 0.85,
                "core_deposits_pct": 0.55,
                "non_interest_deposits_pct": 0.30,
                "loans_to_assets": 0.68,
                "deposits_to_assets": 0.72,
                "avg_3y_assets_yoy_growth": 0.02,
                "avg_3y_loans_yoy_growth": 0.02,
                "loan_mix_cre_pct": 0.35,
                "loan_mix_ci_pct": 0.25,
                "loan_mix_residential_mortgage_pct": 0.10,
                "loan_mix_consumer_pct": 0.08,
                "loan_mix_agricultural_pct": 0.02,
                "loan_mix_construction_pct": 0.10,
                "charter_type": "STATE",
                "has_holding_company": 0,
                "specialty_group": "COMMERCIAL",
            },
            {
                "bank_id": 4,
                "subject_name": "Delta Bank",
                "cert": 404,
                "total_assets": 420.0,
                "log_total_assets": 6.040,
                "log_total_deposits": 5.905,
                "log_total_loans": 5.736,
                "loan_to_deposit_ratio": 0.83,
                "core_deposits_pct": 0.58,
                "non_interest_deposits_pct": 0.28,
                "loans_to_assets": 0.66,
                "deposits_to_assets": 0.74,
                "avg_3y_assets_yoy_growth": 0.03,
                "avg_3y_loans_yoy_growth": 0.03,
                "loan_mix_cre_pct": 0.34,
                "loan_mix_ci_pct": 0.24,
                "loan_mix_residential_mortgage_pct": 0.11,
                "loan_mix_consumer_pct": 0.09,
                "loan_mix_agricultural_pct": 0.02,
                "loan_mix_construction_pct": 0.09,
                "charter_type": "STATE",
                "has_holding_company": 0,
                "specialty_group": "COMMERCIAL",
            },
        ]
    )
    feature_set = FeatureSet(
        frame=frame,
        numeric_features=[
            "log_total_assets",
            "log_total_deposits",
            "log_total_loans",
            "loan_to_deposit_ratio",
            "core_deposits_pct",
            "non_interest_deposits_pct",
        ],
        categorical_features=["charter_type", "has_holding_company", "specialty_group"],
        missing_critical=pd.DataFrame(),
        data_gaps=[],
        feature_status=[],
    )
    geo_engine = GeographicSimilarityEngine(
        locations_df=pd.DataFrame(
            [
                {"CERT": 101, "CBSA_NO": "11100"},
                {"CERT": 202, "CBSA_NO": "11100"},
                {"CERT": 303, "CBSA_NO": "22200"},
                {"CERT": 404, "CBSA_NO": "22200"},
            ]
        ),
        bank_features_df=frame,
        mappings={
            "identifiers": {"cert": "CERT"},
            "geography": {"market_code": "CBSA_NO"},
        },
        geography_config={
            "market_overlap_weight": 0.70,
            "concentration_weight": 0.10,
            "market_count_weight": 0.20,
            "both_missing_score": 0.50,
            "one_missing_score": 0.25,
        },
    )
    recommender = SimilarBankRecommender(
        weights_config={
            "model": {"top_n": 2, "candidate_k": 3},
            "weights": {"financial": 0.7, "geographic": 0.2, "categorical": 0.1},
            "categorical_weights": {
                "charter_type": 0.5,
                "holding_company": 0.25,
                "specialty_group": 0.25,
            },
            "driver_thresholds": {
                "size_pct_diff_max": 0.35,
                "loan_to_deposit_abs_diff_max": 0.15,
                "growth_abs_diff_max": 0.03,
                "overlap_for_strong_geo": 0.50,
                "broad_footprint_min_cbsa": 20,
                "broad_footprint_count_ratio_min": 0.55,
            },
            "numeric_feature_weights": {
                "log_total_assets": 1.0,
                "log_total_deposits": 1.0,
                "log_total_loans": 1.0,
                "loan_to_deposit_ratio": 1.0,
                "core_deposits_pct": 1.0,
                "non_interest_deposits_pct": 1.0,
            },
            "asset_size_filter": {"enabled": False},
            "geography": {
                "market_overlap_weight": 0.70,
                "concentration_weight": 0.10,
                "market_count_weight": 0.20,
                "both_missing_score": 0.50,
                "one_missing_score": 0.25,
            },
        }
    )

    similar_df = recommender.compute(
        feature_set=feature_set,
        geo_engine=geo_engine,
        computed_date=date(2026, 3, 10),
    )
    validation = validate_similarity_output(similar_df, expected_top_n=2)

    assert validation.passed, validation.issues
    assert (similar_df["subject_idrssd"] != similar_df["similar_idrssd"]).all()
    assert (similar_df.groupby("subject_idrssd").size() == 2).all()
