"""
TEMPORAL SPLIT ANOMALY ROOT CAUSE ANALYSIS
===========================================
Investigates why test sets have 3-6x MORE failures than train sets
(opposite of expected pattern with right censoring)
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

print("="*80)
print("TEMPORAL SPLIT ANOMALY - ROOT CAUSE ANALYSIS")
print("="*80)

# 1. Load data
try:
    survival_base = pd.read_csv("data/ara_ciktilar/survival_base.csv")
    print(f"\n[OK] Loaded survival_base.csv: {len(survival_base):,} records")
except FileNotFoundError:
    print("[ERROR] survival_base.csv not found. Run main pipeline first.")
    sys.exit(1)

# Parse dates
survival_base["Kurulum_Tarihi"] = pd.to_datetime(survival_base["Kurulum_Tarihi"], errors="coerce")
survival_base["Ilk_Gercek_Ariza_Tarihi"] = pd.to_datetime(survival_base["Ilk_Gercek_Ariza_Tarihi"], errors="coerce")

# ============================================================================
# HYPOTHESIS 1: Installation Date Data Quality Issues
# ============================================================================
print("\n" + "-"*80)
print("HYPOTHESIS 1: Installation Date Data Quality")
print("-"*80)

install_nulls = survival_base["Kurulum_Tarihi"].isna().sum()
print(f"\nMissing installation dates: {install_nulls:,} ({install_nulls/len(survival_base):.1%})")

if install_nulls == 0:
    # Check for future dates
    from datetime import datetime
    today = datetime.now()
    future_dates = (survival_base["Kurulum_Tarihi"] > today).sum()
    print(f"Future installation dates: {future_dates:,}")

    # Check for unrealistic dates (before 1960)
    ancient_dates = (survival_base["Kurulum_Tarihi"] < pd.Timestamp("1960-01-01")).sum()
    print(f"Pre-1960 installation dates: {ancient_dates:,}")

    # Installation date distribution
    survival_base["install_year"] = survival_base["Kurulum_Tarihi"].dt.year

    print("\nInstallation year distribution:")
    year_dist = survival_base["install_year"].value_counts().sort_index()

    # Show only recent years (2010+)
    recent_years = year_dist[year_dist.index >= 2010]
    for year, count in recent_years.items():
        print(f"  {year}: {count:,} installations")

    # Check for installation spike around cutoff (2018)
    years_2016_2020 = year_dist[(year_dist.index >= 2016) & (year_dist.index <= 2020)]
    print(f"\nInstallation spike check (2016-2020):")
    for year, count in years_2016_2020.items():
        print(f"  {year}: {count:,}")

    if years_2016_2020.max() > years_2016_2020.mean() * 2:
        spike_year = years_2016_2020.idxmax()
        print(f"\n[WARNING] Installation spike detected in {spike_year}")
        print(f"  {years_2016_2020.max():,} installations (2x+ above average)")
else:
    print("[SKIP] Too many null dates to analyze distribution")

# ============================================================================
# HYPOTHESIS 2: Mass Failure Event in Recent Years (2018-2024)
# ============================================================================
print("\n" + "-"*80)
print("HYPOTHESIS 2: Mass Failure Event Analysis (2018-2024)")
print("-"*80)

failed_equipment = survival_base[survival_base["Ilk_Gercek_Ariza_Tarihi"].notna()].copy()

if len(failed_equipment) > 0:
    failed_equipment["failure_year"] = failed_equipment["Ilk_Gercek_Ariza_Tarihi"].dt.year

    yearly_failures = failed_equipment["failure_year"].value_counts().sort_index()

    print(f"\nTotal failed equipment: {len(failed_equipment):,}")
    print("\nYearly failure counts:")

    for year in sorted(yearly_failures.index):
        count = yearly_failures[year]
        pct = count / len(failed_equipment) * 100
        bar = "#" * int(pct / 2)  # Visual bar
        print(f"  {year}: {count:,} ({pct:5.1f}%) {bar}")

    # Check for anomalous spike
    mean_failures = yearly_failures.mean()
    max_failures = yearly_failures.max()
    spike_year = yearly_failures.idxmax()

    print(f"\nFailure spike analysis:")
    print(f"  Mean failures/year: {mean_failures:.0f}")
    print(f"  Max failures/year: {max_failures} (in {spike_year})")

    if max_failures > mean_failures * 3:
        print(f"\n[CRITICAL] MASS FAILURE EVENT DETECTED in {spike_year}!")
        print(f"  {max_failures:,} failures ({max_failures/mean_failures:.1f}x above average)")

        # Analyze what failed
        spike_failures = failed_equipment[failed_equipment["failure_year"] == spike_year]

        if "Ekipman_Turu" in spike_failures.columns:
            print(f"\n  Equipment types in {spike_year} spike:")
            for equip_type, count in spike_failures["Ekipman_Turu"].value_counts().head(5).items():
                print(f"    {equip_type}: {count:,}")

        if "Sehir" in spike_failures.columns:
            print(f"\n  Cities affected in {spike_year}:")
            for city, count in spike_failures["Sehir"].value_counts().head(5).items():
                print(f"    {city}: {count:,}")
    else:
        print("\n[OK] No anomalous failure spikes detected")
else:
    print("[SKIP] No failure data available")

# ============================================================================
# HYPOTHESIS 3: Different Failure Rates by Installation Period
# ============================================================================
print("\n" + "-"*80)
print("HYPOTHESIS 3: Failure Rate by Installation Period")
print("-"*80)

# Simulate the 75/25 temporal split
df_sorted = survival_base.sort_values("Kurulum_Tarihi")
cutoff_pos = int(len(df_sorted) * 0.75)

train_period = df_sorted.iloc[:cutoff_pos]
test_period = df_sorted.iloc[cutoff_pos:]

train_install_range = f"{train_period['Kurulum_Tarihi'].min().date()} to {train_period['Kurulum_Tarihi'].max().date()}"
test_install_range = f"{test_period['Kurulum_Tarihi'].min().date()} to {test_period['Kurulum_Tarihi'].max().date()}"

print(f"\nTrain period installations: {train_install_range}")
print(f"Test period installations:  {test_install_range}")

if "event" in survival_base.columns:
    train_event_rate = train_period["event"].mean()
    test_event_rate = test_period["event"].mean()

    print(f"\nEvent rates:")
    print(f"  Train: {train_event_rate:.1%} ({train_period['event'].sum():,} failures)")
    print(f"  Test:  {test_event_rate:.1%} ({test_period['event'].sum():,} failures)")
    print(f"  Ratio: {test_event_rate/train_event_rate:.2f}x")

    # Analyze by equipment type
    if "Ekipman_Turu" in survival_base.columns:
        print("\nEvent rate by equipment type:")

        for equip_type in survival_base["Ekipman_Turu"].unique()[:5]:  # Top 5
            train_type = train_period[train_period["Ekipman_Turu"] == equip_type]
            test_type = test_period[test_period["Ekipman_Turu"] == equip_type]

            if len(train_type) > 0 and len(test_type) > 0:
                train_rate = train_type["event"].mean()
                test_rate = test_type["event"].mean()

                if train_rate > 0:
                    ratio = test_rate / train_rate
                    print(f"  {equip_type:15s}: Train {train_rate:.1%} | Test {test_rate:.1%} | Ratio {ratio:.2f}x")

# ============================================================================
# HYPOTHESIS 4: Observation Window Bias
# ============================================================================
print("\n" + "-"*80)
print("HYPOTHESIS 4: Observation Window Bias")
print("-"*80)

# Calculate time from installation to data end
DATA_END_DATE = pd.Timestamp("2025-01-01")  # Approximate

survival_base_copy = survival_base.copy()
survival_base_copy["observation_time_years"] = (
    (DATA_END_DATE - survival_base_copy["Kurulum_Tarihi"]).dt.days / 365.25
)

# Re-split with new column
df_sorted_copy = survival_base_copy.sort_values("Kurulum_Tarihi")
cutoff_pos_copy = int(len(df_sorted_copy) * 0.75)
train_period_copy = df_sorted_copy.iloc[:cutoff_pos_copy]
test_period_copy = df_sorted_copy.iloc[cutoff_pos_copy:]

train_obs_time = train_period_copy["observation_time_years"].median()
test_obs_time = test_period_copy["observation_time_years"].median()

print(f"\nMedian observation time:")
print(f"  Train equipment: {train_obs_time:.1f} years")
print(f"  Test equipment:  {test_obs_time:.1f} years")
print(f"  Difference: {train_obs_time - test_obs_time:.1f} years")

if train_obs_time > test_obs_time:
    print("\n[OK] Train has longer observation time (expected)")
else:
    print("\n[ANOMALY] Test has LONGER observation time (unexpected!)")

# Check if newer equipment fails faster
if "event" in survival_base_copy.columns:
    # Bin by observation time
    survival_base_copy["obs_bin"] = pd.cut(
        survival_base_copy["observation_time_years"],
        bins=[0, 1, 3, 5, 10, 100],
        labels=["<1yr", "1-3yr", "3-5yr", "5-10yr", ">10yr"]
    )

    print("\nFailure rate by observation time:")
    for bin_label in ["<1yr", "1-3yr", "3-5yr", "5-10yr", ">10yr"]:
        bin_data = survival_base_copy[survival_base_copy["obs_bin"] == bin_label]
        if len(bin_data) > 0:
            rate = bin_data["event"].mean()
            print(f"  {bin_label:8s}: {rate:.1%} ({len(bin_data):,} equipment)")

# ============================================================================
# FINAL DIAGNOSIS
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

issues = []

# Check 1: Mass failure event
if len(failed_equipment) > 0:
    if yearly_failures.max() > yearly_failures.mean() * 3:
        spike_year = yearly_failures.idxmax()
        issues.append(f"MASS FAILURE EVENT in {spike_year} ({yearly_failures.max():,} failures)")

# Check 2: Event rate ratio
if "event" in survival_base.columns:
    ratio = test_event_rate / train_event_rate
    if ratio > 2.0:
        issues.append(f"SEVERE EVENT RATE MISMATCH (Test {ratio:.1f}x higher than Train)")

# Check 3: Installation spike
if install_nulls == 0 and len(years_2016_2020) > 0:
    if years_2016_2020.max() > years_2016_2020.mean() * 2:
        spike_year = years_2016_2020.idxmax()
        issues.append(f"INSTALLATION SPIKE near split cutoff ({spike_year})")

# Check 4: Observation time
if train_obs_time <= test_obs_time:
    issues.append("OBSERVATION TIME ANOMALY (Test >= Train)")

print("\nIssues detected:")
if issues:
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("  None - data appears normal")

print("\n" + "="*80)
print("RECOMMENDED ACTIONS:")
print("="*80)

if len(issues) > 0:
    print("\n1. VERIFY DATA QUALITY:")
    print("   - Check Kurulum_Tarihi accuracy for equipment installed 2016-2020")
    print("   - Validate Ilk_Gercek_Ariza_Tarihi for spike years")

    print("\n2. INVESTIGATE BUSINESS CONTEXT:")
    print("   - Was there a known event causing mass failures?")
    print("   - Were there changes in maintenance practices?")
    print("   - Was there a data migration or system change?")

    print("\n3. ALTERNATIVE SPLIT STRATEGIES:")
    print("   - Use stratified split by Ekipman_Turu + event")
    print("   - Use cross-validation instead of single split")
    print("   - Filter out anomalous failure period for training")
else:
    print("\nData appears normal. Issue may be in split implementation.")

print("="*80)
