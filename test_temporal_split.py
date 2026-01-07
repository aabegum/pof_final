"""
TEMPORAL SPLIT VALIDATION SCRIPT
Test eder: Zaman bazl blmenin doru alp almadn
"""

import pandas as pd
import numpy as np
import sys

print("="*70)
print("[TEST] TEMPORAL SPLIT VALIDATION")
print("="*70)

# 1. Veriyi ykle
try:
    df = pd.read_csv("data/ara_ciktilar/survival_base.csv")
    print(f"[OK] Data loaded: {len(df)} records")
except FileNotFoundError:
    print("[ERROR] survival_base.csv not found. Run main pipeline first.")
    sys.exit(1)

# 2. Tarih stunlarn parse et
df["Kurulum_Tarihi"] = pd.to_datetime(df["Kurulum_Tarihi"], errors="coerce")

# 3. Temporal split simlasyonu
df_sorted = df.sort_values("Kurulum_Tarihi")
cutoff_pos = int(len(df_sorted) * 0.75)

train_dates = df_sorted.iloc[:cutoff_pos]["Kurulum_Tarihi"]
test_dates = df_sorted.iloc[cutoff_pos:]["Kurulum_Tarihi"]

# 4. Temel kontroller
print("\n" + "-"*70)
print(" DATE RANGE ANALYSIS")
print("-"*70)

print(f"Train date range: {train_dates.min().date()}  {train_dates.max().date()}")
print(f"Test date range:  {test_dates.min().date()}  {test_dates.max().date()}")

#  KONTROL 1: Temporal order doru mu?
train_max = train_dates.max()
test_min = test_dates.min()

if train_max <= test_min:
    print("\n PASS: Temporal order is CORRECT (train < test)")
else:
    print(f"\n FAIL: OVERLAP DETECTED!")
    print(f"   Train max: {train_max.date()}")
    print(f"   Test min:  {test_min.date()}")
    print(f"    {(train_max - test_min).days} days overlap!")

# 5. Event rate analizi
print("\n" + "-"*70)
print(" EVENT RATE ANALYSIS")
print("-"*70)

if "event" in df.columns:
    train_labels = df_sorted.iloc[:cutoff_pos].index
    test_labels = df_sorted.iloc[cutoff_pos:].index

    train_ev = df.loc[train_labels, "event"].mean()
    test_ev = df.loc[test_labels, "event"].mean()

    print(f"Train events: {train_ev:.1%}")
    print(f"Test events:  {test_ev:.1%}")

    if train_ev > 0:
        ratio = test_ev / train_ev
        print(f"Ratio (test/train): {ratio:.2f}x")

        #  KONTROL 2: Event rate makul mu?
        if 0.5 <= ratio <= 2.0:
            print("\n PASS: Event rate ratio is ACCEPTABLE")
        else:
            print(f"\n  WARNING: Event rate mismatch detected")
            if ratio > 2.0:
                print(f"    Test has {ratio:.1f}x MORE events (possible data issue)")
            else:
                print(f"    Test has {1/ratio:.1f}x FEWER events (possible right censoring)")
else:
    print("  'event' column not found")

# 6. Arza tarihi analizi (eer varsa)
print("\n" + "-"*70)
print(" FAILURE DATE DISTRIBUTION")
print("-"*70)

if "Ilk_Gercek_Ariza_Tarihi" in df.columns:
    df["Ilk_Gercek_Ariza_Tarihi"] = pd.to_datetime(df["Ilk_Gercek_Ariza_Tarihi"], errors="coerce")

    failed_df = df[df["Ilk_Gercek_Ariza_Tarihi"].notna()]

    if len(failed_df) > 0:
        # Yllk arza dalm
        failed_df["failure_year"] = failed_df["Ilk_Gercek_Ariza_Tarihi"].dt.year
        yearly_failures = failed_df["failure_year"].value_counts().sort_index()

        print("\nYearly failure distribution:")
        for year, count in yearly_failures.items():
            print(f"  {year}: {count} failures")

        #  KONTROL 3: Anormal patlama var m?
        max_failures = yearly_failures.max()
        mean_failures = yearly_failures.mean()

        if max_failures > mean_failures * 3:
            max_year = yearly_failures.idxmax()
            print(f"\n  WARNING: Anomalous spike detected in {max_year}")
            print(f"    {max_failures} failures (3x above average)")
        else:
            print("\n PASS: No anomalous failure spikes")
else:
    print("  'Ilk_Gercek_Ariza_Tarihi' column not found")

# 7. Train-test overlap kontrol
print("\n" + "-"*70)
print(" OVERLAP CHECK")
print("-"*70)

overlap_count = sum((train_dates >= test_min) & (train_dates <= test_dates.max()))
print(f"Records in train with dates overlapping test range: {overlap_count}")

if overlap_count == 0:
    print(" PASS: No overlap between train and test sets")
else:
    print(f"  WARNING: {overlap_count} records overlap (should be 0 for pure temporal split)")

# 8. FINAL VERDICT
print("\n" + "="*70)
print(" FINAL VERDICT")
print("="*70)

issues = []

if train_max > test_min:
    issues.append(" Temporal order violated")

if "event" in df.columns and train_ev > 0:
    ratio = test_ev / train_ev
    if ratio > 2.0 or ratio < 0.5:
        issues.append(f"  Event rate mismatch (ratio: {ratio:.2f})")

if overlap_count > 0:
    issues.append(f"  Train-test overlap detected ({overlap_count} records)")

if not issues:
    print("\n ALL CHECKS PASSED!")
    print("   Temporal split is working correctly.")
else:
    print("\n  ISSUES DETECTED:")
    for issue in issues:
        print(f"   {issue}")
    print("\n    Review data quality or split logic!")

print("="*70)
