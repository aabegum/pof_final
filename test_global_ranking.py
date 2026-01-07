"""
Test script to verify PURE GLOBAL RANKING fixes PoF-Risk inconsistency.

This test demonstrates that with pure global ranking:
- Lower PoF values ALWAYS have lower/equal Risk_Rank than higher PoF values
- Risk classification is monotonic with respect to PoF values
"""

import pandas as pd
import numpy as np

def compute_health_score_global(
    df: pd.DataFrame,
    pof_col: str = "PoF_Ensemble_12Ay"
) -> pd.DataFrame:
    """Simplified version using PURE GLOBAL RANKING"""

    if pof_col not in df.columns:
        df["Health_Score"] = 90
        df["Risk_Sinifi"] = "BILINMIYOR"
        return df

    # Clean data
    df[pof_col] = df[pof_col].fillna(0).clip(0, 1)

    # ✅ PURE GLOBAL RANKING
    df["Global_Rank"] = df[pof_col].rank(pct=True)
    df["Final_Rank"] = df["Global_Rank"]

    # Health Score
    df["Health_Score"] = 100 * (1 - df["Final_Rank"])

    # Risk Classification
    def assign_risk(p):
        if p >= 0.95: return "KRİTİK"
        if p >= 0.85: return "YÜKSEK"
        if p >= 0.50: return "ORTA"
        return "DÜŞÜK"

    df["Risk_Sinifi"] = df["Final_Rank"].apply(assign_risk)

    return df


def test_consistency():
    """Test that PoF-Risk mapping is consistent"""

    # Create synthetic test data with different equipment types
    np.random.seed(42)
    n = 1000

    test_df = pd.DataFrame({
        'cbs_id': range(n),
        'Ekipman_Tipi': np.random.choice(['Trafo', 'Hücre', 'Kesici'], n),
        'PoF_Ensemble_12Ay': np.random.beta(2, 5, n)  # Realistic PoF distribution
    })

    # Apply pure global ranking
    result = compute_health_score_global(test_df, 'PoF_Ensemble_12Ay')

    print("="*70)
    print("TEST 1: PURE GLOBAL RANKING - Consistency Verification")
    print("="*70)

    # Check 1: Risk distribution
    print("\n1. Risk Distribution:")
    dist = result['Risk_Sinifi'].value_counts().sort_index()
    total = len(result)
    for risk, count in dist.items():
        pct = 100 * count / total
        print(f"   {risk:20s}: {count:4d} ({pct:5.1f}%)")

    # Check 2: Verify monotonicity
    print("\n2. Monotonicity Check:")
    print("   (Lower PoF should never have higher Risk_Rank than higher PoF)")

    # Sort by PoF and check if Risk_Rank is also increasing
    sorted_df = result.sort_values('PoF_Ensemble_12Ay')
    rank_diffs = sorted_df['Final_Rank'].diff()

    # All differences should be >= 0 (non-decreasing)
    violations = (rank_diffs < 0).sum()
    print(f"   Violations: {violations}")

    if violations == 0:
        print("   [OK] PASSED: Risk_Rank is monotonic with PoF")
    else:
        print("   [FAIL] FAILED: Found non-monotonic rankings")

    # Check 3: PoF range by Risk Class
    print("\n3. PoF Range by Risk Class:")
    for risk_class in ["DÜŞÜK", "ORTA", "YÜKSEK", "KRİTİK"]:
        subset = result[result['Risk_Sinifi'] == risk_class]
        if len(subset) > 0:
            min_pof = subset['PoF_Ensemble_12Ay'].min()
            max_pof = subset['PoF_Ensemble_12Ay'].max()
            print(f"   {risk_class:20s}: PoF in [{min_pof:.6f}, {max_pof:.6f}]")

    # Check 4: Verify no overlap between risk classes
    print("\n4. Risk Class Overlap Check:")
    dusuk_max = result[result['Risk_Sinifi']=='DÜŞÜK']['PoF_Ensemble_12Ay'].max()
    orta_min = result[result['Risk_Sinifi']=='ORTA']['PoF_Ensemble_12Ay'].min()
    orta_max = result[result['Risk_Sinifi']=='ORTA']['PoF_Ensemble_12Ay'].max()
    yuksek_min = result[result['Risk_Sinifi']=='YÜKSEK']['PoF_Ensemble_12Ay'].min()
    yuksek_max = result[result['Risk_Sinifi']=='YÜKSEK']['PoF_Ensemble_12Ay'].max()
    kritik_min = result[result['Risk_Sinifi']=='KRİTİK']['PoF_Ensemble_12Ay'].min()

    print(f"   DÜŞÜK max  : {dusuk_max:.6f}")
    print(f"   ORTA min   : {orta_min:.6f}")
    print(f"   ORTA max   : {orta_max:.6f}")
    print(f"   YÜKSEK min : {yuksek_min:.6f}")
    print(f"   YÜKSEK max : {yuksek_max:.6f}")
    print(f"   KRİTİK min : {kritik_min:.6f}")

    # Verify boundaries
    overlap = False
    if dusuk_max > orta_min:
        print(f"   [FAIL] OVERLAP: DUSUK max ({dusuk_max:.6f}) > ORTA min ({orta_min:.6f})")
        overlap = True
    if orta_max > yuksek_min:
        print(f"   [FAIL] OVERLAP: ORTA max ({orta_max:.6f}) > YUKSEK min ({yuksek_min:.6f})")
        overlap = True
    if yuksek_max > kritik_min:
        print(f"   [FAIL] OVERLAP: YUKSEK max ({yuksek_max:.6f}) > KRITIK min ({kritik_min:.6f})")
        overlap = True

    if not overlap:
        print("   [OK] NO OVERLAP: Risk classes have clear PoF boundaries")

    # Check 5: Sample records
    print("\n5. Sample Records (sorted by PoF):")
    print("   Lowest 5 PoF values:")
    sample = result.nsmallest(5, 'PoF_Ensemble_12Ay')[['cbs_id', 'PoF_Ensemble_12Ay', 'Final_Rank', 'Risk_Sinifi']]
    for _, row in sample.iterrows():
        print(f"      CBS {row['cbs_id']:4d}: PoF={row['PoF_Ensemble_12Ay']:.6f}, Rank={row['Final_Rank']:.4f}, Risk={row['Risk_Sinifi']}")

    print("\n   Highest 5 PoF values:")
    sample = result.nlargest(5, 'PoF_Ensemble_12Ay')[['cbs_id', 'PoF_Ensemble_12Ay', 'Final_Rank', 'Risk_Sinifi']]
    for _, row in sample.iterrows():
        print(f"      CBS {row['cbs_id']:4d}: PoF={row['PoF_Ensemble_12Ay']:.6f}, Rank={row['Final_Rank']:.4f}, Risk={row['Risk_Sinifi']}")

    print("\n" + "="*70)
    print("TEST COMPLETED")
    print("="*70)


if __name__ == "__main__":
    test_consistency()
