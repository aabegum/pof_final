"""
DUZELTME: Dengeleme oncesi temporal split yapilmali
====================================================

SORUN:
- Split oncesi dengeleme yapilinca train/test dengesiz oluyor
- Sonuc: Train 7.3% events | Test 41.3% events (YANLIS!)

COZUM:
- ONCE temporal split
- SONRA her kume icin ayri dengeleme (1:5)
- Boylece train ve test ayri ayri 1:5 oluyor
"""

import pandas as pd
import numpy as np

def balance_dataset_stratified(
    df: pd.DataFrame,
    target_col: str = "event",
    target_ratio: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    1:N (arızalı:sağlam) dengeleme yapar

    Args:
        df: 'event' sutunu olan DataFrame (1=faulty, 0=healthy)
        target_ratio: Hedef oran (default 5 = 1:5)
        random_state: Seed

    Returns:
        Dengelenmiş DataFrame
    """

    # Ayır
    faulty = df[df[target_col] == 1].copy()
    healthy = df[df[target_col] == 0].copy()

    n_faulty = len(faulty)
    n_healthy = len(healthy)
    n_healthy_target = n_faulty * target_ratio

    print(f"  Dengeleme oncesi: {n_faulty} arizali | {n_healthy} saglam ({n_healthy/n_faulty:.1f}:1)")

    # Undersample if needed
    if n_healthy > n_healthy_target:
        healthy_sampled = healthy.sample(n=n_healthy_target, random_state=random_state)
        print(f"  Dengeleme sonrasi: {n_faulty} arizali | {n_healthy_target} saglam ({target_ratio}:1)")
    else:
        healthy_sampled = healthy
        print(f"  Dengeleme gereksiz (zaten hedefin altinda)")

    # Birlestir
    balanced = pd.concat([faulty, healthy_sampled], ignore_index=True)

    return balanced


# ============================================================================
# KULLANIM ORNEGI: Split ONCE → Balance SONRA
# ============================================================================

if __name__ == "__main__":

    # 1. Veri yukle
    df = pd.read_csv("data/ara_ciktilar/survival_base.csv")
    df["Kurulum_Tarihi"] = pd.to_datetime(df["Kurulum_Tarihi"], errors="coerce")

    print("="*80)
    print("DUZELTILMIS DENGELEME SIRASI: ONCE SPLIT - SONRA DENGELEME")
    print("="*80)

    # 2. TEMPORAL SPLIT (pipeline ile ayni)
    df_sorted = df.sort_values("Kurulum_Tarihi")
    cutoff_pos = int(len(df_sorted) * 0.75)
    cutoff_date = df_sorted.iloc[cutoff_pos]["Kurulum_Tarihi"]

    train = df_sorted[df_sorted["Kurulum_Tarihi"] < cutoff_date].copy()
    test = df_sorted[df_sorted["Kurulum_Tarihi"] >= cutoff_date].copy()

    print(f"\nOrijinal split:")
    print(f"  Train: {len(train):,} kayit | Event rate: {train['event'].mean():.1%}")
    print(f"  Test:  {len(test):,} kayit | Event rate: {test['event'].mean():.1%}")

    # 3. HER KUME AYRI DENGELENIR
    print(f"\nTrain kume dengeleniyor:")
    train_balanced = balance_dataset_stratified(train, target_ratio=5, random_state=42)

    print(f"\nTest kume dengeleniyor:")
    test_balanced = balance_dataset_stratified(test, target_ratio=5, random_state=42)

    # 4. KONTROL
    print("\n" + "="*80)
    print("SONUCLAR:")
    print("="*80)

    print(f"\nTrain kume:")
    print(f"  Boyut: {len(train_balanced):,} (eski: {len(train):,})")
    print(f"  Event rate: {train_balanced['event'].mean():.1%}")
    print(f"  Oran: {len(train_balanced[train_balanced['event']==0]) / len(train_balanced[train_balanced['event']==1]):.1f}:1")

    print(f"\nTest kume:")
    print(f"  Boyut: {len(test_balanced):,} (eski: {len(test):,})")
    print(f"  Event rate: {test_balanced['event'].mean():.1%}")
    print(f"  Oran: {len(test_balanced[test_balanced['event']==0]) / len(test_balanced[test_balanced['event']==1]):.1f}:1")

    # 5. EVENT RATE ORANI KONTROL
    train_ev = train_balanced['event'].mean()
    test_ev = test_balanced['event'].mean()
    ratio = test_ev / train_ev if train_ev > 0 else 0

    print(f"\nEvent rate orani (test/train): {ratio:.2f}x")

    if 0.5 <= ratio <= 2.0:
        print("  BASARILI: Event rate'ler artik dengeli!")
    else:
        print(f"  UYARI: Hala dengesiz (0.5-2.0x olmali)")

    # 6. Kaydet
    train_balanced.to_csv("data/ara_ciktilar/survival_base_train_balanced.csv", index=False)
    test_balanced.to_csv("data/ara_ciktilar/survival_base_test_balanced.csv", index=False)

    print("\nDengelenmiş veri setleri kaydedildi:")
    print("  - data/ara_ciktilar/survival_base_train_balanced.csv")
    print("  - data/ara_ciktilar/survival_base_test_balanced.csv")

    print("="*80)
