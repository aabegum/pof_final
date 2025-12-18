# -*- coding: utf-8 -*-
"""
DIAGNOSTIC SCRIPT - Run This FIRST
===================================
1. Analyzes fault cause codes (Ariza_Nedeni)
2. Validates date parsing (Kurulum_Tarihi)
3. Identifies data quality issues
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import yaml

# =============================================================================
# IMPROVED DATE PARSER (Deterministic, Handles Turkish Formats)
# =============================================================================

def parse_date_safely(x):
    """
    Deterministic multi-format date parser for Turkish data
    
    Handles:
    - 1.2.2021 16:59 (Turkish format with dots)
    - 07-01-2024 21:17:45 (dashes)
    - 2021-02-01 14:30:00 (ISO format)
    - 01/02/2021 09:30 (slashes)
    - And variations without time
    """
    if pd.isna(x):
        return pd.NaT
    
    x = str(x).strip()
    
    # Priority: Day-first formats for Turkish data
    date_formats = [
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y",
        "%d-%m-%Y %H:%M:%S",
        "%d-%m-%Y %H:%M",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(x, fmt)
        except (ValueError, TypeError):
            pass
    
    # Fallback with explicit dayfirst=True
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except:
        return pd.NaT


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_PATHS = {k: os.path.join(BASE_DIR, v) for k, v in CFG["data_paths"].items()}

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_fault_causes(df_fault: pd.DataFrame):
    """Comprehensive fault cause analysis"""
    
    print("="*80)
    print("FAULT CAUSE CODE ANALYSIS (cause code)")
    print("="*80)
    
    # Try both column names
    cause_col = None
    if "cause code" in df_fault.columns:
        cause_col = "cause code"
    elif "Ariza_Nedeni" in df_fault.columns:
        cause_col = "Ariza_Nedeni"
    
    if cause_col is None:
        print("❌ ERROR: No cause code column found!")
        print(f"   Looking for: 'cause code' or 'Ariza_Nedeni'")
        print(f"   Available columns: {list(df_fault.columns)}")
        return None
    
    print(f"✅ Using column: '{cause_col}'")
    
    # Basic stats
    total = len(df_fault)
    missing = df_fault[cause_col].isna().sum()
    unique = df_fault[cause_col].nunique()
    
    print(f"Total fault records: {total:,}")
    print(f"Missing cause codes: {missing:,} ({100*missing/total:.1f}%)")
    print(f"Unique cause codes: {unique}")
    print()
    
    # Top 30 causes
    cause_counts = df_fault[cause_col].value_counts()
    
    print("TOP 30 FAULT CAUSES:")
    print("-" * 80)
    print(f"{'Count':>8} | {'%':>6} | {'Cause Code'}")
    print("-" * 80)
    
    for cause, count in cause_counts.head(30).items():
        pct = 100 * count / total
        print(f"{count:>8,} | {pct:>5.1f}% | {cause}")
    
    print("-" * 80)
    print()
    
    # Equipment type breakdown
    if "Ekipman_Tipi" in df_fault.columns or "Şebeke Unsuru" in df_fault.columns:
        eq_col = "Ekipman_Tipi" if "Ekipman_Tipi" in df_fault.columns else "Şebeke Unsuru"
        
        print("FAULT CAUSES BY EQUIPMENT TYPE:")
        print("-" * 80)
        
        for eq_type in df_fault[eq_col].value_counts().head(5).index:
            df_eq = df_fault[df_fault[eq_col] == eq_type]
            top3 = df_eq[cause_col].value_counts().head(3)
            
            print(f"\n{eq_type} (N={len(df_eq):,}):")
            for cause, count in top3.items():
                pct = 100 * count / len(df_eq)
                print(f"  {count:>6,} ({pct:>5.1f}%) - {cause}")
    
    print("\n" + "="*80)
    print("RECOMMENDED FILTER CATEGORIES")
    print("="*80)
    
    # Auto-suggest categories based on keywords
    causes_list = cause_counts.index.tolist()
    
    # Real failures (keywords: arıza, hasar, bozulma, yaşlanma)
    failure_keywords = ["arıza", "ariza", "hasar", "bozulma", "yaşlan", "aşın", "kopma", "yanma"]
    real_failures = [c for c in causes_list if any(kw in str(c).lower() for kw in failure_keywords)]
    
    print("\n✅ SUGGESTED REAL_FAILURE_CODES (Equipment actually broke):")
    for cause in real_failures[:15]:
        count = cause_counts[cause]
        print(f"  '{cause}',  # {count:,} records")
    
    # Operational events (keywords: yük, harici, hava, bakım, planlı)
    exclude_keywords = ["yük", "yuk", "harici", "hava", "bakim", "bakım", "planl", "insan"]
    excludes = [c for c in causes_list if any(kw in str(c).lower() for kw in exclude_keywords)]
    
    print("\n❌ SUGGESTED EXCLUDE_CODES (Operational events, not failures):")
    for cause in excludes[:15]:
        count = cause_counts[cause]
        print(f"  '{cause}',  # {count:,} records")
    
    print("\n" + "="*80)
    
    # Save full report
    output_path = os.path.join(BASE_DIR, "fault_cause_analysis.csv")
    cause_df = cause_counts.reset_index()
    cause_df.columns = ["cause_code", "Count"]
    cause_df["Percentage"] = 100 * cause_df["Count"] / total
    cause_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"💾 Full report saved: {output_path}")
    print("="*80)
    
    return cause_df


def analyze_date_parsing(df_fault: pd.DataFrame, df_healthy: pd.DataFrame):
    """Validate date parsing quality"""
    
    print("\n" + "="*80)
    print("DATE PARSING VALIDATION")
    print("="*80)
    
    date_cols = []
    
    # Check fault data
    print("\n📁 FAULT DATA:")
    for col in ["Sebekeye_Baglanma_Tarihi", "started at", "ended at", "Kurulum_Tarihi"]:
        if col in df_fault.columns:
            date_cols.append(("fault", col, df_fault[col]))
    
    # Check healthy data
    print("\n📁 HEALTHY DATA:")
    for col in ["Sebekeye_Baglanma_Tarihi", "Kurulum_Tarihi"]:
        if col in df_healthy.columns:
            date_cols.append(("healthy", col, df_healthy[col]))
    
    # Analyze each date column
    issues_found = False
    
    for source, col_name, series in date_cols:
        print(f"\n  [{source.upper()}] {col_name}:")
        print("  " + "-" * 60)
        
        # Original data sample
        sample = series.dropna().head(10).astype(str).tolist()
        print(f"  Sample formats: {sample[:3]}")
        
        # Parse with improved parser
        parsed = series.apply(parse_date_safely)
        
        # Stats
        total = len(series)
        missing_orig = series.isna().sum()
        missing_parsed = parsed.isna().sum()
        failed_parse = missing_parsed - missing_orig
        
        print(f"  Total records: {total:,}")
        print(f"  Missing (original): {missing_orig:,} ({100*missing_orig/total:.1f}%)")
        print(f"  Missing (after parse): {missing_parsed:,} ({100*missing_parsed/total:.1f}%)")
        
        if failed_parse > 0:
            issues_found = True
            print(f"  ⚠️  PARSE FAILURES: {failed_parse:,} ({100*failed_parse/total:.1f}%)")
            
            # Show failed examples
            failed_mask = series.notna() & parsed.isna()
            failed_examples = series[failed_mask].head(5).tolist()
            print(f"  Failed examples: {failed_examples}")
        else:
            print(f"  ✅ All dates parsed successfully")
        
        # Check for suspicious dates
        if not parsed.isna().all():
            valid_dates = parsed.dropna()
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            
            print(f"  Date range: {min_date.date()} → {max_date.date()}")
            
            # Flag suspicious dates
            if min_date.year < 1900:
                issues_found = True
                print(f"  ⚠️  ISSUE: Dates before 1900 detected")
            
            if max_date > pd.Timestamp.now():
                issues_found = True
                print(f"  ⚠️  ISSUE: Future dates detected")
            
            # Check for placeholder dates
            placeholder_1900 = (parsed.dt.year == 1900).sum()
            if placeholder_1900 > 0:
                issues_found = True
                print(f"  ⚠️  PLACEHOLDER: {placeholder_1900:,} dates set to 1900-01-01")
    
    print("\n" + "="*80)
    
    if issues_found:
        print("❌ DATE QUALITY ISSUES DETECTED")
        print("\nRECOMMENDED ACTIONS:")
        print("1. Use improved parse_date_safely() function")
        print("2. Filter out placeholder dates (1900-01-01)")
        print("3. Handle equipment with missing Kurulum_Tarihi separately")
    else:
        print("✅ ALL DATES PARSE CORRECTLY")
    
    print("="*80)


def analyze_kurulum_tarihi_quality(df_fault: pd.DataFrame, df_healthy: pd.DataFrame):
    """Deep dive on Kurulum_Tarihi quality"""
    
    print("\n" + "="*80)
    print("KURULUM_TARIHI DEEP DIVE")
    print("="*80)
    
    # Combine data
    all_data = []
    
    if "Sebekeye_Baglanma_Tarihi" in df_fault.columns:
        df_fault["Kurulum_Tarihi"] = df_fault["Sebekeye_Baglanma_Tarihi"].apply(parse_date_safely)
        all_data.append(("Fault", df_fault["Kurulum_Tarihi"]))
    
    if "Sebekeye_Baglanma_Tarihi" in df_healthy.columns:
        df_healthy["Kurulum_Tarihi"] = df_healthy["Sebekeye_Baglanma_Tarihi"].apply(parse_date_safely)
        all_data.append(("Healthy", df_healthy["Kurulum_Tarihi"]))
    
    for source, dates in all_data:
        print(f"\n{source} Equipment:")
        
        total = len(dates)
        missing = dates.isna().sum()
        valid = total - missing
        
        print(f"  Total: {total:,}")
        print(f"  Valid dates: {valid:,} ({100*valid/total:.1f}%)")
        print(f"  Missing: {missing:,} ({100*missing/total:.1f}%)")
        
        if valid > 0:
            valid_dates = dates.dropna()
            
            # Year distribution
            year_dist = valid_dates.dt.year.value_counts().sort_index()
            
            print(f"\n  Installation Year Distribution:")
            suspicious_years = []
            for year, count in year_dist.items():
                pct = 100 * count / valid
                marker = ""
                if year < 1950:
                    marker = " ⚠️  SUSPICIOUS"
                    suspicious_years.append(year)
                elif year > 2025:
                    marker = " ⚠️  FUTURE"
                    suspicious_years.append(year)
                
                print(f"    {year}: {count:>6,} ({pct:>5.1f}%){marker}")
            
            # Age analysis
            current_date = pd.Timestamp.now()
            ages = (current_date - valid_dates).dt.days / 365.25
            
            print(f"\n  Equipment Age Statistics:")
            print(f"    Median: {ages.median():.1f} years")
            print(f"    Mean: {ages.mean():.1f} years")
            print(f"    Min: {ages.min():.1f} years")
            print(f"    Max: {ages.max():.1f} years")
            
            # Flag issues
            too_old = (ages > 60).sum()
            negative = (ages < 0).sum()
            
            if too_old > 0:
                print(f"    ⚠️  {too_old:,} equipment older than 60 years")
            if negative > 0:
                print(f"    ⚠️  {negative:,} equipment with future installation dates")
            
            # Placeholder detection
            placeholder_1900 = (valid_dates.dt.year == 1900).sum()
            if placeholder_1900 > 0:
                print(f"    ⚠️  {placeholder_1900:,} dates set to 1900-01-01 (likely placeholders)")
    
    print("\n" + "="*80)


# =============================================================================
# MAIN DIAGNOSTIC
# =============================================================================

def main():
    print("="*80)
    print("POF3 DATA QUALITY DIAGNOSTIC")
    print("="*80)
    print()
    
    # Load data
    print("Loading data...")
    df_fault = pd.read_excel(DATA_PATHS["fault_data"])
    df_healthy = pd.read_excel(DATA_PATHS["healthy_data"])
    
    df_fault.columns = [c.strip() for c in df_fault.columns]
    df_healthy.columns = [c.strip() for c in df_healthy.columns]
    
    print(f"✅ Fault records: {len(df_fault):,}")
    print(f"✅ Healthy records: {len(df_healthy):,}")
    print()
    
    # Run analyses
    print("Running diagnostics...\n")
    
    # 1. Fault cause analysis
    cause_df = analyze_fault_causes(df_fault)
    
    # 2. Date parsing validation
    analyze_date_parsing(df_fault, df_healthy)
    
    # 3. Kurulum_Tarihi deep dive
    analyze_kurulum_tarihi_quality(df_fault, df_healthy)
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nNEXT STEPS:")
    print("1. Review fault_cause_analysis.csv")
    print("2. Update REAL_FAILURE_CODES and EXCLUDE_CODES in your script")
    print("3. Replace parse_date_safely() with the improved version")
    print("4. Run production pipeline with fixes")
    print("="*80)


if __name__ == "__main__":
    main()