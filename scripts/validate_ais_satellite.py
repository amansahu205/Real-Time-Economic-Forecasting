#!/usr/bin/env python3
"""
Validate AIS Data Against Satellite Detections

Compares AIS ship counts with YOLO satellite detections to validate both data sources.
Creates correlation plots and time series comparisons.

Usage:
    python validate_ais_satellite.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def load_satellite_data():
    """Load satellite ship detections from YOLO"""
    csv_path = RESULTS_DIR / "annotations" / "google_earth_ports" / "Port_of_LA" / "all_years_summary.csv"
    
    if not csv_path.exists():
        print(f"❌ Satellite data not found: {csv_path}")
        print("   Please run: python scripts/process_satellite_data.py --dataset ports")
        return None
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['image_name'].str.extract(r'(\d{4})-\d+')[0] + '-01-01')
    
    # Get ship counts
    df = df[['date', 'total_detections']].rename(columns={'total_detections': 'satellite_ships'})
    
    return df


def load_ais_data():
    """Load AIS daily metrics"""
    csv_path = PROJECT_ROOT / "data" / "processed" / "ais" / "Port_of_LA_ais_daily.csv"
    
    if not csv_path.exists():
        print(f"❌ AIS data not found: {csv_path}")
        print("   Please run: python scripts/process_ais_data.py")
        return None
    
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df[['date', 'unique_ships']].rename(columns={'unique_ships': 'ais_ships'})
    
    return df


def merge_datasets(satellite, ais):
    """
    Merge satellite and AIS data on date
    
    Uses nearest-neighbor matching with 3-day tolerance since
    satellite images are sparse and may not align exactly with AIS dates.
    """
    # Sort both datasets
    satellite = satellite.sort_values('date')
    ais = ais.sort_values('date')
    
    # Merge with tolerance (within ±3 days)
    merged = pd.merge_asof(
        satellite,
        ais,
        on='date',
        tolerance=pd.Timedelta('3d'),
        direction='nearest'
    )
    
    # Remove rows with missing data
    merged = merged.dropna()
    
    return merged


def calculate_statistics(merged):
    """Calculate validation statistics"""
    stats_dict = {}
    
    # Correlation
    stats_dict['correlation'] = merged['satellite_ships'].corr(merged['ais_ships'])
    stats_dict['spearman'] = merged['satellite_ships'].corr(merged['ais_ships'], method='spearman')
    
    # Error metrics
    stats_dict['mae'] = np.mean(np.abs(merged['satellite_ships'] - merged['ais_ships']))
    stats_dict['rmse'] = np.sqrt(np.mean((merged['satellite_ships'] - merged['ais_ships'])**2))
    stats_dict['mape'] = np.mean(np.abs((merged['satellite_ships'] - merged['ais_ships']) / merged['ais_ships'])) * 100
    
    # Bias
    stats_dict['mean_diff'] = np.mean(merged['satellite_ships'] - merged['ais_ships'])
    stats_dict['median_diff'] = np.median(merged['satellite_ships'] - merged['ais_ships'])
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(merged['satellite_ships'], merged['ais_ships'])
    stats_dict['t_statistic'] = t_stat
    stats_dict['p_value'] = p_value
    
    return stats_dict


def create_validation_plots(merged, stats_dict):
    """Create comprehensive validation plots"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Scatter plot with regression line
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(merged['ais_ships'], merged['satellite_ships'], alpha=0.6, s=100)
    
    # Add regression line
    z = np.polyfit(merged['ais_ships'], merged['satellite_ships'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(merged['ais_ships'].min(), merged['ais_ships'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.1f}')
    
    # Add perfect match line
    max_val = max(merged['ais_ships'].max(), merged['satellite_ships'].max())
    ax1.plot([0, max_val], [0, max_val], 'k-', linewidth=1, alpha=0.3, label='Perfect match')
    
    ax1.set_xlabel('AIS Ship Count', fontsize=12)
    ax1.set_ylabel('Satellite Ship Count (YOLO)', fontsize=12)
    ax1.set_title(f'AIS vs Satellite Validation\nCorrelation: {stats_dict["correlation"]:.3f}', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time series comparison
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(merged['date'], merged['satellite_ships'], 'o-', label='Satellite (YOLO)', markersize=8, linewidth=2)
    ax2.plot(merged['date'], merged['ais_ships'], 's-', label='AIS', markersize=6, linewidth=2, alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Ship Count', fontsize=12)
    ax2.set_title('Time Series Comparison', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 3. Difference distribution
    ax3 = plt.subplot(2, 3, 3)
    diff = merged['satellite_ships'] - merged['ais_ships']
    ax3.hist(diff, bins=20, edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero difference')
    ax3.axvline(diff.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {diff.mean():.1f}')
    ax3.set_xlabel('Difference (Satellite - AIS)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Bland-Altman plot
    ax4 = plt.subplot(2, 3, 4)
    mean_ships = (merged['satellite_ships'] + merged['ais_ships']) / 2
    diff_ships = merged['satellite_ships'] - merged['ais_ships']
    
    ax4.scatter(mean_ships, diff_ships, alpha=0.6, s=100)
    ax4.axhline(diff_ships.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {diff_ships.mean():.1f}')
    ax4.axhline(diff_ships.mean() + 1.96*diff_ships.std(), color='gray', linestyle='--', linewidth=1, label='±1.96 SD')
    ax4.axhline(diff_ships.mean() - 1.96*diff_ships.std(), color='gray', linestyle='--', linewidth=1)
    ax4.set_xlabel('Mean Ship Count', fontsize=12)
    ax4.set_ylabel('Difference (Satellite - AIS)', fontsize=12)
    ax4.set_title('Bland-Altman Plot', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Percentage error
    ax5 = plt.subplot(2, 3, 5)
    pct_error = ((merged['satellite_ships'] - merged['ais_ships']) / merged['ais_ships']) * 100
    ax5.scatter(merged['date'], pct_error, alpha=0.6, s=100)
    ax5.axhline(0, color='red', linestyle='--', linewidth=2)
    ax5.axhline(pct_error.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {pct_error.mean():.1f}%')
    ax5.set_xlabel('Date', fontsize=12)
    ax5.set_ylabel('Percentage Error (%)', fontsize=12)
    ax5.set_title('Percentage Error Over Time', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 6. Statistics summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    VALIDATION STATISTICS
    {'='*40}
    
    Correlation Metrics:
    • Pearson r: {stats_dict['correlation']:.3f}
    • Spearman ρ: {stats_dict['spearman']:.3f}
    
    Error Metrics:
    • MAE: {stats_dict['mae']:.2f} ships
    • RMSE: {stats_dict['rmse']:.2f} ships
    • MAPE: {stats_dict['mape']:.1f}%
    
    Bias:
    • Mean difference: {stats_dict['mean_diff']:.2f}
    • Median difference: {stats_dict['median_diff']:.2f}
    
    Statistical Test:
    • t-statistic: {stats_dict['t_statistic']:.3f}
    • p-value: {stats_dict['p_value']:.4f}
    
    Sample Size: {len(merged)} matched pairs
    """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    return fig


def print_summary(satellite, ais, merged, stats_dict):
    """Print validation summary"""
    print("\n" + "="*60)
    print("🔍 AIS VS SATELLITE VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\n📊 Data Coverage:")
    print(f"   Satellite data points: {len(satellite)}")
    print(f"   AIS data points: {len(ais)}")
    print(f"   Matched pairs: {len(merged)}")
    print(f"   Match rate: {len(merged)/len(satellite)*100:.1f}%")
    
    print(f"\n📈 Correlation:")
    print(f"   Pearson correlation: {stats_dict['correlation']:.3f}")
    print(f"   Spearman correlation: {stats_dict['spearman']:.3f}")
    
    if stats_dict['correlation'] > 0.8:
        print(f"   ✅ Excellent agreement!")
    elif stats_dict['correlation'] > 0.6:
        print(f"   ✅ Good agreement")
    else:
        print(f"   ⚠️  Moderate agreement")
    
    print(f"\n📊 Error Metrics:")
    print(f"   Mean Absolute Error: {stats_dict['mae']:.2f} ships")
    print(f"   Root Mean Square Error: {stats_dict['rmse']:.2f} ships")
    print(f"   Mean Absolute Percentage Error: {stats_dict['mape']:.1f}%")
    
    print(f"\n⚖️  Bias:")
    print(f"   Mean difference: {stats_dict['mean_diff']:.2f} ships")
    if abs(stats_dict['mean_diff']) < 5:
        print(f"   ✅ Low bias")
    else:
        print(f"   ⚠️  Moderate bias")
    
    print(f"\n📊 Statistical Test:")
    print(f"   t-statistic: {stats_dict['t_statistic']:.3f}")
    print(f"   p-value: {stats_dict['p_value']:.4f}")
    if stats_dict['p_value'] > 0.05:
        print(f"   ✅ No significant difference (p > 0.05)")
    else:
        print(f"   ⚠️  Significant difference detected (p < 0.05)")
    
    print(f"\n💡 Interpretation:")
    if stats_dict['correlation'] > 0.7 and stats_dict['mape'] < 20:
        print(f"   ✅ AIS and satellite data are highly consistent")
        print(f"   ✅ Both sources can be used with confidence")
        print(f"   ✅ Cross-validation successful")
    elif stats_dict['correlation'] > 0.5:
        print(f"   ✅ Moderate agreement between sources")
        print(f"   ⚠️  Some discrepancies may need investigation")
    else:
        print(f"   ⚠️  Low agreement - check data quality")


def main():
    print("="*60)
    print("🔍 VALIDATING AIS VS SATELLITE DATA")
    print("="*60)
    
    # Load data
    print("\n📥 Loading data...")
    satellite = load_satellite_data()
    ais = load_ais_data()
    
    if satellite is None or ais is None:
        print("\n❌ Cannot proceed without both datasets")
        return
    
    print(f"   ✅ Loaded {len(satellite)} satellite observations")
    print(f"   ✅ Loaded {len(ais)} AIS observations")
    
    # Merge datasets
    print("\n🔗 Merging datasets...")
    merged = merge_datasets(satellite, ais)
    print(f"   ✅ Matched {len(merged)} pairs")
    
    if len(merged) < 5:
        print("\n❌ Not enough matched pairs for validation")
        return
    
    # Calculate statistics
    print("\n📊 Calculating statistics...")
    stats_dict = calculate_statistics(merged)
    
    # Create plots
    print("\n📊 Creating validation plots...")
    fig = create_validation_plots(merged, stats_dict)
    
    # Save plot
    output_file = RESULTS_DIR / "ais_satellite_validation.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_file}")
    
    # Print summary
    print_summary(satellite, ais, merged, stats_dict)
    
    # Save statistics
    stats_df = pd.DataFrame([stats_dict])
    stats_file = RESULTS_DIR / "ais_satellite_validation_stats.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"\n✅ Saved statistics: {stats_file}")
    
    # Save merged data
    merged_file = PROJECT_ROOT / "data" / "processed" / "ais" / "ais_satellite_merged.csv"
    merged.to_csv(merged_file, index=False)
    print(f"✅ Saved merged data: {merged_file}")
    
    print("\n" + "="*60)
    print("✅ VALIDATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
