import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data' / 'final'

SPECIES_TO_CHECK = ['Козуля', 'Кабан', 'Лось', 'Олень благородний']
MIN_YEARS = 10

# contamination = лише поріг відсічки. У sklearn він НЕ входить у score_samples,
# тому ранжування аномалій однакове для всіх значень — змінюється тільки
# скільки точок позначено. Grid показує цей trade-off (precision/recall),
# а не "обґрунтовує" вибір. 0.05 = топ-5% точок.
CONTAMINATION_GRID = [0.02, 0.05, 0.10]
DEFAULT_CONTAMINATION = 0.05  # топ-5% як баланс precision/recall


def prepare_features(species='Кабан'):
    """Готує features для Isolation Forest."""
    populations = pd.read_csv(DATA / 'populations_final.csv')

    df = populations[
        (populations['metric'] == 'count') &
        (populations['species_canonical'] == species) &
        (populations['value'].notna())
    ].copy()

    rows = []
    for host in df['host_canonical'].unique():
        ts = df[df['host_canonical'] == host].sort_values('year').set_index('year')['value']

        if len(ts) < MIN_YEARS:
            continue

        mean_val = ts.mean()
        std_val  = ts.std()

        for year in ts.index:
            val = ts[year]
            change_pct = ((val - ts[year - 1]) / ts[year - 1] * 100) if year - 1 in ts.index else np.nan
            deviation  = (val - mean_val) / std_val if std_val > 0 else 0

            rows.append({
                'host':       host,
                'year':       year,
                'value':      val,
                'change_pct': change_pct,
                'deviation':  deviation,
            })

    return pd.DataFrame(rows).dropna()


def sensitivity_analysis(species='Кабан'):
    """Перебір contamination — показує trade-off, а не "доказ" вибору.

    ВАЖЛИВО: contamination не впливає на score_samples у sklearn, тому
    ранжування топ-аномалій ІДЕНТИЧНЕ для всіх значень — змінюється лише
    поріг (скільки точок позначено). Тому "стабільність рангів" тут
    гарантована за побудовою і не є свідченням якості моделі.
    Реальна валідація — збіг аномалій з відомими подіями (АЧС, реформа).

    Для кожного contamination показує:
    - скільки точок ловиться (це й змінюється з порогом)
    - топ-5 найбільш аномальних (за score)
    """
    df = prepare_features(species)
    features = df[['value', 'change_pct', 'deviation']].values

    print(f"\n=== Sensitivity analysis: {species} ===")
    print(f"Точок в датасеті: {len(df)}")
    print(f"{'contam':>8} | {'аномалій':>10} | топ-5 (host, year, change_pct)")
    print("-" * 80)

    results_by_contam = {}

    for contam in CONTAMINATION_GRID:
        model = IsolationForest(
            contamination=contam,
            random_state=42,
            n_estimators=100,
        )
        df_iter = df.copy()
        df_iter['anomaly'] = model.fit_predict(features)
        df_iter['score']   = model.score_samples(features)

        anomalies = df_iter[df_iter['anomaly'] == -1].sort_values('score')

        # топ-5 найгірших
        top5 = anomalies.head(5)
        top5_str = ', '.join([
            f"{r['host'][:15]}/{int(r['year'])}/{r['change_pct']:+.0f}%"
            for _, r in top5.iterrows()
        ])

        print(f"{contam:>8.2f} | {len(anomalies):>10} | {top5_str}")
        results_by_contam[contam] = anomalies

    # Збіг множин при різних порогах = 100% за побудовою (ранг інваріантний).
    # Лишаємо як інформативний вивід, але НЕ як доказ якості.
    strict_anomalies = results_by_contam[CONTAMINATION_GRID[0]]
    if len(strict_anomalies) > 0:
        keys_strict = set(zip(strict_anomalies['host'], strict_anomalies['year']))
        keys_loose = set(zip(results_by_contam[CONTAMINATION_GRID[-1]]['host'],
                             results_by_contam[CONTAMINATION_GRID[-1]]['year']))
        overlap = len(keys_strict & keys_loose) / len(keys_strict) * 100
        print(f"\ncontam=0.02 ⊆ contam=0.10: {overlap:.0f}% "
              f"(очікувано 100% — ранг інваріантний до contamination)")

    return df, results_by_contam


def plot_score_distribution(df, species):
    """← НОВЕ: гістограма score з позначками порогів для різних contamination."""
    features = df[['value', 'change_pct', 'deviation']].values
    model = IsolationForest(contamination=DEFAULT_CONTAMINATION,
                            random_state=42, n_estimators=100)
    model.fit(features)
    scores = model.score_samples(features)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(scores, bins=40, color='#2E86AB', alpha=0.7, edgecolor='black')

    # Позначки порогів для різних contamination
    colors_thr = ['#3BB273', '#F4A261', '#E84855']
    for contam, color in zip(CONTAMINATION_GRID, colors_thr):
        threshold = np.percentile(scores, contam * 100)
        ax.axvline(threshold, color=color, linestyle='--',
                   label=f'поріг contam={contam}')

    ax.set_xlabel('Anomaly score (нижче = більш аномальна точка)')
    ax.set_ylabel('Кількість точок')
    ax.set_title(f'Розподіл anomaly scores — {species}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ROOT / 'reports' / 'figures' / f'anomaly_score_dist_{species}.png', dpi=150)
    plt.show()


def detect_anomalies(species='Кабан', contamination=DEFAULT_CONTAMINATION):
    """Фінальне виявлення аномалій з обраним contamination."""
    df = prepare_features(species)
    features = df[['value', 'change_pct', 'deviation']].values

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )

    df['anomaly'] = model.fit_predict(features)
    df['score']   = model.score_samples(features)

    anomalies = df[df['anomaly'] == -1].sort_values('score')

    print(f"\n=== Аномалії {species} (contam={contamination}) ===")
    print(f"Всього точок: {len(df)}, аномалій: {len(anomalies)}")
    print(anomalies[['host','year','value','change_pct','deviation']].to_string(index=False))

    return df, anomalies


def plot_anomalies_from_data(df, anomalies, species):
    """Графік: популяція по роках з позначкою аномалій."""
    top_hosts = anomalies['host'].value_counts().head(6).index.tolist()

    if not top_hosts:
        print(f"Немає аномалій для візуалізації у {species}")
        return

    n = len(top_hosts)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for i, host in enumerate(top_hosts):
        ax = axes[i]
        host_data = df[df['host'] == host].sort_values('year')
        host_anom = anomalies[anomalies['host'] == host]

        ax.plot(host_data['year'], host_data['value'],
                marker='o', markersize=3, color='#2E86AB')
        ax.scatter(host_anom['year'], host_anom['value'],
                   color='red', zorder=5, s=60, label='аномалія')

        ax.set_title(host, fontsize=9)
        ax.set_ylabel('голів')
        ax.legend(fontsize=8)

    # Прибираємо порожні subplots
    for j in range(len(top_hosts), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'Anomaly Detection — {species}', fontsize=12)
    plt.tight_layout()
    plt.savefig(ROOT / 'reports' / 'figures' / f'anomaly_{species}.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    import os
    import sys
    os.makedirs(ROOT / 'reports' / 'figures', exist_ok=True)

    species_list = sys.argv[1:] if len(sys.argv) > 1 else SPECIES_TO_CHECK

    for species in species_list:
        # Крок 1: sensitivity analysis для обґрунтування contamination
        df_full, results_by_contam = sensitivity_analysis(species)

        # Крок 2: розподіл score
        plot_score_distribution(df_full, species)

        # Крок 3: фінальне виявлення з обраним contamination
        df, anomalies = detect_anomalies(species, contamination=DEFAULT_CONTAMINATION)
        anomalies.to_csv(DATA / f'anomalies_{species}.csv', index=False)

        if len(anomalies) > 0:
            plot_anomalies_from_data(df, anomalies, species)

    print("\nГотово. Результати збережено в data/final/")