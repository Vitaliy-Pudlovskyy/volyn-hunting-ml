import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score  # ← FIX

import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data' / 'final'

RANDOM_STATE = 42
K_RANGE = range(2, 9)  # ← FIX: будемо перебирати k=2..8


def prepare_features():
    """Готує features для кластеризації господарств.
    Одне господарство = один рядок з числовими характеристиками.
    """
    populations = pd.read_csv(DATA / 'populations_final.csv')
    finances    = pd.read_csv(DATA / 'finances_final.csv')

    pop = populations[populations['metric'] == 'count'].copy()

    rows = []
    for host in pop['host_canonical'].unique():
        host_data = pop[pop['host_canonical'] == host]

        koz = host_data[host_data['species_canonical'] == 'Козуля']['value']
        koz_mean = koz.mean() if len(koz) > 0 else 0

        kab = host_data[host_data['species_canonical'] == 'Кабан']['value']
        kab_mean = kab.mean() if len(kab) > 0 else 0

        los = host_data[host_data['species_canonical'] == 'Лось']['value']
        los_mean = los.mean() if len(los) > 0 else 0

        n_years = host_data['year'].nunique()

        koz_ts = host_data[host_data['species_canonical'] == 'Козуля'].set_index('year')['value'].dropna()
        if len(koz_ts) >= 3:
            trend = np.polyfit(koz_ts.index, koz_ts.values, 1)[0]
        else:
            trend = 0

        fin = finances[finances['host_canonical'] == host]
        expenses = fin[fin['metric'] == 'total_expenses']['value'].mean()
        expenses = expenses if not np.isnan(expenses) else 0

        rows.append({
            'host':      host,
            'koz_mean':  koz_mean,
            'kab_mean':  kab_mean,
            'los_mean':  los_mean,
            'n_years':   n_years,
            'koz_trend': trend,
            'expenses':  expenses,
        })

    df = pd.DataFrame(rows).set_index('host')

    # ← FIX: виключаємо резерв і інші domain-outliers перед кластеризацією
    # Державний мисливський резерв — це не мисливське господарство,
    # а централізована установа для переселення тварин.
    # Включення спотворює кластеризацію (singleton cluster з silhouette=0.78).
    EXCLUDE_HOSTS = ['Державний мисливський резерв']
    df = df.drop(index=[h for h in EXCLUDE_HOSTS if h in df.index])

    print(f"Виключено domain-outliers: {EXCLUDE_HOSTS}")
    print(f"Господарств після фільтру: {len(df)}")

    return df


def choose_k(X_scaled):
    """← НОВЕ: вибір оптимального k через Elbow + Silhouette.

    Повертає dict з результатами обох методів і рекомендованим k.
    """
    inertias = []
    silhouettes = []

    for k in K_RANGE:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_scaled)

        inertias.append(model.inertia_)
        # silhouette потребує мінімум 2 кластери і не всі точки в одному
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)

    # Найкраще k за silhouette = максимум score
    best_k_sil = list(K_RANGE)[np.argmax(silhouettes)]

    print("\n=== Вибір оптимального k ===")
    print(f"{'k':>3} | {'inertia':>10} | {'silhouette':>10}")
    print("-" * 35)
    for k, inertia, sil in zip(K_RANGE, inertias, silhouettes):
        marker = " ← найкращий" if k == best_k_sil else ""
        print(f"{k:>3} | {inertia:>10.1f} | {sil:>10.3f}{marker}")

    print(f"\nРекомендоване k за silhouette: {best_k_sil}")

    return {
        'k_range':     list(K_RANGE),
        'inertias':    inertias,
        'silhouettes': silhouettes,
        'best_k':      best_k_sil,
    }


def plot_k_selection(k_results):
    """← НОВЕ: графіки elbow і silhouette."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Elbow
    ax = axes[0]
    ax.plot(k_results['k_range'], k_results['inertias'],
            marker='o', color='#2E86AB')
    ax.axvline(k_results['best_k'], color='#E84855', linestyle='--', alpha=0.5,
               label=f"k={k_results['best_k']} (silhouette)")
    ax.set_xlabel('k (кількість кластерів)')
    ax.set_ylabel('Inertia (сума квадратів відстаней)')
    ax.set_title('Elbow method')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Silhouette
    ax = axes[1]
    ax.plot(k_results['k_range'], k_results['silhouettes'],
            marker='o', color='#3BB273')
    ax.axvline(k_results['best_k'], color='#E84855', linestyle='--', alpha=0.5,
               label=f"k={k_results['best_k']} (макс)")
    ax.set_xlabel('k (кількість кластерів)')
    ax.set_ylabel('Silhouette score')
    ax.set_title('Silhouette method')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Вибір оптимальної кількості кластерів', fontsize=12)
    plt.tight_layout()
    plt.savefig(ROOT / 'reports' / 'figures' / 'clustering_k_selection.png', dpi=150)
    plt.show()


def cluster_hosts(df, n_clusters):
    """Запускає K-Means з обраним k і повертає DataFrame з кластерами."""
    scaler = StandardScaler()
    features = ['koz_mean', 'kab_mean', 'los_mean', 'n_years', 'koz_trend', 'expenses']
    X = scaler.fit_transform(df[features].fillna(0))

    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df['cluster'] = model.fit_predict(X)

    print(f"\n=== Кластери господарств (k={n_clusters}) ===")
    for c in range(n_clusters):
        hosts = df[df['cluster'] == c].index.tolist()
        stats = df[df['cluster'] == c][['koz_mean','kab_mean','n_years','expenses']].mean()
        print(f"\nКластер {c} ({len(hosts)} господарств):")
        print(f"  Козуля середня:  {stats['koz_mean']:.0f}")
        print(f"  Кабан середня:   {stats['kab_mean']:.0f}")
        print(f"  Років звітності: {stats['n_years']:.0f}")
        print(f"  Витрати середні: {stats['expenses']:.0f}")
        print(f"  Господарства: {', '.join(hosts[:5])}{'...' if len(hosts)>5 else ''}")

    return df, model, scaler, X


def plot_clusters(df, X, n_clusters):
    """Візуалізація кластерів через PCA (2D проекція)."""
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(12, 8))

    # Палітра адаптована під число кластерів
    colors = ['#2E86AB', '#E84855', '#3BB273', '#F4A261',
              '#9B5DE5', '#FEE440', '#00BBF9', '#F15BB5']

    for c in range(n_clusters):
        mask = df['cluster'] == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=colors[c % len(colors)], label=f'Кластер {c}',
                   s=80, alpha=0.7)

        for i, host in enumerate(df[mask].index):
            ax.annotate(host[:15], (X_2d[mask][i, 0], X_2d[mask][i, 1]),
                       fontsize=6, alpha=0.7)

    ax.set_xlabel(f'PC1 ({explained[0]:.1f}% варіації)')
    ax.set_ylabel(f'PC2 ({explained[1]:.1f}% варіації)')
    ax.set_title(f'Кластеризація господарств Волині (K-Means k={n_clusters} + PCA)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(ROOT / 'reports' / 'figures' / 'clustering.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    import os
    os.makedirs(ROOT / 'reports' / 'figures', exist_ok=True)

    print("Підготовка features...")
    df = prepare_features()
    print(f"Господарств: {len(df)}")

    # Скейлимо один раз для вибору k
    scaler_init = StandardScaler()
    features = ['koz_mean', 'kab_mean', 'los_mean', 'n_years', 'koz_trend', 'expenses']
    X_scaled = scaler_init.fit_transform(df[features].fillna(0))

    # Крок 1: вибір k
    k_results = choose_k(X_scaled)
    plot_k_selection(k_results)

    best_k = k_results['best_k']

    # Крок 2: кластеризація з обраним k
    print(f"\nКластеризація з k={best_k}...")
    df_clustered, model, scaler, X = cluster_hosts(df, best_k)

    # Крок 3: візуалізація
    print("\nВізуалізація...")
    plot_clusters(df_clustered, X, best_k)

    df_clustered.to_csv(DATA / 'clusters_hosts.csv')
    print("\nЗбережено в data/final/clusters_hosts.csv")