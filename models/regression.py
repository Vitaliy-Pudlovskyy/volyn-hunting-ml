import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit  # ← FIX: додано TimeSeriesSplit
import lightgbm as lgb
import xgboost as xgb
import sys
import os
import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data' / 'final'
MODELS_DIR = ROOT / 'models'

LAG = 1

FEATURE_NAMES_UA = {
    'total_expenses':     'Загальні витрати',
    'expense_protection': 'Витрати на охорону',
    'expense_breeding':   'Витрати на відтворення',
    'expense_feeding':    'Витрати на підгодівлю',
}

# Результати нотбука — найкращі моделі без population_t
# УВАГА: ці моделі підбирались на CV з KFold (leakage).
# Після переходу на TimeSeriesSplit MAE буде вищим — це чесні цифри.
BEST_MODELS = {
    'Олень благородний': {
        'class':  'LightGBM',
        'params': {
            'n_estimators': 71, 'max_depth': 6,
            'learning_rate': 0.01147, 'num_leaves': 35,
            'subsample': 0.966, 'colsample_bytree': 0.990,
            'random_state': 42, 'verbose': -1,
        },
        'mae': 10.2,
    },
    'Козуля': {
        'class':  'MLP',
        'params': {
            'hidden_layer_sizes': (157, 78),
            'max_iter': 500, 'random_state': 42,
        },
        'mae': 18.6,
    },
    'Кабан': {
        'class':  'MLP',
        'params': {
            'hidden_layer_sizes': (249, 124),
            'max_iter': 500, 'random_state': 42,
        },
        'mae': 16.9,
    },
    'Фазан': {
        'class':  'LinearRegression',
        'params': {},
        'mae': 79.0,
    },
}

TARGET_SPECIES = list(BEST_MODELS.keys())
FEATURES_NO_POP = [
    'total_expenses', 'expense_protection',
    'expense_breeding', 'expense_feeding'
]
FEATURES_WITH_POP = ['population_t'] + FEATURES_NO_POP


def prepare_features(species='Козуля'):
    """Лагові features для регресії.
    X: витрати року T (rolling mean 3 роки)
    y: популяція року T+1
    """
    populations = pd.read_csv(DATA / 'populations_final.csv')
    finances    = pd.read_csv(DATA / 'finances_final.csv')

    pop = populations[
        (populations['metric'] == 'count') &
        (populations['species_canonical'] == species)
    ].groupby(['year', 'host_canonical'])['value'].mean().reset_index()
    pop.columns = ['year', 'host', 'population']

    fin = finances.groupby(['year', 'host_canonical', 'metric'])['value'].mean().reset_index()
    fin_pivot = fin.pivot_table(
        index=['year', 'host_canonical'],
        columns='metric',
        values='value'
    ).reset_index()
    fin_pivot.columns.name = None
    fin_pivot = fin_pivot.rename(columns={'host_canonical': 'host'})

    df = pop.merge(fin_pivot, on=['year', 'host'], how='inner')

    rows = []
    for host in df['host'].unique():
        host_df = df[df['host'] == host].sort_values('year')
        for i in range(len(host_df) - LAG):
            curr     = host_df.iloc[i]
            next_row = host_df.iloc[i + LAG]
            if next_row['year'] - curr['year'] != LAG:
                continue
            row = {
                'host':          host,
                'year':          curr['year'],
                'population_t':  curr['population'],
                'population_t1': next_row['population'],
            }
            for col in FEATURES_NO_POP:
                vals = [host_df.iloc[j].get(col, np.nan)
                        for j in range(max(0, i-2), i+1)]
                row[col] = np.nanmean(vals) if vals else np.nan
            rows.append(row)

    if not rows:
        return None

    result = pd.DataFrame(rows).dropna(
        subset=['population_t', 'population_t1', 'total_expenses']
    )
    return result


def clean(df):
    """Стандартне очищення після prepare_features."""
    df = df.copy()
    df['expense_feeding'] = df['expense_feeding'].fillna(0)
    required = ['population_t', 'total_expenses',
                'expense_protection', 'expense_breeding', 'population_t1']
    return df[df[required].notna().all(axis=1)]


def build_model(species):
    """Створює модель за BEST_MODELS конфігом."""
    cfg = BEST_MODELS[species]

    if cfg['class'] == 'LightGBM':
        return lgb.LGBMRegressor(**cfg['params'])

    elif cfg['class'] == 'MLP':
        return Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPRegressor(**cfg['params']))
        ])

    elif cfg['class'] == 'LinearRegression':
        return LinearRegression()

    elif cfg['class'] == 'XGBoost':
        return xgb.XGBRegressor(**cfg['params'])

    raise ValueError(f"Невідомий клас: {cfg['class']}")


def train_regression(species):
    """Навчає дві моделі для виду:
    1. з population_t  — прогноз абсолютної популяції
    2. без population_t — вплив інвестицій (delta target)

    FIX: TimeSeriesSplit замість дефолтного KFold —
    дані сортуються за роком, train завжди до року test.
    """
    df = prepare_features(species)
    if df is None:
        print(f"{species}: недостатньо даних")
        return None

    df = clean(df)
    df['delta'] = df['population_t1'] - df['population_t']

    # ← FIX: сортуємо за роком (всередині року — за host для детермінізму)
    # TimeSeriesSplit йде по порядку рядків, тому це КРИТИЧНО
    df = df.sort_values(['year', 'host']).reset_index(drop=True)

    # ← FIX: TimeSeriesSplit замість cv=5
    tscv = TimeSeriesSplit(n_splits=5)

    # ── Модель 1: прогноз з population_t ──────────────────
    X_with = df[FEATURES_WITH_POP].fillna(0)
    y_abs  = df['population_t1']

    m1 = LinearRegression()
    scores1 = cross_val_score(m1, X_with, y_abs,
                              cv=tscv,  # ← FIX
                              scoring='neg_mean_absolute_error')
    m1.fit(X_with, y_abs)
    mae1 = round(-scores1.mean(), 1)
    mae1_std = round(scores1.std(), 1)

    # ── Модель 2: вплив інвестицій (delta) ─────────────────
    X_no  = df[FEATURES_NO_POP].fillna(0)
    y_delta = df['delta']

    m2 = build_model(species)
    scores2 = cross_val_score(m2, X_no, y_delta,
                              cv=tscv,  # ← FIX
                              scoring='neg_mean_absolute_error')
    m2.fit(X_no, y_delta)
    mae2 = round(-scores2.mean(), 1)
    mae2_std = round(scores2.std(), 1)

    print(f"{species}:")
    print(f"  Прогноз популяції (з pop_t):  MAE={mae1} ± {mae1_std}")
    print(f"  Вплив інвестицій  (delta):    MAE={mae2} ± {mae2_std} "
          f"[{BEST_MODELS[species]['class']}]")

    return {
        'species':       species,
        'df':            df,
        'model_forecast': m1,
        'model_invest':   m2,
        'mae_forecast':   mae1,
        'mae_invest':     mae2,
        'features_forecast': FEATURES_WITH_POP,
        'features_invest':   FEATURES_NO_POP,
    }


def get_investment_impact(result):
    """SHAP values для моделі інвестицій."""
    species = result['species']
    model   = result['model_invest']
    df      = result['df']
    X       = df[FEATURES_NO_POP].fillna(0)
    cfg     = BEST_MODELS[species]

    if cfg['class'] in ('LightGBM', 'XGBoost'):
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)

    elif cfg['class'] == 'MLP':
        X_scaled   = model.named_steps['scaler'].transform(X)
        explainer  = shap.KernelExplainer(
            model.named_steps['model'].predict,
            shap.sample(X_scaled, 50)
        )
        shap_vals  = explainer.shap_values(X_scaled, nsamples=100)

    elif cfg['class'] == 'LinearRegression':
        explainer  = shap.LinearExplainer(model, X)
        shap_vals  = explainer.shap_values(X)

    impact = pd.DataFrame({
        'feature':    FEATURES_NO_POP,
        'feature_ua': [FEATURE_NAMES_UA[f] for f in FEATURES_NO_POP],
        'shap_mean':  np.abs(shap_vals).mean(axis=0),
        'shap_dir':   shap_vals.mean(axis=0),
    }).sort_values('shap_mean', ascending=False)

    return impact, shap_vals, explainer


def get_stability_threshold(result):
    """Мінімальні витрати щоб delta >= 0."""
    df = result['df']

    stable = df[df['delta'] >= 0]
    threshold = {}
    for f in FEATURES_NO_POP:
        threshold[f] = {
            'median_stable':   round(stable[f].median(), 1),
            'median_all':      round(df[f].median(), 1),
            'feature_ua':      FEATURE_NAMES_UA[f],
        }

    return threshold


def plot_results(result, impact, threshold):
    """3 графіки."""
    species = result['species']
    df      = result['df']

    os.makedirs(ROOT / 'reports' / 'figures', exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    by_year = df.groupby('year')['population_t'].mean()
    ax.plot(by_year.index, by_year.values,
            marker='o', markersize=3, color='#2E86AB', label='факт')

    X_with = df[FEATURES_WITH_POP].fillna(0)
    df['forecast'] = result['model_forecast'].predict(X_with)
    fc_year = df.groupby('year')['forecast'].mean()
    ax.plot(fc_year.index, fc_year.values,
            linestyle='--', color='#E84855', label='прогноз')

    ax.axvline(2022, color='gray', linestyle=':', alpha=0.7,
               label='заборона полювання')
    ax.set_title(f'Популяція {species}')
    ax.set_ylabel('голів')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = ['#3BB273' if d >= 0 else '#E84855'
              for d in impact['shap_dir']]
    ax.barh(impact['feature_ua'], impact['shap_mean'],
            color=colors, alpha=0.8)
    ax.set_title(f'Вплив інвестицій на зміну популяції\n{species}')
    ax.set_xlabel('середній |SHAP| (голів)')
    ax.axvline(0, color='black', linewidth=0.5)

    ax = axes[2]
    features_ua   = [v['feature_ua']    for v in threshold.values()]
    median_all    = [v['median_all']    for v in threshold.values()]
    median_stable = [v['median_stable'] for v in threshold.values()]

    x = np.arange(len(features_ua))
    w = 0.35
    ax.bar(x - w/2, median_all,    w, label='середнє всі роки',
           color='#2E86AB', alpha=0.7)
    ax.bar(x + w/2, median_stable, w, label='стабільні роки (delta≥0)',
           color='#3BB273', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(features_ua, rotation=20, ha='right', fontsize=8)
    ax.set_title(f'Мінімальні витрати для стабільності\n{species}')
    ax.set_ylabel('тис. грн')
    ax.legend(fontsize=8)

    plt.suptitle(f'Регресійний аналіз — {species}', fontsize=13)
    plt.tight_layout()
    plt.savefig(
        ROOT / 'reports' / 'figures' / f'regression_{species}.png',
        dpi=150
    )
    plt.show()


if __name__ == '__main__':
    species_list = sys.argv[1:] if len(sys.argv) > 1 else TARGET_SPECIES

    for species in species_list:
        print(f"\n{'='*50}")
        print(f"ВИД: {species}")
        print('='*50)

        result = train_regression(species)
        if result is None:
            continue

        impact, shap_vals, explainer = get_investment_impact(result)
        threshold = get_stability_threshold(result)

        print(f"\nВплив інвестицій (SHAP):")
        print(impact[['feature_ua', 'shap_mean', 'shap_dir']].to_string(index=False))

        print(f"\nПоріг стабільності:")
        for f, v in threshold.items():
            print(f"  {v['feature_ua']}: "
                  f"факт={v['median_all']} / стабільні роки={v['median_stable']}")

        plot_results(result, impact, threshold)

    print("\nГотово.")