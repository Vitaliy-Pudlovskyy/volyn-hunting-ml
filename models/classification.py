import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, TimeSeriesSplit  # FIX: TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data' / 'final'

HARVEST_THRESHOLD = 0.15
TARGET_SPECIES = ['Козуля', 'Кабан']

FEATURE_NAMES_UA = {
    'harvest_rate_lag': 'Відстріл/попул. (минулий рік)',
    'population':       'Поточна популяція',
    'pop_trend':        'Тренд популяції',
}


def prepare_features(species='Козуля'):
    populations = pd.read_csv(DATA / 'populations_final.csv')
    harvest     = pd.read_csv(DATA / 'harvest_final.csv')

    pop = populations[
        (populations['metric'] == 'count') &
        (populations['species_canonical'] == species)
    ].groupby(['year', 'host_canonical'])['value'].mean().reset_index()
    pop.columns = ['year', 'host', 'population']

    harv = harvest[
        (harvest['metric'] == 'shot_heads') &
        (harvest['species_canonical'] == species) &
        (harvest['year'] <= 2021)
    ].groupby(['year', 'host_canonical'])['value'].mean().reset_index()
    harv.columns = ['year', 'host', 'shot_heads']

    df = pop.merge(harv, on=['year', 'host'], how='inner')
    df = df.dropna(subset=['population', 'shot_heads'])
    df = df[df['population'] > 0]

    df['harvest_rate'] = df['shot_heads'] / df['population']
    df['label'] = (df['harvest_rate'] >= HARVEST_THRESHOLD).astype(int)

    trends = []
    for host in df['host'].unique():
        ts = df[df['host'] == host].sort_values('year')
        trend = np.polyfit(ts['year'], ts['population'], 1)[0] if len(ts) >= 3 else 0
        trends.append({'host': host, 'pop_trend': trend})
    df = df.merge(pd.DataFrame(trends), on='host', how='left')

    # лаг = минулий рік того ж господарства (без витоку)
    df = df.sort_values(['host', 'year'])
    df['harvest_rate_lag'] = df.groupby('host')['harvest_rate'].shift(1)
    return df.dropna(subset=['harvest_rate_lag'])


def temporal_cv_evaluate(pipe, X, y, n_splits=5):
    """Чесна CV по часу (TimeSeriesSplit): train завжди СТРОГО до test.
    Модель не вчиться на майбутньому, щоб передбачити минуле — саме це робив
    shuffle=True (= витік, завищений F1). cross_val_predict тут не можна
    (не кожен рядок потрапляє в test), тому збираємо OOS-прогнози вручну.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    f1_per_fold = []
    oos = pd.Series(np.nan, index=X.index)

    for k, (tr, te) in enumerate(tscv.split(X), 1):
        if y.iloc[tr].nunique() < 2:
            # Реальність, не баг: позитивів мало й вони в пізніх роках,
            # тому ранні фолди їх не бачать. shuffle це маскував.
            print(f"    fold {k}: ПРОПУЩЕНО — у train один клас "
                  f"(поз. train={int(y.iloc[tr].sum())}, test={int(y.iloc[te].sum())})")
            continue
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        oos.iloc[te] = pred
        f1_k = f1_score(y.iloc[te], pred, zero_division=0)
        f1_per_fold.append(f1_k)
        print(f"    fold {k}: F1={f1_k:.3f} | поз. train={int(y.iloc[tr].sum())}, "
              f"test={int(y.iloc[te].sum())}")

    if not f1_per_fold:
        print("    -> жоден фолд не придатний: замало позитивів для часової CV")
        return float('nan'), float('nan'), oos
    return float(np.mean(f1_per_fold)), float(np.std(f1_per_fold)), oos


def train_and_evaluate(species='Козуля'):
    df = prepare_features(species)
    # FIX (критично): сортуємо за роком, щоб TimeSeriesSplit мав сенс
    df = df.sort_values(['year', 'host']).reset_index(drop=True)

    print(f"\n{species}: {len(df)} рядків")
    print(f"Стійких (0): {(df['label']==0).sum()}, Надмірних (1): {(df['label']==1).sum()}")

    features = ['harvest_rate_lag', 'population', 'pop_trend']
    X = df[features].fillna(0)
    y = df['label']

    pipelines = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(random_state=42, max_iter=1000,
                                       class_weight='balanced')),
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42,
                                           class_weight='balanced')),
        ]),
    }

    leaky_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, pipe in pipelines.items():
        print(f"\n{name}")
        leaky_f1 = cross_val_score(pipe, X, y, cv=leaky_cv, scoring='f1').mean()
        f1, f1_std, oos = temporal_cv_evaluate(pipe, X, y)

        print(f"  -> ЧЕСНИЙ F1 (TimeSeriesSplit): {f1:.3f} ± {f1_std:.3f}")
        print(f"     leaky baseline (shuffle):    {leaky_f1:.3f}"
              f"   <- завищення {leaky_f1 - f1:+.3f}")

        covered = oos.notna()
        if covered.sum() == 0:
            print("     (немає OOS-прогнозів — звіт неможливий)")
        else:
            print(classification_report(
                y[covered], oos[covered].astype(int),
                labels=[0, 1], target_names=['Стійкий', 'Надмірний'],
                digits=3, zero_division=0,
            ))

        pipe.fit(X, y)
        results[name] = {'pipeline': pipe, 'f1': round(f1, 3),
                         'f1_std': round(f1_std, 3), 'leaky_f1': round(leaky_f1, 3),
                         'y_pred_cv': oos}
    return df, results, features


def get_feature_importance(results, features):
    importance = {}
    rf_pipe = results['Random Forest']['pipeline']
    rf_imp = rf_pipe.named_steps['clf'].feature_importances_
    importance['Random Forest'] = pd.Series(rf_imp, index=features)

    lr_pipe = results['Logistic Regression']['pipeline']
    lr_imp = np.abs(lr_pipe.named_steps['clf'].coef_[0])
    if lr_imp.max() > 0:
        lr_imp = lr_imp / lr_imp.max() * rf_imp.max()
    importance['Logistic Regression'] = pd.Series(lr_imp, index=features)
    return importance


def plot_feature_importance(importance, species):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, (name, imp) in zip(axes, importance.items()):
        imp_sorted = imp.sort_values(ascending=True)
        labels_ua = [FEATURE_NAMES_UA.get(f, f) for f in imp_sorted.index]
        ax.barh(labels_ua, imp_sorted.values, color='#2E86AB', alpha=0.8)
        ax.set_title(f'{name}')
        ax.set_xlabel('важливість')
        ax.grid(True, alpha=0.3, axis='x')
    plt.suptitle(f'Feature importance — {species}', fontsize=12)
    plt.tight_layout()
    plt.savefig(ROOT / 'reports' / 'figures' / f'classification_importance_{species}.png', dpi=150)
    plt.close()


def plot_results(df, species):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.hist(df[df['label']==0]['harvest_rate'], bins=20, alpha=0.7, color='#3BB273', label='Стійкий')
    ax.hist(df[df['label']==1]['harvest_rate'], bins=20, alpha=0.7, color='#E84855', label='Надмірний')
    ax.axvline(HARVEST_THRESHOLD, color='black', linestyle='--', label=f'Поріг {HARVEST_THRESHOLD}')
    ax.set_title('Розподіл harvest rate')
    ax.set_xlabel('shot_heads / population')
    ax.legend()
    ax = axes[1]
    by_year = df.groupby('year')['label'].mean() * 100
    ax.bar(by_year.index, by_year.values, color='#E84855', alpha=0.7)
    ax.set_title('% надмірного відстрілу по роках')
    ax.set_ylabel('%'); ax.set_xlabel('рік')
    plt.suptitle(f'Стійкість відстрілу — {species}', fontsize=12)
    plt.tight_layout()
    plt.savefig(ROOT / 'reports' / 'figures' / f'classification_{species}.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    import os, sys
    os.makedirs(ROOT / 'reports' / 'figures', exist_ok=True)
    species_list = sys.argv[1:] if len(sys.argv) > 1 else TARGET_SPECIES

    for species in species_list:
        print(f"\n{'='*50}\nВИД: {species}\n{'='*50}")
        df, results, features = train_and_evaluate(species)

        importance = get_feature_importance(results, features)
        plot_results(df, species)
        plot_feature_importance(importance, species)

        # зберігаємо OOS-прогнози (NaN для ранніх рядків, що не тестувались)
        df['predicted'] = results['Random Forest']['y_pred_cv']
        df[['host', 'year', 'harvest_rate', 'label', 'predicted']].to_csv(
            DATA / f'classification_{species}.csv', index=False)

    print("\nГотово.")