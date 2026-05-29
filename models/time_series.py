import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from pmdarima import auto_arima

import warnings
warnings.filterwarnings('ignore')


ROOT = Path(__file__).parent.parent
DATA = ROOT / 'data' / 'final'

TRAIN_END         = 2018
TEST_START        = 2019
TEST_END          = 2021
FORECAST_DATA_END = 2025
FORECAST_YEARS    = [2026, 2027, 2028]

# ── STABLE_HOSTS автогенерація ─────────────────────────────
# Критерій: господарство мало звітність ≥18 років з 22-річного вікна
# (2000-2021, до заборони полювання). Поріг 18/22 = ~82% покриття.
# Це баланс між повнотою даних і кількістю господарств для аналізу.

MIN_YEARS_COVERAGE = 22
COVERAGE_WINDOW = (2000, 2021)


def get_stable_hosts(species_for_check='Козуля'):
    """Знаходить господарства зі стабільною звітністю для time series.

    Замість хардкоду повертає список господарств що мають дані
    мінімум MIN_YEARS_COVERAGE років у вікні COVERAGE_WINDOW
    для заданого виду (за замовчуванням Козуля — представлена скрізь).
    """
    populations = pd.read_csv(DATA / 'populations_final.csv')

    df = populations[
        (populations['metric'] == 'count') &
        (populations['species_canonical'] == species_for_check) &
        (populations['year'].between(*COVERAGE_WINDOW)) &
        (populations['value'].notna())
    ]

    years_per_host = df.groupby('host_canonical')['year'].nunique()
    stable_hosts = years_per_host[years_per_host >= MIN_YEARS_COVERAGE].index.tolist()

    return sorted(stable_hosts)


# Генеруємо список при імпорті
STABLE_HOSTS = get_stable_hosts()
print(f"STABLE_HOSTS згенеровано: {len(STABLE_HOSTS)} господарств "
      f"(≥{MIN_YEARS_COVERAGE} років з {COVERAGE_WINDOW[0]}-{COVERAGE_WINDOW[1]})")

NO_FORECAST_SPECIES = ['Кабан']  # волатильні через АЧС


def load_data(species='Козуля'):
    populations = pd.read_csv(DATA / 'populations_final.csv')
    df = populations[
        (populations['metric'] == 'count') &
        (populations['species_canonical'] == species) &
        (populations['host_canonical'].isin(STABLE_HOSTS)) &
        (populations['year'] <= TEST_END)
    ].copy()
    return df


def fit_auto_arima(train):
    """auto_arima замість хардкоду (1,1,0)."""
    return auto_arima(
        train,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        max_d=2,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        information_criterion='aic',
        trace=False,
    )


def evaluate_arima(df):
    """Оцінка на train/test split. Train: 2000-2018, Test: 2019-2021."""
    results = []

    for host in STABLE_HOSTS:
        ts = (df[df['host_canonical'] == host]
              .sort_values('year').set_index('year')['value'].dropna())
        train = ts.loc[:TRAIN_END]
        test  = ts.loc[TEST_START:TEST_END]

        if len(train) < 5 or len(test) == 0:
            results.append({'host': host, 'order': None, 'mae': None, 'status': 'мало даних'})
            continue

        try:
            model = fit_auto_arima(train)
            pred = model.predict(n_periods=len(test))
            mae = mean_absolute_error(test.values, pred)
            results.append({
                'host':   host,
                'order':  str(model.order),
                'mae':    round(mae, 1),
                'status': 'ok',
            })
        except Exception as e:
            results.append({'host': host, 'order': None, 'mae': None, 'status': str(e)[:50]})

    return pd.DataFrame(results).sort_values('mae', na_position='last')


def forecast_arima(species='Козуля'):
    """Прогноз на 2026-2028. Auto-ARIMA на всіх даних 2000-2025."""
    populations = pd.read_csv(DATA / 'populations_final.csv')
    df = populations[
        (populations['metric'] == 'count') &
        (populations['species_canonical'] == species) &
        (populations['host_canonical'].isin(STABLE_HOSTS)) &
        (populations['year'] <= FORECAST_DATA_END)
    ].copy()

    forecasts = []
    for host in STABLE_HOSTS:
        ts = (df[df['host_canonical'] == host]
              .sort_values('year').set_index('year')['value'].dropna())

        if len(ts) < 5:
            continue

        try:
            model = fit_auto_arima(ts)
            pred  = model.predict(n_periods=3)
            for year, value in zip(FORECAST_YEARS, pred):
                forecasts.append({
                    'host':     host,
                    'species':  species,
                    'year':     year,
                    'forecast': round(float(value), 1),
                    'order':    str(model.order),
                })
        except Exception as e:
            print(f"Помилка {host}: {str(e)[:80]}")

    return pd.DataFrame(forecasts)


def forecast_trend(yearly_totals, raw_df=None):
    """Лінійна екстраполяція тренду 2022→2025 на 2026-2028.

    ВАЖЛИВО: беремо тільки господарства що звітували у ОБОХ якірних роках
    (2022 і 2025). Інакше різний склад → суми незіставні → хибний тренд.

    Якщо raw_df передано — використовуємо його для коректного фільтру.
    Якщо ні — fallback до старого методу (може бути неточно).
    """
    if raw_df is not None:
        # Господарства що звітували і у 2022, і у 2025
        h_2022 = set(raw_df[(raw_df['year'] == 2022) &
                            (raw_df['value'].notna())]['host_canonical'])
        h_2025 = set(raw_df[(raw_df['year'] == 2025) &
                            (raw_df['value'].notna())]['host_canonical'])
        consistent_hosts = h_2022 & h_2025

        if not consistent_hosts:
            return None

        val_2022 = raw_df[(raw_df['year'] == 2022) &
                          (raw_df['host_canonical'].isin(consistent_hosts))]['value'].sum()
        val_2025 = raw_df[(raw_df['year'] == 2025) &
                          (raw_df['host_canonical'].isin(consistent_hosts))]['value'].sum()

        anchor_years = [2022, 2025]
        anchor_values = [val_2022, val_2025]

        print(f"  (Тренд base на {len(consistent_hosts)} господарствах "
              f"з даними у 2022 і 2025)")
    else:
        # старий метод як fallback
        anchor_years = [y for y in [2022, 2025] if y in yearly_totals.index]
        if len(anchor_years) < 2:
            return None
        anchor_values = [yearly_totals.loc[y] for y in anchor_years]

    slope, intercept = np.polyfit(anchor_years, anchor_values, 1)

    trend_forecasts = {}
    for year in FORECAST_YEARS:
        trend_forecasts[year] = slope * year + intercept

    return pd.Series(trend_forecasts)


def analyze_recovery(species='Козуля'):
    """Аналіз відновлення популяції після заборони полювання 2022."""
    populations = pd.read_csv(DATA / 'populations_final.csv')
    df = populations[
        (populations['metric'] == 'count') &
        (populations['species_canonical'] == species)
    ].copy()

    years_check = [2019, 2020, 2021, 2022, 2025]
    stable = []
    for host in STABLE_HOSTS:
        host_data = df[
            (df['host_canonical'] == host) &
            (df['year'].isin(years_check)) &
            (df['value'].notna())
        ]
        if len(host_data) == len(years_check):
            stable.append(host)

    if not stable:
        print(f"\n=== Recovery {species}: немає господарств з повними даними ===")
        return None

    before = df[(df['year'].isin([2019, 2020, 2021])) & (df['host_canonical'].isin(stable))]
    after_2022 = df[(df['year'] == 2022) & (df['host_canonical'].isin(stable))]
    after_2025 = df[(df['year'] == 2025) & (df['host_canonical'].isin(stable))]

    # ← FIX: дедуплікація per-host перед сумою.
    # entity resolution може давати кілька сирих рядків на одне
    # (host, year) → сирий .sum() подвоїв би. Беремо mean по господарству,
    # потім суму — узгоджено з рештою пайплайну (groupby+mean).
    avg_after_2022 = after_2022.groupby('host_canonical')['value'].mean().sum()
    avg_after_2025 = after_2025.groupby('host_canonical')['value'].mean().sum()
    avg_before = (before.groupby(['year', 'host_canonical'])['value'].mean()
                        .groupby('year').sum().mean())
    change_2022 = avg_after_2022 - avg_before
    change_2025 = avg_after_2025 - avg_before

    print(f"\n=== Відновлення популяції {species} після заборони полювання ===")
    print(f"Стабільних господарств: {len(stable)}")
    print(f"Середнє 2019-2021: {avg_before:.0f} голів")
    print(f"2022:              {avg_after_2022:.0f} голів  ({change_2022:+.0f}, {change_2022/avg_before*100:+.1f}%)")
    print(f"2025:              {avg_after_2025:.0f} голів  ({change_2025:+.0f}, {change_2025/avg_before*100:+.1f}%)")

    return {
        'species':         species,
        'n_hosts':         len(stable),
        'before':          round(avg_before, 1),
        'after_2022':      float(avg_after_2022),
        'after_2025':      float(avg_after_2025),
        'change_2022':     round(change_2022, 1),
        'change_pct_2022': round(change_2022/avg_before*100, 1),
        'change_2025':     round(change_2025, 1),
        'change_pct_2025': round(change_2025/avg_before*100, 1),
    }


def plot_results(species='Козуля'):
    """Графік: історія + (для стабільних видів) прогноз."""
    populations = pd.read_csv(DATA / 'populations_final.csv')
    df = populations[
        (populations['metric'] == 'count') &
        (populations['species_canonical'] == species) &
        (populations['host_canonical'].isin(STABLE_HOSTS))
    ].copy()

    yearly = df.groupby('year')['value'].sum()
    yearly_clean = yearly[~yearly.index.isin([2023, 2024])]
    yearly_gap   = yearly[yearly.index.isin([2023, 2024])]

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(yearly_clean.index, yearly_clean.values,
            marker='o', markersize=4, color='#2E86AB', label='факт')
    ax.plot(yearly_gap.index, yearly_gap.values,
            marker='o', markersize=4, color='#2E86AB',
            linestyle='--', alpha=0.4, label='неповні дані (2023-2024)')

    if species not in NO_FORECAST_SPECIES:
        forecast_df = forecast_arima(species)
        arima_forecast = forecast_df.groupby('year')['forecast'].sum()
        trend_forecast = forecast_trend(yearly, raw_df=df)

        ax.plot(arima_forecast.index, arima_forecast.values,
                marker='o', markersize=4, color='#E84855',
                linestyle='--', label='прогноз ARIMA (baseline)')
        if 2025 in yearly.index and 2026 in arima_forecast.index:
            ax.plot([2025, 2026],
                    [yearly.loc[2025], arima_forecast.loc[2026]],
                    color='#E84855', linestyle='--', alpha=0.5)

        if trend_forecast is not None:
            ax.plot(trend_forecast.index, trend_forecast.values,
                    marker='s', markersize=5, color='#3BB273',
                    linestyle='--', label='прогноз з трендом заборони')
            if 2025 in yearly.index and 2026 in trend_forecast.index:
                ax.plot([2025, 2026],
                        [yearly.loc[2025], trend_forecast.loc[2026]],
                        color='#3BB273', linestyle='--', alpha=0.5)

            years = list(FORECAST_YEARS)
            lower = [min(arima_forecast.get(y, np.nan), trend_forecast.get(y, np.nan))
                     for y in years]
            upper = [max(arima_forecast.get(y, np.nan), trend_forecast.get(y, np.nan))
                     for y in years]
            ax.fill_between(years, lower, upper, alpha=0.15, color='gray',
                            label='діапазон невизначеності')
    else:
        ax.text(0.98, 0.95,
                'Прогноз не надається:\nвисока волатильність через АЧС',
                transform=ax.transAxes,
                ha='right', va='top',
                fontsize=9, style='italic',
                bbox=dict(boxstyle='round', facecolor='#FFF3CD',
                          edgecolor='#856404', alpha=0.8))

    ax.axvline(2014, color='brown', linestyle=':', alpha=0.6, label='АЧС (кабан)')
    ax.axvline(2022, color='red',   linestyle=':', alpha=0.6, label='заборона полювання')

    ax.set_title(f'{species} — Волинська область, {len(STABLE_HOSTS)} господарств')
    ax.set_ylabel('голів')
    ax.set_xlabel('рік')
    ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(ROOT / 'reports' / 'figures' / f'timeseries_{species}.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    import os
    import sys
    os.makedirs(ROOT / 'reports' / 'figures', exist_ok=True)

    print("=== Time Series — Мисливські господарства Волині ===\n")

    species_list = sys.argv[1:] if len(sys.argv) > 1 else ['Козуля', 'Кабан']

    for species in species_list:
        print(f"\n{'='*50}")
        print(f"ВИД: {species}")
        print('='*50)

        df = load_data(species)
        eval_results = evaluate_arima(df)
        print(eval_results.to_string(index=False))

        valid_mae = eval_results['mae'].dropna()
        print(f"\nСередній MAE: {valid_mae.mean():.1f} голів  (по {len(valid_mae)} господарствах)")

        order_counts = eval_results['order'].value_counts()
        print(f"\nРозподіл вибраних (p,d,q) auto-ARIMA:")
        print(order_counts.to_string())

        recovery = analyze_recovery(species)

        # ← FIX: блок прогнозу тепер УСЕРЕДИНІ циклу for
        populations_full = pd.read_csv(DATA / 'populations_final.csv')
        df_full = populations_full[
            (populations_full['metric'] == 'count') &
            (populations_full['species_canonical'] == species) &
            (populations_full['host_canonical'].isin(STABLE_HOSTS))
        ].copy()
        yearly_totals = df_full.groupby('year')['value'].sum()

        if species in NO_FORECAST_SPECIES:
            print(f"\nПрогноз не надається для {species}: волатильність через АЧС")
        else:
            print(f"\nПрогноз 2026-2028:")

            forecast_df = forecast_arima(species)
            arima_total = forecast_df.groupby('year')['forecast'].sum()
            trend_total = forecast_trend(yearly_totals, raw_df=df_full)

            print(f"  Сценарій ARIMA (baseline):")
            for year, val in arima_total.items():
                print(f"    {year}: {val:.0f} голів")

            if trend_total is not None:
                print(f"  Сценарій тренду (якщо заборона діє):")
                for year, val in trend_total.items():
                    print(f"    {year}: {val:.0f} голів")

            forecast_df.to_csv(DATA / f'arima_forecast_{species}.csv', index=False)

        print(f"\nГрафік...")
        plot_results(species)

        eval_results.to_csv(DATA / f'arima_mae_{species}.csv', index=False)

    print("\nГотово. Результати збережено в data/final/")