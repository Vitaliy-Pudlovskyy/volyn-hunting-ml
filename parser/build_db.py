"""
build_db.py — збирає нормалізовану базу volyn.db з фінальних CSV хантінг-проекту.

Логіка:
  1. Довідники species / hosts — унікальні види й господарства з власними id.
  2. Факти (populations, harvest, finances, hosts_meta, relocation_events) —
     назви замінюються на id через словники, «довгий» формат розвертається у «широкий».

Запуск:  python build_db.py   (перезбирає всю базу з нуля)
"""

import sqlite3
import pandas as pd
from pathlib import Path

# ── Шляхи ──────────────────────────────────────────────────────────────
# ROOT — корінь проєкту (від файлу скрипта на дві папки вгору), тож шлях
# не залежить від того, звідки запускати.
ROOT = Path(__file__).parent.parent
SRC = ROOT / "data" / "final"       # де лежать вихідні CSV

DB_DIR = ROOT / "db"
DB_DIR.mkdir(exist_ok=True)         # створити папку db, якщо її ще нема
DB = DB_DIR / "volyn.db"

con = sqlite3.connect(DB)           # відкриваємо (або створюємо) файл бази

# ── Довідники ──────────────────────────────────────────────────────────
# populations містить і господарства, і види, тож довідники будуємо з нього.
df = pd.read_csv(SRC / "populations_final.csv")

# species: беремо canonical-назви → унікальні → сортуємо → додаємо id
species_names = df["species_canonical"].dropna().unique()
species = pd.DataFrame({"name": sorted(species_names)})
species.insert(0, "species_id", range(1, len(species) + 1))
species.to_sql("species", con, if_exists="replace", index=False)

# hosts: те саме для господарств
host_names = df["host_canonical"].dropna().unique()
hosts = pd.DataFrame({"name": sorted(host_names)})
hosts.insert(0, "host_id", range(1, len(hosts) + 1))
hosts.to_sql("hosts", con, if_exists="replace", index=False)

# Словники перекладу «назва → id». Ними факти підставлятимуть id замість тексту.
host_to_id = dict(zip(hosts["name"], hosts["host_id"]))
species_to_id = dict(zip(species["name"], species["species_id"]))


# ── Факт: populations (одна метрика — count) ────────────────────────────
pop = pd.read_csv(SRC / "populations_final.csv")
pop["host_id"] = pop["host_canonical"].map(host_to_id)
pop["species_id"] = pop["species_canonical"].map(species_to_id)
# лишаємо потрібні колонки; value за змістом — це кількість, тож перейменовуємо
pop = pop[["year", "host_id", "species_id", "value"]].rename(columns={"value": "count"})
pop = pop.dropna(subset=["count"])          # викидаємо порожні (виду не було в господарстві)
pop.to_sql("populations", con, if_exists="replace", index=False)


# ── Факт: harvest (4 метрики — розворот «довге → широке») ───────────────
harv = pd.read_csv(SRC / "harvest_final.csv")

# pivot_table: значення з колонки metric стають окремими стовпцями
wide = harv.pivot_table(
    index=["year", "host_canonical", "species_canonical"],  # ключ рядка (унікальна подія)
    columns="metric",                                       # метрики → колонки
    values="value",
    aggfunc="mean"                                          # дублі усереднити, NaN ігнорувати
).reset_index()

wide["host_id"] = wide["host_canonical"].map(host_to_id)
wide["species_id"] = wide["species_canonical"].map(species_to_id)

harvest = wide[["year", "host_id", "species_id",
                "shot_heads", "shot_tons", "illegal_shot", "found_dead"]]
# рядок лишаємо, якщо є хоч одна метрика; викидаємо тільки повністю порожні
harvest = harvest.dropna(how="all",
                         subset=["shot_heads", "shot_tons", "illegal_shot", "found_dead"])
harvest.to_sql("harvest", con, if_exists="replace", index=False)


# ── Факт: finances (без видів; метрики беремо автоматично) ──────────────
fin = pd.read_csv(SRC / "finances_final.csv")

wide_fin = fin.pivot_table(
    index=["year", "host_canonical"],       # фінанси — на господарство за рік, без видів
    columns="metric",
    values="value",
    aggfunc="mean"
).reset_index()

wide_fin["host_id"] = wide_fin["host_canonical"].map(host_to_id)

# усі метрики автоматично — все, крім службових колонок
# (перевага: додасться нова метрика — код підхопить її сам)
metric_cols = [c for c in wide_fin.columns
               if c not in ["year", "host_canonical", "host_id"]]
finances = wide_fin[["year", "host_id"] + metric_cols]
finances = finances.dropna(how="all", subset=metric_cols)
finances.to_sql("finances", con, if_exists="replace", index=False)


# ── Факт: hosts_meta (площі + персонал; без видів) ──────────────────────
meta = pd.read_csv(SRC / "hosts_meta_final.csv")

wide_host = meta.pivot_table(
    index=["year", "host_canonical"],
    columns="metric",
    values="value",
    aggfunc="mean"
).reset_index()

wide_host["host_id"] = wide_host["host_canonical"].map(host_to_id)

metric_cols = [c for c in wide_host.columns
               if c not in ["year", "host_canonical", "host_id"]]
hosts_meta = wide_host[["year", "host_id"] + metric_cols]
hosts_meta = hosts_meta.dropna(how="all", subset=metric_cols)
hosts_meta.to_sql("hosts_meta", con, if_exists="replace", index=False)


# ── Факт: relocation_events (вже «широкий» — без розвороту) ─────────────
# Має готову колонку count і два текстові поля: location (куди) та origin (звідки).
rel = pd.read_csv(SRC / "relocation_events_final.csv")
rel["host_id"] = rel["host_canonical"].map(host_to_id)
rel["species_id"] = rel["species_canonical"].map(species_to_id)
rel = rel[["year", "host_id", "species_id", "count", "location", "origin"]]
rel.to_sql("relocation_events", con, if_exists="replace", index=False)


con.close()