import pandas as pd
from pathlib import Path

DATA    = Path(__file__).parent.parent / 'data'
PROC    = DATA / 'processed'
FINAL   = DATA / 'final'
MAPPING = DATA / 'entity_mapping' / 'host_mapping_draft_v2.csv'

FINAL.mkdir(exist_ok=True)

# Завантажуємо маппінг
mapping_df = pd.read_csv(MAPPING)
mapping = dict(zip(mapping_df['host_raw'], mapping_df['host_canonical_draft']))
print(f"Маппінг: {len(mapping)} raw → {len(set(mapping.values()))} канонічних")


SUFFIXES = ['', '_mid', '_2018', '_new', '_2025']
TABLES   = ['hosts_meta', 'finances', 'populations', 'harvest']

def build_table(table_name):
    parts = []
    for suffix in SUFFIXES:
        f = PROC / f'{table_name}{suffix}.csv'
        if f.exists():
            parts.append(pd.read_csv(f))
    
    if not parts:
        print(f"  {table_name}: файлів не знайдено")
        return pd.DataFrame()
    
    df = pd.concat(parts, ignore_index=True)
    
    # Застосовуємо маппінг
    df['host_canonical'] = df['host'].map(mapping)
    
    # Фільтруємо SKIP і немаповані
    before = len(df)
    df = df[df['host_canonical'].notna()]
    df = df[df['host_canonical'] != 'SKIP']
    after = len(df)
    
    print(f"  {table_name}: {before} → {after} рядків (прибрано {before-after})")
    return df


SPECIES_MAP = {
    "Ведмідь": "Ведмідь бурий",
    "Лебеді": "Лебідь",
    "Олень благ.": "Олень благородний",
    "Олень плям.": "Олень плямистий",
    "Олень європ.": "Олень європейський",
    "Фазани": "Фазан",
    "Дикий кролик": "Кріль дикий",
    "Норка вільна": "Норка європейська",
    "Качка": "Качки",
    "Крижень": "Качки",
    "Байбак": "Бабак",
    "Єноиовидний собака": "Єнотовидний собака",
    "Бобер": "Бобер річковий",
    "Лиска": "Лиска",
    "Гуска сіра": "Гуси",
    "Гуменник": "Гуси",
    "Чирянка мала(чирок свистунок)": "Качки",
    "Нерозень (сіра качка)": "Качки",
}

def apply_species_map(df):
    if 'species' not in df.columns:
        return df
    df['species_canonical'] = df['species'].map(
        lambda x: SPECIES_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x
    )
    return df

if __name__ == '__main__':
    print("=== Генерація фінальних CSV ===\n")

    for table in TABLES:
        print(f"\n{table}:")
        df = build_table(table)
        if df.empty:
            continue
        df = apply_species_map(df)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        out = FINAL / f'{table}_final.csv'
        df.to_csv(out, index=False)
        print(f"  Збережено: {out.name}")

    print("\nГотово.")