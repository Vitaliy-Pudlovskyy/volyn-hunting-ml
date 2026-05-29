"""
Парсер для файлу Форми 2-ТП за 2025 рік (newest формат).

Відмінності від new_format (2019-2024):
- Лист "Чисельність": host у col 1, species у двох колонках кожен 
  (Чисельність всього + в неволі), шапка 5 рядків
- Harvest на цьому листі немає — повертаємо порожній DataFrame
- 8.ОП користувачів і 13.Розселення — структура як у new

Для hosts_meta, finances і relocation_events перевикористовуємо 
існуючу логіку з new_format.
"""

import pandas as pd
from parser.new_format import (
    get_engine,
    normalize_species_name,
    is_aggregate,
    is_invalid_host,
    parse_relocation_events,
    find_sheet_by_keyword,
)


metrics_hosts_meta = {
    1: "area_total",
    2: "area_forest",
    3: "area_field",
    4: "area_water",
    5: "area_managed",
    6: "staff_total",
    7: "staff_biologists",
    8: "staff_rangers",
}

metrics_finances = {
    10: "total_expenses",
    12: "gov_funding",
    14: "salary",
    16: "expense_protection",
    18: "expense_breeding",
    20: "revenue",
}


def parse_hosts_finances_2025(filepath, year, engine):
    """8. ОП користувачів 2025 — структура як у new_format."""
    
    sheet = f"8. ОП користувачів {year}"
    df = pd.read_excel(filepath, engine=engine, sheet_name=sheet, header=None)
    
    header_row = None
    for i in range(20):
        val = df.iloc[i, 0]
        if pd.notna(val) and "Користувач" in str(val):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Не знайдено 'Користувач' у 8.ОП")
    
    start_row = None
    for j in range(header_row + 1, df.shape[0]):
        val = df.iloc[j, 0]
        if pd.notna(val):
            start_row = j
            break
    if start_row is None:
        raise ValueError("Не знайдено початок даних")
    
    rows_meta = []
    rows_finances = []
    for i in range(start_row, df.shape[0]):
        host_name = df.iloc[i, 0]
        if pd.isna(host_name):
            continue
        if is_aggregate(host_name) or is_invalid_host(host_name):
            continue
        host_name = str(host_name).strip()
        
        for col_index, metric_name in metrics_hosts_meta.items():
            rows_meta.append({
                "year": year, "host": host_name,
                "metric": metric_name, "value": df.iloc[i, col_index],
            })
        
        for col_index, metric_name in metrics_finances.items():
            rows_finances.append({
                "year": year, "host": host_name,
                "metric": metric_name, "value": df.iloc[i, col_index],
            })
    
    hosts_meta = pd.DataFrame(rows_meta)
    hosts_meta["value"] = pd.to_numeric(hosts_meta["value"], errors="coerce")
    
    finances = pd.DataFrame(rows_finances)
    finances["value"] = pd.to_numeric(finances["value"], errors="coerce")
    
# 2025: автор звіту переплутав одиницю — площі написані в га, 
    # а шапка каже тис.га. Приводимо до тис.га (як у 2019-2024).
    area_metrics = {"area_total", "area_forest", "area_field", 
                    "area_water", "area_managed"}
    mask = hosts_meta["metric"].isin(area_metrics)
    hosts_meta.loc[mask, "value"] = hosts_meta.loc[mask, "value"] / 1000

    return hosts_meta, finances


def parse_populations_2025(filepath, year, engine):
    """
    Лист "Чисельність" 2025: host у col 1, species у 2 колонках 
    (всього + в неволі). Беремо тільки "Чисельність всього".
    """
    
    df = pd.read_excel(filepath, engine=engine, sheet_name="Чисельність", header=None)
    
    species_row = 3
    sub_metric_row = 4
    start_data_row = 5
    
    # Знайти колонки де sub_metric = "Чисельність всього"
    species_cols = {}
    for col in range(df.shape[1]):
        sub = df.iloc[sub_metric_row, col]
        species_raw = df.iloc[species_row, col]
        
        if pd.isna(sub) or pd.isna(species_raw):
            continue
        if "Чисельність всього" not in str(sub):
            continue
        
        species = normalize_species_name(species_raw)
        if species is None:  # фільтрує "Всього копитних" та ін.
            continue
        
        species_cols[col] = species
    
    rows = []
    for i in range(start_data_row, df.shape[0]):
        host_name = df.iloc[i, 1]  # ← col 1, не col 0
        
        if pd.isna(host_name):
            continue
        if is_aggregate(host_name) or is_invalid_host(host_name):
            continue
        host_name = str(host_name).strip()
        
        for col_index, species in species_cols.items():
            value = df.iloc[i, col_index]
            rows.append({
                "year": year, "host": host_name, "species": species,
                "metric": "count", "value": value,
            })
    
    populations = pd.DataFrame(rows)
    populations["value"] = pd.to_numeric(populations["value"], errors="coerce")
    
    return populations


def parse_new_2025(filepath, year=2025):
    """Парсить файл Форми 2-ТП за 2025 рік."""
    
    engine = get_engine(filepath)
    
    hosts_meta, finances = parse_hosts_finances_2025(filepath, year, engine)
    populations = parse_populations_2025(filepath, year, engine)
    
    # harvest відсутній у форматі 2025 — повертаємо порожній DataFrame з правильною схемою
    harvest = pd.DataFrame(columns=["year", "host", "species", "metric", "value"])
    
    relocation_events = parse_relocation_events(filepath, year, engine)
    
    return hosts_meta, finances, populations, harvest, relocation_events