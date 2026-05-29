"""
Парсер для файлу Форми 2-ТП за 2018 рік (гібридний формат).

2018 — перехідний рік між mid (2013-2017) і new (2019+):
- hosts_meta/finances/populations/harvest усі на листі "2-тп 1" (як mid)
- але fin метрики ближче до new (з'явився expense_breeding)
- relocation/mortality винесені на окремі листи (як new)
- площі ще в га, витрати в грн (як mid)
"""

import pandas as pd
from parser.new_format import (
    normalize_species_name,
    is_aggregate,
    is_invalid_host,
    parse_relocation_events,
)
from parser.new_format import find_section_columns, build_species_columns

metrics_hosts_meta_2018 = {
    1: "area_total",
    2: "area_forest",
    3: "area_field",
    4: "area_water",
    5: "area_managed",
    6: "area_managed_this_year",
    7: "staff_total",
    8: "staff_biologists",
    9: "staff_rangers",
}

metrics_finances_2018 = {
    10: "total_expenses",
    11: "gov_funding",
    12: "salary",
    13: "expense_protection",
    14: "expense_breeding",
    15: "expense_arrangement",
    16: "revenue",
}


def parse_hybrid_2018(filepath, year=2018):
    """Парсить файл 2018 (гібридний формат)."""
    
    engine = "xlrd"  # .xls файл
    
    # ============ 2-тп 1: hosts_meta + finances ============
    df = pd.read_excel(filepath, engine=engine, sheet_name="2-тп 1", header=None)
    
    # Знайти стартовий рядок ("Користувач" → дані з наступного)
    start_row = None
    for i in range(20):
        val = df.iloc[i, 0]
        if pd.notna(val) and "Користувач" in str(val):
            start_row = i + 1
            break
    
    if start_row is None:
        raise ValueError("Не знайдено рядок 'Користувач' у 2-тп 1")
    
    rows_meta = []
    rows_finances = []
    for i in range(start_row, df.shape[0]):
        host_name = df.iloc[i, 0]
        
        if pd.isna(host_name):
            continue
        if is_aggregate(host_name) or is_invalid_host(host_name):
            continue
        
        host_name = str(host_name).strip()
        
        for col_index, metric_name in metrics_hosts_meta_2018.items():
            value = df.iloc[i, col_index]
            rows_meta.append({
                "year": year, "host": host_name,
                "metric": metric_name, "value": value,
            })
        
        for col_index, metric_name in metrics_finances_2018.items():
            value = df.iloc[i, col_index]
            rows_finances.append({
                "year": year, "host": host_name,
                "metric": metric_name, "value": value,
            })

    species_row = 2
    header_row = 0
    
    # populations
    kop = find_section_columns(df, header_row, "Чисельність копитних")
    hut = find_section_columns(df, header_row, "Чисельність хутрових")
    per = find_section_columns(df, header_row, "Чисельність пернатих")
    species_cols_pop = build_species_columns(df, species_row, [kop, hut, per])
    
    # harvest
    kop_h = find_section_columns(df, header_row, "Кількість добутих (вилучених) копитних")
    hut_h = find_section_columns(df, header_row, "Кількість добутих (вилучених) хутрових")
    per_h = find_section_columns(df, header_row, "Кількість добутих (вилучених) пернатих")
    species_cols_har = build_species_columns(df, species_row, [kop_h, hut_h, per_h])
    
    rows_populations = []
    rows_harvest = []
    for i in range(start_row, df.shape[0]):
        host_name = df.iloc[i, 0]
        if pd.isna(host_name):
            continue
        if is_aggregate(host_name) or is_invalid_host(host_name):
            continue
        host_name = str(host_name).strip()
        
        for col_index, species in species_cols_pop.items():
            value = df.iloc[i, col_index]
            rows_populations.append({
                "year": year, "host": host_name, "species": species,
                "metric": "count", "value": value,
            })
        
        for col_index, species in species_cols_har.items():
            value = df.iloc[i, col_index]
            rows_harvest.append({
                "year": year, "host": host_name, "species": species,
                "metric": "shot_heads", "value": value,
            })
    
    populations = pd.DataFrame(rows_populations)
    populations["value"] = pd.to_numeric(populations["value"], errors="coerce")
    
    harvest = pd.DataFrame(rows_harvest)
    harvest["value"] = pd.to_numeric(harvest["value"], errors="coerce")



    hosts_meta = pd.DataFrame(rows_meta)
    hosts_meta["value"] = pd.to_numeric(hosts_meta["value"], errors="coerce")
    
    finances = pd.DataFrame(rows_finances)
    finances["value"] = pd.to_numeric(finances["value"], errors="coerce")
    
    relocation_events = parse_relocation_events(filepath, year, engine)
    return hosts_meta, finances, populations, harvest, relocation_events