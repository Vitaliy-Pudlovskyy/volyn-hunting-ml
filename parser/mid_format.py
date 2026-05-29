"""
Парсер для файлів Форми 2-ТП формату 2013-2018.
"""

import pandas as pd

def normalize_species_name(name):
    if pd.isna(name):
        return None
    
    name = name.strip()  
    while "  " in name:
        name = name.replace("  ", " ")
    
    name = name.rstrip("*").strip()

    name = name.replace("- ", "-")

    name = name.replace('"', "'")

    if name == "Інші":
        return None
    if "всього" in name.lower() or "усього" in name.lower():
        return None
    return name

def is_aggregate(name):
    """Чи це рядок-агрегат (а не справжнє господарство)"""
    name_lower = str(name).lower()
    if "всього" in name_lower or "усього" in name_lower:
        return True
    if "по області" in name_lower:
        return True
    return False


# Маппінг колонок → імена метрик для секції hosts_meta
metrics_hosts_meta = {
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


def parse_mid_format(filepath, year):
    """
    Парсить файл Форми 2-ТП формату 2013-2018.
    """
    
    # --- Завантаження ---
    df = pd.read_excel(filepath, engine="xlrd", sheet_name="2-тп 1", header=None)
    
    start_row = None
    for i in range(20):
        val = df.iloc[i, 0]
        if pd.notna(val) and "Користувач" in str(val):
            start_row = i + 1
            break
    
    if start_row is None:
        raise ValueError("Не знайдено рядок 'Користувач'")
    
    # --- Парсинг hosts_meta ---
    rows = []
    for i in range(start_row, df.shape[0]):
        host_name = df.iloc[i, 0]
        
        # Пропускаємо порожні рядки
        if pd.isna(host_name):
            continue
        
        # Пропускаємо агрегати ("ДКЛГ - усього", "ІНШІ - всього", "Волинська - всього")
        if is_aggregate(host_name):
            continue
        
        # Очищаємо імʼя (бувають зайві пробіли)
        host_name = str(host_name).strip()
        
        # Для кожної метрики беремо значення з відповідної колонки
        for col_index, metric_name in metrics_hosts_meta.items():
            value = df.iloc[i, col_index]
            rows.append({
                "year": year,
                "host": host_name,
                "metric": metric_name,
                "value": value,
            })
    
    hosts_meta = pd.DataFrame(rows)
    hosts_meta["value"] = pd.to_numeric(hosts_meta["value"], errors="coerce")
    
    
    df_oblik = None
    xl_sheets = pd.ExcelFile(filepath, engine="xlrd").sheet_names
    for sheet_name in ["облік\n", "облік ", "облік1", "облік"]:
        if sheet_name in xl_sheets:
            df_oblik = pd.read_excel(filepath, engine="xlrd", sheet_name=sheet_name, header=None)
            break
    
    
    header_row_oblik = None
    for i in range(10):
        val = df_oblik.iloc[i, 0]
        if pd.notna(val) and "Користувач" in str(val):
            header_row_oblik = i
            break



    if df_oblik is None:
        raise ValueError(f"Не знайдено лист з обліком чисельності")

    if header_row_oblik is None:
        raise ValueError("Не знайдено шапку у листі 'облік'")
    
    species_columns = {}
    for col in range(2, df_oblik.shape[1]):
        raw_name = df_oblik.iloc[header_row_oblik, col]
        species = normalize_species_name(raw_name)
        if species is not None:
            species_columns[col] = species

    populations_rows = []
    for i in range(header_row_oblik +1, df_oblik.shape[0]):
        host_name = df_oblik.iloc[i , 0]

        if pd.isna(host_name):
            continue
        
        if is_aggregate(host_name):
            continue
        
        host_name = str(host_name).strip()


        for col_index, species in species_columns.items():
            value = df_oblik.iloc[i, col_index]
            populations_rows.append({
                "year": year,
                "host": host_name,
                "species": species,
                "metric": "count",
                "value": value,
            })

    populations = pd.DataFrame(populations_rows)
    populations["value"] = pd.to_numeric(populations["value"], errors = "coerce")
    
    metrics_finances = {
                        10:"total_expenses",
                        11:"gov_funding",
                        12:"salary",
                        14:"expense_counting",
                        15:"expense_protection",
                        20:"expense_arrangement",
                        21:"revenue",
                       }
    
    # --- Парсинг finances ---
    finance_rows = []
    for i in range(start_row, df.shape[0]):
        host_name = df.iloc[i, 0]
        
        # Пропускаємо порожні рядки
        if pd.isna(host_name):
            continue
        
        # Пропускаємо агрегати ("ДКЛГ - усього", "ІНШІ - всього", "Волинська - всього")
        if is_aggregate(host_name):
            continue
        
        # Очищаємо імʼя (бувають зайві пробіли)
        host_name = str(host_name).strip()
        
        # Для кожної метрики беремо значення з відповідної колонки
        for col_index, metric_name in metrics_finances.items():
            value = df.iloc[i, col_index]
            finance_rows.append({
                "year": year,
                "host": host_name,
                "metric": metric_name,
                "value": value,
            })
    
    finances = pd.DataFrame(finance_rows)
    finances["value"] = pd.to_numeric(finances["value"], errors="coerce")
    
    metric_ranges = [
        (80,99, "relocated"),
        (99, 114, "caught"),
        (114, 124, "found_dead" ),
        (124, 181, "shot_heads"),
        (181, 238, "illegal_shot"), 
    ]

    combined_rows= []
    for i in range(start_row, df.shape[0]):
        host_name = df.iloc[i, 0]
        if pd.isna(host_name):
            continue
        if is_aggregate(host_name):
            continue
        host_name = str(host_name).strip()

        for col_start, col_end, metric in metric_ranges:
            for col in range(col_start, col_end):
                species = normalize_species_name(df.iloc[2, col])
                if species is None:
                    continue

                value = df.iloc[i, col]
                combined_rows.append({
                    "year": year,
                    "host": host_name,
                    "species": species,
                    "metric": metric,
                    "value": value,
                })
    combined = pd.DataFrame(combined_rows)
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")

    harvest = combined[combined["metric"].isin(["found_dead", "shot_heads", "illegal_shot"])].copy()
    relocation = combined[combined["metric"].isin(["relocated", "caught"])].copy()


    return hosts_meta, populations, finances, harvest, relocation





