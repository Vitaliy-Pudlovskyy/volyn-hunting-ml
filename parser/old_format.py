"""
Парсер для файлів Форми 2-ТП старого формату (2000-2012).
"""

import pandas as pd


#===============ФУНКЦІЇ-ПОМІЧНИКИ===========
"""
Парсер для файлів Форми 2-ТП старого формату (2000-2012).
"""

import pandas as pd


# ============ ФУНКЦІЇ-ПОМІЧНИКИ ============

def is_aggregate(name):
    """Чи це рядок-агрегат (а не справжнє господарство)"""
    name_lower = name.lower()
    if "всього" in name_lower or "усього" in name_lower:
        return True
    if "по області" in name_lower:
        return True
    return False


_metric_mapping = {
    "загальна кількість":  "count",
    "розселено":           "relocated",
    "відловлено":          "caught",
    "виявлено загиблими":  "found_dead",
}


def detect_metric(name, unit):
    """Визначити метрику за текстом рядка та одиницею виміру."""
    name_lower = name.lower() if pd.notna(name) else ""
    unit_str = str(unit).lower() if pd.notna(unit) else ""
    
    if "добуто" in name_lower:
        if "тонн" in unit_str:
            return "shot_tons"
        else:
            return "shot_heads"
    
    for keyword, metric_name in _metric_mapping.items():
        if keyword in name_lower:
            return metric_name
    
    return None


def is_species_header(code, name):
    if pd.isna(code):
        return False
    code_str = str(code).strip()
    if code_str == "":
        return False
    if code_str == "Код тварини":
        return False  # підзаголовок секції
    return True


def is_aggregate_species(code, name):
    """Чи це рядок-агрегат типу 'Копитні всього'? Код кратний 100."""
    return int(float(code)) % 100 == 0


def extract_species_name(name):
    """'Зубр - загальна кількість' → 'Зубр'"""
    if " - " in name:
        first = name.split(" - ")[0]
    elif " -" in name:
        first = name.split(" -")[0]
    else:
        first = name
    return first.strip()

def find_section_iv_start(df):
    for i in range(201):
        value = df.iloc[i, 0]
        if pd.notna(value) and ("ІV" in str(value) or "ШТУЧНЕ РОЗВЕДЕННЯ" in str(value)):
            return i 
    raise ValueError("Не знайдено початок Секції IV")


def find_header_row(df):
    for i in range(20):
        value = df.iloc[i, 1]
        if pd.notna(value) and "Найменування" in str(value):
            return i
    raise ValueError ("Не знайдено рядка з шапкою")


# ============ ОСНОВНА ФУНКЦІЯ ПАРСИНГУ ============

def parse_old_format(filepath, year):
    """
    Парсить файл Форми 2-ТП старого формату (2000-2012).
    
    Args:
        filepath: шлях до .xls файлу
        year: рік звіту (наприклад, 2005)
    
    Returns:
        hosts_meta, finances, populations, harvest, relocation - 5 DataFrame-ів
    """
    
    # --- Завантаження ---
    df = pd.read_excel(filepath, engine="xlrd", header=None)
    header_row = find_header_row(df)
    section_iv_start = find_section_iv_start(df)


    # --- Аналіз шапки ---
    hosts_raw = df.iloc[header_row, 3:].tolist()
    real_hosts = []
    for i, h in enumerate(hosts_raw):
        if not is_aggregate(h):
            df_col = i + 3
            real_hosts.append((df_col, h))
    
    # --- Секція I: hosts_meta ---
    metrics_section1 = {
        2: "area_total",
        3: "area_managed",
        4:  "area_counted",
        5:  "staff_total",
        6:  "staff_biologists",
        7: "staff_rangers",
    }
    
    rows = []
    for offset, metric_name in metrics_section1.items():
        df_row = header_row + offset
        for df_col, host_name in real_hosts:
            value = df.iloc[df_row, df_col]
            rows.append({
                "year": year,
                "host": host_name,
                "metric": metric_name,
                "value": value,
            })
    hosts_meta = pd.DataFrame(rows)
    
    # --- Секція II: finances ---
    metrics_section2 = {
        11: "total_expenses",
        12: "gov_funding",
        13: "salary",
        16: "expense_counting",
        17: "expense_protection",
        22: "expense_feeding",
        25: "revenue",
    }
    
    rows = []
    for offset, metric_name in metrics_section2.items():
        df_row = header_row + offset
        for df_col, host_name in real_hosts:
            value = df.iloc[df_row, df_col]
            rows.append({
                "year": year,
                "host": host_name,
                "metric": metric_name,
                "value": value,
            })
    finances = pd.DataFrame(rows)
    finances["value"] = pd.to_numeric(finances["value"], errors="coerce")
    
    # --- Секція III: state machine ---
    rows = []
    current_species = None
    
    for i in range(header_row + 29, section_iv_start):
        if i>= df.shape[0]:
            break
        code = df.iloc[i, 0]
        name = df.iloc[i, 1]
        unit = df.iloc[i, 2]
        if pd.notna(code) and str(code).strip() == "Код тварини":
            current_species = None
            continue
        if pd.isna(name):
            continue
        
        if is_species_header(code, name):
            if is_aggregate_species(code, name):
                current_species = None
                continue
            else:
                current_species = extract_species_name(name)
                if " - " in name or " -" in name:
                    save_count = True
                else:
                    if i + 1 <df.shape[0]:
                        next_code = df.iloc[i + 1, 0]
                        next_name = df.iloc[i + 1, 1]

                        if pd.notna(next_code) and str(next_code).strip() != "":
                            save_count = True
                        elif pd.isna(next_name):
                            save_count = True    
                        else:
                            save_count = False
                    else:
                        save_count = True
                if save_count:
                    metric = detect_metric(name, unit)
                    if metric is None:
                        metric = "count"
                    for df_col, host_name in real_hosts:
                        value = df.iloc[i, df_col]
                        rows.append({
                            "year": year,
                            "host": host_name,
                            "species": current_species,
                            "metric": metric,
                            "value": value,
                        })
        else:
            if current_species is None:
                continue
            metric = detect_metric(name, unit)
            if metric is None:
                continue
            for df_col, host_name in real_hosts:
                value = df.iloc[i, df_col]
                rows.append({
                    "year": year,
                    "host": host_name,
                    "species": current_species,
                    "metric": metric,
                    "value": value,
                })
    
    populations_harvest = pd.DataFrame(rows)
    populations_harvest["value"] = pd.to_numeric(populations_harvest["value"], errors="coerce")
    # --- Розділення на 3 таблиці ---
    populations = populations_harvest[populations_harvest["metric"] == "count"].copy()
    harvest = populations_harvest[populations_harvest["metric"].isin(["shot_heads", "shot_tons", "found_dead"])].copy()
    relocation = populations_harvest[populations_harvest["metric"].isin(["relocated", "caught"])].copy()
    
    return hosts_meta, finances, populations, harvest, relocation
