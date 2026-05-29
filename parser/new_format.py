"""
Парсер для файлів Форми 2-ТП формату 2019-2025.
"""

import pandas as pd

def get_engine(filepath):
    if filepath.endswith(".xlsx"):
        return "openpyxl"
    elif filepath.endswith(".xls"):
        return "xlrd"
    else:
        raise ValueError(f"Невідоме розширення файлу: {filepath}")

SPECIES_CANONICAL = {
        "Байбак": "Бабак",
    }


def normalize_species_name(name):
    if pd.isna(name):
        return None
    
    name = name.strip()  
    while "  " in name:
        name = name.replace("  ", " ")
    
    name = name.rstrip("*").strip()
    name = name.replace("- ", "-")
    name = name.replace('"', "'")
    
    name_lower = name.lower()
    if not name:
        return None  
    name = name[0].upper() + name[1:]


    if name in SPECIES_CANONICAL:
        name = SPECIES_CANONICAL[name]

    if name == "Інші":
        return None
    
    
    if name_lower.startswith("всього") or name_lower.startswith("усього"):
        return None
    if "в тому числі" in name_lower or "у тому числі" in name_lower:
        return None
    
    
    if name_lower.endswith("-всього") or name_lower.endswith("-усього"):
        name = name.rsplit("-", 1)[0].strip()
    
    return name

def is_aggregate(name):
    name_lower = str(name).lower()
    if "всього" in name_lower or "усього" in name_lower:
        return True
    if "по області" in name_lower:
        return True
    if "разом" in name_lower:
        return True
    if "в тому числі" in name_lower or "у тому числі" in name_lower:  # ← нове
        return True
    return False


def is_invalid_host(name):
    """Чи це забруднювач — оператор звіту, телефон, тощо"""
    name_str = str(name)
    name_upper = name_str.upper()
    
    if "ПУДЛОВСЬКА" in name_upper:
        return True
    
    # Телефонний номер — 10+ цифр підряд
    digit_runs = "".join(c if c.isdigit() else " " for c in name_str).split()
    if any(len(run) >= 10 for run in digit_runs):
        return True
    
    return False

def build_species_columns(df, species_row, section_ranges):
    species_columns = {}
    for start , end  in section_ranges:
        for col in range(start, end+1):
            raw_name = df.iloc[species_row, col]
            species = normalize_species_name(raw_name)
            if species is not None:
                species_columns[col]= species
    return species_columns

def find_header_row_by_keyword(df, keyword, max_rows = 10):
    for i in range(max_rows):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            if pd.notna(value) and keyword in str(value):
                return i
    raise ValueError(f"Не знайдено рядок з '{keyword}'")

def find_sheet_by_keyword(filepath, engine, keyword):
    xl = pd.ExcelFile(filepath, engine=engine)
    for sheet_name in xl.sheet_names:
        if keyword.lower() in sheet_name.lower():
            return sheet_name  
    raise ValueError(f"Не знайдено лист з '{keyword}' у файлі {filepath}")     

def find_section_columns(df, header_row, section_keyword):
    start_col = None
    for col in range(df.shape[1]):
        value = df.iloc[header_row, col]
        if pd.notna(value) and section_keyword in str(value):
         start_col = col 
         break

    if start_col is None:
       raise ValueError (f"Не знайдено секцію: {section_keyword}")
    
    end_col = None
    for col in range(start_col + 1, df.shape[1]):
        value = df.iloc[header_row, col]
        if pd.notna(value):
           end_col = col - 1
           break
    if end_col is None:
       end_col= df.shape[1] - 1    
    return start_col, end_col


def parse_relocation_events(filepath, year, engine):
    """Парсить лист 12.Розселення / 13.Розселення (event-log)."""
    
    sheet_name = find_sheet_by_keyword(filepath, engine, "озселенн")
    df = pd.read_excel(filepath, engine=engine, sheet_name=sheet_name, header=None)
    
    header_row = find_header_row_by_keyword(df, "Користувач")
    
    # критично перед циклом
    df.iloc[:, 0] = df.iloc[:, 0].ffill()
    
    rows = []
    for i in range(header_row + 1, df.shape[0]):
        host = df.iloc[i, 0]
        species_raw = df.iloc[i, 1]
        count = df.iloc[i, 2]
        location = df.iloc[i, 3]
        origin = df.iloc[i, 4]
        
        if pd.isna(species_raw) or pd.isna(host):
            continue
        
        host = str(host).strip()
        if is_aggregate(host) or is_invalid_host(host):
            continue
        
        species = normalize_species_name(species_raw)
        if species is None:
            continue
        
        location = str(location).strip() if pd.notna(location) else None
        origin = str(origin).strip() if pd.notna(origin) else None
        
        rows.append({
            "year": year,
            "host": host,
            "species": species,
            "count": count,
            "location": location,
            "origin": origin,
        })
    
    result = pd.DataFrame(rows)
    if not result.empty:
        result["count"] = pd.to_numeric(result["count"], errors="coerce")
    
    return result

metrics_hosts_meta = {
    1:"area_total",
    2:"area_forest",
    3:"area_field",
    4:"area_water",
    5:"area_managed",
    6:"staff_total",
    7:"staff_biologists",
    8:"staff_rangers",
}

metrics_finances = {    
    10:"total_expenses",
    12:"gov_funding",
    14:"salary",
    16:"expense_protection",
    18:"expense_breeding",
    20:"revenue",

}



def parse_new_format(filepath, year):
    """Парсить файл Форми 2-ТП формату 2019-2025."""
    
    engine = get_engine(filepath)
    
    sheet_candidates = [
        "8. ОП користувачів ",            # 2019-2021
        f"8. ОП користувачів {year}",     # 2022+ з роком
        f"8.ОП користувачів {year}",      # 2022+ без пробілу
    ]
    
    target_sheet = None
    x1 = pd.ExcelFile(filepath, engine=engine)
    for candidate in sheet_candidates:
        if candidate in x1.sheet_names:
                target_sheet = candidate
                break
        
    if target_sheet is None:
        raise ValueError(f"Не знайдено лист ОП користувачів у {filepath}")
    df = pd.read_excel(filepath, engine=engine, sheet_name=target_sheet, header=None)
    start_row = None
    header_row = None
    for i in range(20):
        val = df.iloc[i, 0]
        if pd.notna(val) and "Користувач" in str(val):
            header_row = i
            break
    
    if header_row is None:
        raise ValueError("Не знайдено рядок 'Користувач'")
    
    for j in range(header_row +1, df.shape[0]):
        val = df.iloc[j,0]
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

        if is_aggregate(host_name):
            continue

        if is_invalid_host(host_name):    # ← нове
            continue


        host_name = str(host_name).strip()

        for col_index,metric_name, in metrics_hosts_meta.items():
            value = df.iloc[i, col_index]
            rows_meta.append({
                "year": year, 
                "host":host_name,
                "metric":metric_name,
                "value":value,
            })


        for col_index,metric_name, in metrics_finances.items():
            value = df.iloc[i, col_index]
            rows_finances.append({
                "year": year, 
                "host":host_name,
                "metric":metric_name,
                "value":value,
            })
    
    sheet_chys = find_sheet_by_keyword(filepath, engine, "чисельн")
    df_15 = pd.read_excel(filepath,engine = engine, sheet_name =sheet_chys, 
    header = None )
        
    header_row_15 = find_header_row_by_keyword(df_15,"Чисельність копитних")
    species_row_15 = find_header_row_by_keyword(df_15, "Користувач") - 1
    start_data_row_15 = find_header_row_by_keyword(df_15, "Користувач")+ 1
    
    kop = find_section_columns(df_15 , header_row_15,"Чисельність копитних")
    hut = find_section_columns(df_15, header_row_15, "Чисельність хутрових")
    per = find_section_columns(df_15, header_row_15, "Чисельність пернатих")
    species_cols = build_species_columns(df_15, species_row_15, [kop, hut, per])


    kop_h = find_section_columns(df_15, header_row_15,  "Кількість добутих (вилучених) копитних")
    hut_h = find_section_columns(df_15, header_row_15, "Кількість добутих (вилучених) хутрових")
    per_h = find_section_columns(df_15, header_row_15, "Кількість добутих (вилучених) пернатих")
    species_cols_harvest = build_species_columns(df_15, species_row_15, [kop_h, hut_h, per_h])
    rows_populations = []
    rows_harvest = []

    for i in range(start_data_row_15, df_15.shape[0]):
        host_name = df_15.iloc[i, 0]
        
        if pd.isna(host_name):
            continue
        if is_aggregate(host_name):
            continue
        if is_invalid_host(host_name):
            continue

        host_name = str(host_name).strip()


        for col_index, species in species_cols.items():
            value = df_15.iloc[i, col_index]
            rows_populations.append({
                "year": year, "host": host_name, "species": species,
                "metric": "count", "value": value,           
            })

        for col_index, species in species_cols_harvest.items():
            value = df_15.iloc[i, col_index]
            rows_harvest.append({
                "year": year, "host": host_name, "species": species,
                "metric": "shot_heads", "value": value,            
            })



    hosts_meta = pd.DataFrame(rows_meta)
    hosts_meta["value"] = pd.to_numeric(hosts_meta["value"], errors="coerce")

    finances = pd.DataFrame(rows_finances)
    finances["value"] = pd.to_numeric(finances["value"], errors="coerce")

    populations = pd.DataFrame(rows_populations)
    populations["value"] = pd.to_numeric(populations["value"], errors="coerce")

    harvest = pd.DataFrame(rows_harvest)
    harvest["value"] = pd.to_numeric(harvest["value"], errors = "coerce")    

    relocation_events = parse_relocation_events(filepath, year, engine)

    return hosts_meta, finances, populations, harvest, relocation_events 
    
    
    


