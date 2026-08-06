"""
Оновлена версія analyze_question:
- LLM-виклик №1 (розпізнавання) отримує повний список видів/угідь
  у system-промпті і використовує потужнішу модель (gpt-oss-120b).
- rapidfuzz лишається як другий шар захисту.
- raw: true/false — пропустити другий LLM-виклик, повернути сирі дані.
- topic "all" — усі п'ять категорій по заданому host одразу.
- Другий LLM-виклик тепер отримує документований контекст (заборона
  полювання, війна, реформа 2023, АЧС для кабана) — щоб не робити
  висновків "з голови", яких у сирих даних нема.
"""

import sys
import os
from dotenv import load_dotenv
from groq import Groq
import json
import sqlite3
from rapidfuzz import process
from pathlib import Path
import streamlit as st

DB_PATH = Path(__file__).parent / "db" / "volyn.db"

@st.cache_resource
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

con = get_connection()

cursor = con.cursor()
cursor.execute("SELECT name FROM species")
species_names = [row[0] for row in cursor.fetchall()]

cursor.execute("SELECT name FROM hosts")
species_hosts = [row[0] for row in cursor.fetchall()]

SPECIES_LIST_TEXT = ", ".join(species_names)
HOSTS_LIST_TEXT = ", ".join(species_hosts)

UNITS = {
    "population": "голів",
    "harvest": "голів",
    "finance": "грн",
    "staff": "осіб",
    "relocation": "голів",
}

ALL_TOPICS = ["population", "harvest", "finance", "staff", "relocation"]

GENERAL_CONTEXT = """
Заборона полювання діє з 2022 року. Повномасштабна війна триває з лютого 2022 року;
реформа Лісового агентства 2023 року спричинила блекаут/неповноту звітності за 2023-2024 роки.
"""

SPECIES_CONTEXT = {
    "Кабан": "Африканська чума свиней (АЧС) спричиняє масову смертність кабанів, з періодичними спалахами з середини 2010-х.",
}

load_dotenv()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def build_query(topic, species, host):
    """Повертає SQL-текст для однієї теми, з урахуванням наявності species/host.

    Для population/harvest/relocation: якщо species не вказано —
    сумуємо по ВСІХ видах разом (без JOIN на species), а не підставляємо
    буквальний рядок 'None' у WHERE.
    """
    if topic in ("population", "harvest", "relocation"):
        table = {
            "population": ("populations", "SUM(populations.count)"),
            "harvest": ("harvest", "SUM(harvest.shot_heads)"),
            "relocation": ("relocation_events", "SUM(relocation_events.count)"),
        }[topic]
        table_name, sum_expr = table

        joins = []
        wheres = []

        if species is not None:
            joins.append(f"JOIN species ON {table_name}.species_id = species.species_id")
            wheres.append(f"species.name = '{species}'")

        if host is not None:
            joins.append(f"JOIN hosts ON {table_name}.host_id = hosts.host_id")
            wheres.append(f"hosts.name = '{host}'")

        join_clause = "\n".join(joins)
        where_clause = ("WHERE " + " AND ".join(wheres)) if wheres else ""

        return f"""
        SELECT {table_name}.year, {sum_expr}
        FROM {table_name}
        {join_clause}
        {where_clause}
        GROUP BY {table_name}.year
        """

    if topic == "finance":
        if host is not None:
            return f"""
            SELECT finances.year, SUM(finances.total_expenses)
            FROM finances
            JOIN hosts ON hosts.host_id = finances.host_id
            WHERE hosts.name = '{host}'
            GROUP BY finances.year
            """
        return """
        SELECT finances.year, SUM(finances.total_expenses)
        FROM finances
        GROUP BY finances.year
        """

    if topic == "staff":
        if host is not None:
            return f"""
            SELECT hosts_meta.year, SUM(hosts_meta.staff_total)
            FROM hosts_meta
            JOIN hosts ON hosts.host_id = hosts_meta.host_id
            WHERE hosts.name = '{host}'
            GROUP BY hosts_meta.year
            """
        return """
        SELECT hosts_meta.year, SUM(hosts_meta.staff_total)
        FROM hosts_meta
        GROUP BY hosts_meta.year
        """

    return None


def run_query_to_text(topic, species, host):
    """Виконує запит для однієї теми, повертає готовий текстовий блок (чи None, якщо даних нема)."""
    query = build_query(topic, species, host)
    if query is None:
        return None

    cursor = con.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()

    if not rows:
        return None

    unit = UNITS[topic]
    text = ""
    for year, total in rows:
        text += f"Рік {year}: {total} {unit}\n"
    return text


def analyze_question(user_question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "Визнач тему запитання (поле topic): "
                    "'population' — чисельність, кількість голів, популяція; "
                    "'harvest' — відстріл, скільки застрелили/здобули; "
                    "'finance' — витрати, гроші, фінансування, зарплата; "
                    "'staff' — персонал, рейнджери, біологи, штат; "
                    "'relocation' — переселення, розселення, завезення тварин; "
                    "'all' — користувач хоче ВСІ категорії одразу по угіддю "
                    "(ключові слова: 'всі дані', 'все по', 'повну інформацію'). "
                    "Поле host — конкретна назва угіддя/господарства, тільки якщо користувач явно її називає. "
                    "Якщо йдеться про Волинську область/Волинь загалом, без конкретного угіддя — host: null. "
                    "Визнач також поле raw (true/false): true, якщо користувач явно хоче ЛИШЕ дані/цифри "
                    "без аналізу тренду (ключові слова: 'виведи дані', 'без аналізу', 'просто цифри', "
                    "'сирі дані', 'дай мені дані'). Інакше — false. "
                    f"Список реальних видів тварин (обери species ТОЧНО з цього списку, "
                    f"навіть якщо в питанні вжито іншу форму слова, синонім чи скорочення): {SPECIES_LIST_TEXT}. "
                    f"Список реальних угідь (обери host ТОЧНО з цього списку, "
                    f"навіть якщо в питанні вжито іншу форму слова чи скорочення): {HOSTS_LIST_TEXT}. "
                    "Поверни лише JSON з полями species, host, topic і raw (species/host — null, якщо не згадано). "
                    "Відповідь має бути у форматі JSON. "
                    'Приклад: {"species": "Кабан", "host": "Ішів", "topic": "harvest", "raw": false} '
                    'або {"species": null, "host": "Ішів", "topic": "all", "raw": true}. '
                ),
            },
            {"role": "user", "content": user_question},
        ],
        response_format={"type": "json_object"},
        model="openai/gpt-oss-120b",
        temperature=0,
    )

    parsed_data = json.loads(chat_completion.choices[0].message.content)
    species = parsed_data.get("species")
    host = parsed_data.get("host")
    topic = parsed_data.get("topic")
    raw = parsed_data.get("raw", False)

    if isinstance(species, list):
        species = species[0] if species else None
    if isinstance(host, list):
        host = host[0] if host else None

    # --- rapidfuzz як другий шар захисту ---
    if species is not None:
        result = process.extractOne(species, species_names)
        species_matched, score, _ = result
        if score < 70:
            return "вид тварини не розпізнано"
        species = species_matched

    if host is not None:
        result = process.extractOne(host, species_hosts)
        host_matched, score, _ = result
        if score < 70:
            host = None
        else:
            host = host_matched
    else:
        host = None

    # контекст (заповнюється тут, після того як species вже уточнено rapidfuzz-ом)
    species_context = SPECIES_CONTEXT.get(species, "")

    # ============ ГІЛКА "all" ============
    if topic == "all":
        if host is None:
            return "Для теми 'all' потрібно вказати конкретне угіддя."

        combined_text = ""
        for t in ALL_TOPICS:
            block = run_query_to_text(t, species, host)
            if block is not None:
                combined_text += f"\n--- {t} ---\n{block}"

        if not combined_text:
            return f"Дані по '{host}' відсутні в базі."

        if raw:
            return combined_text

        chat_completion_2 = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Ось усі наявні дані по угіддю {host}:\n{combined_text}\n"
                        f"{GENERAL_CONTEXT}\n{species_context}\n"
                        "Дай короткий загальний огляд по кожній категорії, спираючись на наведені числа та "
                        "вказаний контекст (якщо він релевантний). "
                        "Не вигадуй причин, яких немає ні в цифрах, ні в наведеному контексті. "
                        "Відповідай українською мовою."
                    ),
                },
            ],
            model="openai/gpt-oss-120b",
        )
        return chat_completion_2.choices[0].message.content

    # ============ ОДНА КОНКРЕТНА ТЕМА (як і раніше) ============
    if topic not in ALL_TOPICS:
        return "тему запитання не розпізнано"

    text = run_query_to_text(topic, species, host)
    if text is None:
        return f"Дані по темі '{topic}' відсутні в базі."

    if raw:
        return text

    label = species if species else (host if host else "по всій області")
    chat_completion_2 = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Ось дані по темі {topic} для {label}:\n{text}\n"
                    f"{GENERAL_CONTEXT}\n{species_context}\n"
                    "Проаналізуй тренд, спираючись на наведені числа та вказаний контекст (якщо він релевантний). "
                    "Не вигадуй причин, яких немає ні в цифрах, ні в наведеному контексті. "
                    "Відповідай українською мовою."
                ),
            },
        ],
        model="openai/gpt-oss-120b",
    )
    return chat_completion_2.choices[0].message.content


if __name__ == "__main__":
    while True:
        question = input("Твоє питання (або 'exit' щоб вийти): ")
        if question == "exit":
            break
        print(analyze_question(question))