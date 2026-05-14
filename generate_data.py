"""
Этап 1: Генерация синтетических данных о продажах.
Каждый товар имеет свою дату последних данных — без общего "декабрьского обрыва".
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
(BASE / "data").mkdir(exist_ok=True)
(BASE / "models").mkdir(exist_ok=True)
(BASE / "charts").mkdir(exist_ok=True)

np.random.seed(42)

# ─── Ассортимент: 4 категории, 12 товаров ─────────────────────────────────
# base         - базовый дневной объём продаж
# trend        - линейный тренд (рост в день)
# season_amp   - амплитуда сезонности
# season_phase - день года с максимумом продаж
# noise        - стандартное отклонение шума
# weekend      - мультипликатор продаж в выходные
# category     - категория товара
# end          - дата последней записи (у каждого своя)

PRODUCTS = {
    # ── Техника ──────────────────────────────────────────────────────────
    "Смартфон": {
        "base": 120, "trend": 0.05, "season_amp": 30,
        "season_phase": 355, "noise": 12,
        "weekend": 1.15, "category": "Техника",
        "end": "2024-10-31",   # прогноз → ноябрь
    },
    "Ноутбук": {
        "base": 40, "trend": 0.03, "season_amp": 10,
        "season_phase": 355, "noise": 5,
        "weekend": 1.10, "category": "Техника",
        "end": "2024-07-31",   # прогноз → август (летнее затишье)
    },
    "Наушники": {
        "base": 75, "trend": 0.02, "season_amp": 20,
        "season_phase": 355, "noise": 8,
        "weekend": 1.12, "category": "Техника",
        "end": "2024-12-31",   # прогноз → январь (один из немногих с декабрём)
    },
    "Умные часы": {
        "base": 50, "trend": 0.04, "season_amp": 15,
        "season_phase": 355, "noise": 6,
        "weekend": 1.10, "category": "Техника",
        "end": "2024-05-31",   # прогноз → июнь
    },
    # ── Продукты питания ─────────────────────────────────────────────────
    "Хлеб": {
        "base": 500, "trend": 0.01, "season_amp": 20,
        "season_phase": 1, "noise": 25,
        "weekend": 1.20, "category": "Продукты",
        "end": "2024-08-31",   # прогноз → сентябрь
    },
    "Молоко": {
        "base": 380, "trend": 0.01, "season_amp": 30,
        "season_phase": 1, "noise": 20,
        "weekend": 1.18, "category": "Продукты",
        "end": "2024-11-30",   # прогноз → декабрь (предпраздничный рост)
    },
    "Кофе": {
        "base": 150, "trend": 0.03, "season_amp": 35,
        "season_phase": 30, "noise": 15,
        "weekend": 1.25, "category": "Продукты",
        "end": "2024-06-30",   # прогноз → июль (летний спад)
    },
    "Йогурт": {
        "base": 220, "trend": 0.02, "season_amp": 40,
        "season_phase": 172, "noise": 18,
        "weekend": 1.15, "category": "Продукты",
        "end": "2024-09-30",   # прогноз → октябрь (конец летнего пика)
    },
    # ── Товары первой необходимости / гигиена ─────────────────────────────
    "Шампунь": {
        "base": 90, "trend": 0.01, "season_amp": 8,
        "season_phase": 90, "noise": 7,
        "weekend": 1.08, "category": "Гигиена",
        "end": "2024-12-31",   # прогноз → январь
    },
    "Стиральный порошок": {
        "base": 70, "trend": 0.01, "season_amp": 10,
        "season_phase": 90, "noise": 6,
        "weekend": 1.05, "category": "Гигиена",
        "end": "2024-04-30",   # прогноз → май (весенняя уборка)
    },
    # ── Одежда и спорт ────────────────────────────────────────────────────
    "Спортивная одежда": {
        "base": 60, "trend": 0.02, "season_amp": 25,
        "season_phase": 90, "noise": 8,
        "weekend": 1.30, "category": "Одежда",
        "end": "2024-03-31",   # прогноз → апрель (начало сезона)
    },
    "Зимняя куртка": {
        "base": 25, "trend": 0.01, "season_amp": 22,
        "season_phase": 335, "noise": 5,
        "weekend": 1.20, "category": "Одежда",
        "end": "2024-10-31",   # прогноз → ноябрь (начало зимнего сезона)
    },
}

records = []

for product, cfg in PRODUCTS.items():
    date_range = pd.date_range(start="2022-01-01", end=cfg["end"], freq="D")

    for i, date in enumerate(date_range):
        trend_val = cfg["base"] + cfg["trend"] * i

        season_val = cfg["season_amp"] * np.sin(
            2 * np.pi * (date.dayofyear - cfg["season_phase"]) / 365
        )

        holiday_boost = 0
        if date.month == 12:
            holiday_boost = cfg["base"] * 0.40
        elif date.month == 11 and date.day >= 20:
            holiday_boost = cfg["base"] * 0.25
        elif date.month == 3 and 1 <= date.day <= 10:
            holiday_boost = cfg["base"] * 0.15
        elif date.month == 1 and date.day <= 10:
            if cfg["category"] == "Продукты":
                holiday_boost = cfg["base"] * 0.10
            else:
                holiday_boost = -cfg["base"] * 0.10

        is_weekend = date.weekday() >= 5
        weekend_boost = (cfg["weekend"] - 1.0) * trend_val if is_weekend else 0

        sales = trend_val + season_val + holiday_boost + weekend_boost
        sales += np.random.normal(0, cfg["noise"])
        sales = max(0, round(sales))

        records.append({
            "date":        date,
            "product":     product,
            "category":    cfg["category"],
            "sales":       int(sales),
            "month":       date.month,
            "day_of_week": date.weekday(),
            "is_weekend":  int(is_weekend),
            "quarter":     date.quarter,
        })

df = pd.DataFrame(records)
df.to_csv(BASE / "data" / "sales_data.csv", index=False)

print(f"✅ Данные сгенерированы: {len(df)} строк, {df['product'].nunique()} товаров")
print()

summary = (df.groupby(["category", "product"])
           .agg(end=("date", "max"), months=("date", lambda x: x.dt.to_period("M").nunique()),
                mean_sales=("sales", "mean"))
           .round(1))
summary["прогноз"] = pd.to_datetime(summary["end"]) + pd.DateOffset(months=1)
summary["прогноз"] = summary["прогноз"].dt.strftime("%b %Y")
summary["end"]     = pd.to_datetime(summary["end"]).dt.strftime("%b %Y")
summary.columns    = ["данные до", "месяцев", "ср. продажи/день", "след. прогноз"]
print(summary.to_string())
