"""
Этап 3: Streamlit-приложение для прогнозирования спроса.
Запуск: python -m streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, json
from pathlib import Path

st.set_page_config(
    page_title="Прогнозирование спроса | AI Analytics",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.stButton button {
    background: linear-gradient(90deg, #667eea, #764ba2);
    color: white; border: none; padding: 0.6rem 2rem;
    border-radius: 25px; font-size: 1.1rem; font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent

FEATURES = [
    "product_enc", "month", "quarter", "year", "t",
    "weekend_ratio", "days_in_month",
    "lag_1", "lag_2", "lag_3",
    "roll_mean_3", "roll_mean_6",
]

CATEGORY_ICONS = {
    "Техника":  "💻",
    "Продукты": "🛒",
    "Гигиена":  "🧴",
    "Одежда":   "👕",
}

MONTH_RU = {
    1:"январь",2:"февраль",3:"март",4:"апрель",5:"май",6:"июнь",
    7:"июль",8:"август",9:"сентябрь",10:"октябрь",11:"ноябрь",12:"декабрь"
}

@st.cache_resource
def load_artifacts():
    model  = joblib.load(BASE / "models" / "demand_forecast_model.pkl")
    le     = joblib.load(BASE / "models" / "label_encoder.pkl")
    df     = pd.read_csv(BASE / "data" / "monthly_features.csv", parse_dates=["date"])
    with open(BASE / "models" / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    return model, le, df, metrics


model, le, df_monthly, metrics = load_artifacts()
categories = df_monthly[["product","category"]].drop_duplicates().set_index("product")["category"].to_dict()

# ─── Шапка ─────────────────────────────────────────────────────────────
st.title("📈 Система прогнозирования спроса на основе ИИ")
st.markdown("*Анализ исторических продаж и прогноз на следующий месяц*")
st.divider()

# ─── Боковая панель ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Настройки")

    cats = {}
    for product in sorted(le.classes_):
        cat = categories.get(product, "Прочее")
        cats.setdefault(cat, []).append(product)

    cat_choice   = st.selectbox("📦 Категория:", ["Все категории"] + sorted(cats.keys()))
    product_list = sorted(le.classes_) if cat_choice == "Все категории" else cats[cat_choice]
    selected_product = st.selectbox("🛍️ Товар:", product_list)

    st.divider()
    st.subheader("📊 Метрики моделей")
    best = metrics["best_model"]
    for name in ["Линейная регрессия","SARIMA","Random Forest","XGBoost","Gradient Boosting"]:
        m = metrics.get(name)
        if not m:
            continue
        with st.expander(name + (" 🏆" if name == best else ""), expanded=(name == best)):
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE",  f"{m['MAE']:.0f}")
            c2.metric("RMSE", f"{m['RMSE']:.0f}")
            c3.metric("MAPE", f"{m['MAPE']:.1f}%")
    st.info(f"✅ Активна: **{best}**")
    st.caption("Цель паспорта: MAPE < 15% ✅")

# ─── Основной блок ──────────────────────────────────────────────────────
product_enc = le.transform([selected_product])[0]
sp          = df_monthly[df_monthly["product_enc"] == product_enc].copy().reset_index(drop=True)
cat_icon    = CATEGORY_ICONS.get(categories.get(selected_product, ""), "📦")
last_date   = sp["date"].max()
next_date   = last_date + pd.DateOffset(months=1)

st.subheader(f"{cat_icon} {selected_product}  —  {categories.get(selected_product,'—')}")

# Инфо о данных
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Всего продано",      f"{sp['sales'].sum():,} ед.")
c2.metric("📈 Среднее/месяц",      f"{sp['sales'].mean():.0f} ед.")
c3.metric("📅 Данные по",          last_date.strftime("%b %Y"))
c4.metric("🔮 Прогноз на",         next_date.strftime("%b %Y"))

# История продаж
st.subheader("История продаж")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sp["date"], sp["sales"], color="#667eea", linewidth=2, marker="o", markersize=3)
ax.fill_between(sp["date"], sp["sales"], alpha=0.18, color="#667eea")
ax.axvline(last_date, color="#E91E63", linewidth=1.5, linestyle="--", alpha=0.7,
           label=f"Конец данных ({last_date.strftime('%b %Y')})")
ax.set_ylabel("Продажи (ед./месяц)")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)

# ─── Прогноз ────────────────────────────────────────────────────────────
st.divider()
if st.button(f"🚀 Построить прогноз на {next_date.strftime('%B %Y')}"):

    next_dates_range = pd.date_range(next_date, periods=int(next_date.days_in_month))
    wr = (next_dates_range.weekday >= 5).sum() / int(next_date.days_in_month)

    last_row = sp.iloc[-1]
    next_features = {
        "product_enc":   product_enc,
        "month":         next_date.month,
        "quarter":       next_date.quarter,
        "year":          next_date.year,
        "t":             last_row["t"] + 1,
        "weekend_ratio": wr,
        "days_in_month": int(next_date.days_in_month),
        "lag_1":         last_row["sales"],
        "lag_2":         sp.iloc[-2]["sales"],
        "lag_3":         sp.iloc[-3]["sales"],
        "roll_mean_3":   sp["sales"].iloc[-3:].mean(),
        "roll_mean_6":   sp["sales"].iloc[-6:].mean(),
    }
    forecast   = int(model.predict(pd.DataFrame([next_features])[FEATURES])[0])
    prev       = int(last_row["sales"])
    change_pct = (forecast - prev) / prev * 100

    st.subheader("🔮 Результат прогноза")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"📦 {next_date.strftime('%B %Y')}",
              f"{forecast:,} ед.", f"{forecast-prev:+,} ед.")
    c2.metric("📊 Изменение", f"{change_pct:+.1f}%")
    c3.metric("🎯 MAPE модели", f"{metrics[best]['MAPE']:.1f}%")
    c4.metric("📐 Выходных в месяце",
              f"{int(wr * next_date.days_in_month)} дн. ({wr*100:.0f}%)")

    # График прогноза
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    history = sp.tail(12)
    ax3.plot(history["date"], history["sales"], color="#667eea",
             linewidth=2.5, marker="o", markersize=5, label="Исторические продажи")
    ax3.plot([history["date"].iloc[-1], next_date],
             [history["sales"].iloc[-1], forecast],
             color="#E91E63", linewidth=2.5, linestyle="--",
             marker="*", markersize=14, label="Прогноз")
    ax3.annotate(f"{forecast:,} ед.",
                 xy=(next_date, forecast), xytext=(-70, 22), textcoords="offset points",
                 fontsize=13, fontweight="bold", color="#E91E63",
                 arrowprops=dict(arrowstyle="->", color="#E91E63"))
    ax3.set_title(f"Прогноз — {selected_product}  |  {next_date.strftime('%B %Y')}",
                  fontsize=13, fontweight="bold")
    ax3.set_ylabel("Продажи (ед./месяц)")
    ax3.legend(); ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig3)

    st.success(f"✅ Ожидаемый спрос на {next_date.strftime('%B %Y')}: "
               f"**{forecast:,} единиц** ({change_pct:+.1f}%)")

    with st.expander("💡 Бизнес-рекомендации"):
        if change_pct > 10:
            st.warning(f"📈 Ожидается рост (+{change_pct:.1f}%). "
                       "Увеличьте складские запасы и усильте маркетинговую активность.")
        elif change_pct < -10:
            st.info(f"📉 Ожидается снижение ({change_pct:.1f}%). "
                    "Рассмотрите акции для стимулирования спроса, сократите ближайшие закупки.")
        else:
            st.success(f"➡️ Спрос стабилен ({change_pct:+.1f}%). "
                       "Поддерживайте текущий уровень запасов.")
