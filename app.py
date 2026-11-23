"""Personal Finance Analytics Dashboard (Streamlit). Run with: streamlit run app.py"""
import os
import json
import sqlite3
from contextlib import closing

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="My Finance Dashboard", page_icon="💰", layout="wide")

CURRENCY = "₹"
HERE = os.path.dirname(os.path.abspath(__file__))
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
STATUS_COLORS = {"Safe": "#2e8b57", "Warning": "#f5a623", "Exceeded": "#dc143c"}


# ------------------------------------------------------------------ files and database helpers
def find_file(name):
    """Look for a project file next to app.py, in ./outputs or in /content/outputs."""
    for folder in (HERE, os.path.join(HERE, "outputs"), "/content/outputs"):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


DB_PATH = find_file("finance.db")
if DB_PATH is None:
    st.error("The database file finance.db was not found. Please run the notebook first.")
    st.stop()


def run_sql(query, params=()):
    """Read data from SQLite and return a DataFrame."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query(query, conn, params=params)


def execute_sql(statement, params=(), many=False):
    """Run an INSERT / UPDATE statement and commit."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        if many:
            conn.executemany(statement, params)
        else:
            conn.execute(statement, params)
        conn.commit()


def money(x):
    """Format a number as currency."""
    return f"{CURRENCY}{x:,.0f}"


@st.cache_data(show_spinner=False)
def load_transactions(version):
    """All transactions. `version` is the database file time, so the cache refreshes after any change."""
    df = run_sql("SELECT * FROM transactions")
    df["date"] = pd.to_datetime(df["date"])
    df["date_only"] = pd.to_datetime(df["date_only"])
    return df


@st.cache_data(show_spinner=False)
def load_anomalies(version):
    """Flagged transactions with their explanations."""
    df = run_sql("SELECT * FROM anomalies")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_budget(version):
    """Monthly budget per category."""
    return run_sql("SELECT category, monthly_budget, source FROM budget")


@st.cache_resource(show_spinner=False)
def load_pickle(path):
    """Load a saved model bundle once."""
    return joblib.load(path) if path else None


def load_metrics():
    """Read the saved model metrics (JSON)."""
    path = find_file("metrics.json")
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def monthly_table(df):
    """Month-wise income, expense, savings and savings rate (months without data are filled with 0)."""
    if df.empty:
        return pd.DataFrame(columns=["year_month", "income", "expense", "savings", "savings_rate"])
    m = df.pivot_table(index="year_month", columns="type", values="amount", aggfunc="sum", fill_value=0)
    for c in ("Income", "Expense"):
        if c not in m.columns:
            m[c] = 0.0
    idx = pd.period_range(pd.Period(m.index.min(), "M"), pd.Period(m.index.max(), "M"), freq="M")
    m.index = pd.PeriodIndex(m.index, freq="M")
    m = m.reindex(idx, fill_value=0.0)
    out = pd.DataFrame({"year_month": idx.astype(str), "income": m["Income"].values, "expense": m["Expense"].values})
    out["savings"] = out["income"] - out["expense"]
    out["savings_rate"] = np.where(out["income"] > 0, out["savings"] / out["income"].replace(0, np.nan) * 100, np.nan)
    out["cum_savings"] = out["savings"].cumsum()
    return out


def forecast_next(model_bundle, expense_values, last_period, steps=3):
    """Predict the next months one at a time (each prediction feeds the next month's lag features)."""
    model, features = model_bundle["model"], model_bundle["features"]
    hist = list(map(float, expense_values))
    rows = []
    t_next = len(hist)
    for k in range(1, steps + 1):
        period = last_period + k
        feats = {"lag1": hist[-1], "lag2": hist[-2], "roll3_prev": float(np.mean(hist[-3:])),
                 "month_sin": np.sin(2 * np.pi * period.month / 12), "month_cos": np.cos(2 * np.pi * period.month / 12),
                 "t": t_next}
        pred = max(float(model.predict(pd.DataFrame([feats])[features])[0]), 0.0)
        rows.append({"year_month": str(period), "predicted_expense": pred})
        hist.append(pred)
        t_next += 1
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ load data and sidebar filters
version = os.path.getmtime(DB_PATH)
tx = load_transactions(version)
anomalies = load_anomalies(version)
model_bundle = load_pickle(find_file("model.pkl"))
anomaly_bundle = load_pickle(find_file("anomaly_model.pkl"))
metrics = load_metrics()

st.title("💰 My Personal Finance Dashboard")
st.caption("Explore your income, spending, savings and unusual transactions. Use the filters on the left to focus on a period, category or payment mode.")

if tx.empty:
    st.warning("There are no transactions in the database yet.")
    st.stop()

st.sidebar.header("🔎 Filters")
min_d, max_d = tx["date"].min().date(), tx["date"].max().date()
date_sel = st.sidebar.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
                                 help="Only transactions inside this period are shown.")
if isinstance(date_sel, (list, tuple)):
    start, end = (date_sel[0], date_sel[-1]) if len(date_sel) else (min_d, max_d)
else:
    start = end = date_sel
types = st.sidebar.multiselect("Transaction type", ["Income", "Expense"], default=["Income", "Expense"],
                               help="Income = money received, Expense = money spent.")
all_cats = sorted(tx["category"].unique())
cats = st.sidebar.multiselect("Category", all_cats, default=all_cats, help="Pick one or more spending/income categories.")
all_modes = sorted(tx["payment_mode"].dropna().unique())
modes = st.sidebar.multiselect("Payment mode", all_modes, default=all_modes, help="How the payment was made (cash, card, UPI...).")

mask = (tx["date_only"].dt.date >= start) & (tx["date_only"].dt.date <= end) & tx["type"].isin(types) \
    & tx["category"].isin(cats) & tx["payment_mode"].isin(modes)
f = tx[mask].copy()
exp_f = f[f["type"] == "Expense"]
st.sidebar.markdown(f"**{len(f):,}** transactions match your filters.")
if f.empty:
    st.warning("No transactions match the current filters. Try widening the date range or selecting more categories.")

tabs = st.tabs(["📊 Overview", "🛒 Spending", "🎯 Budget", "💵 Savings", "🚨 Unusual spending",
                "🔮 Prediction", "🔎 Data explorer", "➕ Add transaction"])

# ------------------------------------------------------------------ (a) Overview
with tabs[0]:
    income = f.loc[f["type"] == "Income", "amount"].sum()
    expense = exp_f["amount"].sum()
    net = income - expense
    rate = (net / income * 100) if income > 0 else None
    n_unusual = int(anomalies["txn_id"].isin(f["txn_id"]).sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total income", money(income), help="All money received in the selected period.")
    c2.metric("Total expense", money(expense), help="All money spent in the selected period.")
    c3.metric("Net savings", money(net), help="Income minus expense. Negative means you spent more than you earned.")
    c4.metric("Savings rate", f"{rate:.1f}%" if rate is not None else "n/a",
              help="Share of income you kept. Not available when no income is recorded in the period.")
    c5.metric("Unusual transactions", n_unusual, help="Payments that look very different from your normal spending.")
    if not exp_f.empty:
        top_cat = exp_f.groupby("category")["amount"].sum().sort_values(ascending=False)
        st.info(f"Your biggest spending category is **{top_cat.index[0]}** ({money(top_cat.iloc[0])}, "
                f"{top_cat.iloc[0] / top_cat.sum() * 100:.0f}% of all spending). "
                f"On average you spend **{money(exp_f['amount'].sum() / max(exp_f['year_month'].nunique(), 1))}** in a month.")
    mt = monthly_table(f)
    if not mt.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mt["year_month"], y=mt["income"], name="Income", line=dict(color="#2e8b57")))
        fig.add_trace(go.Scatter(x=mt["year_month"], y=mt["expense"], name="Expense", line=dict(color="#dc143c")))
        fig.update_layout(title="Income and expense by month", template="plotly_white", hovermode="x unified",
                          xaxis_title="Month", yaxis_title=f"Amount ({CURRENCY})")
        st.plotly_chart(fig)

# ------------------------------------------------------------------ (b) Spending analysis
with tabs[1]:
    if exp_f.empty:
        st.info("No expenses in the current selection.")
    else:
        left, right = st.columns(2)
        cs = exp_f.groupby("category")["amount"].sum().sort_values().tail(12).reset_index()
        fig = px.bar(cs, x="amount", y="category", orientation="h", title="Where your money goes (top 12 categories)",
                     labels={"amount": f"Total spent ({CURRENCY})", "category": "Category"}, template="plotly_white")
        left.plotly_chart(fig)
        wd = exp_f.groupby("weekday")["amount"].sum().reindex(WEEKDAYS).fillna(0).reset_index()
        fig = px.bar(wd, x="weekday", y="amount", title="Spending by day of the week",
                     labels={"amount": f"Total spent ({CURRENCY})", "weekday": "Day"}, template="plotly_white")
        right.plotly_chart(fig)
        mm = exp_f.groupby("year_month")["amount"].sum().reset_index()
        fig = px.line(mm, x="year_month", y="amount", markers=True, title="Monthly spending trend",
                      labels={"amount": f"Total spent ({CURRENCY})", "year_month": "Month"}, template="plotly_white")
        st.plotly_chart(fig)
        st.caption("Tip: hover over any bar or point to see exact values.")

# ------------------------------------------------------------------ (c) Budget
with tabs[2]:
    st.subheader("How much of your monthly budget have you used?")
    st.write("A category is **Safe** below 80% of its budget, in **Warning** between 80% and 100%, and **Exceeded** above 100%.")
    budget_df = load_budget(version)
    with st.expander("✏️ Edit your monthly budgets", expanded=False):
        st.caption("Change the numbers, then press Save to keep them. The chart below updates immediately.")
        edited = st.data_editor(
            budget_df[["category", "monthly_budget"]], hide_index=True, disabled=["category"], key="budget_editor",
            column_config={"category": st.column_config.TextColumn("Category"),
                           "monthly_budget": st.column_config.NumberColumn(f"Monthly budget ({CURRENCY})", min_value=1.0, step=100.0, format="%.0f")})
        if st.button("💾 Save my budgets"):
            rows = [(float(r.monthly_budget), "user (dashboard)", r.category) for r in edited.itertuples() if r.monthly_budget and r.monthly_budget > 0]
            execute_sql("UPDATE budget SET monthly_budget = ?, source = ? WHERE category = ?", rows, many=True)
            st.cache_data.clear()
            st.success("Budgets saved.")
    months_avail = sorted(exp_f["year_month"].unique(), reverse=True)
    if not months_avail:
        st.info("No expenses in the current selection.")
    else:
        month = st.selectbox("Month", months_avail, help="Pick the month to check against your budget.")
        actual_m = exp_f[exp_f["year_month"] == month].groupby("category")["amount"].sum()
        bud = edited.set_index("category")["monthly_budget"]
        table = pd.DataFrame({"spent": actual_m}).join(bud, how="left").dropna(subset=["monthly_budget"])
        table["used_pct"] = table["spent"] / table["monthly_budget"] * 100
        table["status"] = np.where(table["used_pct"] < 80, "Safe", np.where(table["used_pct"] <= 100, "Warning", "Exceeded"))
        table = table.sort_values("used_pct", ascending=False).reset_index().rename(columns={"index": "category"})
        c1, c2, c3 = st.columns(3)
        c1.metric("✅ Safe", int((table["status"] == "Safe").sum()))
        c2.metric("⚠️ Warning", int((table["status"] == "Warning").sum()))
        c3.metric("🚨 Exceeded", int((table["status"] == "Exceeded").sum()))
        fig = px.bar(table, x="category", y="used_pct", color="status", color_discrete_map=STATUS_COLORS,
                     category_orders={"status": ["Safe", "Warning", "Exceeded"]}, hover_data={"spent": ":,.0f", "monthly_budget": ":,.0f"},
                     labels={"used_pct": "Budget used (%)", "category": "Category"}, title=f"Budget used in {month}", template="plotly_white")
        fig.add_hline(y=100, line_dash="dash", line_color="red")
        fig.add_hline(y=80, line_dash="dot", line_color="orange")
        st.plotly_chart(fig)
        show = table.rename(columns={"category": "Category", "spent": "Spent", "monthly_budget": "Budget", "used_pct": "Used %", "status": "Status"})
        st.dataframe(show.round(1), hide_index=True)

# ------------------------------------------------------------------ (d) Savings
with tabs[3]:
    mt = monthly_table(f)
    if mt.empty or mt["income"].sum() == 0:
        st.info("Savings need income. No income is recorded in the current selection (check the Type filter).")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=mt["year_month"], y=mt["savings"], name="Savings",
                             marker_color=np.where(mt["savings"] >= 0, "#2e8b57", "#cd5c5c")), secondary_y=False)
        fig.add_trace(go.Scatter(x=mt["year_month"], y=mt["savings_rate"].clip(-200, 100), name="Savings rate (%)",
                                 mode="lines+markers", line=dict(color="navy")), secondary_y=True)
        fig.update_layout(title="Monthly savings and savings rate", template="plotly_white", hovermode="x unified", xaxis_title="Month")
        fig.update_yaxes(title_text=f"Savings ({CURRENCY})", secondary_y=False)
        fig.update_yaxes(title_text="Savings rate (%)", secondary_y=True)
        st.plotly_chart(fig)
        fig = px.area(mt, x="year_month", y="cum_savings", title="Total savings built up over time",
                      labels={"cum_savings": f"Cumulative savings ({CURRENCY})", "year_month": "Month"}, template="plotly_white")
        st.plotly_chart(fig)
        st.caption("Green bars are months where you saved money, red bars are months where spending was higher than recorded income.")

# ------------------------------------------------------------------ (e) Anomalies
with tabs[4]:
    st.subheader("Transactions that look unusual")
    st.write("A machine-learning model (Isolation Forest) compared every payment with your normal behaviour. "
             "These payments stood out. A flag does not mean something is wrong: it means *take a second look*.")
    an_f = anomalies[anomalies["txn_id"].isin(f["txn_id"])].sort_values("anomaly_score", ascending=False)
    if an_f.empty:
        st.success("No unusual transactions in the current selection.")
    else:
        show = an_f.merge(tx[["txn_id", "payment_mode", "subcategory", "note"]], on="txn_id", how="left")
        show["date"] = show["date"].dt.strftime("%d %b %Y")
        st.dataframe(show[["date", "category", "subcategory", "amount", "payment_mode", "note", "explanation"]]
                     .rename(columns={"date": "Date", "category": "Category", "subcategory": "Sub-category", "amount": "Amount",
                                      "payment_mode": "Paid with", "note": "Note", "explanation": "Why it was flagged"}),
                     hide_index=True)
    if not exp_f.empty:
        sc = exp_f.merge(anomalies[["txn_id", "anomaly_score"]], on="txn_id", how="left")
        sc["Status"] = np.where(sc["anomaly_score"].notna(), "Unusual", "Normal")
        fig = px.scatter(sc, x="date", y="amount", color="Status", log_y=True, color_discrete_map={"Normal": "#b0b0b0", "Unusual": "#e60000"},
                         hover_data={"category": True, "note": True, "date": "|%d %b %Y"},
                         title="Your expenses over time (unusual ones in red)", labels={"amount": f"Amount ({CURRENCY}, log scale)", "date": "Date"},
                         template="plotly_white")
        st.plotly_chart(fig)

# ------------------------------------------------------------------ (f) Prediction
with tabs[5]:
    st.subheader("What will I probably spend in the next months?")
    full_m = monthly_table(tx)
    if model_bundle is None or len(full_m) < 4:
        st.info("The prediction model is not available yet. Run the notebook up to the model-saving step.")
    else:
        fc = forecast_next(model_bundle, full_m["expense"].tolist(), pd.Period(full_m["year_month"].iloc[-1], freq="M"), 3)
        rmse = float(model_bundle.get("heldout_rmse", 0))
        fc["lower"] = (fc["predicted_expense"] - 1.96 * rmse).clip(lower=0)
        fc["upper"] = fc["predicted_expense"] + 1.96 * rmse
        c1, c2, c3 = st.columns(3)
        for col, (_, r) in zip((c1, c2, c3), fc.iterrows()):
            col.metric(f"Expected spending in {r['year_month']}", money(r["predicted_expense"]),
                       help=f"Likely range: {money(r['lower'])} to {money(r['upper'])}")
        hist = full_m.tail(18)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["year_month"], y=hist["expense"], name="Your spending", mode="lines+markers", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=[hist["year_month"].iloc[-1]] + fc["year_month"].tolist(),
                                 y=[hist["expense"].iloc[-1]] + fc["predicted_expense"].tolist(), name="Forecast",
                                 mode="lines+markers", line=dict(color="crimson", dash="dash")))
        fig.add_trace(go.Scatter(x=fc["year_month"].tolist() + fc["year_month"].tolist()[::-1],
                                 y=fc["upper"].tolist() + fc["lower"].tolist()[::-1], fill="toself", fillcolor="rgba(220,20,60,0.12)",
                                 line=dict(color="rgba(0,0,0,0)"), name="Likely range", hoverinfo="skip"))
        fig.update_layout(title="Monthly spending: history and forecast", template="plotly_white", xaxis_title="Month",
                          yaxis_title=f"Spending ({CURRENCY})", hovermode="x unified")
        st.plotly_chart(fig)
        st.markdown("#### How reliable is this forecast?")
        if metrics:
            hm = metrics["heldout_metrics"]
            st.write(f"The model (**{metrics['best_model']}**) was tested on {metrics['n_test_months']} months it had never seen.")
            st.markdown(f"- On average, its monthly guess was off by about **{money(hm['MAE'])}**.\n"
                        f"- A typical miss, with big misses counted more heavily, was **{money(hm['RMSE'])}**.\n"
                        f"- In percentage terms it was off by roughly **{hm['MAPE_%']:.0f}%** per month.\n"
                        f"- Score (R²): **{hm['R2']:.2f}**. Closer to 1 is better; zero or below means it was no better than guessing your average month.")
        st.caption("This is an estimate based only on your past spending. Big one-off purchases, travel or income changes are not known to the model.")

# ------------------------------------------------------------------ (g) Data explorer
with tabs[6]:
    st.subheader("Browse your transactions")
    query = st.text_input("Search (category, note, payment mode...)", help="Type any word, e.g. 'food' or 'cash'.")
    view = f.copy()
    if query:
        q = query.strip().lower()
        hay = (view["category"].astype(str) + " " + view["subcategory"].astype(str) + " " + view["note"].astype(str)
               + " " + view["payment_mode"].astype(str)).str.lower()
        view = view[hay.str.contains(q, regex=False)]
    cols = ["date", "type", "category", "subcategory", "amount", "payment_mode", "note"]
    view = view.sort_values("date", ascending=False)[cols]
    st.write(f"Showing **{len(view):,}** transactions.")
    st.dataframe(view, hide_index=True)
    st.download_button("⬇️ Download as CSV", data=view.to_csv(index=False).encode("utf-8"), file_name="my_transactions.csv",
                       mime="text/csv", help="Saves the table you see here (with filters and search applied).")

# ------------------------------------------------------------------ (h) Add a transaction
with tabs[7]:
    st.subheader("Add a new transaction")
    st.write("Enter a payment below. It is saved to your database and checked right away to see whether it looks unusual.")
    if "add_msg" in st.session_state:
        level, text = st.session_state.pop("add_msg")
        getattr(st, level)(text)
    NEW = "➕ New category..."
    with st.form("add_txn"):
        a, b = st.columns(2)
        t_date = a.date_input("Date", value=max_d, help="When the payment happened.")
        t_type = b.selectbox("Type", ["Expense", "Income"])
        t_cat = a.selectbox("Category", sorted(tx["category"].unique()) + [NEW])
        t_newcat = b.text_input("New category name (only if you chose 'New category')")
        t_mode = a.selectbox("Payment mode", sorted(tx["payment_mode"].dropna().unique()))
        t_amount = b.number_input(f"Amount ({CURRENCY})", min_value=0.0, step=10.0, format="%.2f")
        t_note = st.text_input("Note (optional)")
        submitted = st.form_submit_button("Save transaction")

    if submitted:
        category = t_newcat.strip().title() if t_cat == NEW else t_cat
        if t_amount <= 0:
            st.session_state["add_msg"] = ("error", "Please enter an amount greater than zero.")
        elif not category:
            st.session_state["add_msg"] = ("error", "Please type a name for the new category.")
        else:
            try:
                ts = pd.Timestamp(t_date)
                hist_c = tx[(tx["type"] == t_type) & (tx["category"] == category)]["amount"]
                mean_c, std_c = (hist_c.mean(), hist_c.std()) if len(hist_c) else (np.nan, np.nan)
                z = float((t_amount - mean_c) / std_c) if len(hist_c) > 1 and std_c and std_c > 0 else 0.0
                p95 = hist_c.quantile(0.95) if len(hist_c) >= 5 else np.nan
                reasons = []
                if len(hist_c) >= 5 and z > 2:
                    reasons.append(f"it is {z:.1f} standard deviations above your usual {category} payment of {money(mean_c)}")
                if len(hist_c) >= 5 and t_amount > p95:
                    reasons.append(f"it is larger than 95% of your past {category} payments")
                score, model_flag = None, False
                if anomaly_bundle is not None and t_type == "Expense":
                    cat_code = anomaly_bundle["category_map"].get(category, -1)
                    freq = float((tx["category"] == category).mean())
                    row = {"amount": t_amount, "log_amount": np.log1p(t_amount), "category_code": cat_code, "weekday_num": ts.dayofweek,
                           "month": ts.month, "is_weekend": int(ts.dayofweek >= 5), "amount_zscore_cat": z, "category_freq": freq}
                    X = pd.DataFrame([row])[anomaly_bundle["features"]]
                    Xs = anomaly_bundle["scaler"].transform(X)
                    score = float(-anomaly_bundle["model"].score_samples(Xs)[0])
                    model_flag = bool(anomaly_bundle["model"].predict(Xs)[0] == -1)
                    if model_flag and not reasons:
                        reasons.append("the anomaly model finds this combination of amount, category and timing rare")
                unusual = bool(reasons) and t_type == "Expense"
                new_id = int(run_sql("SELECT COALESCE(MAX(txn_id), 0) + 1 AS n FROM transactions")["n"].iloc[0])
                known = tx.drop_duplicates("category").set_index("category")["category_code"]
                code = int(known.get(category, tx["category_code"].max() + 1))
                execute_sql("""INSERT INTO transactions (txn_id, date, date_only, year, month, year_month, weekday, weekday_num, is_weekend,
                               quarter, category, subcategory, payment_mode, type, amount, log_amount, amount_zscore_cat, category_freq,
                               category_code, is_extreme_outlier, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (new_id, ts.strftime("%Y-%m-%d %H:%M:%S"), ts.strftime("%Y-%m-%d"), ts.year, ts.month, ts.strftime("%Y-%m"),
                             ts.day_name(), ts.dayofweek, int(ts.dayofweek >= 5), ts.quarter, category, "Unknown", t_mode, t_type,
                             float(t_amount), float(np.log1p(t_amount)), z, 0.0, code, int(unusual), t_note.strip()))
                if unusual:
                    explanation = f"Newly added {category} payment of {t_amount:,.0f}: " + "; ".join(reasons) + "."
                    execute_sql("""INSERT OR REPLACE INTO anomalies (txn_id, date, category, amount, anomaly_score, anomaly_flag, category_mean,
                                   category_std, z_score, explanation) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                                (new_id, ts.strftime("%Y-%m-%d %H:%M:%S"), category, float(t_amount), score if score is not None else 0.0, 1,
                                 None if np.isnan(mean_c) else float(mean_c), None if np.isnan(std_c) else float(std_c), z, explanation))
                    st.session_state["add_msg"] = ("warning", f"Saved. ⚠️ This payment looks **unusual**: {'; '.join(reasons)}. "
                                                   "If it is correct, no action is needed.")
                elif len(hist_c) < 5:
                    st.session_state["add_msg"] = ("info", f"Saved. You have very few past '{category}' payments, so there is not enough history to judge it yet.")
                else:
                    st.session_state["add_msg"] = ("success", f"Saved. ✅ This looks like a normal {category} payment (usual average: {money(mean_c)}).")
                st.cache_data.clear()
                st.rerun()
            except Exception as exc:
                st.session_state["add_msg"] = ("error", f"Sorry, the transaction could not be saved: {exc}")
                st.rerun()
    st.caption("If you cannot see a new transaction in the other tabs, widen the date range in the sidebar.")
