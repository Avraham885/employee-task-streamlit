# -*- coding: utf-8 -*-
"""
Assignment App (Hungarian Algorithm) — RTL v5
- Dummy-preferred blocking: illegal pairs set to hard_constraint*10 to ensure "לא שובצו" מזוהה
- Unassigned reasons are alias-aware ("לא נמצא <Alias> מתאים")
- Hide index in all tables for readability
- Prettify Taxi headers to Hebrew (Rating, Car_Type, Is_Silent, ...)
"""

import io
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linear_sum_assignment
import plotly.express as px

DEFAULT_HARD_CONSTRAINT = 1_000_000.0

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_excel_sheet_from_bytes(xlsx_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    with io.BytesIO(xlsx_bytes) as bio:
        df = pd.read_excel(bio, sheet_name=sheet_name)
    return clean_column_names(df)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")

def vectorized_distance_cost(df_rows: pd.DataFrame, df_cols: pd.DataFrame) -> Optional[np.ndarray]:
    if not all(col in df_rows.columns for col in ["Location_X", "Location_Y"]):
        return None
    if not all(col in df_cols.columns for col in ["Location_X", "Location_Y"]):
        return None
    try:
        A = df_rows[["Location_X", "Location_Y"]].to_numpy(dtype=float)
        B = df_cols[["Location_X", "Location_Y"]].to_numpy(dtype=float)
    except Exception:
        return None
    dists = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    return dists

def _hebrew_header_map(alias_rows: str, alias_cols: str) -> Dict[str, str]:
    base = {
        "Rating": "דירוג",
        "Car_Type": "סוג רכב",
        "Is_Silent": "נסיעה שקטה",
        "Max_Distance": "מרחק מקס'",
        "Location_X": "מיקום X",
        "Location_Y": "מיקום Y",
        "Car_Type_Required": "סוג רכב נדרש",
        "Is_Silent_Preference": "העדפת שקט",
    }
    # Also map variants with aliases in parentheses
    mapped = {}
    for k, v in base.items():
        mapped[k] = v
        mapped[f"{k} ({alias_rows})"] = f"{v} ({alias_rows})"
        mapped[f"{k} ({alias_cols})"] = f"{v} ({alias_cols})"
    return mapped

def prettify_headers(df: pd.DataFrame, alias_rows: str, alias_cols: str) -> pd.DataFrame:
    mapping = _hebrew_header_map(alias_rows, alias_cols)
    cols = {c: mapping.get(c, c) for c in df.columns}
    return df.rename(columns=cols)

def build_taxi_cost_matrix(df_drivers, df_passengers, weight_distance, weight_rating_match,
                           hard_constraint, match_car_type, match_silent_pref) -> np.ndarray:
    n_rows, n_cols = len(df_drivers), len(df_passengers)
    base = np.zeros((n_rows, n_cols), dtype=float)

    dvec = vectorized_distance_cost(df_drivers, df_passengers)
    if dvec is not None:
        base += dvec * float(weight_distance)
    else:
        for i, driver in df_drivers.iterrows():
            for j, passenger in df_passengers.iterrows():
                cost = 0.0
                try:
                    if {"Location_X", "Location_Y"} <= set(driver.index) and {"Location_X", "Location_Y"} <= set(passenger.index):
                        if pd.notna(driver["Location_X"]) and pd.notna(driver["Location_Y"]) and \
                           pd.notna(passenger["Location_X"]) and pd.notna(passenger["Location_Y"]):
                            dist = float(np.sqrt((driver["Location_X"] - passenger["Location_X"])**2 +
                                                 (driver["Location_Y"] - passenger["Location_Y"])**2))
                            cost += dist * float(weight_distance)
                except Exception:
                    pass
                base[i, j] = cost

    if "Rating" in df_drivers.columns and "Rating" in df_passengers.columns:
        for i, driver in df_drivers.iterrows():
            for j, passenger in df_passengers.iterrows():
                if pd.notna(driver.get("Rating")) and pd.notna(passenger.get("Rating")):
                    base[i, j] += abs(float(driver["Rating"]) - float(passenger["Rating"])) * float(weight_rating_match)

    # Use a harsher block than dummy padding so algorithm prefers dummy for illegal pairs
    HARD_BLOCK = float(hard_constraint) * 10.0

    for i, driver in df_drivers.iterrows():
        for j, passenger in df_passengers.iterrows():
            if match_car_type and "Car_Type_Required" in df_passengers.columns and "Car_Type" in df_drivers.columns:
                req = passenger.get("Car_Type_Required")
                drt = driver.get("Car_Type")
                if pd.notna(req) and pd.notna(drt) and req != drt:
                    base[i, j] = base[i, j] + HARD_BLOCK
            if match_silent_pref and "Is_Silent_Preference" in df_passengers.columns and "Is_Silent" in df_drivers.columns:
                pref = passenger.get("Is_Silent_Preference")
                drv_silent = driver.get("Is_Silent")
                if pd.notna(pref) and bool(pref) and not bool(drv_silent):
                    base[i, j] = base[i, j] + HARD_BLOCK
            if "Max_Distance" in df_drivers.columns and dvec is not None:
                if pd.notna(driver.get("Max_Distance")) and dvec[i, j] > float(driver["Max_Distance"]):
                    base[i, j] = base[i, j] + HARD_BLOCK
    return base

def safe_compare(a, b, comp: str) -> Optional[bool]:
    if pd.isna(a) or pd.isna(b):
        return None
    try:
        if comp == "שווה (==)":
            return a == b
        if comp == "לא שווה (!=)":
            return a != b
        if comp == "קטן מ (<)":
            return a < b
        if comp == "גדול מ (>)":
            return a > b
    except Exception:
        return None
    return None

def build_custom_cost_matrix(df_a, df_b, rules: List[Dict], penalty_mode: str) -> Optional[np.ndarray]:
    if df_a.empty or df_b.empty:
        return None
    n_rows, n_cols = len(df_a), len(df_b)
    M = np.zeros((n_rows, n_cols), dtype=float)
    cols_a = set(df_a.columns); cols_b = set(df_b.columns)
    for i, row in df_a.iterrows():
        for j, col in df_b.iterrows():
            cost = 0.0
            for rule in rules:
                col_a = rule.get("col_a"); col_b = rule.get("col_b")
                comp = rule.get("comp"); penalty = float(rule.get("penalty", 0.0))
                if col_a not in cols_a or col_b not in cols_b:
                    continue
                res = safe_compare(row.get(col_a), col.get(col_b), comp)
                if penalty_mode == "כשלא מתקיים":
                    if not res: cost += penalty
                else:
                    if res: cost += penalty
            M[i, j] = cost
    return M

def _rename_with_aliases(df_assign: pd.DataFrame, row_label: str, col_label: str,
                         keep_rows: List[str], keep_cols: List[str],
                         alias_rows: str, alias_cols: str) -> pd.DataFrame:
    dup = set(keep_rows) & set(keep_cols)
    ren = {"ID_A": f"מזהה {alias_rows}".strip(),
           "ID_B": f"מזהה {alias_cols}".strip()}
    for c in keep_rows:
        key = f"A_{c}"
        val = f"{c} ({alias_rows})" if c in dup else c
        ren[key] = val
    for c in keep_cols:
        key = f"B_{c}"
        val = f"{c} ({alias_cols})" if c in dup else c
        ren[key] = val
    return df_assign.rename(columns=ren)

def solve_assignment_problem(row_ids, col_ids, cost_matrix_base: np.ndarray, row_label: str, col_label: str,
                             df_rows=None, df_cols=None, display_cols_rows=None, display_cols_cols=None,
                             alias_rows: str = "A", alias_cols: str = "B",
                             hard_constraint: float = DEFAULT_HARD_CONSTRAINT):
    start = time.perf_counter()
    n_rows, n_cols = cost_matrix_base.shape
    dim = max(n_rows, n_cols)
    padded = np.full((dim, dim), float(hard_constraint), dtype=float)
    padded[:n_rows, :n_cols] = cost_matrix_base
    row_ind, col_ind = linear_sum_assignment(padded)
    run_time = time.perf_counter() - start

    assignments = []; unassigned_records = []
    for r, c in zip(row_ind, col_ind):
        is_dummy_row = (r >= n_rows); is_dummy_col = (c >= n_cols)
        cost_val = padded[r, c]
        if not is_dummy_row and not is_dummy_col and cost_val < hard_constraint:
            assignments.append({row_label: row_ids[r], col_label: col_ids[c], "עלות": float(cost_val)})
        else:
            # Unassigned cases: matched to dummy OR illegal real-real with big cost
            if not is_dummy_row:
                unassigned_records.append({
                    "צד": row_label, "ID": row_ids[r],
                    "סיבה": f"לא נמצא {alias_cols} מתאים"
                })
            if not is_dummy_col:
                unassigned_records.append({
                    "צד": col_label, "ID": col_ids[c],
                    "סיבה": f"לא נמצא {alias_rows} מתאים"
                })

    df_assign = pd.DataFrame(assignments)
    total_cost = float(df_assign["עלות"].sum()) if not df_assign.empty else 0.0
    df_unassigned = pd.DataFrame(unassigned_records)

    # Enrich display for assignments
    if df_rows is not None and df_cols is not None:
        keep_rows = [c for c in (display_cols_rows or []) if c in df_rows.columns]
        keep_cols = [c for c in (display_cols_cols or []) if c in df_cols.columns]
        if keep_rows or keep_cols:
            tmp = df_assign.rename(columns={row_label: "ID_A", col_label: "ID_B"})
            if keep_rows:
                left = df_rows.set_index("ID")[keep_rows].add_prefix("A_")
                tmp = tmp.merge(left, left_on="ID_A", right_index=True, how="left")
            if keep_cols:
                right = df_cols.set_index("ID")[keep_cols].add_prefix("B_")
                tmp = tmp.merge(right, left_on="ID_B", right_index=True, how="left")
            ordered = ["ID_A"] + [f"A_{c}" for c in keep_rows] + ["ID_B"] + [f"B_{c}" for c in keep_cols] + ["עלות"]
            df_assign = tmp[ordered]
            df_assign = _rename_with_aliases(df_assign, row_label, col_label, keep_rows, keep_cols, alias_rows, alias_cols)
        else:
            df_assign = df_assign.rename(columns={row_label: f"מזהה {alias_rows}", col_label: f"מזהה {alias_cols}"})

    # Build detailed unassigned tables (safe when none exist)
    df_un_rows = pd.DataFrame(columns=["מי לא שובץ?", f"מזהה {alias_rows}", "סיבה"])
    df_un_cols = pd.DataFrame(columns=["מי לא שובץ?", f"מזהה {alias_cols}", "סיבה"])
    if not df_unassigned.empty and "צד" in df_unassigned.columns:
        mask_r = df_unassigned["צד"] == row_label
        if mask_r.any():
            df_un_rows = df_unassigned.loc[mask_r, ["ID", "סיבה"]].copy()
            df_un_rows.rename(columns={"ID": f"מזהה {alias_rows}"}, inplace=True)
            df_un_rows.insert(0, "מי לא שובץ?", f"{alias_rows}")
            extra_cols = [c for c in (display_cols_rows or []) if (df_rows is not None and c in df_rows.columns)]
            if extra_cols:
                df_un_rows = df_un_rows.merge(df_rows.set_index("ID")[extra_cols],
                                              left_on=f"מזהה {alias_rows}", right_index=True, how="left")
        mask_c = df_unassigned["צד"] == col_label
        if mask_c.any():
            df_un_cols = df_unassigned.loc[mask_c, ["ID", "סיבה"]].copy()
            df_un_cols.rename(columns={"ID": f"מזהה {alias_cols}"}, inplace=True)
            df_un_cols.insert(0, "מי לא שובץ?", f"{alias_cols}")
            extra_cols = [c for c in (display_cols_cols or []) if (df_cols is not None and c in df_cols.columns)]
            if extra_cols:
                df_un_cols = df_un_cols.merge(df_cols.set_index("ID")[extra_cols],
                                              left_on=f"מזהה {alias_cols}", right_index=True, how="left")

    return df_assign.sort_values(by="עלות", ascending=True), total_cost, run_time, df_un_rows, df_un_cols

# ---------------- UI ----------------
st.set_page_config(page_title="Hungarian Assignment App", layout="wide")
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] * { direction: rtl; text-align: right; }
[data-testid="stSidebar"] * { direction: rtl; text-align: right; }
/* Make table headers render LTR text correctly as needed */
[data-testid="stDataFrame"] thead, [data-testid="stDataFrame"] th { unicode-bidi: plaintext; }
</style>
""", unsafe_allow_html=True)

st.title("🧭 מערכת שיבוץ — האלגוריתם ההונגרי")
st.caption("גמיש, רספונסיבי, ונוח לדו\"ח.")

with st.sidebar:
    st.header("ℹ️ על האלגוריתם ההונגרי")
    st.write("• האלגוריתם ההונגרי פותר בעיית התאמה מינימלית בגרף דו־חלקי באמצעות חיפוש התאמה מיטבית במטריצת עלויות.")
    st.write("• הסיבוכיות בזמן היא סדר גודל O(n³); לכן ב־Benchmark נראה עקומה קובייתית בקירוב.")
    st.write("• כאשר המטריצה אינה ריבועית, מרפדים ב'דמי' ו/או מוסיפים ענישה גדולה כדי לחסום שיבוצים לא חוקיים.")
    st.divider()
    HARD_CONSTRAINT = st.number_input("ענישה עבור אילוץ קשיח (גבוה=חוסם)",
                                      min_value=1_000.0, max_value=100_000_000.0,
                                      value=DEFAULT_HARD_CONSTRAINT, step=1_000.0, format="%.0f")

tab_taxi, tab_custom, tab_bench = st.tabs(["🚕 שיבוץ נהגים לנוסעים", "🧩 שיבוץ מותאם אישית", "⏱️ מבחן ביצועים"])

# ------- Taxi -------
with tab_taxi:
    st.subheader("🚕 שיבוץ נהגים לנוסעים")
    st.markdown("**מבנה גיליונות נדרש**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Drivers**")
        st.code("ID, Location_X, Location_Y, Rating, Car_Type, Is_Silent, Max_Distance", language="text")
    with c2:
        st.markdown("**Passengers**")
        st.code("ID, Location_X, Location_Y, Rating, Car_Type_Required, Is_Silent_Preference", language="text")

    uploaded = st.file_uploader("בחר קובץ Excel עם הגיליונות: Drivers, Passengers", type=["xlsx"], key="taxi_xlsx")

    if uploaded:
        try:
            xbytes = uploaded.getvalue()
            df_drivers = load_excel_sheet_from_bytes(xbytes, "Drivers")
            df_passengers = load_excel_sheet_from_bytes(xbytes, "Passengers")

            if df_drivers.empty or df_passengers.empty:
                st.error("❌ הגיליונות חייבים להכיל נתונים.")
            elif "ID" not in df_drivers.columns or "ID" not in df_passengers.columns:
                st.error("❌ בכל גיליון חייבת להיות עמודת 'ID'.")
            else:
                with st.expander("בחר עמודות להצגה בתוצאות (אופציונלי)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        disp_dr = st.multiselect("עמודות מ-Drivers", [c for c in df_drivers.columns if c != "ID"])
                    with c2:
                        disp_ps = st.multiselect("עמודות מ-Passengers", [c for c in df_passengers.columns if c != "ID"])

                st.markdown("#### משקולות ואילוצים")
                with st.form("taxi_form"):
                    c1, c2 = st.columns(2)
                    with c1:
                        weight_distance = st.number_input("משקל למרחק", min_value=0.0, value=5.0)
                    with c2:
                        weight_rating_match = st.number_input("משקל לפער דירוגים", min_value=0.0, value=5.0)

                    cc1, cc2 = st.columns(2)
                    with cc1:
                        match_car_type = st.checkbox("דרישת סוג רכב (אם קיימת)", value=False)
                    with cc2:
                        match_silent_pref = st.checkbox("העדפת נסיעה שקטה", value=False)

                    submitted = st.form_submit_button("הרץ שיבוץ")

                if submitted:
                    with st.spinner("בונה מטריצת עלויות ומחשב שיבוץ..."):
                        rows = df_drivers["ID"].tolist()
                        cols = df_passengers["ID"].tolist()
                        base = build_taxi_cost_matrix(df_drivers, df_passengers,
                                                      weight_distance, weight_rating_match,
                                                      HARD_CONSTRAINT, match_car_type, match_silent_pref)
                        df_res, total_cost, runtime, df_un_drivers, df_un_passengers = solve_assignment_problem(
                            rows, cols, base, "נהג", "נוסע",
                            df_rows=df_drivers, df_cols=df_passengers,
                            display_cols_rows=disp_dr, display_cols_cols=disp_ps,
                            alias_rows="נהג", alias_cols="נוסע",
                            hard_constraint=HARD_CONSTRAINT
                        )
                    st.session_state.taxi_output = {
                        "df_res": df_res, "total_cost": total_cost, "runtime": runtime,
                        "df_un_drivers": df_un_drivers, "df_un_passengers": df_un_passengers
                    }

                if "taxi_output" in st.session_state:
                    out = st.session_state.taxi_output
                    df_show = prettify_headers(out["df_res"], "נהג", "נוסע")
                    st.success(f"💵 עלות כוללת: {out['total_cost']:.2f} | ⏱️ זמן ריצה: {out['runtime']:.4f} שניות")
                    st.dataframe(df_show, use_container_width=True, hide_index=True)
                    if not df_show.empty:
                        st.download_button("⬇️ הורד תוצאות שיבוץ (CSV)",
                                           data=to_csv_bytes(df_show),
                                           file_name="assignments_taxi.csv", mime="text/csv")
                    if out.get("df_un_drivers") is not None and not out["df_un_drivers"].empty:
                        st.warning("🚫 נהגים שלא שובצו")
                        st.dataframe(prettify_headers(out["df_un_drivers"], "נהג", "נוסע"), use_container_width=True, hide_index=True)
                        st.download_button("⬇️ הורד 'נהגים שלא שובצו' (CSV)",
                                           data=to_csv_bytes(prettify_headers(out["df_un_drivers"], "נהג", "נוסע")),
                                           file_name="unassigned_drivers.csv", mime="text/csv")
                    if out.get("df_un_passengers") is not None and not out["df_un_passengers"].empty:
                        st.warning("🚫 נוסעים שלא שובצו")
                        st.dataframe(prettify_headers(out["df_un_passengers"], "נהג", "נוסע"), use_container_width=True, hide_index=True)
                        st.download_button("⬇️ הורד 'נוסעים שלא שובצו' (CSV)",
                                           data=to_csv_bytes(prettify_headers(out["df_un_passengers"], "נהג", "נוסע")),
                                           file_name="unassigned_passengers.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ שגיאה: {e}")

# ------- Custom -------
with tab_custom:
    st.subheader("🧩 שיבוץ מותאם אישית")
    st.markdown("**מבנה גיליונות מומלץ:** Items_A ו-Items_B חייב להכיל 'ID'. יתר העמודות — לפי הכללים שתגדיר (למשל 'התמחות', 'ניסיון', 'מחיר').")

    uploaded2 = st.file_uploader("בחר קובץ Excel עם הגיליונות: Items_A, Items_B", type=["xlsx"], key="custom_xlsx")

    if uploaded2:
        try:
            xbytes2 = uploaded2.getvalue()
            df_a = load_excel_sheet_from_bytes(xbytes2, "Items_A")
            df_b = load_excel_sheet_from_bytes(xbytes2, "Items_B")

            if df_a.empty or df_b.empty:
                st.error("❌ הגיליונות חייבים להכיל נתונים.")
            elif "ID" not in df_a.columns or "ID" not in df_b.columns:
                st.error("❌ בכל גיליון חייבת להיות עמודת 'ID'.")
            else:
                alias_cols = st.columns(2)
                with alias_cols[0]:
                    alias_a = st.text_input("שם ישות A (לכותרות)", value="מרצה")
                with alias_cols[1]:
                    alias_b = st.text_input("שם ישות B (לכותרות)", value="קורס")

                with st.expander("בחר עמודות להצגה בתוצאות (אופציונלי)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        disp_a = st.multiselect(f"עמודות מ-Items_A ({alias_a})", [c for c in df_a.columns if c != "ID"], key="disp_a")
                    with c2:
                        disp_b = st.multiselect(f"עמודות מ-Items_B ({alias_b})", [c for c in df_b.columns if c != "ID"], key="disp_b")

                if "rules" not in st.session_state:
                    st.session_state.rules = []

                with st.form("add_rule"):
                    c1, c2 = st.columns(2)
                    with c1:
                        col_a = st.selectbox("עמודה מ-Items_A", [c for c in df_a.columns if c != "ID"])
                    with c2:
                        col_b = st.selectbox("עמודה מ-Items_B", [c for c in df_b.columns if c != "ID"])
                    comp = st.selectbox("סוג השוואה", ["שווה (==)", "לא שווה (!=)", "קטן מ (<)", "גדול מ (>)"])
                    penalty = st.number_input("ערך ענישה", min_value=0.0, value=1000.0, step=100.0)
                    c3, c4 = st.columns(2)
                    with c3:
                        penalty_mode = st.selectbox("מתי להעניש?", ["כשלא מתקיים", "כשמתקיים"])
                    with c4:
                        add_btn = st.form_submit_button("➕ הוסף כלל")
                if add_btn:
                    st.session_state.rules.append({"col_a": col_a, "col_b": col_b, "comp": comp, "penalty": penalty, "mode": penalty_mode})
                    st.rerun()

                if st.session_state.rules:
                    st.markdown("##### איך נבנית מטריצת העלויות כאן?")
                    st.markdown("- אין 'עלות בסיס'. כל כלל מוסיף קנס בהתאם לתוצאה (לפי בחירתך: כשמתקיים/כשלא מתקיים).")
                    st.markdown("- כך אפשר לבטא אילוצים קשיחים (קנס גדול מאוד) או רכים (קנס קטן).")
                    st.markdown(f"- לדוגמה: אם **התמחות {alias_a} ≠ התמחות {alias_b}** → הוסף 1e6 (חוסם). אם **ניסיון {alias_a} < נדרש** → הוסף 500.")

                    st.markdown("##### כללים שהוגדרו")
                    for i, rule in enumerate(st.session_state.rules):
                        st.write(f"**כלל {i+1}:** אם '{rule['col_a']}' ב-{alias_a} **{rule['comp']}** '{rule['col_b']}' ב-{alias_b} — הוסף {rule['penalty']}. (ענישה: {rule.get('mode','כשלא מתקיים')})")
                        if st.button("מחק", key=f"del_rule_{i}"):
                            st.session_state.rules.pop(i); st.rerun()

                st.divider()
                if st.button("בנה מטריצת עלויות והרץ שיבוץ"):
                    with st.spinner("בונה מטריצה ומחשב שיבוץ..."):
                        rows = df_a["ID"].tolist(); cols = df_b["ID"].tolist()
                        mode_final = st.session_state.rules[-1].get("mode", "כשלא מתקיים") if st.session_state.rules else "כשלא מתקיים"
                        rules_clean = [{k: v for k, v in r.items() if k in ("col_a", "col_b", "comp", "penalty")} for r in st.session_state.rules]
                        base = build_custom_cost_matrix(df_a, df_b, rules_clean, penalty_mode=mode_final)
                        if base is None:
                            st.error("לא ניתן לבנות מטריצה — בדוק את הנתונים.")
                        else:
                            df_res, total_cost, runtime, df_un_A, df_un_B = solve_assignment_problem(
                                rows, cols, base, "A_ID", "B_ID",
                                df_rows=df_a, df_cols=df_b,
                                display_cols_rows=disp_a, display_cols_cols=disp_b,
                                alias_rows=alias_a, alias_cols=alias_b,
                                hard_constraint=HARD_CONSTRAINT
                            )
                    st.session_state.custom_output = {
                        "df_res": df_res, "total_cost": total_cost, "runtime": runtime,
                        "df_un_A": df_un_A, "df_un_B": df_un_B
                    }

                if "custom_output" in st.session_state:
                    out = st.session_state.custom_output
                    st.success(f"💵 עלות כוללת: {out['total_cost']:.2f} | ⏱️ זמן ריצה: {out['runtime']:.4f} שניות")
                    st.dataframe(out["df_res"], use_container_width=True, hide_index=True)
                    if not out["df_res"].empty:
                        st.download_button("⬇️ הורד תוצאות (CSV)",
                                           data=to_csv_bytes(out["df_res"]), file_name="assignments_custom.csv", mime="text/csv")

                    if out.get("df_un_A") is not None and not out["df_un_A"].empty:
                        st.warning(f"🚫 פריטי {alias_a} שלא שובצו")
                        st.dataframe(out["df_un_A"], use_container_width=True, hide_index=True)
                        st.download_button(f"⬇️ הורד '{alias_a} שלא שובצו' (CSV)",
                                           data=to_csv_bytes(out["df_un_A"]), file_name="unassigned_A.csv", mime="text/csv")
                    if out.get("df_un_B") is not None and not out["df_un_B"].empty:
                        st.warning(f"🚫 פריטי {alias_b} שלא שובצו")
                        st.dataframe(out["df_un_B"], use_container_width=True, hide_index=True)
                        st.download_button(f"⬇️ הורד '{alias_b} שלא שובצו' (CSV)",
                                           data=to_csv_bytes(out["df_un_B"]), file_name="unassigned_B.csv", mime="text/csv")
        except Exception as e:
            st.error(f"❌ שגיאה: {e}")

# ------- Benchmark -------
with tab_bench:
    st.subheader("⏱️ מבחן ביצועים (לעקומת סיבוכיות בדו\"ח)")
    c1, c2 = st.columns(2)
    with c1:
        max_size = st.number_input("גודל מטריצה מקסימלי (N)", min_value=10, max_value=1500, value=200, step=10)
    with c2:
        num_runs = st.number_input("מספר הרצות לממוצע", min_value=1, value=5, step=1)

    if st.button("הרץ מבחן ביצועים"):
        with st.spinner("מריץ..."):
            sizes = list(range(10, int(max_size) + 1, 10))
            all_runs = []
            avg_times = []
            for n in sizes:
                rt = []
                for r in range(int(num_runs)):
                    M = np.random.rand(n, n) * 100.0
                    t0 = time.perf_counter()
                    linear_sum_assignment(M)
                    sec = time.perf_counter() - t0
                    rt.append(sec); all_runs.append((n, r+1, sec))
                avg_times.append(float(np.mean(rt)))
            df_perf = pd.DataFrame({"N": sizes, "avg_sec": avg_times})
            df_perf.rename(columns={
                "N": "גודל מטריצה (N)",
                "avg_sec": f"זמן ממוצע ({int(num_runs)} הרצות) [שניות]"
            }, inplace=True)

            st.success("✅ מבחן הביצועים הסתיים בהצלחה.")
            fig = px.line(df_perf, x="גודל מטריצה (N)", y=df_perf.columns[1], markers=True)
            fig.update_layout(title=dict(text="זמן ריצה של האלגוריתם ההונגרי כתלות בגודל המטריצה", x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)

            df_raw = pd.DataFrame(all_runs, columns=["N", "הרצה", "שניות"])
            agg = df_raw.groupby("N")["שניות"].agg(["mean", "min", "max", "std"]).reset_index()
            agg["מספר הרצות"] = int(num_runs)
            agg.rename(columns={"N": "גודל מטריצה (N)", "mean": "ממוצע (שניות)", "min": "מינימום (שניות)",
                                "max": "מקסימום (שניות)", "std": "סטיית תקן (שניות)"}, inplace=True)

            st.markdown("#### נתוני מבחן הביצועים")
            st.dataframe(agg, use_container_width=True, hide_index=True)
            st.download_button("⬇️ הורד נתוני מבחן ביצועים (CSV)", data=to_csv_bytes(agg), file_name="benchmark.csv", mime="text/csv")

            with st.expander("ראה את כל ההרצות (Raw)"):
                st.dataframe(df_raw, use_container_width=True, hide_index=True)
                st.download_button("⬇️ הורד את כל ההרצות (CSV)", data=to_csv_bytes(df_raw), file_name="benchmark_raw.csv", mime="text/csv")
