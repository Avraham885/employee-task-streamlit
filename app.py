# -*- coding: utf-8 -*-
"""
Shift Scheduler — Hungarian (v7.8a)
Base: v7.7i with bug fixes and per-day 'required' roles.
Changes:
1) Fix: added gini() to avoid NameError.
2) NEW: Per-day required roles (e.g., Sunday morning "אחמש" חובה). UI stores per day; algorithm uses min-1 when required.
   - Backward compatible: falls back to legacy per-shift REQUIRED_BY_SHIFT if per-day not set.
3) Stronger file validation & normalization for Employees sheet (columns, dtypes).
4) Control tab now shows dispersion (CV/Gini) as fairness KPIs.
5) Share/print table respects required min staffing display ('לא שובץ' when empty).
6) General hardening and small UX copy edits.
"""
st.set_page_config(page_title="שיבוץ משמרות לבתי קפה ומסעדות")

# ---- Hotfix guard: ensure effective_plan_for_day exists before any call ----
try:
    effective_plan_for_day  # type: ignore  # noqa: F821
except Exception:
    pass
else:
    _EPFD_ALREADY = True
if not locals().get("_EPFD_ALREADY", False):
    from typing import Dict

# --- Effective plan resolver (custom overrides defaults + business days) ---
from typing import Dict
def effective_plan_for_day(day_code: str) -> Dict[str, Dict[str,int]]:
    custom = WEEK_PLAN.get(day_code, {}) if 'WEEK_PLAN' in globals() else {}
    if custom:
        return custom
    # Respect business days config (default Sun-Fri); if day is not active and no custom plan -> no schedule
    biz = st.session_state.get("CONFIG", {}).get("business_days", ["Sun","Mon","Tue","Wed","Thu","Fri"])
    if day_code not in biz:
        return {}
    eff: Dict[str, Dict[str,int]] = {}
    for s in SHIFTS:
        base = DEFAULT_STAFFING.get(s, {}) if 'DEFAULT_STAFFING' in globals() else {}
        has_demand = any(int(base.get(r, 0)) > 0 for r in ROLES)
        has_required = any(bool(REQUIRED_BY_SHIFT.get(s, {}).get(r, False)) for r in ROLES) if 'REQUIRED_BY_SHIFT' in globals() else False
        if has_demand or has_required:
            eff[s] = {r: int(base.get(r, 0)) for r in ROLES}
    return eff
# --- End ---


def effective_plan_for_day(day_code: str) -> Dict[str, Dict[str,int]]:  # fallback (minimal)
        custom = WEEK_PLAN.get(day_code, {}) if 'WEEK_PLAN' in globals() else {}
        if custom:
            return custom
        eff: Dict[str, Dict[str,int]] = {}
        for s in SHIFTS if 'SHIFTS' in globals() else []:
            base = DEFAULT_STAFFING.get(s, {}) if 'DEFAULT_STAFFING' in globals() else {}
            has_demand = any(int(base.get(r, 0)) > 0 for r in (ROLES if 'ROLES' in globals() else []))
            has_required = any(bool(REQUIRED_BY_SHIFT.get(s, {}).get(r, False)) for r in (ROLES if 'ROLES' in globals() else [])) if 'REQUIRED_BY_SHIFT' in globals() else False
            if has_demand or has_required:
                eff[s] = {r: int(base.get(r, 0)) for r in (ROLES if 'ROLES' in globals() else [])}
        _EPFD_ALREADY = True
        return eff
# ---------------------------------------------------------------------------
import io
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linear_sum_assignment

# ----------------- Constants & Layout -----------------
HARD_CONSTRAINT = 1_000_000.0
HARD_BLOCK = HARD_CONSTRAINT * 10.0

DAYS_ORDER = [
    ("Sun", "ראשון"),
    ("Mon", "שני"),
    ("Tue", "שלישי"),
    ("Wed", "רביעי"),
    ("Thu", "חמישי"),
    ("Fri", "שישי"),
    ("Sat", "שבת"),
]

st.set_page_config(page_title="🗓️ שיבוץ משמרות — v7.8a", layout="wide")

def rtl_css():
    st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] * { direction: rtl; text-align: right; }
    [data-testid="stSidebar"]{display:none;}
    .chipline{display:flex;align-items:center;gap:8px;padding:6px 10px;border:1px solid #e5e7eb;border-radius:16px;background:#f6f8ff;}
    .badge { display:inline-block; padding:6px 10px; border-radius:12px; background:#f0f2f6; margin-left:6px; }
    .badge.ok { background:#e6ffed; }
    .badge.warn { background:#fff5e6; }
    .hsep{height:1px;background:#eee;margin:8px 0;}
    </style>
    """, unsafe_allow_html=True)

rtl_css()

st.title("🗓️ שיבוץ משמרות לבתי קפה ומסעדות ")
st.caption("תפקידים/משמרות בהתאמה אישית, תכנית שבועית דינמית, איזון הוגן ותצוגות תוצאות עשירות — עם תמיכה ב'חובה' לפי יום.")

def ss_get(k, default):
    if k not in st.session_state: st.session_state[k] = default
    return st.session_state[k]

# ----------------- Session Structures -----------------
ROLES: List[str] = ss_get("ROLES", [])                 # user-defined, start empty
SHIFTS: List[str] = ss_get("SHIFTS", [])               # user-defined, start empty
# Legacy global requirement: {shift: {role: bool}} (kept for backward compatibility / defaults)
REQUIRED_BY_SHIFT: Dict[str, Dict[str, bool]] = ss_get("REQ_BY_SHIFT", {})
# NEW per-day requirement: {day_code: {shift: {role: bool}}}
REQUIRED_BY_DAY: Dict[str, Dict[str, Dict[str,bool]]] = ss_get("REQ_BY_DAY", {code:{} for code,_ in DAYS_ORDER})
# Default staffing per shift: {shift: {role: qty}} used as initial suggestion
DEFAULT_STAFFING: Dict[str, Dict[str,int]] = ss_get("DEFAULT_STAFFING", {})
# Week plan: {day_code: {shift: {role: qty}}} only for active shifts
WEEK_PLAN: Dict[str, Dict[str, Dict[str,int]]] = ss_get("WEEK_PLAN", {code:{} for code,_ in DAYS_ORDER})
CONFIG: Dict = ss_get("CONFIG", {"shift_hours":8.0, "fairness":"ללא חלוקה הוגנת", "max_override":None})
EMP_BYTES = st.session_state.get("EMP_FILE_BYTES")

tabs = st.tabs(["הגדרות", "תוצאות", "לא שובצו", "בקרה", "שיתוף"])

# ----------------- Utils -----------------
def parse_csv_like(val) -> List[str]:
    if pd.isna(val): return []
    if not isinstance(val, str): val = str(val)
    parts = [p.strip() for p in val.replace(";", ",").split(",")]
    return [p for p in parts if p]

# --- Added: Hebrew-to-English day mapping & normalizer ---
HE2EN_DAYS = {
    "ראשון": "Sun", "שני": "Mon", "שלישי": "Tue",
    "רביעי": "Wed", "חמישי": "Thu", "שישי": "Fri", "שבת": "Sat",
    # Accept also short Hebrew variants if appear
    "א'": "Sun", "ב'": "Mon", "ג'": "Tue", "ד'": "Wed", "ה'": "Thu", "ו'": "Fri", "שבתון": "Sat"
}

def normalize_days_tokens(tokens):
    out = []
    for t in tokens:
        t = str(t).strip()
        out.append(HE2EN_DAYS.get(t, t))
    return out
# --- End Added ---


def gini(x: np.ndarray) -> float:
    """Gini coefficient for non-negative array x; returns 0 if empty or all zeros."""
    if x is None: return 0.0
    x = np.asarray(x, dtype=float).flatten()
    if x.size == 0: return 0.0
    if np.allclose(x, 0): return 0.0
    if np.any(x < 0):  # shift to non-negative
        x = x - np.min(x)
    x_sorted = np.sort(x)
    n = x_sorted.size
    cumx = np.cumsum(x_sorted)
    # Relative mean absolute difference method
    g = (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    return float(g)

@st.cache_data(show_spinner=False)
def read_excel_sheet(xbytes: bytes, sheet_name: str) -> pd.DataFrame:
    with io.BytesIO(xbytes) as bio:
        df = pd.read_excel(bio, sheet_name=sheet_name)
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    return df

def validate_employees_df(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate Employees sheet schema and normalize key columns."""
    required = ["ID", "Name", "HourlyCost", "DaysAvailable"]
    optional = ["ShiftTypesAvailable", "RolesQualified", "Role", "MaxShiftsPerWeek"]
    missing = [c for c in required if c not in df.columns]
    errs = []
    if missing:
        errs.append(f"עמודות חסרות: {', '.join(missing)}")
    # Normalize dtypes
    try:
        if "ID" in df.columns:
            df["ID"] = df["ID"].astype(str).str.strip()
        if "Name" in df.columns:
            df["Name"] = df["Name"].astype(str).str.strip()
        if "HourlyCost" in df.columns:
            df["HourlyCost"] = pd.to_numeric(df["HourlyCost"], errors="coerce").fillna(0.0).astype(float)
        if "MaxShiftsPerWeek" in df.columns:
            df["MaxShiftsPerWeek"] = pd.to_numeric(df["MaxShiftsPerWeek"], errors="coerce")
    except Exception as e:
        errs.append(f"שגיאת המרה/נירמול: {e}")
    # Hints
    hints = []
    if "ShiftTypesAvailable" not in df.columns:
        hints.append("מומלץ לכלול ShiftTypesAvailable כדי להגביל משמרות (התאמה לשמות המשמרות שהוגדרו).")
    if "RolesQualified" not in df.columns and "Role" not in df.columns:
        hints.append("מומלץ לכלול RolesQualified או Role כדי לקבוע כשירות לתפקידים.")
    if hints:
        st.info("המלצות סכימה: " + " ".join(hints))
    ok = (len(errs) == 0) and (len(missing) == 0)
    return ok, errs

def coverage_badge(assigned: int, total: int):
    pct = 0 if total==0 else int(round(100*assigned/total))
    cls = "ok" if assigned==total else "warn"
    st.markdown(f'<span class="badge {cls}">כיסוי תקנים: {assigned}/{total} ({pct}%)</span>', unsafe_allow_html=True)

def is_required(day_code: str, shift_name: str, role: str) -> bool:
    """Check per-day required; fallback to legacy global per-shift if not set."""
    day_map = REQUIRED_BY_DAY.get(day_code, {})
    if shift_name in day_map and role in day_map[shift_name]:
        return bool(day_map[shift_name][role])
    # Fallback to legacy per-shift default
    return bool(REQUIRED_BY_SHIFT.get(shift_name, {}).get(role, False))

# ----------------- Algorithm -----------------
def build_rows_employee_days(df_emp: pd.DataFrame, active_days: List[str], max_override: Optional[int]) -> List[Tuple[str,str]]:
    # Each row is (employee_id, day_code) up to cap
    avail = {}
    for _, r in df_emp.iterrows():
        emp_id = str(r.get("ID"))
        days = set(normalize_days_tokens(parse_csv_like(r.get("DaysAvailable",""))))
        avail[emp_id] = [d for d in active_days if d in days]
    rows = []
    for _, r in df_emp.iterrows():
        emp_id = str(r.get("ID"))
        days_av = avail.get(emp_id, [])
        if not days_av: continue
        cap = None
        if max_override and max_override > 0:
            cap = int(max_override)
        elif "MaxShiftsPerWeek" in df_emp.columns and pd.notna(r.get("MaxShiftsPerWeek")):
            try: cap = int(r.get("MaxShiftsPerWeek"))
            except: cap = None
        if cap is None:
            for d in days_av: rows.append((emp_id, d))
        else:
            # Greedy: top-demand days first
            demand_sorted = st.session_state.get("DEMAND_SORT", active_days)
            ordered = [d for d in demand_sorted if d in days_av]
            for d in ordered[:cap]:
                rows.append((emp_id, d))
    return rows

def build_cols_from_weekplan(active_days: List[str]) -> List[Tuple[str,str,int,str]]:
    """Columns = demand slots (day, shift, k, role). Quantity respects per-day required (min-1)."""
    cols = []
    for d in active_days:
        plan = effective_plan_for_day(d)
        for s, comp in plan.items():
            for r in ROLES:
                qty = int(comp.get(r, 0))
                # Enforce per-day min-1 required roles
                if is_required(d, s, r):
                    qty = max(1, qty)
                for k in range(1, qty+1):
                    cols.append((d, s, k, r))
    return cols

def build_base_costs(rows, cols, df_emp, shift_hours: float):
    # Pre-index employees
    emp_index = {}
    for _, r in df_emp.iterrows():
        roles_q = set(parse_csv_like(r.get("RolesQualified","")))
        # Fallback single Role if provided and non-empty
        role_val = str(r.get("Role","")).strip()
        if not roles_q and role_val:
            roles_q = {role_val}
        emp_index[str(r.get("ID"))] = {
            "Name": r.get("Name",""),
            "HourlyCost": float(r.get("HourlyCost",0.0) or 0.0),
            "ShiftTypesAvailable": set(parse_csv_like(r.get("ShiftTypesAvailable",""))),
            "RolesQualified": roles_q,
        }
    n_rows, n_cols = len(rows), len(cols)
    M0 = np.full((n_rows, n_cols), HARD_BLOCK, dtype=float)
    for i, (emp, d) in enumerate(rows):
        e = emp_index.get(emp)
        if e is None: continue
        stypes = e["ShiftTypesAvailable"]
        rolesq = e["RolesQualified"]
        for j, (dj, s, _k, r) in enumerate(cols):
            if dj != d: continue
            if stypes and (s not in stypes): continue
            if rolesq and (r not in rolesq): continue
            M0[i,j] = e["HourlyCost"] * float(shift_hours)
    return M0, emp_index

def solve_hungarian(M: np.ndarray):
    n_rows, n_cols = M.shape
    dim = max(n_rows, n_cols)
    padded = np.full((dim, dim), HARD_CONSTRAINT, dtype=float)
    padded[:n_rows, :n_cols] = M
    r_idx, c_idx = linear_sum_assignment(padded)
    return r_idx, c_idx

def iterative_fair_assignment(rows, cols, M0, fairness: str):
    # penalties: total-load, per-shift, per-role + jitter
    if fairness == "ללא חלוקה הוגנת":
        iters, lam_total, lam_shift, lam_role, lam_run = 1, 0.0, 0.0, 0.0, 0.0
    elif fairness == "חלוקה הוגנת (מתונה)":
        iters, lam_total, lam_shift, lam_role, lam_run = 3, 1.0, 0.0, 0.2, 1.6
    else:
        iters, lam_total, lam_shift, lam_role, lam_run = 5, 2.0, 0.0, 0.3, 3.0

    n_rows, n_cols = M0.shape
    valid = M0[M0 < HARD_BLOCK]
    base_scale = float(np.median(valid)) if valid.size else 100.0
    row_emp = [emp for emp,_ in rows]
    col_shift = [s for (_d,s,_k,_r) in cols]
    col_role  = [r for (_d,_s,_k,r) in cols]

    rng = np.random.default_rng(42)
    M = M0.copy()
    assign = []
    total_true = 0.0; total_pen = 0.0

    for t in range(1, iters+1):
        jitter = rng.normal(0.0, 1e-6, size=M.shape)
        r_idx, c_idx = solve_hungarian(M + jitter)
        assign = []
        used_rows, used_cols = set(), set()
        for r,c in zip(r_idx, c_idx):
            if r<n_rows and c<n_cols and M[r,c] < HARD_BLOCK:
                assign.append((r,c)); used_rows.add(r); used_cols.add(c)
        if t == iters:
            total_true = float(np.sum([M0[r,c] for r,c in assign if M0[r,c] < HARD_BLOCK]))
            total_pen  = float(np.sum([M[r,c]  for r,c in assign if M[r,c]  < HARD_BLOCK]))
        if t < iters and (lam_total>0 or lam_shift>0 or lam_role>0 or lam_run>0):
            cnt_total, cnt_shift, cnt_role = {}, {}, {}
            for r,c in assign:
                e = row_emp[r]
                cnt_total[e] = cnt_total.get(e,0)+1
                s = col_shift[c]; cnt_shift[(e,s)] = cnt_shift.get((e,s),0)+1
                ro = col_role[c]; cnt_role[(e,ro)] = cnt_role.get((e,ro),0)+1
            M = M0.copy()
            for i,(emp,_day) in enumerate(rows):
                ct = cnt_total.get(emp,0)
                for j in range(n_cols):
                    if M0[i,j] >= HARD_BLOCK: continue
                    s = col_shift[j]; ro = col_role[j]
                    cs = cnt_shift.get((emp,s),0)
                    cr = cnt_role.get((emp,ro),0)
                    pen = lam_total*base_scale*ct + lam_shift*base_scale*cs + lam_role*base_scale*cr
                    M[i,j] = M0[i,j] + pen
    
            # --- Run/streak penalty: discourage same shift on consecutive days for the same employee ---
            if lam_run > 0:
                day_to_idx = {c:i for i,(c,_) in enumerate(DAYS_ORDER)}
                per_emp = {}
                for (ri, ci) in assign:
                    dcode, sname, _k, _role = cols[ci]
                    emp_id = row_emp[ri]
                    per_emp.setdefault(emp_id, []).append((day_to_idx.get(dcode,0), sname, ri, ci))
                for emp_id, items in per_emp.items():
                    items.sort(key=lambda x: x[0])
                    prev_shift = None
                    for (_di, sname, ri, ci) in items:
                        if prev_shift is not None and sname == prev_shift:
                            if M[ri, ci] < HARD_BLOCK:
                                M[ri, ci] = M[ri, ci] + lam_run * base_scale
                        prev_shift = sname
            # --- end run penalty ---
    return assign, total_true, total_pen

def cap_violations_report(df_assign: pd.DataFrame, df_emp: pd.DataFrame, max_override: Optional[int]):
    if df_assign is None or df_assign.empty:
        return pd.DataFrame(columns=["ID עובד","שם עובד","מגבלת מקס","מס' שיבוצים","חריגה?"])
    caps = {}
    for _, r in df_emp.iterrows():
        emp_id = str(r.get("ID"))
        cap = None
        if max_override is not None and max_override > 0:
            cap = int(max_override)
        elif "MaxShiftsPerWeek" in df_emp.columns and pd.notna(r.get("MaxShiftsPerWeek")):
            try: cap = int(r.get("MaxShiftsPerWeek"))
            except: cap = None
        caps[emp_id] = cap
    counts = df_assign.groupby(["ID עובד","שם עובד"]).size().reset_index(name="מס' שיבוצים")
    out = []
    for _, rr in counts.iterrows():
        emp_id = str(rr["ID עובד"]); name = rr["שם עובד"]; c = int(rr["מס' שיבוצים"])
        cap = caps.get(emp_id, None)
        status = "—" if cap is None else ("כן" if c > cap else "לא")
        out.append({"ID עובד": emp_id, "שם עובד": name, "מגבלת מקס": cap if cap is not None else "לא מוגדר", "מס' שיבוצים": c, "חריגה?": status})
    return pd.DataFrame(out)

# ----------------- Settings Tab -----------------
with tabs[0]:
    st.subheader("הגדרות כלליות")
    c1,c2,c3 = st.columns([1,1,1])
    with c1:
        CONFIG["shift_hours"] = st.number_input("⌛ אורך משמרת (שעות)", min_value=1.0, max_value=16.0, value=float(CONFIG.get("shift_hours",8.0)), step=0.5, format="%.1f")
    with c2:
        CONFIG["fairness"] = st.radio("חלוקה הוגנת", ["ללא חלוקה הוגנת","חלוקה הוגנת (מתונה)","חלוקה הוגנת (חזקה)"],
                                      index={"ללא חלוקה הוגנת":0,"חלוקה הוגנת (מתונה)":1,"חלוקה הוגנת (חזקה)":2}.get(CONFIG.get("fairness","ללא חלוקה הוגנת"),0), horizontal=True)
    with c3:
        mv = CONFIG.get("max_override", 0)
        CONFIG["max_override"] = st.number_input("מקס' משמרות לעובד (גלובלי, 0=ללא)", min_value=0, value=int(mv or 0), step=1)
        if CONFIG["max_override"] == 0: CONFIG["max_override"] = None

    st.markdown("---")
    st.markdown("#### 1) הגדרת תפקידים ומשמרות")
    c_role, c_shift = st.columns(2)
    with c_role:
        st.markdown("##### תפקידים")
        new_role = st.text_input("➕ תפקיד חדש", key="add_role_input")
        if st.button("הוסף תפקיד", key="add_role_btn"):
            name = (new_role or "").strip()
            if name and name not in ROLES:
                ROLES.append(name); st.success(f"נוסף תפקיד: {name}"); st.rerun()
            else:
                st.warning("שם ריק או קיים.")
        if ROLES:
            st.write("---")
            for r in list(ROLES):
                cA,cB = st.columns([8,1])
                cA.markdown(f"<div class='chipline'>{r}</div>", unsafe_allow_html=True)
                if cB.button("✖", key=f"del_role_{r}"):
                    ROLES.remove(r)
                    for s in list(REQUIRED_BY_SHIFT.keys()):
                        REQUIRED_BY_SHIFT[s].pop(r, None)
                    for d in list(REQUIRED_BY_DAY.keys()):
                        for s in list(REQUIRED_BY_DAY[d].keys()):
                            REQUIRED_BY_DAY[d][s].pop(r, None)
                    for s in list(DEFAULT_STAFFING.keys()):
                        DEFAULT_STAFFING[s].pop(r, None)
                    for d in WEEK_PLAN.keys():
                        for s in list(WEEK_PLAN[d].keys()):
                            WEEK_PLAN[d][s].pop(r, None)
                    st.rerun()
        else:
            st.caption("טרם הוגדרו תפקידים.")

    with c_shift:
        st.markdown("##### משמרות")
        new_shift = st.text_input("➕ משמרת חדשה", key="add_shift_input")
        if st.button("הוסף משמרת", key="add_shift_btn"):
            name = (new_shift or "").strip()
            if name and name not in SHIFTS:
                SHIFTS.append(name)
                REQUIRED_BY_SHIFT.setdefault(name, {})
                DEFAULT_STAFFING.setdefault(name, {})
                st.session_state["NEW_SHIFT_ADDED"] = name
                st.success(f"נוספה משמרת: {name}"); st.rerun()
            else:
                st.warning("שם ריק או קיים.")

        # New feature: set default staffing for a newly added shift
        if st.session_state.get("NEW_SHIFT_ADDED") and ROLES:
            shift_to_set = st.session_state["NEW_SHIFT_ADDED"]
            st.markdown(f"**הגדר ברירת מחדל לצוות עבור '{shift_to_set}':**")
            with st.form(key="default_staffing_form"):
                current_defaults = DEFAULT_STAFFING.get(shift_to_set, {})
                cols = st.columns(max(1, len(ROLES)))
                for r, col in zip(ROLES, cols):
                    with col:
                        v = int(current_defaults.get(r, 0))
                        current_defaults[r] = int(st.number_input(f"כמות ל{r}", min_value=0, value=v, step=1, key=f"default_{shift_to_set}_{r}"))
                if st.form_submit_button("שמור ברירת מחדל"):
                    DEFAULT_STAFFING[shift_to_set] = current_defaults
                    st.success("ברירת המחדל נשמרה.")
                    st.session_state["NEW_SHIFT_ADDED"] = None
                    st.rerun()

        if SHIFTS:
            st.write("---")
            for s in list(SHIFTS):
                cA,cB = st.columns([8,1])
                cA.markdown(f"<div class='chipline'>{s}</div>", unsafe_allow_html=True)
                if cB.button("✖", key=f"del_shift_{s}"):
                    SHIFTS.remove(s)
                    REQUIRED_BY_SHIFT.pop(s, None)
                    for d in REQUIRED_BY_DAY.keys(): REQUIRED_BY_DAY[d].pop(s, None)
                    DEFAULT_STAFFING.pop(s, None)
                    for d in WEEK_PLAN.keys(): WEEK_PLAN[d].pop(s, None)
                    st.rerun()
        else:
            st.caption("טרם הוגדרו משמרות.")

    st.markdown("---")
    
    st.markdown("---")
    st.markdown("#### ימי פעילות עסקיים")
    # שמירת ימי פעילות (ברירת מחדל: ראשון–שישי)
    default_biz = ["Sun","Mon","Tue","Wed","Thu","Fri"]
    current_biz = st.session_state.get("CONFIG", {}).get("business_days", default_biz)
    human = {c: h for c,h in DAYS_ORDER}
    sel = st.multiselect("בחר ימי פעילות", options=[c for c,_ in DAYS_ORDER],
                         default=current_biz, format_func=lambda c: human.get(c,c))
    CONFIG["business_days"] = sel
    st.markdown("#### 2) תכנית שבועית (מותאם-אישית לימים)")
    if SHIFTS and ROLES:
        day_he = st.selectbox("בחר יום", [heb for _,heb in DAYS_ORDER], key="day_select")
        day_code = next(c for c,h in DAYS_ORDER if h==day_he)
        WEEK_PLAN.setdefault(day_code, {})
        REQUIRED_BY_DAY.setdefault(day_code, {})

        st.info("בחר משמרות פעילות ביום זה, קבע תקנים לכל תפקיד, וסמן 'חובה' לתפקידים שחייבים אדם אחד לפחות.")

        active_for_day = st.multiselect(f"משמרות פעילות ב{day_he}", options=SHIFTS, default=list(WEEK_PLAN[day_code].keys()))
        # purge deselected
        for s in list(WEEK_PLAN[day_code].keys()):
            if s not in active_for_day: WEEK_PLAN[day_code].pop(s, None)

        for s in active_for_day:
            st.markdown(f"##### {s}")
            comp = WEEK_PLAN[day_code].setdefault(s, {})
            req_roles_day = REQUIRED_BY_DAY[day_code].setdefault(s, {})
            req_roles_global = REQUIRED_BY_SHIFT.get(s, {})

            cols = st.columns(max(1, len(ROLES)))
            for r, col in zip(ROLES, cols):
                with col:
                    # Use default staffing as initial value if no custom value exists
                    default_val = DEFAULT_STAFFING.get(s, {}).get(r, 0)
                    v = int(comp.get(r, default_val))

                    # default checkbox value = per-day if set else global legacy
                    default_req = bool(req_roles_day.get(r, req_roles_global.get(r, False)))
                    is_req_new = st.checkbox(f"חובה {r}", value=default_req, key=f"req_{day_code}_{s}_{r}")
                    req_roles_day[r] = is_req_new

                    comp[r] = int(st.number_input(f"תקן ל{r}", min_value=0, value=v, step=1, key=f"{day_code}_{s}_{r}"))
            REQUIRED_BY_DAY[day_code][s] = req_roles_day
    else:
        st.info("הגדר תחילה תפקידים ומשמרות כדי לקבוע תכנית שבועית.")

    st.markdown("---")
    st.markdown("#### 3) קובץ עובדים")
    st.markdown("חובה גיליון **Employees**: ID, Name, HourlyCost, DaysAvailable[, ShiftTypesAvailable][, RolesQualified][, Role][, MaxShiftsPerWeek]")
    # הורדת תבנית לפני ההעלאה
    try:
        with open("Templet_Assighment.xlsx", "rb") as f:
            tmpl_bytes = f.read()
        st.download_button("⬇️ הורד תבנית אקסל", data=tmpl_bytes,
                           file_name="Templet_Assighment.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except FileNotFoundError:
        st.warning("קובץ התבנית Templet_Assighment.xlsx לא נמצא בתיקייה לצד קובץ ה־PY.")
    up = st.file_uploader("בחר Excel", type=["xlsx"], key="emp_xlsx_roles")
    if up is not None:
        try:
            EMP_BYTES = up.getvalue()
            st.session_state["EMP_FILE_BYTES"] = EMP_BYTES
            df_prev = read_excel_sheet(EMP_BYTES, "Employees")
            ok, errs = validate_employees_df(df_prev)
            if not ok:
                for e in errs: st.error(f"❌ {e}")
            # Preview (trimmed) regardless, if loaded
            prev = []
            head_df = df_prev.head(20)
            for _, r in head_df.iterrows():
                roles_q = ",".join(parse_csv_like(r.get("RolesQualified",""))) or (str(r.get("Role","")).strip())
                prev.append({
                    "ID": r.get("ID"), "שם": r.get("Name"),
                    "ימים": ",".join(parse_csv_like(r.get("DaysAvailable",""))),
                    "משמרות": ",".join(parse_csv_like(r.get("ShiftTypesAvailable",""))),
                    "תפקידים": roles_q,
                    "מקס'": int(r.get("MaxShiftsPerWeek",0) or 0),
                })
            st.success("✅ קובץ עובדים נטען.")
            st.dataframe(pd.DataFrame(prev), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"❌ שגיאה בטעינת הקובץ: {e}")

    if st.button("💾 שמור הגדרות", key="save_settings_btn"):
        st.session_state["CONFIG"] = CONFIG
        st.success("נשמר. עבור ללשונית 'תוצאות' כדי להריץ.")

# ----------------- Core Runner -----------------
def run_assignment():
    cfg = st.session_state.get("CONFIG", CONFIG)
    xbytes = st.session_state.get("EMP_FILE_BYTES", None)
    if not xbytes:
        st.session_state["RESULTS"] = {"error":"צריך לטעון קובץ עובדים."}
        return
    df_emp = read_excel_sheet(xbytes, "Employees")
    ok, errs = validate_employees_df(df_emp)
    if not ok:
        st.session_state["RESULTS"] = {"error":"קובץ העובדים אינו תקין. תקן בלשונית הגדרות."}
        return

    # Active days are those with selected shifts
    biz = st.session_state.get("CONFIG", {}).get("business_days", ["Sun","Mon","Tue","Wed","Thu","Fri"])
    active_days = [c for c,_ in DAYS_ORDER if (c in biz) and effective_plan_for_day(c)]
    if not active_days:
        st.session_state["RESULTS"] = {"error":"לא הוגדרו משמרות פעילות בשום יום."}
        return

    # demand by day (for cap ordering)
    day_totals = {}
    for d in active_days:
        total = 0
        for s, comp in effective_plan_for_day(d).items():
            for r, qty in comp.items():
                total += max(qty, 1) if is_required(d, s, r) else qty
        day_totals[d] = total
    st.session_state["DEMAND_SORT"] = sorted(active_days, key=lambda dd: day_totals.get(dd,0), reverse=True)
    st.session_state["MAX_OVERRIDE"] = cfg.get("max_override", None)

    rows = build_rows_employee_days(df_emp, active_days, cfg.get("max_override", None))
    cols = build_cols_from_weekplan(active_days)

    if len(rows) == 0:
        st.session_state["RESULTS"] = {"error":"אין עובד-יום זמין בשום משמרת פעילה שהוגדרה."}
        return
    if len(cols) == 0:
        st.session_state["RESULTS"] = {"error":"לא הוגדרו תקנים (כמויות עובדים) בשום משמרת פעילה."}
        return

    M0, emp_index = build_base_costs(rows, cols, df_emp, shift_hours=cfg.get("shift_hours",8.0))
    assign, true_cost, pen_cost = iterative_fair_assignment(rows, cols, M0, cfg.get("fairness","ללא חלוקה הוגנת"))

    # Build assignment DF and structures
    recs, used_rows, used_cols = [], set(), set()
    for r,c in assign:
        if r<len(rows) and c<len(cols) and M0[r,c] < HARD_BLOCK:
            emp, d = rows[r]
            (dj, s, k, role) = cols[c]
            recs.append({
                "יום": next(h for c2,h in DAYS_ORDER if c2==dj),
                "משמרת": s, "תפקיד": role, "תקן": k,
                "ID עובד": emp, "שם עובד": emp_index[emp]["Name"],
                "עלות אמת": round(float(M0[r,c]),2),
            })
            used_rows.add(r); used_cols.add(c)
    df_assign = pd.DataFrame(recs)

    # Build schedule for share
    schedule = {(code,heb): {s: [] for s in SHIFTS} for code,heb in DAYS_ORDER}
    for _, row in df_assign.iterrows():
        day_he = row["יום"]; code = next(c for c,h in DAYS_ORDER if h==day_he)
        schedule[(code,day_he)][row["משמרת"]].append((row["תפקיד"], row["שם עובד"]))

    # Unfilled slots with reasons
    df_unfilled_rows = []
    for j,(d,s,k,role) in enumerate(cols):
        if j in used_cols: continue
        reason = "לא נמצא עובד מתאים"
        # check potential candidates
        candidates = 0
        for _, e in df_emp.iterrows():
            if d not in set(normalize_days_tokens(parse_csv_like(e.get("DaysAvailable","")))): continue
            stypes = set(parse_csv_like(e.get("ShiftTypesAvailable","")))
            if stypes and s not in stypes: continue
            roles_q = set(parse_csv_like(e.get("RolesQualified","")))
            role_val = str(e.get("Role","")).strip()
            if not roles_q and role_val:
                roles_q = {role_val}
            if roles_q and role in roles_q:
                candidates += 1
        if candidates == 0:
            reason = "אין מועמדים זמינים/כשירים"
        else:
            reason = "הגעת מועמדים למכסה/העדפת עלות-איזון"
        df_unfilled_rows.append({"יום": next(h for c,h in DAYS_ORDER if c==d), "משמרת": s, "תפקיד": role, "סיבה": reason})
    df_unfilled = pd.DataFrame(df_unfilled_rows)

    # Unused employee-days with reasons
    df_unused_rows = []
    used_count = {}
    for r in used_rows:
        emp,_ = rows[r]; used_count[emp] = used_count.get(emp,0)+1
    # cap map
    cap_map = {}
    for _, e in df_emp.iterrows():
        emp_id = str(e.get("ID"))
        cap = None
        if cfg.get("max_override", None) is not None:
            cap = int(cfg["max_override"])
        elif "MaxShiftsPerWeek" in df_emp.columns and pd.notna(e.get("MaxShiftsPerWeek")):
            try: cap = int(e.get("MaxShiftsPerWeek"))
            except: cap=None
        cap_map[emp_id] = cap
    for i,(emp, d) in enumerate(rows):
        if i in used_rows: continue
        reason = "לא נדרש ביום זה"
        if cap_map.get(emp) is not None and used_count.get(emp,0) >= cap_map[emp]:
            reason = "הגיע למקסימום משמרות"
        else:
            # qualify any open slot?
            qualify = False
            row_e_all = df_emp[df_emp["ID"].astype(str)==emp]
            if not row_e_all.empty:
                row_e = row_e_all.iloc[0]
                stypes = set(parse_csv_like(row_e.get("ShiftTypesAvailable","")))
                roles_q = set(parse_csv_like(row_e.get("RolesQualified","")))
                role_val = str(row_e.get("Role","")).strip()
                if not roles_q and role_val:
                    roles_q = {role_val}
                for j,(dj,s,k,role) in enumerate(cols):
                    if dj!=d or j in used_cols: continue
                    if (not stypes or s in stypes) and (not roles_q or role in roles_q):
                        qualify = True; break
            reason = "נמצא מועמד מתאים יותר לפי עלות/איזון" if qualify else "אין התאמה למשמרות/תפקידים ביום זה"
        df_unused_rows.append({"ID עובד": emp, "שם עובד": (row_e_all["Name"].iloc[0] if not row_e_all.empty else ""), "יום": next(h for c,h in DAYS_ORDER if c==d), "סיבה": reason})
    df_unused = pd.DataFrame(df_unused_rows)

    # Cap violations
    cap_df = cap_violations_report(df_assign, df_emp, cfg.get("max_override", None)) if not df_assign.empty else pd.DataFrame()

    # Share table with ' | ' and 'לא שובץ'
    biz = st.session_state.get("CONFIG", {}).get("business_days", ["Sun","Mon","Tue","Wed","Thu","Fri"])
    cols_order = [heb for code,heb in DAYS_ORDER if (code in biz) and effective_plan_for_day(code)]
    # rows_order = union of all active shifts across the week, in user SHIFTS order
    rows_order = SHIFTS

    def build_cell(day_he, shift_name):
        code = next(c for c,h in DAYS_ORDER if h==day_he)
        items = schedule.get((code,day_he),{}).get(shift_name, [])
        demand_map = effective_plan_for_day(code).get(shift_name, {})
        parts = []
        for r in ROLES:
            if r in demand_map or is_required(code, shift_name, r):
                names = [n for (rr,n) in items if rr==r]
                # Respect min-1 for required; if no names show 'לא שובץ'
                show_empty = (len(names) == 0) and (is_required(code, shift_name, r) or demand_map.get(r,0) > 0)
                parts.append(f"{r}: {', '.join(names) if (names and not show_empty) else 'לא שובץ' if show_empty else ', '.join(names)}")
        return " | ".join(parts)

    data = []
    for s in rows_order:
        row_vals = []
        for heb in cols_order:
            code = next(c for c,h in DAYS_ORDER if h==heb)
            row_vals.append(build_cell(heb, s))
        data.append(row_vals)
    df_share = pd.DataFrame(data, columns=cols_order, index=rows_order).reset_index().rename(columns={"index":"משמרות/יום"})
    df_share = df_share[["משמרות/יום"] + [c for c in df_share.columns if c in cols_order]]

    # Dispersion metrics
    per_emp = df_assign.groupby("שם עובד").size().reset_index(name="מספר שיבוצים") if not df_assign.empty else pd.DataFrame(columns=["שם עובד","מספר שיבוצים"])
    arr = per_emp["מספר שיבוצים"].values if not per_emp.empty else np.array([])
    cv = float(np.std(arr)/np.mean(arr)) if arr.size>0 and np.mean(arr)>0 else 0.0
    g = gini(arr) if arr.size>0 else 0.0
    per_emp_chart_df = per_emp.set_index("שם עובד")[["מספר שיבוצים"]] if not per_emp.empty else pd.DataFrame()

    # Costs
    cost_by_day = df_assign.groupby("יום")["עלות אמת"].sum().reset_index().sort_values("יום") if not df_assign.empty else pd.DataFrame(columns=["יום","עלות אמת"])
    cost_by_shift = df_assign.groupby("משמרת")["עלות אמת"].sum().reset_index().sort_values("משמרת") if not df_assign.empty else pd.DataFrame(columns=["משמרת","עלות אמת"])
    cost_by_emp = df_assign.groupby("שם עובד")["עלות אמת"].sum().reset_index().sort_values("עלות אמת", ascending=False) if not df_assign.empty else pd.DataFrame(columns=["שם עובד","עלות אמת"])

    st.session_state["RESULTS"] = {
        "df_assign": df_assign,
        "df_unfilled": df_unfilled,
        "df_unused": df_unused,
        "cap_df": cap_df,
        "schedule": schedule,
        "df_share": df_share,
        "total_true_cost": float(true_cost),
        "total_penalized_cost": float(pen_cost),
        "assigned_slots": len(used_cols),
        "total_slots": len(cols),
        "cv": cv, "gini": g,
        "per_emp_chart_df": per_emp_chart_df,
        "cost_by_day": cost_by_day,
        "cost_by_shift": cost_by_shift,
        "cost_by_emp": cost_by_emp,
    }

# ----------------- Tabs -----------------
with tabs[1]:
    st.subheader("תוצאות — שיבוץ אופטימלי")
    if st.button("🚀 הרץ שיבוץ", key="run_assignment_btn"):
        run_assignment()
    res = st.session_state.get("RESULTS")
    if not res:
        st.info("הגדר בלשונית 'הגדרות' ואז הרץ.")
    elif "error" in res:
        st.error(res["error"])
    else:
        coverage_badge(res["assigned_slots"], res["total_slots"])
        st.success(f"💵 עלות כוללת (אמת): {round(res['total_true_cost'],2)}")
        if CONFIG.get("fairness") != "ללא חלוקה הוגנת":
            st.caption(f"עלות עם חלוקה הוגנת (קריטריון): {round(res['total_penalized_cost'],2)}")

        # --- Detailed assignments at the top ---
        st.subheader("שיבוצים מפורטים")
        st.dataframe(res["df_assign"].sort_values(by=["יום","משמרת","תפקיד","תקן"]), use_container_width=True, hide_index=True)
        st.download_button("⬇️ הורד תוצאות (CSV)", data=res["df_assign"].to_csv(index=False).encode("utf-8-sig"),
                           file_name="shift_assignments.csv", mime="text/csv")

        st.markdown("---")
        # --- Three cost tables side-by-side ---
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("#### עלות לפי יום")
            st.dataframe(res["cost_by_day"], use_container_width=True, hide_index=True)
        with c2:
            st.markdown("#### עלות לפי משמרת")
            st.dataframe(res["cost_by_shift"], use_container_width=True, hide_index=True)
        with c3:
            st.markdown("#### עלות שבועית לפי עובד")
            st.dataframe(res["cost_by_emp"], use_container_width=True, hide_index=True)

        st.markdown("---")
        # --- Dispersion metrics at bottom ---
        st.subheader("מדדי פיזור")
        c1,c2 = st.columns(2)
        c1.metric("CV", f"{res['cv']:.3f}")
        c2.metric("Gini", f"{res['gini']:.3f}")
        if res["per_emp_chart_df"] is not None and not res["per_emp_chart_df"].empty:
            st.bar_chart(res["per_emp_chart_df"], use_container_width=True)
with tabs[2]:
    st.subheader("לא שובצו — ניתוח סיבות")
    res = st.session_state.get("RESULTS")
    if not res or "error" in (res or {}):
        st.info("אין נתונים להצגה. הרץ בלשונית תוצאות.")
    else:
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("#### עובדים-יום שלא שובצו + סיבה")
            df_unused = res["df_unused"]
            if df_unused is None or df_unused.empty:
                st.success("כל העובדים-יום שובצו או לא נדרשו.")
            else:
                biz = st.session_state.get("CONFIG", {}).get("business_days", ["Sun","Mon","Tue","Wed","Thu","Fri"])
                active_heb = [heb for code,heb in DAYS_ORDER if (code in biz) and effective_plan_for_day(code)]
                if df_unused is not None and not df_unused.empty and "יום" in df_unused.columns:
                    df_unused = df_unused[df_unused["יום"].isin(active_heb)]
                st.dataframe(df_unused.sort_values(by=["שם עובד","יום"]), use_container_width=True, hide_index=True)
                st.download_button("⬇️ הורד (CSV)", data=df_unused.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="unused_employee_days.csv", mime="text/csv")
        with c2:
            st.markdown("#### תקנים שלא התמלאו + סיבה")
            df_unfilled = res["df_unfilled"]
            if df_unfilled is None or df_unfilled.empty:
                st.success("כל התקנים מולאו.")
            else:
                biz = st.session_state.get("CONFIG", {}).get("business_days", ["Sun","Mon","Tue","Wed","Thu","Fri"])
                active_heb = [heb for code,heb in DAYS_ORDER if (code in biz) and effective_plan_for_day(code)]
                if "יום" in df_unfilled.columns:
                    df_unfilled = df_unfilled[df_unfilled["יום"].isin(active_heb)]
                st.dataframe(df_unfilled.sort_values(by=["יום","משמרת","תפקיד"]), use_container_width=True, hide_index=True)
                st.download_button("⬇️ הורד (CSV)", data=df_unfilled.to_csv(index=False).encode("utf-8-sig"),
                                   file_name="unfilled_slots.csv", mime="text/csv")

with tabs[3]:
    st.subheader("דשבורד בקרה")
    res = st.session_state.get("RESULTS")
    if not res or "error" in res:
        st.info("הרץ שיבוץ להצגת דוחות.")
    else:
        # Preflight audit — demand vs potential supply
        xbytes = st.session_state.get("EMP_FILE_BYTES")
        if xbytes:
            df_emp = read_excel_sheet(xbytes, "Employees")
            rows = []
            biz = st.session_state.get("CONFIG", {}).get("business_days", ["Sun","Mon","Tue","Wed","Thu","Fri"])
            for code, heb in [(c,h) for (c,h) in DAYS_ORDER if (c in biz) and effective_plan_for_day(c)]:
                plan = effective_plan_for_day(code)
                for s, comp in plan.items():
                    for r in ROLES:
                        need = int(comp.get(r, 0))
                        if is_required(code, s, r): need = max(1, need)
                        if need > 0:
                            raw = 0
                            for _, e in df_emp.iterrows():
                                if code not in set(normalize_days_tokens(parse_csv_like(e.get("DaysAvailable","")))): continue
                                stypes = set(parse_csv_like(e.get("ShiftTypesAvailable","")))
                                if stypes and s not in stypes: continue
                                roles_q = set(parse_csv_like(e.get("RolesQualified","")))
                                role_val = str(e.get("Role","")).strip()
                                if not roles_q and role_val:
                                    roles_q = {role_val}
                                if roles_q and r in roles_q:
                                    raw += 1
                            rows.append({"יום": heb, "משמרת": s, "תפקיד": r, "נדרש": int(need), "היצע פוטנציאלי": raw})
            df_audit = pd.DataFrame(rows)
            st.markdown("#### בדיקת היתכנות (ביקוש מול היצע)")
            if df_audit.empty:
                st.info("אין נתונים להצגה.")
            else:
                st.dataframe(df_audit.sort_values(by=["יום","משמרת","תפקיד"]), use_container_width=True, hide_index=True)
                totals = df_audit[["נדרש","היצע פוטנציאלי"]].sum()
                st.bar_chart(totals, use_container_width=True)

        st.markdown("---")
        st.markdown("#### חריגות ממגבלת משמרות")
        cap_df = res.get("cap_df", pd.DataFrame())
        st.dataframe(cap_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### מדדי הוגנות (פיזור)")
        c1,c2 = st.columns(2)
        c1.metric("CV", f"{res['cv']:.3f}")
        c2.metric("Gini", f"{res['gini']:.3f}")
        if res.get("per_emp_chart_df") is not None and not res["per_emp_chart_df"].empty:
            st.bar_chart(res["per_emp_chart_df"], use_container_width=True)

with tabs[4]:
    st.subheader("טבלת שיתוף לעובדים (להדפסה/שילוח)")
    res = st.session_state.get("RESULTS")
    if not res or "error" in (res or {}):
        st.info("הרץ שיבוץ בלשונית 'תוצאות' כדי לבנות טבלה.")
    else:
        st.dataframe(res["df_share"], use_container_width=True, hide_index=True)
        csv = res["df_share"].to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ הורד CSV", data=csv, file_name="shareable_schedule.csv", mime="text/csv")
        xbytes = io.BytesIO()
        with pd.ExcelWriter(xbytes, engine="openpyxl") as writer:
            res["df_share"].to_excel(writer, index=False, sheet_name="Schedule")
        st.download_button("⬇️ הורד XLSX", data=xbytes.getvalue(),
                           file_name="shareable_schedule.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.caption("פיזור הוגן חזק פועל בכפוף לזמינות/כשירות. כדי לראות ערבוב טוב יותר — הגדל כשירות/זמינות, איזן דרישות, או העלה מגבלת מקס'.")
# --- Added: Effective plan resolver (custom overrides defaults) ---
def effective_plan_for_day(day_code: str) -> Dict[str, Dict[str,int]]:
    """
    Return the plan for a given day:
    - If a custom plan exists in WEEK_PLAN, use it.
    - Else, derive from DEFAULT_STAFFING; activate a shift if it has demand (>0)
      or any role is globally required for that shift.
    """
    custom = WEEK_PLAN.get(day_code, {}) if 'WEEK_PLAN' in globals() else {}
    if custom:
        return custom

    eff: Dict[str, Dict[str,int]] = {}
    for s in SHIFTS:
        base = DEFAULT_STAFFING.get(s, {}) if 'DEFAULT_STAFFING' in globals() else {}
        if not isinstance(base, dict):
            continue
        has_demand = any(int(base.get(r, 0)) > 0 for r in ROLES)
        has_required = any(bool(REQUIRED_BY_SHIFT.get(s, {}).get(r, False)) for r in ROLES) if 'REQUIRED_BY_SHIFT' in globals() else False
        if has_demand or has_required:
            eff[s] = {r: int(base.get(r, 0)) for r in ROLES}
    return eff
# --- End Added ---

