import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import io
st.title("🧠 מערכת שיבוץ עובדים לשבוע")


# הסבר והוראות מימין לשמאל
st.markdown("""
<div dir="rtl">

#### נא לפעול לפי ההוראות הבאות:

1. העלאת קובץ אקסל תקני מסוג `.xlsx`  
2. הקובץ צריך להכיל שני גיליונות בשמות:  
   - `Employees`  
   - `Shifts`  
3. מבנה גיליון **Employees**:

</div>
""", unsafe_allow_html=True)

# טבלת Employees לדוגמה
example_employees = pd.DataFrame({
    "FullName": ["John Doe"],
    "AvailableDays": ["Sunday,Monday"],
    "PreferredShifts": ["Morning"],
    "MaxShifts": [6],
    "MinShifts": [4],
    "QualifiedFor": ["Sales Man"],
    "HourlyRate": [40]
})
st.table(example_employees)

# סעיף 4 גם מימין לשמאל
st.markdown("""
<div dir="rtl">
4. מבנה גיליון <strong>Shifts</strong>:
</div>
""", unsafe_allow_html=True)

# טבלת Shifts לדוגמה
example_shifts = pd.DataFrame({
    "Day": ["Sunday"],
    "ShiftType": ["Morning"],
    "RequiredRole": ["Sales Man"],
    "Duration (hours)": [8]
})
st.table(example_shifts)

# טופס להזנת פרמטרים + העלאת קובץ
with st.form("input_form"):
    min_employees_evening = st.number_input("כמה עובדים מינימום בכל משמרת ערב?", min_value=1, value=2, step=1)
    uploaded_file = st.file_uploader("העלה את קובץ ה-Excel שלך", type=["xlsx"])
    submitted = st.form_submit_button("שבץ עובדים")

if submitted and uploaded_file:
    # קריאת הקובץ
    df_employees = pd.read_excel(uploaded_file, sheet_name="Employees")
    df_shifts = pd.read_excel(uploaded_file, sheet_name="Shifts")

    # חישוב סה"כ דרישות מינימום משמרות
    total_min_shifts_required = df_employees["MinShifts"].sum()

    # חישוב מספר המשמרות בפועל עם ההרחבה של ערב
    num_evening_shifts = len(df_shifts[df_shifts["ShiftType"].str.lower() == "evning"]) * min_employees_evening
    num_morning_shifts = len(df_shifts[df_shifts["ShiftType"].str.lower() == "morning"])
    total_available_shifts = num_morning_shifts + num_evening_shifts

    # בדיקה
    if total_min_shifts_required > total_available_shifts:
        st.error(f"""❗ לא ניתן לבצע שיבוץ:
    נדרש לפחות {total_min_shifts_required} משמרות כדי לעמוד בדרישות המינימום של העובדים,
    אבל קיימות רק {total_available_shifts} משמרות זמינות.""")
        st.stop()



    def parse_list(cell):
        if pd.isna(cell):
            return []
        return [x.strip() for x in str(cell).split(',')]

    df_employees["AvailableDays"] = df_employees["AvailableDays"].apply(parse_list)
    df_employees["PreferredShifts"] = df_employees["PreferredShifts"].apply(parse_list)
    df_employees["QualifiedFor"] = df_employees["QualifiedFor"].apply(parse_list)
    df_shifts["RequiredRole"] = df_shifts["RequiredRole"].apply(parse_list)

    # הרחבת משמרות ערב
    expanded_shifts = []
    for _, row in df_shifts.iterrows():
        repeat = min_employees_evening if row["ShiftType"].lower() == "evning" else 1
        for i in range(repeat):
            shift_copy = row.copy()
            shift_copy["InstanceID"] = i
            expanded_shifts.append(shift_copy)

    shifts = expanded_shifts
    employees = df_employees.to_dict("records")
    num_employees = len(employees)
    num_shifts = len(shifts)

    shift_counts = {emp["FullName"]: 0 for emp in employees}
    assigned_days = {emp["FullName"]: set() for emp in employees}

    def calculate_cost(emp, shift):
        return emp["HourlyRate"] * shift["Duration (hours)"]

    cost_matrix = np.full((num_employees, num_shifts), 1e6)
    for i, emp in enumerate(employees):
        for j, shift in enumerate(shifts):
            if shift["Day"] in emp["AvailableDays"] and any(role in emp["QualifiedFor"] for role in shift["RequiredRole"]):
                cost_matrix[i][j] = calculate_cost(emp, shift)

    # שלב 1: Hungarian
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignments = []
    assigned_shifts = set()

    for i, j in zip(row_ind, col_ind):
        emp_name = employees[i]["FullName"]
        shift_day = shifts[j]["Day"]

        if cost_matrix[i][j] >= 1e6 or shift_counts[emp_name] >= employees[i]["MaxShifts"] or shift_day in assigned_days[emp_name]:
            continue

        assignments.append({
            "Day": shift_day,
            "ShiftType": shifts[j]["ShiftType"],
            "AssignedTo": emp_name
        })

        shift_counts[emp_name] += 1
        assigned_days[emp_name].add(shift_day)
        assigned_shifts.add(j)

    # שלב 2: השלמה ידנית
    for j, shift in enumerate(shifts):
        if j in assigned_shifts:
            continue
        best_emp = None
        best_score = float("inf")
        for i, emp in enumerate(employees):
            name = emp["FullName"]
            if shift["Day"] not in emp["AvailableDays"] or shift["Day"] in assigned_days[name]:
                continue
            if not any(role in emp["QualifiedFor"] for role in shift["RequiredRole"]):
                continue
            if shift_counts[name] >= emp["MaxShifts"]:
                continue
            score = calculate_cost(emp, shift)
            if score < best_score:
                best_score = score
                best_emp = emp

        if best_emp:
            name = best_emp["FullName"]
            assignments.append({
                "Day": shift["Day"],
                "ShiftType": shift["ShiftType"],
                "AssignedTo": name
            })
            shift_counts[name] += 1
            assigned_days[name].add(shift["Day"])
        else:
            assignments.append({
                "Day": shift["Day"],
                "ShiftType": shift["ShiftType"],
                "AssignedTo": "לא שובץ - אין מספיק עובדים"
            })

    # שלב 3: איזון MinShifts
    df_assignments = pd.DataFrame(assignments)
    for emp in employees:
        name = emp["FullName"]
        while shift_counts[name] < emp["MinShifts"]:
            possible_days = [d for d in emp["AvailableDays"] if d not in assigned_days[name]]
            found = False
            for day in possible_days:
                unassigned_shift = df_assignments[
                    (df_assignments["Day"] == day) & 
                    (df_assignments["AssignedTo"] == "לא שובץ - אין התאמה")
                ]
                if not unassigned_shift.empty:
                    idx = unassigned_shift.index[0]
                    shift_type = df_assignments.loc[idx, "ShiftType"]
                    shift_info = next((s for s in shifts if s["Day"] == day and s["ShiftType"] == shift_type), None)
                    if shift_info and any(role in emp["QualifiedFor"] for role in shift_info["RequiredRole"]):
                        df_assignments.loc[idx, "AssignedTo"] = name
                        shift_counts[name] += 1
                        assigned_days[name].add(day)
                        found = True
                        break
            if not found:
                break

    df_assignments = df_assignments.sort_values(by=["Day", "ShiftType"])

    # טבלת עלות כוללת לפי יום
    merged = df_assignments.merge(df_shifts, on=["Day", "ShiftType"], how="left")
    merged = merged.merge(df_employees[["FullName", "HourlyRate"]], left_on="AssignedTo", right_on="FullName", how="left")
    merged["ShiftCost"] = merged["HourlyRate"] * merged["Duration (hours)"]
    cost_summary = merged[merged["AssignedTo"] != "לא שובץ - אין התאמה"].groupby("Day")["ShiftCost"].sum().reset_index()
    cost_summary.columns = ["Day", "Total Cost"]

    # הצגת טבלאות
    st.subheader("📋 טבלת עובדים מהקובץ")
    st.dataframe(df_employees)

    st.subheader("📅 טבלת השמה אופטימלית")
    st.dataframe(df_assignments[["Day", "ShiftType", "AssignedTo"]])
    
    pivot_table = df_assignments.pivot(index="AssignedTo", columns="Day", values="ShiftType").fillna("")
    st.subheader("📊 טבלת שיבוץ בפורמט סופי (לפי עובד ויום)")
    st.dataframe(pivot_table)

    st.subheader("💰 טבלת עלות כוללת לפי יום")
    st.dataframe(cost_summary)



    to_download = df_assignments.copy()
    to_download = to_download.sort_values(by=["Day", "ShiftType"])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        to_download.to_excel(writer, index=False, sheet_name="Schedule")

    st.download_button(
        label="📥 הורד את טבלת השיבוץ כ-Excel",
        data=buffer.getvalue(),
        file_name="שיבוץ_עובדים.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
