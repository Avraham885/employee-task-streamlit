import streamlit as st
import pandas as pd

st.set_page_config(page_title="מערכת שיבוץ אופטימלית", layout="centered")

st.title("📋 מערכת שיבוץ אופטימלית")
st.write("ברוך הבא! העלה את קובץ האקסל עם רשימת העובדים והמשימות לפי התבנית.")

uploaded_file = st.file_uploader("בחר קובץ Excel (xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df_employees = pd.read_excel(uploaded_file, sheet_name="Employees")
        df_tasks = pd.read_excel(uploaded_file, sheet_name="Tasks")

        st.subheader("👥 רשימת עובדים")
        st.dataframe(df_employees)

        st.subheader("📂 רשימת משימות")
        st.dataframe(df_tasks)

        # 📌 שיבוץ פשוט: עובד לכל משימה לפי תפקיד תואם (אחד לאחד)
        assignments = []
        for _, task in df_tasks.iterrows():
            matched = df_employees[
                (df_employees["Role"] == task["RequiredRole"]) & 
                (df_employees["Availability"].str.lower() == "yes")
            ]
            if not matched.empty:
                assigned_employee = matched.iloc[0]["FullName"]
                assignments.append({
                    "Task": task["TaskName"],
                    "AssignedTo": assigned_employee,
                    "Duration": task["Duration (hours)"],
                    "HourlyRate": df_employees[df_employees["FullName"] == assigned_employee]["HourlyRate"].values[0],
                })
        if assignments:
            df_assignments = pd.DataFrame(assignments)
            st.subheader("✅ שיבוץ אופטימלי (פשוט)")
            st.dataframe(df_assignments)
        else:
            st.warning("לא נמצאו שיבוצים מתאימים עם עובדים זמינים.")
    
    except Exception as e:
        st.error(f"שגיאה בקריאת הקובץ: {e}")
