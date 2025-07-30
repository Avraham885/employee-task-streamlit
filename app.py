import streamlit as st
import pandas as pd

st.set_page_config(page_title="שיבוץ אופטימלי", layout="centered")

st.title("📋 מערכת שיבוץ אופטימלית לעובדים")
st.write("ברוך הבא! טען קובץ Excel עם גיליון עובדים (`Employees`) ומשימות (`Tasks`) לפי התבנית שהגדרנו.")

uploaded_file = st.file_uploader("📤 העלה קובץ Excel", type=["xlsx"])

if uploaded_file is not None:
    try:
        # קריאת הגליונות
        df_employees = pd.read_excel(uploaded_file, sheet_name="Employees")
        df_tasks = pd.read_excel(uploaded_file, sheet_name="Tasks")

        st.subheader("👨‍🔧 רשימת עובדים")
        st.dataframe(df_employees, use_container_width=True)

        st.subheader("📝 רשימת משימות")
        st.dataframe(df_tasks, use_container_width=True)

        # סינון עובדים זמינים
        available_employees = df_employees[df_employees["Availability"] == "Yes"]

        # רשימת השיבוץ
        assignments = []

        for _, task in df_tasks.iterrows():
            role_needed = task["RequiredRole"]

            # מסננים עובדים לפי התפקיד הנדרש
            suitable_workers = available_employees[available_employees["Role"] == role_needed]

            if not suitable_workers.empty:
                # לוקחים את העובד עם השכר לשעה הכי נמוך
                chosen = suitable_workers.sort_values("HourlyRate").iloc[0]
                assignments.append({
                    "Task": task["TaskName"],
                    "AssignedEmployee": chosen["FullName"],
                    "HourlyRate": chosen["HourlyRate"],
                    "Duration (hours)": task["Duration (hours)"],
                    "TotalCost": chosen["HourlyRate"] * task["Duration (hours)"]
                })

        if assignments:
            df_assignments = pd.DataFrame(assignments)

            st.subheader("✅ שיבוץ אופטימלי")
            st.dataframe(df_assignments, use_container_width=True)
        else:
            st.warning("לא נמצאו עובדים מתאימים לשיבוץ.")

    except Exception as e:
        st.error(f"שגיאה בקריאת הקובץ: {str(e)}")
