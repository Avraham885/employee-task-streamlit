import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import matplotlib.pyplot as plt
import re
import unicodedata

HARD_CONSTRAINT = 1e9

st.set_page_config(page_title="מערכת לשיבוץ אופטימלי", layout="wide")

st.markdown("""
<style>
    body {
        direction: rtl;
        unicode-bidi: embed;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stLabel, .stRadio, .stSelectbox, .stFileUploader, .stButton, .st-bh, .st-bv {
        direction: rtl;
        text-align: right;
    }
    .stDataFrame {
        direction: rtl;
        text-align: right;
    }
    div.st-bv {
        text-align: right;
    }
    [data-testid="stSidebar"] {
        direction: rtl;
    }
    .st-bu {
        direction: rtl;
    }
    [data-testid="stForm"] > div {
        direction: rtl;
    }
    .stButton > button {
        direction: rtl;
    }
</style>
""", unsafe_allow_html=True)

st.title("מערכת לשיבוץ אופטימלי - באמצעות האלגוריתם ההונגרי")
st.markdown("""
ברוכים הבאים למערכת לפתרון בעיות שיבוץ.
בחרו את סוג הבעיה שברצונכם לפתור.
""")

def solve_assignment_problem(rows_list, cols_list, cost_matrix_base, rows_label, cols_label, df_rows, df_cols, display_cols_a, display_cols_b, **kwargs):
    """פותר בעיית השמה באמצעות האלגוריתם ההונגרי ומציג את התוצאות."""
    start_time = time.perf_counter()
    
    num_rows, num_cols = cost_matrix_base.shape
    size = max(num_rows, num_cols)
    cost_matrix = np.full((size, size), fill_value=HARD_CONSTRAINT, dtype=float)
    cost_matrix[:num_rows, :num_cols] = cost_matrix_base

    if num_rows > num_cols:
        rows_list_extended = rows_list.copy()
        cols_list_extended = cols_list.copy()
        for i in range(num_rows - num_cols):
            cols_list_extended.append(f"פריט דמה ({cols_label}_{i+1})")
    elif num_cols > num_rows:
        rows_list_extended = rows_list.copy()
        cols_list_extended = cols_list.copy()
        for i in range(num_cols - num_rows):
            rows_list_extended.append(f"פריט דמה ({rows_label}_{i+1})")
    else:
        rows_list_extended = rows_list
        cols_list_extended = cols_list
    
    st.info(f"✅ נתוני הקלט נקראו בהצלחה. מריץ את האלגוריתם ההונגרי על מטריצה בגודל {cost_matrix.shape}...")
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    assignments = []
    total_cost = 0
    unassigned_items = []

    for r, c in zip(row_ind, col_ind):
        cost = cost_matrix[r, c]
        if cost < HARD_CONSTRAINT:
            assignments.append({
                rows_label: rows_list_extended[r],
                cols_label: cols_list_extended[c],
                'עלות': round(cost, 2)
            })
            total_cost += cost
        else:
            unassigned_items.append({
                'פריט שנותר ללא שיבוץ': rows_list_extended[r],
                'הותאם לפריט דמה': cols_list_extended[c]
            })

    end_time = time.perf_counter()
    run_time = end_time - start_time
    
    st.subheader("📋 טבלת שיבוץ אופטימלי:")
    df_assignments = pd.DataFrame(assignments)
    
    if df_rows is not None and df_cols is not None:
        if display_cols_a or display_cols_b:
            df_display_rows = df_rows.set_index('ID')[display_cols_a]
            df_display_cols = df_cols.set_index('ID')[display_cols_b]
            df_assignments = df_assignments.rename(columns={rows_label: 'ID_A', cols_label: 'ID_B'})
            
            df_assignments = pd.merge(df_assignments, df_display_rows, 
                                      left_on='ID_A', right_index=True, how='left')
            df_assignments = pd.merge(df_assignments, df_display_cols, 
                                      left_on='ID_B', right_index=True, how='left')

            final_cols = ['ID_A'] + display_cols_a + ['ID_B'] + display_cols_b + ['עלות']
            df_assignments = df_assignments[final_cols].rename(columns={'ID_A': rows_label, 'ID_B': cols_label})
        else:
            st.warning("לא נבחרו עמודות תצוגה מגיליון A או B – יוצגו רק שדות ID ועלות.")
    
    st.dataframe(df_assignments.sort_values(by="עלות"), use_container_width=True)
    st.success(f"💵 העלות הכוללת של השיבוץ האופטימלי: {total_cost:.2f}")
    st.info(f"⏱️ זמן ריצת האלגוריתם: {run_time:.4f} שניות.")

    if unassigned_items:
        st.warning("⚠️ לא נמצא שיבוץ עבור הפריטים הבאים:")
        df_unassigned = pd.DataFrame(unassigned_items)
        if df_rows is not None:
            df_unassigned = df_unassigned.rename(columns={'פריט שנותר ללא שיבוץ': 'ID_A', 'הותאם לפריט דמה': 'דמה'})
            
            reasons = []
            
            # בדיקת סיבות לאי-שיבוץ ואיסוף
            match_car_type = kwargs.get('match_car_type', False)
            match_silent_pref = kwargs.get('match_silent_pref', False)
            
            for item_id_a in df_unassigned['ID_A'].tolist():
                driver = df_rows[df_rows['ID'] == item_id_a].iloc[0]
                
                # מצא את האילוצים הקשיחים שנכשלו עבור הנהג מול כל הנוסעים
                failed_reasons = set()
                for _, passenger in df_cols.iterrows():
                    distance = np.sqrt(
                        (driver['Location_X'] - passenger['Location_X'])**2 +
                        (driver['Location_Y'] - passenger['Location_Y'])**2
                    )

                    if match_car_type and 'Car_Type_Required' in passenger and 'Car_Type' in driver:
                        if pd.notna(passenger['Car_Type_Required']) and passenger['Car_Type_Required'] != driver['Car_Type']:
                            failed_reasons.add(f"לא עונה על דרישת סוג רכב של נוסע {passenger['ID']} (נדרש: {passenger['Car_Type_Required']})")
                    
                    if match_silent_pref and 'Is_Silent_Preference' in passenger and 'Is_Silent' in driver:
                        if pd.notna(passenger['Is_Silent_Preference']) and passenger['Is_Silent_Preference'] and not driver['Is_Silent']:
                            failed_reasons.add(f"לא עונה על העדפת שקט של נוסע {passenger['ID']}")
                    
                    if 'Max_Distance' in driver:
                        if distance > driver['Max_Distance']:
                            failed_reasons.add(f"מרחק הנסיעה ({distance:.2f}) עולה על המרחק המקסימלי לנהג ({driver['Max_Distance']}) עבור נוסע {passenger['ID']}")
                
                if failed_reasons:
                    reasons.append(" | ".join(failed_reasons))
                else:
                    reasons.append("לא נמצאה סיבה ברורה לאי-שיבוץ. ייתכן שאף שיבוץ לא היה משתלם.")
            
            df_unassigned['סיבה לאי-שיבוץ'] = reasons

            df_display_rows = df_rows.set_index('ID')[display_cols_a]
            df_unassigned = pd.merge(df_unassigned, df_display_rows, 
                                     left_on='ID_A', right_index=True, how='left')
            df_unassigned = df_unassigned.rename(columns={'ID_A': 'פריט שנותר ללא שיבוץ'})
            
        st.dataframe(df_unassigned, use_container_width=True)
    else:
        st.info("✅ כל הפריטים שובצו בהצלחה.")

def build_cost_matrix(df_items_a, df_items_b, rules):
    """בונה מטריצת עלויות בהתאם לכללים שהוגדרו."""
    cost_matrix_base = np.zeros((len(df_items_a), len(df_items_b)))

    for i, row_a in df_items_a.iterrows():
        for j, row_b in df_items_b.iterrows():
            cost = 0
            for rule in rules:
                try:
                    val_a = row_a[rule['col_a']]
                    val_b = row_b[rule['col_b']]
                    
                    is_match = False
                    if rule['comp'] == "שווה (==)":
                        is_match = (val_a == val_b)
                    elif rule['comp'] == "לא שווה (!=)":
                        is_match = (val_a != val_b)
                    elif rule['comp'] == "קטן מ (<)":
                        is_match = (val_a < val_b)
                    elif rule['comp'] == "גדול מ (>)":
                        is_match = (val_a > val_b)
                    
                    if not is_match:
                        cost += rule['penalty']
                except KeyError as e:
                    st.error(f"❌ שגיאה: העמודה {e} לא קיימת באחד הגיליונות.")
                    return None
            
            cost_matrix_base[i, j] = cost
    
    return cost_matrix_base

def clean_column_names(df):
    """
    מנקה שמות עמודות מרווחים מיותרים ותווים בלתי נראים.
    הפונקציה משפרת את הניקוי כדי להתמודד עם בעיות העתקה-הדבקה מורכבות.
    """
    cleaned_cols = {}
    for col in df.columns:
        # נרמול Unicode וטיפול בתווים מיוחדים
        col = unicodedata.normalize('NFKC', col)
        # הסרת תווים בלתי נראים כמו תווים של כיווניות טקסט
        col = re.sub(r'[\u200e\u200f\u202a\u202b\u202c\u2066\u2067\u2068]+', '', col)
        # החלפת כל סוגי הרווחים ברווח יחיד ותבנית
        col = re.sub(r'\s+', ' ', col).strip()
        cleaned_cols[col] = col
    
    df.rename(columns=cleaned_cols, inplace=True)
    return df

def handle_duplicate_column_names(df_a, df_b):
    """
    מטפל בשמות עמודות זהים בין שני הגיליונות על ידי הוספת סיומת.
    """
    common_cols = set(df_a.columns).intersection(df_b.columns)
    common_cols.discard('ID')  # לא מתייחס לעמודת ה-ID ככפילות

    if common_cols:
        st.warning(f"⚠️ שימו לב: נמצאו שמות עמודות זהים בשני הגיליונות: {list(common_cols)}. המערכת שינתה את שמות העמודות באופן אוטומטי כדי למנוע התנגשויות.")
        
        rename_dict_a = {col: f"{col} (A)" for col in common_cols}
        rename_dict_b = {col: f"{col} (B)" for col in common_cols}
        
        df_a.rename(columns=rename_dict_a, inplace=True)
        df_b.rename(columns=rename_dict_b, inplace=True)
    
    return df_a, df_b

problem_type = st.radio("בחר את סוג הבעיה:", ["שיבוץ נהגים לנוסעים", "שיבוץ מותאם אישית (הכנת מטריצה)", "מבחן ביצועים (לצורך דוח)"])

if problem_type == "שיבוץ נהגים לנוסעים":
    st.subheader("שיבוץ נהגים לנוסעים")
    st.markdown("""
        אנא העלה קובץ Excel המכיל את הגיליונות: `Drivers` ו-`Passengers`.
        הקוד יחשב את העלות האופטימלית לשיבוץ על בסיס מרחק וכללים נוספים שתגדיר.
    """)
    uploaded_file = st.file_uploader("בחר קובץ Excel", type=["xlsx"], key="taxi_uploader")
    
    if uploaded_file:
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheets = excel_file.sheet_names
            
            if 'Drivers' not in sheets or 'Passengers' not in sheets:
                st.error("❌ הקובץ חייב להכיל את שני הגיליונות 'Drivers' ו-'Passengers'.")
            else:
                df_drivers = pd.read_excel(excel_file, sheet_name="Drivers")
                df_passengers = pd.read_excel(excel_file, sheet_name="Passengers")
                
                df_drivers = clean_column_names(df_drivers)
                df_passengers = clean_column_names(df_passengers)

                if df_drivers.empty or df_passengers.empty:
                    st.error("❌ הגיליונות חייבים להכיל נתונים.")
                elif 'ID' not in df_drivers.columns or 'ID' not in df_passengers.columns:
                    st.error("❌ הגיליונות חייבים לכלול עמודה בשם 'ID'.")
                else:
                    st.markdown("---")
                    
                    with st.expander("בחר נתונים נוספים להצגה בתוצאות"):
                        display_cols_drivers = st.multiselect("בחר עמודות מגיליון 'Drivers' להצגה:", 
                                                              [col for col in df_drivers.columns if col != 'ID'],
                                                              key='display_drivers')
                        display_cols_passengers = st.multiselect("בחר עמודות מגיליון 'Passengers' להצגה:", 
                                                                 [col for col in df_passengers.columns if col != 'ID'],
                                                                 key='display_passengers')
                    
                    st.markdown("#### הגדרת כללים ומשקולות לשיבוץ")
                    st.markdown("השתמש בשדות הקלט כדי להקצות משקל לחשיבות של כל גורם. משקל גבוה = חשיבות גבוהה.")
                    
                    with st.form("driver_assignment_rules"):
                        
                        st.markdown("##### משקולות לחישוב העלות:")
                        # משקלים לגורמים שונים
                        weight_distance = st.number_input("משקל למרחק (מרחק קצר עדיף)", min_value=0.0, value=5.0)
                        weight_rating_match = st.number_input("משקל להתאמת דירוגים", min_value=0.0, value=3.0)
                        
                        st.markdown("##### הגדרת אילוצים קשיחים:")
                        # אילוצים קשיחים שמונעים שיבוץ
                        match_car_type = st.checkbox("התאמת סוג רכב (אם נדרש)", value=True)
                        match_silent_pref = st.checkbox("התאמת העדפה לנסיעה שקטה", value=True)
                        
                        if st.form_submit_button("הרץ שיבוץ"):
                            with st.spinner('בניית מטריצת עלויות וביצוע שיבוץ...'):
                                rows = df_drivers['ID'].tolist()
                                cols = df_passengers['ID'].tolist()
                                
                                cost_matrix_base = np.zeros((len(df_drivers), len(df_passengers)))
                                
                                for i, driver in df_drivers.iterrows():
                                    for j, passenger in df_passengers.iterrows():
                                        cost = 0
                                        
                                        # 1. חישוב עלות על בסיס מרחק
                                        distance = np.sqrt(
                                            (driver['Location_X'] - passenger['Location_X'])**2 +
                                            (driver['Location_Y'] - passenger['Location_Y'])**2
                                        )
                                        cost += distance * weight_distance
                                        
                                        # 2. חישוב עלות על בסיס דירוגים (פער קטן עדיף)
                                        if 'Rating' in driver and 'Rating' in passenger:
                                            rating_diff = abs(driver['Rating'] - passenger['Rating'])
                                            cost += rating_diff * weight_rating_match
                                        
                                        # 3. אילוצים קשיחים
                                        # אילוץ סוג רכב
                                        if match_car_type and 'Car_Type_Required' in passenger and 'Car_Type' in driver:
                                            if pd.notna(passenger['Car_Type_Required']) and passenger['Car_Type_Required'] != driver['Car_Type']:
                                                cost += HARD_CONSTRAINT
                                        
                                        # אילוץ נסיעה שקטה
                                        if match_silent_pref and 'Is_Silent_Preference' in passenger and 'Is_Silent' in driver:
                                            if pd.notna(passenger['Is_Silent_Preference']) and passenger['Is_Silent_Preference'] and not driver['Is_Silent']:
                                                cost += HARD_CONSTRAINT

                                        # אילוץ מרחק מקסימלי לנהג
                                        if 'Max_Distance' in driver:
                                            if distance > driver['Max_Distance']:
                                                cost += HARD_CONSTRAINT
                                        
                                        cost_matrix_base[i, j] = cost
                                
                                solve_assignment_problem(rows, cols, cost_matrix_base, "נהג", "נוסע", 
                                                        df_drivers, df_passengers, display_cols_drivers, display_cols_passengers,
                                                        match_car_type=match_car_type, match_silent_pref=match_silent_pref)

        except Exception as e:
            st.error(f"❌ שגיאה בקריאת הקובץ: {e}")

elif problem_type == "שיבוץ מותאם אישית (הכנת מטריצה)":
    st.subheader("שיבוץ מותאם אישית")
    st.markdown("""
        אנא העלה קובץ Excel עם שני גיליונות: `Items_A` ו-`Items_B`.
        המערכת תאפשר לך להגדיר כללים כדי לחשב את העלויות ביניהם.
    """)
    uploaded_file = st.file_uploader("בחר קובץ Excel", type=["xlsx"])
    
    if uploaded_file:
        try:
            excel_file = pd.ExcelFile(uploaded_file)
            sheets = excel_file.sheet_names
            
            if 'Items_A' not in sheets or 'Items_B' not in sheets:
                st.error("❌ הקובץ חייב להכיל את שני הגיליונות 'Items_A' ו-'Items_B'.")
            else:
                df_items_a = pd.read_excel(excel_file, sheet_name="Items_A")
                df_items_b = pd.read_excel(excel_file, sheet_name="Items_B")

                df_items_a = clean_column_names(df_items_a)
                df_items_b = clean_column_names(df_items_b)

                # הוספה של פונקציית הטיפול בכפילות
                df_items_a, df_items_b = handle_duplicate_column_names(df_items_a, df_items_b)

                if df_items_a.empty or df_items_b.empty:
                    st.error("❌ הגיליונות חייבים להכיל נתונים.")
                elif 'ID' not in df_items_a.columns or 'ID' not in df_items_b.columns:
                    st.error("❌ הגיליונות חייבים לכלול עמודה בשם 'ID'.")
                else:
                    st.markdown("#### שלב 1: הגדרת פרטי השיבוץ")
                    rows_label = st.text_input("שם הפריטים בשורות (לדוגמה: 'מרצים')", "מרצה")
                    cols_label = st.text_input("שם הפריטים בעמודות (לדוגמה: 'קורסים')", "קורס")
                    
                    st.markdown("---")
                    with st.expander("בחר נתונים נוספים להצגה בתוצאות"):
                        display_cols_a = st.multiselect("בחר עמודות מגיליון A להצגה:", 
                                                         [col for col in df_items_a.columns if col != 'ID'],
                                                         key='display_a')
                        display_cols_b = st.multiselect("בחר עמודות מגיליון B להצגה:", 
                                                         [col for col in df_items_b.columns if col != 'ID'],
                                                         key='display_b')

                    st.markdown("---")
                    st.markdown("#### שלב 2: הגדרת כללי העלויות והאילוצים")
                    st.markdown("הגדר כאן אילוצים שונים. אי התאמה באילוץ תוסיף עלות לשיבוץ.")

                    if 'rules' not in st.session_state:
                        st.session_state.rules = []

                    with st.form("add_rule_form"):
                        # עכשיו הרשימות מכילות שמות ייחודיים שנוצרו על ידי הפונקציה החדשה
                        col_a_options = [col for col in df_items_a.columns.tolist() if col != 'ID']
                        col_b_options = [col for col in df_items_b.columns.tolist() if col != 'ID']
                        
                        col_a_selected = st.selectbox("בחר עמודה מגיליון Items_A:", col_a_options)
                        col_b_selected = st.selectbox("בחר עמודה מגיליון Items_B:", col_b_options)
                        
                        comparison_type = st.selectbox("בחר סוג השוואה:", 
                                                       ["שווה (==)", "לא שווה (!=)", "קטן מ (<)", "גדול מ (>)"])
                        
                        penalty_value = st.number_input("ערך ענישה במקרה של אי התאמה:", min_value=0.0, value=1000.0)

                        if st.form_submit_button("הוסף כלל"):
                            st.session_state.rules.append({
                                'col_a': col_a_selected,
                                'col_b': col_b_selected,
                                'comp': comparison_type,
                                'penalty': penalty_value
                            })
                            st.rerun() 
                    
                    if st.session_state.rules:
                        st.markdown("---")
                        st.markdown("#### כללים שהוגדרו:")
                        for idx, rule in enumerate(st.session_state.rules):
                            col1, col2 = st.columns([0.9, 0.1])
                            with col1:
                                st.write(f"**כלל {idx+1}:** אם '{rule['col_a']}' ב-A **{rule['comp']}** '{rule['col_b']}' ב-B, הוסף עלות של **{rule['penalty']}**.")
                            with col2:
                                if st.button("מחק", key=f"delete_rule_{idx}"):
                                    st.session_state.rules.pop(idx)
                                    st.rerun()

                        if st.button("בנה מטריצת עלויות והרץ שיבוץ"):
                            with st.spinner('בניית מטריצת עלויות וביצוע שיבוץ...'):
                                rows = df_items_a['ID'].tolist()
                                cols = df_items_b['ID'].tolist()
                                
                                cost_matrix_base = build_cost_matrix(df_items_a, df_items_b, st.session_state.rules)

                                if cost_matrix_base is not None:
                                    solve_assignment_problem(rows, cols, cost_matrix_base, rows_label, cols_label, df_items_a, df_items_b, display_cols_a, display_cols_b)

        except Exception as e:
            st.error(f"❌ שגיאה בקריאת הקובץ: {e}")

elif problem_type == "מבחן ביצועים (לצורך דוח)":
    st.subheader("מבחן ביצועים")
    st.markdown("""
        כאן תוכל להריץ את האלגוריתם על מטריצות בגדלים שונים כדי לבחון את ביצועיו.
        התוצאות יסייעו בכתיבת דוח הפרויקט על הסיבוכיות של האלגוריתם.
    """)
    
    max_size = st.slider("בחר את גודל המטריצה המקסימלי (N):", min_value=10, max_value=500, value=100, step=10)
    num_runs = st.number_input("בחר מספר הרצות (כדי לקבל ממוצע):", min_value=1, value=5)
    
    if st.button("הרץ מבחן ביצועים"):
        with st.spinner("מריץ את מבחן הביצועים..."):
            sizes = range(10, max_size + 1, 10)
            avg_times = []
            
            for n in sizes:
                run_times = []
                for _ in range(num_runs):
                    cost_matrix = np.random.rand(n, n) * 100
                    start_time = time.perf_counter()
                    linear_sum_assignment(cost_matrix)
                    end_time = time.perf_counter()
                    run_times.append(end_time - start_time)
                avg_times.append(np.mean(run_times))
            
            fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
            ax.plot(sizes, avg_times, marker='o', linestyle='-')
            ax.set_title("זמן ריצה של האלגוריתם ההונגרי כתלות בגודל המטריצה")
            ax.set_xlabel("גודל מטריצה (N)")
            ax.set_ylabel(f"זמן ריצה ממוצע ({num_runs} הרצות) בשניות")
            ax.grid(True)
            st.pyplot(fig)
            st.success("✅ מבחן הביצועים הסתיים בהצלחה.")
            
            df_performance = pd.DataFrame({
                'גודל מטריצה (N)': sizes,
                'זמן ריצה ממוצע (שניות)': [f'{t:.6f}' for t in avg_times]
            })
            st.markdown("#### נתוני מבחן הביצועים:")
            st.dataframe(df_performance, use_container_width=True)
