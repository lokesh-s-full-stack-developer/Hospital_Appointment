import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from datetime import datetime, timedelta
import pickle
import sqlite3

# Database setup
def init_db():
    conn = sqlite3.connect('hospital_appointments.db')
    c = conn.cursor()

    # Create tables if they don't exist
    c.execute('''CREATE TABLE IF NOT EXISTS patients
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  age INTEGER,
                  gender TEXT,
                  phone TEXT,
                  email TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS doctors
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  specialization TEXT,
                  available_days TEXT,
                  available_hours TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS appointments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  patient_id INTEGER,
                  doctor_id INTEGER,
                  appointment_date TEXT,
                  appointment_time TEXT,
                  status TEXT,
                  symptoms TEXT,
                  diagnosis TEXT,
                  treatment TEXT,
                  no_show INTEGER DEFAULT 0,
                  FOREIGN KEY(patient_id) REFERENCES patients(id),
                  FOREIGN KEY(doctor_id) REFERENCES doctors(id))''')

    conn.commit()
    conn.close()

# Initialize database
init_db()

# Sample data insertion (for demo)
def insert_sample_data():
    conn = sqlite3.connect('hospital_appointments.db')
    c = conn.cursor()

    # Check if data already exists
    c.execute("SELECT COUNT(*) FROM patients")
    if c.fetchone()[0] == 0:
        # Insert sample patients
        patients = [
            ('John Doe', 35, 'Male', '1234567890', 'john@example.com'),
            ('Jane Smith', 28, 'Female', '0987654321', 'jane@example.com'),
            ('Mike Johnson', 45, 'Male', '1122334455', 'mike@example.com'),
            ('Sarah Williams', 32, 'Female', '2233445566', 'sarah@example.com'),
            ('David Brown', 60, 'Male', '3344556677', 'david@example.com'),
            ('Emily Davis', 25, 'Female', '4455667788', 'emily@example.com'),
            ('Robert Wilson', 50, 'Male', '5566778899', 'robert@example.com')
        ]
        c.executemany("INSERT INTO patients (name, age, gender, phone, email) VALUES (?, ?, ?, ?, ?)", patients)

        # Insert sample doctors - Expanded list with more specialties
        doctors = [
            # Cardiology
            ('Dr. Sarah Chen', 'Cardiology', 'Mon,Wed,Fri', '09:00-17:00'),
            ('Dr. Michael Rodriguez', 'Cardiology', 'Tue,Thu,Sat', '10:00-18:00'),

            # Neurology
            ('Dr. Robert Brown', 'Neurology', 'Tue,Thu,Sat', '10:00-18:00'),
            ('Dr. Emily Zhang', 'Neurology', 'Mon,Wed,Fri', '08:00-16:00'),

            # Pediatrics
            ('Dr. Lisa Johnson', 'Pediatrics', 'Mon-Fri', '08:00-16:00'),
            ('Dr. James Wilson', 'Pediatrics', 'Mon-Fri', '09:00-17:00'),

            # Orthopedics
            ('Dr. Mark Taylor', 'Orthopedics', 'Mon,Wed,Fri', '08:00-17:00'),
            ('Dr. Anna Martinez', 'Orthopedics', 'Tue,Thu', '09:00-18:00'),

            # Dermatology
            ('Dr. Rachel Kim', 'Dermatology', 'Mon,Tue,Wed', '10:00-16:00'),
            ('Dr. Daniel Park', 'Dermatology', 'Thu,Fri', '11:00-19:00'),

            # Ophthalmology
            ('Dr. Susan Lee', 'Ophthalmology', 'Mon,Wed,Fri', '09:00-17:00'),
            ('Dr. Kevin White', 'Ophthalmology', 'Tue,Thu,Sat', '08:00-15:00'),

            # General Practice
            ('Dr. Jennifer Adams', 'General Practice', 'Mon-Fri', '08:00-18:00'),
            ('Dr. Thomas Clark', 'General Practice', 'Mon-Sat', '07:00-19:00'),

            # Gastroenterology
            ('Dr. Olivia Green', 'Gastroenterology', 'Tue,Thu', '09:00-17:00'),
            ('Dr. Richard Scott', 'Gastroenterology', 'Mon,Wed,Fri', '10:00-18:00'),

            # Endocrinology
            ('Dr. Patricia King', 'Endocrinology', 'Mon,Wed', '08:00-16:00'),
            ('Dr. Charles Young', 'Endocrinology', 'Tue,Thu,Fri', '09:00-17:00'),

            # Psychiatry
            ('Dr. Amanda Hall', 'Psychiatry', 'Mon-Fri', '10:00-18:00'),
            ('Dr. Brian Allen', 'Psychiatry', 'Tue,Thu,Sat', '09:00-17:00')
        ]
        c.executemany("INSERT INTO doctors (name, specialization, available_days, available_hours) VALUES (?, ?, ?, ?)",
                      doctors)

        # Insert sample appointments
        appointments = [
            (1, 1, '2023-06-15', '10:00', 'Completed', 'Chest pain', 'Angina', 'Medication', 0),
            (2, 2, '2023-06-16', '14:00', 'Completed', 'Headaches', 'Migraine', 'Therapy', 0),
            (3, 3, '2023-06-17', '11:00', 'No-show', 'Fever', '', '', 1),
            (1, 1, '2023-06-20', '15:00', 'Scheduled', 'Follow-up', '', '', 0),
            (4, 5, '2023-06-18', '10:30', 'Completed', 'Back pain', 'Muscle strain', 'Physical therapy', 0),
            (5, 8, '2023-06-19', '13:00', 'Completed', 'Skin rash', 'Eczema', 'Topical cream', 0),
            (2, 10, '2023-06-21', '14:30', 'Scheduled', 'Annual checkup', '', '', 0),
            (3, 15, '2023-06-22', '11:00', 'Scheduled', 'Diabetes consultation', '', '', 0),
            (6, 12, '2023-06-23', '09:00', 'Completed', 'Eye exam', 'Myopia', 'Prescription glasses', 0),
            (7, 18, '2023-06-24', '16:00', 'Completed', 'Anxiety', 'Generalized anxiety disorder', 'Therapy', 0)
        ]
        c.executemany("""INSERT INTO appointments 
                      (patient_id, doctor_id, appointment_date, appointment_time, status, symptoms, diagnosis, treatment, no_show) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", appointments)

        conn.commit()

    conn.close()

insert_sample_data()

# Machine Learning Model for No-show Prediction
def train_no_show_model():
    conn = sqlite3.connect('hospital_appointments.db')

    # Load historical data
    query = """
    SELECT 
        p.age, p.gender,
        a.appointment_date, a.appointment_time, a.status, a.no_show,
        julianday(a.appointment_date) - julianday(date('now')) as days_until_appointment,
        strftime('%w', a.appointment_date) as day_of_week
    FROM appointments a
    JOIN patients p ON a.patient_id = p.id
    WHERE a.status != 'Scheduled'
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if len(df) == 0:
        return None

    # Feature engineering
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'Male' else 0)
    df['day_of_week'] = df['day_of_week'].astype(int)

    # Prepare features and target
    X = df[['age', 'gender', 'days_until_appointment', 'day_of_week']]
    y = df['no_show']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # Save model
    with open('no_show_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return model

# Try to load existing model or train a new one
try:
    with open('no_show_model.pkl', 'rb') as f:
        no_show_model = pickle.load(f)
except:
    no_show_model = train_no_show_model()

# Streamlit App
def main():
    st.title("🏥 Hospital Appointment Booking & Analysis System")

    menu = ["Book Appointment", "View Appointments", "Patient Management",
            "Doctor Management", "No-show Analysis", "Appointment Analytics"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Book Appointment":
        st.subheader("📅 Book New Appointment")

        # Patient selection
        conn = sqlite3.connect('hospital_appointments.db')
        patients = pd.read_sql_query("SELECT id, name, age FROM patients", conn)
        patient_id = st.selectbox("Select Patient", patients['id'], format_func=lambda
            x: f"{patients[patients['id'] == x]['name'].values[0]} (Age: {patients[patients['id'] == x]['age'].values[0]})")

        # Doctor selection by specialization
        doctors = pd.read_sql_query("SELECT id, name, specialization FROM doctors", conn)
        specialization = st.selectbox("Select Specialization", sorted(doctors['specialization'].unique()))
        filtered_doctors = doctors[doctors['specialization'] == specialization]

        if not filtered_doctors.empty:
            doctor_id = st.selectbox("Select Doctor", filtered_doctors['id'],
                                     format_func=lambda x: filtered_doctors[filtered_doctors['id'] == x]['name'].values[0])

            # Get doctor availability
            doctor_info = pd.read_sql_query(f"SELECT available_days, available_hours FROM doctors WHERE id = {doctor_id}", conn).iloc[0]
            available_days = doctor_info['available_days']
            available_hours = doctor_info['available_hours']

            # Date selection
            st.write(f"📅 Doctor Availability: {available_days}, ⏰ {available_hours}")
            appointment_date = st.date_input("Select Date", min_value=datetime.today())

            # Check if selected day is available
            day_name = appointment_date.strftime('%a')
            if available_days != 'Mon-Fri' and day_name not in available_days.split(','):
                st.error("❌ Doctor not available on this day")
            else:
                # Time selection
                start_hour = int(available_hours.split('-')[0].split(':')[0])
                end_hour = int(available_hours.split('-')[1].split(':')[0])
                available_times = [f"{h:02d}:00" for h in range(start_hour, end_hour)]

                # Check existing appointments
                existing_appts = pd.read_sql_query(f"""
                    SELECT appointment_time FROM appointments 
                    WHERE doctor_id = {doctor_id} AND appointment_date = '{appointment_date}' AND status = 'Scheduled'
                """, conn)

                booked_times = existing_appts['appointment_time'].tolist()
                available_times = [t for t in available_times if t not in booked_times]

                if available_times:
                    appointment_time = st.selectbox("Select Time", available_times)
                    symptoms = st.text_area("Describe Symptoms")

                    if st.button("Book Appointment"):
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO appointments 
                            (patient_id, doctor_id, appointment_date, appointment_time, status, symptoms)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (patient_id, doctor_id, appointment_date, appointment_time, 'Scheduled', symptoms))
                        conn.commit()
                        st.success("✅ Appointment Booked Successfully!")

                        # Predict no-show probability
                        if no_show_model:
                            patient_data = pd.read_sql_query(f"SELECT age, gender FROM patients WHERE id = {patient_id}", conn).iloc[0]
                            day_of_week = appointment_date.weekday()
                            days_until = (appointment_date - datetime.today().date()).days

                            features = np.array([[patient_data['age'], 1 if patient_data['gender'] == 'Male' else 0,
                                                  days_until, day_of_week]])
                            proba = no_show_model.predict_proba(features)[0][1]
                            st.warning(f"⚠️ No-show probability: {proba * 100:.1f}%")
                else:
                    st.error("❌ No available time slots for this doctor on the selected date")
        else:
            st.error("❌ No doctors available for this specialization")

        conn.close()

    elif choice == "View Appointments":
        st.subheader("📋 Appointment Schedule")

        view_option = st.radio("View Options", ["Upcoming", "Completed", "No-shows", "All"], horizontal=True)

        conn = sqlite3.connect('hospital_appointments.db')

        if view_option == "Upcoming":
            query = """
                SELECT a.id, p.name as patient, d.name as doctor, d.specialization, 
                       a.appointment_date, a.appointment_time, a.symptoms
                FROM appointments a
                JOIN patients p ON a.patient_id = p.id
                JOIN doctors d ON a.doctor_id = d.id
                WHERE a.status = 'Scheduled' AND date(a.appointment_date) >= date('now')
                ORDER BY a.appointment_date, a.appointment_time
            """
        elif view_option == "Completed":
            query = """
                SELECT a.id, p.name as patient, d.name as doctor, d.specialization, 
                       a.appointment_date, a.appointment_time, a.symptoms, a.diagnosis, a.treatment
                FROM appointments a
                JOIN patients p ON a.patient_id = p.id
                JOIN doctors d ON a.doctor_id = d.id
                WHERE a.status = 'Completed'
                ORDER BY a.appointment_date DESC
            """
        elif view_option == "No-shows":
            query = """
                SELECT a.id, p.name as patient, d.name as doctor, d.specialization, 
                       a.appointment_date, a.appointment_time, a.symptoms
                FROM appointments a
                JOIN patients p ON a.patient_id = p.id
                JOIN doctors d ON a.doctor_id = d.id
                WHERE a.no_show = 1
                ORDER BY a.appointment_date DESC
            """
        else:
            query = """
                SELECT a.id, p.name as patient, d.name as doctor, d.specialization, 
                       a.appointment_date, a.appointment_time, a.status, a.symptoms
                FROM appointments a
                JOIN patients p ON a.patient_id = p.id
                JOIN doctors d ON a.doctor_id = d.id
                ORDER BY a.appointment_date DESC
            """

        appointments = pd.read_sql_query(query, conn)
        st.dataframe(appointments)

        # Update appointment status
        if view_option == "Upcoming" and not appointments.empty:
            st.subheader("Update Appointment Status")
            appt_id = st.selectbox("Select Appointment", appointments['id'])
            new_status = st.selectbox("New Status", ["Completed", "No-show"])

            if st.button("Update Status"):
                if new_status == "Completed":
                    diagnosis = st.text_input("Diagnosis")
                    treatment = st.text_input("Treatment")
                    c = conn.cursor()
                    c.execute("""
                        UPDATE appointments 
                        SET status = ?, diagnosis = ?, treatment = ?, no_show = 0
                        WHERE id = ?
                    """, ("Completed", diagnosis, treatment, appt_id))
                else:
                    c = conn.cursor()
                    c.execute("""
                        UPDATE appointments 
                        SET status = 'No-show', no_show = 1
                        WHERE id = ?
                    """, (appt_id,))

                conn.commit()
                st.success("✅ Appointment status updated!")
                st.experimental_rerun()

        conn.close()

    elif choice == "Patient Management":
        st.subheader("👨‍⚕️ Patient Management")

        option = st.radio("Options", ["View Patients", "Add New Patient", "Update Patient"], horizontal=True)

        conn = sqlite3.connect('hospital_appointments.db')

        if option == "View Patients":
            patients = pd.read_sql_query("SELECT * FROM patients", conn)
            st.dataframe(patients)

        elif option == "Add New Patient":
            with st.form("patient_form"):
                name = st.text_input("Full Name")
                age = st.number_input("Age", min_value=0, max_value=120)
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                phone = st.text_input("Phone Number")
                email = st.text_input("Email")

                if st.form_submit_button("Add Patient"):
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO patients (name, age, gender, phone, email)
                        VALUES (?, ?, ?, ?, ?)
                    """, (name, age, gender, phone, email))
                    conn.commit()
                    st.success("✅ Patient added successfully!")

        elif option == "Update Patient":
            patients = pd.read_sql_query("SELECT id, name FROM patients", conn)
            patient_id = st.selectbox("Select Patient", patients['id'],
                                      format_func=lambda x: patients[patients['id'] == x]['name'].values[0])

            patient_data = pd.read_sql_query(f"SELECT * FROM patients WHERE id = {patient_id}", conn).iloc[0]

            with st.form("update_patient_form"):
                name = st.text_input("Full Name", value=patient_data['name'])
                age = st.number_input("Age", min_value=0, max_value=120, value=patient_data['age'])
                gender = st.selectbox("Gender", ["Male", "Female", "Other"],
                                      index=0 if patient_data['gender'] == 'Male' else 1 if patient_data['gender'] == 'Female' else 2)
                phone = st.text_input("Phone Number", value=patient_data['phone'])
                email = st.text_input("Email", value=patient_data['email'])

                if st.form_submit_button("Update Patient"):
                    c = conn.cursor()
                    c.execute("""
                        UPDATE patients 
                        SET name = ?, age = ?, gender = ?, phone = ?, email = ?
                        WHERE id = ?
                    """, (name, age, gender, phone, email, patient_id))
                    conn.commit()
                    st.success("✅ Patient updated successfully!")

        conn.close()

    elif choice == "Doctor Management":
        st.subheader("👩‍⚕️ Doctor Management")

        option = st.radio("Options", ["View Doctors", "Add New Doctor", "Update Doctor"], horizontal=True)

        conn = sqlite3.connect('hospital_appointments.db')

        if option == "View Doctors":
            doctors = pd.read_sql_query("SELECT * FROM doctors", conn)
            st.dataframe(doctors)

        elif option == "Add New Doctor":
            with st.form("doctor_form"):
                name = st.text_input("Doctor Name")
                specialization = st.text_input("Specialization")
                available_days = st.multiselect("Available Days",
                                                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                                default=["Mon", "Wed", "Fri"])
                start_hour = st.slider("Start Hour", 8, 18, 9)
                end_hour = st.slider("End Hour", 9, 20, 17)

                if st.form_submit_button("Add Doctor"):
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO doctors (name, specialization, available_days, available_hours)
                        VALUES (?, ?, ?, ?)
                    """, (name, specialization, ",".join(available_days), f"{start_hour:02d}:00-{end_hour:02d}:00"))
                    conn.commit()
                    st.success("✅ Doctor added successfully!")

        elif option == "Update Doctor":
            doctors = pd.read_sql_query("SELECT id, name FROM doctors", conn)
            doctor_id = st.selectbox("Select Doctor", doctors['id'],
                                     format_func=lambda x: doctors[doctors['id'] == x]['name'].values[0])

            doctor_data = pd.read_sql_query(f"SELECT * FROM doctors WHERE id = {doctor_id}", conn).iloc[0]

            with st.form("update_doctor_form"):
                name = st.text_input("Doctor Name", value=doctor_data['name'])
                specialization = st.text_input("Specialization", value=doctor_data['specialization'])

                current_days = doctor_data['available_days'].split(',')
                available_days = st.multiselect("Available Days",
                                                ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                                default=current_days)

                current_hours = doctor_data['available_hours'].split('-')
                start_hour = int(current_hours[0].split(':')[0])
                end_hour = int(current_hours[1].split(':')[0])

                new_start = st.slider("Start Hour", 8, 18, start_hour)
                new_end = st.slider("End Hour", new_start + 1, 20, end_hour)

                if st.form_submit_button("Update Doctor"):
                    c = conn.cursor()
                    c.execute("""
                        UPDATE doctors 
                        SET name = ?, specialization = ?, available_days = ?, available_hours = ?
                        WHERE id = ?
                    """, (name, specialization, ",".join(available_days), f"{new_start:02d}:00-{new_end:02d}:00",
                          doctor_id))
                    conn.commit()
                    st.success("✅ Doctor updated successfully!")

        conn.close()

    elif choice == "No-show Analysis":
        st.subheader("📊 No-show Analysis and Prediction")

        if no_show_model:
            st.write("### No-show Prediction Model Performance")

            conn = sqlite3.connect('hospital_appointments.db')
            df = pd.read_sql_query("""
                SELECT 
                    p.age, p.gender,
                    a.appointment_date, a.appointment_time, a.status, a.no_show,
                    julianday(a.appointment_date) - julianday(date('now')) as days_until_appointment,
                    strftime('%w', a.appointment_date) as day_of_week
                FROM appointments a
                JOIN patients p ON a.patient_id = p.id
                WHERE a.status != 'Scheduled'
            """, conn)
            conn.close()

            if not df.empty:
                # Calculate no-show rate
                no_show_rate = df['no_show'].mean() * 100
                st.metric("Overall No-show Rate", f"{no_show_rate:.1f}%")

                # Show feature importance
                st.write("### Feature Importance")
                feature_importance = pd.DataFrame({
                    'Feature': ['Age', 'Gender (Male=1)', 'Days Until Appointment', 'Day of Week'],
                    'Importance': no_show_model.feature_importances_
                }).sort_values('Importance', ascending=False)

                st.bar_chart(feature_importance.set_index('Feature'))

                # Show patterns
                st.write("### No-show Patterns")

                col1, col2 = st.columns(2)

                with col1:
                    # By age
                    age_groups = pd.cut(df['age'], bins=[0, 18, 30, 50, 70, 100])
                    age_no_show = df.groupby(age_groups)['no_show'].mean().reset_index()
                    age_no_show['age'] = age_no_show['age'].astype(str)
                    st.write("No-show Rate by Age Group")
                    st.bar_chart(age_no_show.set_index('age'))

                with col2:
                    # By days until appointment
                    days_groups = pd.cut(df['days_until_appointment'],
                                         bins=[0, 1, 7, 14, 30, 365],
                                         labels=['Same day', '1-7 days', '1-2 weeks', '2-4 weeks', '>4 weeks'])
                    days_no_show = df.groupby(days_groups)['no_show'].mean().reset_index()
                    st.write("No-show Rate by Booking Lead Time")
                    st.bar_chart(days_no_show.set_index('days_until_appointment'))

                # Prediction interface
                st.write("### Predict No-show Probability for New Appointment")

                with st.form("prediction_form"):
                    age = st.number_input("Patient Age", min_value=0, max_value=120)
                    gender = st.selectbox("Patient Gender", ["Female", "Male"])
                    days_until = st.number_input("Days Until Appointment", min_value=0)
                    day_of_week = st.selectbox("Day of Week",
                                               ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                                                "Sunday"])

                    if st.form_submit_button("Predict"):
                        features = np.array([[age, 1 if gender == 'Male' else 0,
                                              days_until,
                                              ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                                               "Sunday"].index(day_of_week)]])
                        proba = no_show_model.predict_proba(features)[0][1]
                        st.warning(f"⚠️ Predicted No-show Probability: {proba * 100:.1f}%")
            else:
                st.warning("Not enough data for analysis")
        else:
            st.warning("No-show prediction model not available. Please book some appointments first.")

    elif choice == "Appointment Analytics":
        st.subheader("📈 Appointment Analytics Dashboard")

        conn = sqlite3.connect('hospital_appointments.db')

        # Load data
        appointments = pd.read_sql_query("""
            SELECT 
                a.*, 
                p.name as patient_name, p.age, p.gender,
                d.name as doctor_name, d.specialization
            FROM appointments a
            JOIN patients p ON a.patient_id = p.id
            JOIN doctors d ON a.doctor_id = d.id
        """, conn, parse_dates=['appointment_date'])

        conn.close()

        if not appointments.empty:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(appointments['appointment_date']):
                appointments['appointment_date'] = pd.to_datetime(appointments['appointment_date'])

            # Time period selection
            st.sidebar.header("Filters")
            min_date = appointments['appointment_date'].min().date()
            max_date = appointments['appointment_date'].max().date()

            start_date = st.sidebar.date_input("Start Date", min_date)
            end_date = st.sidebar.date_input("End Date", max_date)

            # Filter data
            filtered = appointments[
                (appointments['appointment_date'].dt.date >= start_date) &
                (appointments['appointment_date'].dt.date <= end_date)
                ]

            # KPI metrics
            st.write("### Key Performance Indicators")

            col1, col2, col3, col4 = st.columns(4)

            total_appointments = len(filtered)
            completed = len(filtered[filtered['status'] == 'Completed'])
            no_shows = len(filtered[filtered['no_show'] == 1])
            scheduled = len(filtered[filtered['status'] == 'Scheduled'])

            with col1:
                st.metric("Total Appointments", total_appointments)

            with col2:
                st.metric("Completed", completed, f"{completed / total_appointments * 100:.1f}%")

            with col3:
                st.metric("No-shows", no_shows, f"{no_shows / total_appointments * 100:.1f}%")

            with col4:
                st.metric("Scheduled", scheduled, f"{scheduled / total_appointments * 100:.1f}%")

            # Charts
            st.write("### Appointment Trends")

            # By date
            daily_counts = filtered.groupby(filtered['appointment_date'].dt.date).size().reset_index(name='count')
            st.line_chart(daily_counts.set_index('appointment_date'))

            # By specialization
            st.write("### Appointments by Specialization")
            spec_counts = filtered.groupby('specialization').size().reset_index(name='count')
            st.bar_chart(spec_counts.set_index('specialization'))

            # Patient demographics
            st.write("### Patient Demographics")

            col1, col2 = st.columns(2)

            with col1:
                age_groups = pd.cut(filtered['age'], bins=[0, 18, 30, 50, 70, 100])
                age_counts = filtered.groupby(age_groups).size().reset_index(name='count')
                age_counts['age'] = age_counts['age'].astype(str)
                st.bar_chart(age_counts.set_index('age'))

            with col2:
                gender_counts = filtered.groupby('gender').size().reset_index(name='count')
                st.bar_chart(gender_counts.set_index('gender'))

            # Doctor performance
            st.write("### Doctor Performance")
            doctor_stats = filtered.groupby(['doctor_name', 'specialization']).agg({
                'id': 'count',
                'no_show': 'mean'
            }).rename(columns={'id': 'appointments', 'no_show': 'no_show_rate'})
            doctor_stats['no_show_rate'] = doctor_stats['no_show_rate'] * 100
            st.dataframe(doctor_stats.sort_values('appointments', ascending=False))
        else:
            st.warning("No appointment data available for the selected period")

if __name__ == "__main__":
    main()