from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, send_file, Response
)
import os
import io
import random
import smtplib
import sqlite3
from datetime import datetime, timedelta
from email.mime.text import MIMEText

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.security import generate_password_hash, check_password_hash

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "predictions.db")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "heart_project_secret_key")

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "").strip()
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "").strip()
OTP_MODE = os.getenv("OTP_MODE", "console").strip().lower()

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
metadata = joblib.load(os.path.join(BASE_DIR, "metadata.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "heart.csv"))

# Based on your observed outputs, class 0 represents disease and class 1 represents no disease.
DISEASE_CLASS_LABEL = 0

SEX_DISPLAY = {0: "Female", 1: "Male"}
CP_DISPLAY = {
    0: "Typical Angina",
    1: "Atypical Angina",
    2: "Non-anginal Pain",
    3: "Asymptomatic",
}
RESTECG_DISPLAY = {
    0: "Normal",
    1: "ST-T Abnormality",
    2: "Left Ventricular Hypertrophy",
}
SLOPE_DISPLAY = {
    0: "Upsloping",
    1: "Flat",
    2: "Downsloping",
}
THAL_DISPLAY = {
    1: "Normal",
    2: "Fixed Defect",
    3: "Reversible Defect",
}


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column_exists(conn, table_name, column_name, column_def):
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cur.fetchall()]
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}")
        conn.commit()


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            user_email TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            age REAL,
            sex TEXT,
            cp REAL,
            trestbps REAL,
            chol REAL,
            fbs TEXT,
            restecg TEXT,
            thalach REAL,
            exang TEXT,
            oldpeak REAL,
            slope TEXT,
            ca REAL,
            thal TEXT,
            prediction TEXT,
            probability REAL,
            risk_level TEXT,
            risk_message TEXT,
            doctor_notes TEXT
        )
    """)

    conn.commit()

    ensure_column_exists(conn, "users", "is_verified", "INTEGER DEFAULT 0")
    ensure_column_exists(conn, "predictions", "patient_id", "TEXT")
    ensure_column_exists(conn, "predictions", "risk_message", "TEXT")
    ensure_column_exists(conn, "predictions", "doctor_notes", "TEXT")

    conn.close()


init_db()


def send_otp_email(receiver_email: str, otp: str, purpose: str = "OTP"):
    if OTP_MODE == "console":
        print(f"[DEV OTP] {purpose} OTP for {receiver_email}: {otp}")
        return True, "OTP generated in console mode. Check terminal."

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        return False, "Email credentials missing in .env file."

    subject = f"Heart Disease Risk Prediction System - {purpose}"
    body = f"""
Hello,

Your OTP for {purpose.lower()} is: {otp}

This OTP is valid for 5 minutes.

If you did not request this, please ignore this email.

Regards,
Heart Disease Risk Prediction System
"""

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = receiver_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        return True, "OTP sent successfully."
    except smtplib.SMTPAuthenticationError:
        print(f"[DEV OTP FALLBACK] SMTP auth failed. OTP for {receiver_email}: {otp}")
        return False, "Gmail rejected the login. Use OTP_MODE=console for local demo."
    except Exception as e:
        print(f"[DEV OTP FALLBACK] SMTP error. OTP for {receiver_email}: {otp}")
        return False, f"Email send failed: {str(e)}"


def is_logged_in():
    return "user_email" in session


def get_risk_level(disease_probability: float):
    if disease_probability < 35:
        return "Low Risk"
    if disease_probability < 70:
        return "Moderate Risk"
    return "High Risk"


def get_risk_message(risk_level: str):
    if risk_level == "Low Risk":
        return "Preventive monitoring recommended."
    if risk_level == "Moderate Risk":
        return "Clinical review is advised."
    return "Specialist consultation is strongly recommended."


def generate_patient_id():
    conn = get_db_connection()
    count = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"] + 1
    conn.close()
    return f"HDP2026-{count:04d}"


def save_prediction(data: dict):
    conn = get_db_connection()
    conn.execute(
        """
        INSERT INTO predictions (
            patient_id, user_email, timestamp, age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal,
            prediction, probability, risk_level, risk_message, doctor_notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data["patient_id"], data["user_email"], data["timestamp"], data["age"], data["sex"],
            data["cp"], data["trestbps"], data["chol"], data["fbs"], data["restecg"],
            data["thalach"], data["exang"], data["oldpeak"], data["slope"], data["ca"],
            data["thal"], data["prediction"], data["probability"], data["risk_level"],
            data["risk_message"], data["doctor_notes"]
        ),
    )
    conn.commit()
    conn.close()


def validate_inputs(age, trestbps, chol, thalach, oldpeak):
    if not (1 <= age <= 120):
        return "Age must be between 1 and 120."
    if not (50 <= trestbps <= 250):
        return "Resting blood pressure must be between 50 and 250."
    if not (100 <= chol <= 700):
        return "Cholesterol must be between 100 and 700."
    if not (50 <= thalach <= 250):
        return "Maximum heart rate must be between 50 and 250."
    if not (0 <= oldpeak <= 10):
        return "Oldpeak must be between 0 and 10."
    return None


def get_probability_by_class(probability_vector, class_label):
    classes = list(getattr(model, "classes_", []))
    if class_label in classes:
        return float(probability_vector[classes.index(class_label)])

    # Fallback if model.classes_ is unavailable.
    if class_label == 1:
        return float(probability_vector[1])
    return float(probability_vector[0])


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        full_name = request.form["full_name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if len(full_name) < 3:
            flash("Full name must be at least 3 characters.", "danger")
            return render_template("signup.html")

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("signup.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("signup.html")

        conn = get_db_connection()
        existing_user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if existing_user:
            conn.close()
            flash("Email already registered. Please login.", "danger")
            return render_template("signup.html")

        hashed_password = generate_password_hash(password)
        conn.execute(
            """
            INSERT INTO users (full_name, email, password, is_verified, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                full_name,
                email,
                hashed_password,
                0,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        conn.commit()
        conn.close()

        otp = str(random.randint(100000, 999999))
        session["signup_email"] = email
        session["signup_otp"] = otp
        session["signup_otp_expiry"] = (
            datetime.now() + timedelta(minutes=5)
        ).strftime("%Y-%m-%d %H:%M:%S")

        sent, message = send_otp_email(email, otp, purpose="Signup Email Verification")
        flash(message, "success" if sent else "warning")
        return redirect(url_for("verify_signup_otp"))

    return render_template("signup.html")


@app.route("/verify-signup-otp", methods=["GET", "POST"])
def verify_signup_otp():
    if "signup_email" not in session:
        return redirect(url_for("signup"))

    if request.method == "POST":
        entered_otp = request.form["otp"].strip()
        saved_otp = session.get("signup_otp")
        expiry = session.get("signup_otp_expiry")

        if not saved_otp or not expiry:
            flash("Signup OTP expired. Please sign up again.", "danger")
            return redirect(url_for("signup"))

        expiry_time = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S")
        if datetime.now() > expiry_time:
            flash("Signup OTP expired. Please sign up again.", "danger")
            return redirect(url_for("signup"))

        if entered_otp == saved_otp:
            conn = get_db_connection()
            conn.execute("UPDATE users SET is_verified = 1 WHERE email = ?", (session["signup_email"],))
            conn.commit()
            conn.close()

            session.pop("signup_email", None)
            session.pop("signup_otp", None)
            session.pop("signup_otp_expiry", None)

            flash("Email verified successfully. Please login.", "success")
            return redirect(url_for("login"))

        flash("Invalid OTP.", "danger")

    return render_template("verify_signup_otp.html")


@app.route("/resend-signup-otp")
def resend_signup_otp():
    if "signup_email" not in session:
        return redirect(url_for("signup"))

    otp = str(random.randint(100000, 999999))
    session["signup_otp"] = otp
    session["signup_otp_expiry"] = (
        datetime.now() + timedelta(minutes=5)
    ).strftime("%Y-%m-%d %H:%M:%S")

    sent, message = send_otp_email(session["signup_email"], otp, purpose="Signup Email Verification")
    flash(message, "success" if sent else "warning")
    return redirect(url_for("verify_signup_otp"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            is_verified = user["is_verified"] if "is_verified" in user.keys() else 1

            if is_verified != 1:
                flash("Please verify your email before login.", "warning")
                return redirect(url_for("signup"))

            session["user_email"] = user["email"]
            session["user_name"] = user["full_name"]
            flash("Login successful.", "success")
            return redirect(url_for("home"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form["email"].strip().lower()

        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()

        if not user:
            flash("This email is not registered.", "danger")
            return render_template("forgot_password.html")

        otp = str(random.randint(100000, 999999))
        session["reset_email"] = email
        session["reset_otp"] = otp
        session["otp_expiry"] = (
            datetime.now() + timedelta(minutes=5)
        ).strftime("%Y-%m-%d %H:%M:%S")

        sent, message = send_otp_email(email, otp, purpose="Password Reset")
        flash(message, "success" if sent else "warning")
        return redirect(url_for("verify_otp"))

    return render_template("forgot_password.html")


@app.route("/resend-reset-otp")
def resend_reset_otp():
    if "reset_email" not in session:
        return redirect(url_for("forgot_password"))

    otp = str(random.randint(100000, 999999))
    session["reset_otp"] = otp
    session["otp_expiry"] = (
        datetime.now() + timedelta(minutes=5)
    ).strftime("%Y-%m-%d %H:%M:%S")

    sent, message = send_otp_email(session["reset_email"], otp, purpose="Password Reset")
    flash(message, "success" if sent else "warning")
    return redirect(url_for("verify_otp"))


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    if "reset_email" not in session:
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        entered_otp = request.form["otp"].strip()
        saved_otp = session.get("reset_otp")
        expiry = session.get("otp_expiry")

        if not saved_otp or not expiry:
            flash("OTP session expired. Please request again.", "danger")
            return redirect(url_for("forgot_password"))

        expiry_time = datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S")

        if datetime.now() > expiry_time:
            session.pop("reset_otp", None)
            session.pop("otp_expiry", None)
            flash("OTP expired. Please request a new one.", "danger")
            return redirect(url_for("forgot_password"))

        if entered_otp == saved_otp:
            session["otp_verified"] = True
            flash("OTP verified successfully.", "success")
            return redirect(url_for("reset_password"))

        flash("Invalid OTP.", "danger")

    return render_template("verify_otp.html")


@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    if "reset_email" not in session or not session.get("otp_verified"):
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form["new_password"]
        confirm_password = request.form["confirm_password"]

        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return render_template("reset_password.html")

        if len(new_password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("reset_password.html")

        hashed_password = generate_password_hash(new_password)

        conn = get_db_connection()
        conn.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_password, session["reset_email"]))
        conn.commit()
        conn.close()

        session.pop("reset_email", None)
        session.pop("reset_otp", None)
        session.pop("otp_expiry", None)
        session.pop("otp_verified", None)

        flash("Password changed successfully. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
def home():
    if "user_email" not in session:
        return redirect(url_for("login"))

    prediction_text = None
    probability = None
    result_class = None
    suggestions = []
    patient_data = {}
    confidence_chart = None
    risk_level = None
    risk_message = None
    validation_error = None
    doctor_notes = ""
    patient_id = None

    total_records = len(df)
    positive_cases = int(df["target"].sum())
    negative_cases = total_records - positive_cases

    if request.method == "POST":
        try:
            age = float(request.form["age"])
            sex = int(float(request.form["sex"]))
            cp = int(float(request.form["cp"]))
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = int(float(request.form["fbs"]))
            restecg = int(float(request.form["restecg"]))
            thalach = float(request.form["thalach"])
            exang = int(float(request.form["exang"]))
            oldpeak = float(request.form["oldpeak"])
            slope = int(float(request.form["slope"]))
            ca = int(float(request.form["ca"]))
            thal = int(float(request.form["thal"]))
            doctor_notes = request.form.get("doctor_notes", "").strip()

            validation_error = validate_inputs(age, trestbps, chol, thalach, oldpeak)

            if validation_error:
                return render_template(
                    "index.html",
                    prediction_text=prediction_text,
                    probability=probability,
                    result_class=result_class,
                    suggestions=suggestions,
                    patient_data=patient_data,
                    confidence_chart=confidence_chart,
                    risk_level=risk_level,
                    risk_message=risk_message,
                    validation_error=validation_error,
                    doctor_notes=doctor_notes,
                    patient_id=patient_id,
                    model_name=metadata["best_model_name"],
                    accuracy=metadata["best_model_accuracy"],
                    total_records=total_records,
                    positive_cases=positive_cases,
                    negative_cases=negative_cases,
                    model_accuracies=metadata["all_model_accuracies"],
                    feature_importance=metadata["feature_importance"],
                    all_model_metrics=metadata.get("all_model_metrics", {}),
                    confusion_matrix=metadata.get("confusion_matrix", [[0, 0], [0, 0]]),
                    user_name=session.get("user_name"),
                )

            patient_id = generate_patient_id()

            patient_data = {
                "Patient ID": patient_id,
                "Age": age,
                "Sex": SEX_DISPLAY.get(sex, str(sex)),
                "Chest Pain Type": CP_DISPLAY.get(cp, str(cp)),
                "Resting Blood Pressure": trestbps,
                "Cholesterol": chol,
                "Fasting Blood Sugar": "High" if fbs == 1 else "Normal",
                "Resting ECG": RESTECG_DISPLAY.get(restecg, str(restecg)),
                "Maximum Heart Rate": thalach,
                "Exercise Induced Angina": "Yes" if exang == 1 else "No",
                "Oldpeak": oldpeak,
                "Slope": SLOPE_DISPLAY.get(slope, str(slope)),
                "Major Vessels": ca,
                "Thal": THAL_DISPLAY.get(thal, str(thal)),
            }

            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            features_scaled = scaler.transform(features)

            prediction = model.predict(features_scaled)[0]
            probability_vector = model.predict_proba(features_scaled)[0]

            disease_probability = round(get_probability_by_class(probability_vector, DISEASE_CLASS_LABEL) * 100, 2)
            no_disease_probability = round(100 - disease_probability, 2)

            probability = disease_probability
            confidence_chart = [disease_probability, no_disease_probability]
            risk_level = get_risk_level(disease_probability)
            risk_message = get_risk_message(risk_level)

            if prediction == DISEASE_CLASS_LABEL:
                prediction_text = "Heart Disease Likely Detected"
                result_class = "danger"
                suggestions = [
                    "Consult a cardiologist for detailed clinical evaluation.",
                    "Perform ECG, echo, and blood-related screening tests.",
                    "Reduce excess salt, sugar, and saturated fat intake.",
                    "Adopt medically supervised exercise.",
                    "Schedule periodic health monitoring.",
                ]
            else:
                prediction_text = "Heart Disease Not Likely Detected"
                result_class = "safe"
                suggestions = [
                    "Maintain a balanced diet and regular physical activity.",
                    "Continue preventive health checkups.",
                    "Track blood pressure and cholesterol periodically.",
                    "Avoid smoking and manage stress effectively.",
                    "Follow healthy sleep and hydration habits.",
                ]

            save_prediction(
                {
                    "patient_id": patient_id,
                    "user_email": session["user_email"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "age": age,
                    "sex": SEX_DISPLAY.get(sex, str(sex)),
                    "cp": CP_DISPLAY.get(cp, str(cp)),
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": "High" if fbs == 1 else "Normal",
                    "restecg": RESTECG_DISPLAY.get(restecg, str(restecg)),
                    "thalach": thalach,
                    "exang": "Yes" if exang == 1 else "No",
                    "oldpeak": oldpeak,
                    "slope": SLOPE_DISPLAY.get(slope, str(slope)),
                    "ca": ca,
                    "thal": THAL_DISPLAY.get(thal, str(thal)),
                    "prediction": prediction_text,
                    "probability": probability,
                    "risk_level": risk_level,
                    "risk_message": risk_message,
                    "doctor_notes": doctor_notes,
                }
            )

            session["latest_report"] = {
                "patient_id": patient_id,
                "patient_data": patient_data,
                "prediction_text": prediction_text,
                "probability": probability,
                "risk_level": risk_level,
                "risk_message": risk_message,
                "model_name": metadata["best_model_name"],
                "suggestions": suggestions,
                "doctor_notes": doctor_notes,
            }

        except Exception as e:
            validation_error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction_text=prediction_text,
        probability=probability,
        result_class=result_class,
        suggestions=suggestions,
        patient_data=patient_data,
        confidence_chart=confidence_chart,
        risk_level=risk_level,
        risk_message=risk_message,
        validation_error=validation_error,
        doctor_notes=doctor_notes,
        patient_id=patient_id,
        model_name=metadata["best_model_name"],
        accuracy=metadata["best_model_accuracy"],
        total_records=total_records,
        positive_cases=positive_cases,
        negative_cases=negative_cases,
        model_accuracies=metadata["all_model_accuracies"],
        feature_importance=metadata["feature_importance"],
        all_model_metrics=metadata.get("all_model_metrics", {}),
        confusion_matrix=metadata.get("confusion_matrix", [[0, 0], [0, 0]]),
        user_name=session.get("user_name"),
    )


@app.route("/report")
def report():
    if "latest_report" not in session:
        flash("No report available. Please generate a prediction first.", "warning")
        return redirect(url_for("home"))
    return render_template("report.html", report=session["latest_report"])


@app.route("/dashboard")
def dashboard():
    if "user_email" not in session:
        return redirect(url_for("login"))

    risk_filter = request.args.get("risk_level", "").strip()
    email_filter = request.args.get("user_email", "").strip()
    date_filter = request.args.get("date", "").strip()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if risk_filter:
        query += " AND risk_level = ?"
        params.append(risk_filter)

    if email_filter:
        query += " AND user_email LIKE ?"
        params.append(f"%{email_filter}%")

    if date_filter:
        query += " AND DATE(timestamp) = ?"
        params.append(date_filter)

    query += " ORDER BY id DESC"

    conn = get_db_connection()
    history = pd.read_sql_query(query, conn, params=params)
    all_history = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()

    total_predictions = len(all_history)
    high_risk = len(all_history[all_history["risk_level"] == "High Risk"])
    moderate_risk = len(all_history[all_history["risk_level"] == "Moderate Risk"])
    low_risk = len(all_history[all_history["risk_level"] == "Low Risk"])

    records = history.to_dict(orient="records")

    chart_dates = []
    chart_counts = []
    if not all_history.empty and "timestamp" in all_history.columns:
        history_dates = all_history["timestamp"].astype(str).str[:10].value_counts().sort_index()
        chart_dates = list(history_dates.index)
        chart_counts = list(history_dates.values)

    return render_template(
        "dashboard.html",
        records=records,
        total_predictions=total_predictions,
        high_risk=high_risk,
        moderate_risk=moderate_risk,
        low_risk=low_risk,
        chart_dates=chart_dates,
        chart_counts=chart_counts,
        selected_risk=risk_filter,
        selected_email=email_filter,
        selected_date=date_filter,
    )


@app.route("/download-pdf")
def download_pdf():
    if "latest_report" not in session:
        return redirect(url_for("home"))

    report = session["latest_report"]

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _, height = A4

    y = height - 50
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "Heart Disease Risk Prediction Report")
    y -= 25
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, "Ensemble Learning Based Clinical Assessment")
    y -= 30

    pdf.drawString(50, y, f"Patient ID: {report['patient_id']}")
    y -= 20
    pdf.drawString(50, y, f"Model: {report['model_name']}")
    y -= 20
    pdf.drawString(50, y, f"Prediction: {report['prediction_text']}")
    y -= 20
    pdf.drawString(50, y, f"Confidence: {report['probability']}%")
    y -= 20
    pdf.drawString(50, y, f"Risk Level: {report['risk_level']}")
    y -= 20
    pdf.drawString(50, y, f"Interpretation: {report['risk_message']}")
    y -= 30

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Patient Summary")
    y -= 20
    pdf.setFont("Helvetica", 10)

    for key, value in report["patient_data"].items():
        pdf.drawString(60, y, f"{key}: {value}")
        y -= 16
        if y < 100:
            pdf.showPage()
            y = height - 50

    y -= 10
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Recommendations")
    y -= 20
    pdf.setFont("Helvetica", 10)

    for item in report["suggestions"]:
        pdf.drawString(60, y, f"- {item}")
        y -= 16
        if y < 100:
            pdf.showPage()
            y = height - 50

    if report.get("doctor_notes"):
        y -= 15
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Doctor Notes")
        y -= 20
        pdf.setFont("Helvetica", 10)
        pdf.drawString(60, y, str(report["doctor_notes"]))
        y -= 16

    y -= 20
    pdf.drawString(50, y, "Note: This report is for academic and educational use only.")
    pdf.save()

    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name="heart_disease_risk_prediction_report.pdf",
        mimetype="application/pdf",
    )


@app.route("/export-csv")
def export_csv():
    if "user_email" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    history = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
    conn.close()

    output = io.StringIO()
    history.to_csv(output, index=False)
    output.seek(0)

    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_history.csv"},
    )


if __name__ == "__main__":
    print("Using database:", DB_PATH)
    init_db()
    app.run(debug=True)
