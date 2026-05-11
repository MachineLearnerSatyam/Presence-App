import streamlit as st

# ==========================================
# 1. IN-MEMORY CRASH PATCHER (Safe & Clean)
# Swallows all Streamlit-WebRTC and Aioice teardown crashes
# ==========================================
import logging

logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

try:
    import streamlit_webrtc.shutdown

    _orig_stop = streamlit_webrtc.shutdown.SessionShutdownObserver.stop

    def _safe_stop(self):
        try:
            if getattr(self, "_polling_thread", None) is not None:
                _orig_stop(self)
        except Exception:
            pass

    streamlit_webrtc.shutdown.SessionShutdownObserver.stop = _safe_stop
except Exception:
    pass

try:
    import aioice.stun

    _orig_retry = aioice.stun.Transaction._Transaction__retry

    def _safe_retry(self, *args, **kwargs):
        try:
            _orig_retry(self, *args, **kwargs)
        except Exception:
            pass

    aioice.stun.Transaction._Transaction__retry = _safe_retry
except Exception:
    pass

try:
    import aioice.ice

    _orig_send_stun = aioice.ice.Connection.send_stun

    def _safe_send_stun(self, message, addr):
        try:
            _orig_send_stun(self, message, addr)
        except Exception:
            pass

    aioice.ice.Connection.send_stun = _safe_send_stun
except Exception:
    pass


# ==========================================
# STANDARD IMPORTS & SETUP
# ==========================================
import cv2
import face_recognition
import sqlite3
import numpy as np
from PIL import Image
import pickle
import io
import qrcode
from datetime import datetime, timedelta
import pandas as pd
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

st.set_page_config(page_title="Hybrid Attendance System", page_icon="🏫", layout="wide")

# ==========================================
# 2. FIREWALL BYPASS (Free Public TURN Server)
# This fixes the "Connection taking longer than expected" error
# ==========================================
RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [
            # Keep the Google STUN as a backup
            {"urls": ["stun:stun.l.google.com:19302"]},
            # Your Active Metered TURN server
            {
                "urls": [
                    "turn:global.relay.metered.ca:80",
                    "turn:global.relay.metered.ca:443",
                    "turn:global.relay.metered.ca:80?transport=tcp",
                    "turns:global.relay.metered.ca:443?transport=tcp",
                ],
                "username": "68501a35dac303646919b7e0",
                "credential": "VDKnDhoB8+MnOUNp",
            },
        ]
    }
)

# ==========================================
# DATABASE CONFIGURATION & GLOBAL HELPERS
# ==========================================
DB_PATH = "attendance.db"


def init_db():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS students (student_id TEXT PRIMARY KEY, name TEXT, face_encoding BLOB)"""
        )
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, name TEXT, date TEXT, time_in TEXT, time_out TEXT)"""
        )
        conn.commit()
        conn.close()
    except Exception as e:
        st.sidebar.error(f"Database Error: {e}")


try:
    init_db()
except Exception:
    pass


def load_student_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT student_id, name, face_encoding FROM students")
        rows = cursor.fetchall()
    except sqlite3.Error:
        rows = []

    known_ids, known_names, known_encodings = [], [], []
    for row in rows:
        known_ids.append(row[0])
        known_names.append(row[1])
        known_encodings.append(pickle.loads(row[2]))
    conn.close()
    return known_ids, known_names, known_encodings


def calculate_ear(eye):
    a = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    b = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    c = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (a + b) / (2.0 * c)


def mark_attendance(std_id, name):
    # check_same_thread=False allows background video threads to access SQLite safely
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    now = datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    current_time_str = now.strftime("%H:%M:%S")

    cursor.execute(
        "SELECT id, time_in, time_out FROM attendance WHERE student_id = ? AND date = ?",
        (std_id, today_date),
    )
    record = cursor.fetchone()

    status_message = ""
    if record is None:
        cursor.execute(
            "INSERT INTO attendance (student_id, name, date, time_in) VALUES (?, ?, ?, ?)",
            (std_id, name, today_date, current_time_str),
        )
        conn.commit()
        status_message = f"IN: {current_time_str}"
    else:
        att_id, time_in, time_out = record
        if time_out is None:
            if isinstance(time_in, str):
                last_scan_time = datetime.combine(
                    now.date(), datetime.strptime(time_in, "%H:%M:%S").time()
                )
            else:
                last_scan_time = datetime.combine(now.date(), time_in)

            if now - last_scan_time > timedelta(minutes=1):
                cursor.execute(
                    "UPDATE attendance SET time_out = ? WHERE id = ?",
                    (current_time_str, att_id),
                )
                conn.commit()
                status_message = f"OUT: {current_time_str}"
            else:
                remaining = 60 - (now - last_scan_time).seconds
                status_message = f"Wait {remaining}s"
        else:
            status_message = "Done for today!"

    conn.close()
    return status_message


# ==========================================
# MAIN APP SETUP & NAVIGATION
# ==========================================
st.sidebar.title("🏫 Attendance System")
menu = ["📸 Register Student", "🟢 Automatic Cloud Scanner", "📊 Admin Dashboard"]
choice = st.sidebar.radio("Navigation", menu)

# ==========================================
# MODULE 1: REGISTER STUDENT
# ==========================================
if choice == "📸 Register Student":
    st.title("📸 Advanced Student Registration")
    st.write(
        "Register a new student by entering their details and capturing their face."
    )

    col1, col2 = st.columns(2)
    with col1:
        student_id = st.text_input("Enter Student ID (e.g., CS101)")
    with col2:
        student_name = st.text_input("Enter Full Name")

    st.write("### Capture Face")
    img_file_buffer = st.camera_input("Look straight into the camera")

    if st.button("Save & Generate QR Code"):
        if not student_id or not student_name:
            st.warning("⚠️ Please enter both Student ID and Name.")
        elif img_file_buffer is None:
            st.warning("⚠️ Please take a picture first.")
        else:
            with st.spinner("Processing face data..."):
                try:
                    image = Image.open(img_file_buffer)
                    img_array = np.array(image)
                    face_encodings = face_recognition.face_encodings(img_array)

                    if len(face_encodings) == 0:
                        st.error(
                            "🚨 No face detected! Please ensure you are well-lit and looking at the camera."
                        )
                    else:
                        encoding = face_encodings[0]
                        encoding_bytes = pickle.dumps(encoding)

                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT OR REPLACE INTO students (student_id, name, face_encoding) VALUES (?, ?, ?)",
                            (student_id, student_name, encoding_bytes),
                        )
                        conn.commit()
                        conn.close()

                        qr = qrcode.QRCode(box_size=10, border=4)
                        qr.add_data(student_id)
                        qr.make(fit=True)
                        qr_img = qr.make_image(fill_color="black", back_color="white")

                        buf = io.BytesIO()
                        qr_img.save(buf, format="PNG")
                        byte_im = buf.getvalue()

                        st.success(f"✅ Successfully registered {student_name}!")
                        st.image(byte_im, caption=f"QR for {student_id}", width=200)
                        st.download_button(
                            "Download QR Code",
                            data=byte_im,
                            file_name=f"{student_id}_QR.png",
                            mime="image/png",
                        )

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ==========================================
# MODULE 2: CLOUD LIVE SCANNER (Automatic & Thread-Safe)
# ==========================================
elif choice == "🟢 Automatic Cloud Scanner":
    st.title("🟢 Automatic Live Cloud Scanner")
    st.write("Automatically scans QR codes or faces with blink detection.")

    known_ids, known_names, known_encodings = load_student_data()

    if len(known_encodings) == 0:
        st.warning(
            "⚠️ No students found in the database. Please register students first."
        )
    else:
        # Process frames continuously in the background
        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 1. QR Code Logic
            detector = cv2.QRCodeDetector()
            qr_data, _, _ = detector.detectAndDecode(img)
            attendance_marked = False

            if qr_data and qr_data in known_ids:
                index = known_ids.index(qr_data)
                name = known_names[index]
                status = mark_attendance(qr_data, name)
                cv2.rectangle(
                    img,
                    (50, 50),
                    (img.shape[1] - 50, img.shape[0] - 50),
                    (255, 255, 0),
                    4,
                )
                cv2.putText(
                    img,
                    f"QR: {name}",
                    (60, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    status,
                    (60, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                )
                attendance_marked = True

            # 2. Face + EAR Blink Detection Logic
            if not attendance_marked:
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)
                face_locations = face_recognition.face_locations(small_frame)

                if face_locations:
                    face_encodings_in_frame = face_recognition.face_encodings(
                        small_frame, face_locations
                    )
                    face_landmarks_list = face_recognition.face_landmarks(
                        small_frame, face_locations
                    )

                    for (
                        (top, right, bottom, left),
                        face_encoding,
                        face_landmarks,
                    ) in zip(
                        face_locations, face_encodings_in_frame, face_landmarks_list
                    ):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        matches = face_recognition.compare_faces(
                            known_encodings, face_encoding, tolerance=0.5
                        )

                        if True in matches:
                            match_index = matches.index(True)
                            name = known_names[match_index]
                            student_id = known_ids[match_index]

                            # Calculate Eye Aspect Ratio (EAR) for blinking
                            avg_ear = (
                                calculate_ear(face_landmarks["left_eye"])
                                + calculate_ear(face_landmarks["right_eye"])
                            ) / 2.0

                            if avg_ear < 0.22:
                                status = mark_attendance(student_id, name)
                                cv2.rectangle(
                                    img, (left, top), (right, bottom), (0, 255, 0), 2
                                )
                                cv2.putText(
                                    img,
                                    f"{name}: {status}",
                                    (left, bottom + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                )
                            else:
                                cv2.rectangle(
                                    img, (left, top), (right, bottom), (0, 165, 255), 2
                                )
                                cv2.putText(
                                    img,
                                    f"{name}: BLINK TO VERIFY",
                                    (left, bottom + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 165, 255),
                                    2,
                                )
                        else:
                            cv2.rectangle(
                                img, (left, top), (right, bottom), (0, 0, 255), 2
                            )
                            cv2.putText(
                                img,
                                "Unknown",
                                (left, bottom + 25),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="live-scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,  # <--- Uses the Public TURN server to bypass firewalls
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

# ==========================================
# MODULE 3: ADMIN DASHBOARD
# ==========================================
elif choice == "📊 Admin Dashboard":
    st.title("📊 Attendance Admin Dashboard")
    st.write("View, filter, and export daily student attendance records.")

    def fetch_attendance(selected_date=None):
        try:
            conn = sqlite3.connect(DB_PATH)
            if selected_date:
                query = "SELECT student_id, name, date, time_in, time_out FROM attendance WHERE date = ? ORDER BY time_in DESC"
                df = pd.read_sql(query, conn, params=(selected_date,))
            else:
                query = "SELECT student_id, name, date, time_in, time_out FROM attendance ORDER BY date DESC, time_in DESC"
                df = pd.read_sql(query, conn)
            conn.close()

            if not df.empty:
                df["time_in"] = df["time_in"].apply(
                    lambda val: "" if pd.isnull(val) else str(val).split()[-1]
                )
                df["time_out"] = df["time_out"].apply(
                    lambda val: "" if pd.isnull(val) else str(val).split()[-1]
                )
            return df
        except Exception as err:
            st.error(f"Database Error: {err}")
            return pd.DataFrame()

    col1, col2 = st.columns([1, 3])
    with col1:
        st.write("### Filter Records")
        filter_by_date = st.checkbox("Filter by Date")
        if filter_by_date:
            selected_date = st.date_input("Select Date", datetime.today())
            date_str = selected_date.strftime("%Y-%m-%d")
            df = fetch_attendance(date_str)
        else:
            date_str = "All Time"
            df = fetch_attendance()

    with col2:
        st.write(f"### Records: {date_str}")
        if df.empty:
            st.info("No attendance records found for this selection.")
        else:
            st.dataframe(df, width="stretch", hide_index=True)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Data as CSV",
                data=csv,
                file_name=f"Attendance_{date_str}.csv",
                mime="text/csv",
            )
