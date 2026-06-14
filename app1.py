from flask import (
    Flask, render_template, request, Response, redirect,
    url_for, session, flash
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mysql.connector
import numpy as np
import cv2, os, time, torch, logging
from PIL import Image

# ── Flask & paths ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'wiredefect'

app.config.update(
    UPLOAD_FOLDER='static/uploads',
    RESULT_FOLDER='static/results',
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# ── Logging ─────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Database details ────────────────────────────────────────────
db_config = dict(
    host=os.environ.get('DB_HOST', 'localhost'),
    user=os.environ.get('DB_USER', 'root'),
    password=os.environ.get('DB_PASSWORD', 'root'),
    database=os.environ.get('DB_NAME', 'wire_rope_detection_db'),
    port=int(os.environ.get('DB_PORT', 3306))
)

# ── CNN / YOLO models ───────────────────────────────────────────
# ── CNN / YOLO models ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cnn_model_path = os.environ.get(
    'CNN_MODEL_PATH',
    os.path.join(BASE_DIR, 'model.h5')
)

yolo_model_path = os.environ.get(
    'YOLO_MODEL_PATH',
    os.path.join(BASE_DIR, 'best.pt')
)

try:
    cnn_model = load_model(cnn_model_path)
    logger.info("CNN model loaded")
except Exception as e:
    logger.warning(f"CNN model not loaded: {e}")
    cnn_model = None

try:
    yolo_model = YOLO(yolo_model_path)
    logger.info("YOLO model loaded")
except Exception as e:
    logger.warning(f"YOLO model not loaded: {e}")
    yolo_model = None

class_names = ['break', 'thunderbolt', 'unknown']

defect_severity = {
    'break'      : {'level': 'Critical', 'color': '#FF0000'},
    'thunderbolt': {'level': 'High',     'color': '#FF8C00'},
    'unknown'    : {'level': 'Medium',   'color': '#FFD700'}
}

safety_rec = {
    'break'      : 'IMMEDIATE ACTION REQUIRED: Replace wire rope.',
    'thunderbolt': 'Inspect thoroughly; consider replacement.',
    'unknown'    : 'Unknown defect; perform professional inspection.'
}

risk_level = {
    'break'      : {'risk': 'Extreme',  'action': 'Stop operations now'},
    'thunderbolt': {'risk': 'High',     'action': 'Detailed inspection'},
    'unknown'    : {'risk': 'Moderate', 'action': 'Monitor condition'}
}

# ── Helper functions ────────────────────────────────────────────
def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_db():
    return mysql.connector.connect(**db_config)

def create_schema():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(80) UNIQUE,
            email VARCHAR(120) UNIQUE,
            password VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cnn_inspections(
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            filename VARCHAR(255),
            defect_type VARCHAR(50),
            confidence FLOAT,
            severity_level VARCHAR(20),
            recommendation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit(); cur.close(); conn.close()

def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, 0)

# ── App start-up ────────────────────────────────────────────────
@app.before_first_request
def init():
    create_schema()

# ── Routes ──────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', logged_in=('username' in session))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        u, e, p = (request.form[k] for k in ('username', 'email', 'password'))
        conn, cur = None, None
        try:
            conn = get_db(); cur = conn.cursor()
            cur.execute(
                "INSERT INTO users(username,email,password) VALUES(%s,%s,%s)",
                (u, e, generate_password_hash(p))
            )
            conn.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            flash('Username or email already exists.', 'danger')
            logger.error(err)
        finally:
            if cur: cur.close()
            if conn: conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        u, p = request.form['username'], request.form['password']
        conn, cur = None, None
        try:
            conn = get_db(); cur = conn.cursor(dictionary=True)
            cur.execute("SELECT * FROM users WHERE username=%s", (u,))
            user = cur.fetchone()
            if user and check_password_hash(user['password'], p):
                session.update(username=u, user_id=user['id'])
                flash('Logged in!', 'success')
                return redirect(url_for('dashboard'))
            flash('Invalid credentials', 'danger')
        finally:
            if cur:  cur.close()
            if conn: conn.close()
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html')

# ── CNN detection ───────────────────────────────────────────────
@app.route('/cnn_detection', methods=['GET', 'POST'])
def cnn_detection():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename and allowed_file(file.filename):
            ts = int(time.time())
            fname = f'cnn_{ts}_{secure_filename(file.filename)}'
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            file.save(path)

            img = Image.open(path)
            if cnn_model is None:
                return "CNN model not found"
if cnn_model is None:
    return "CNN model not found"
preds = cnn_model.predict(preprocess(img))[0]
            idx = int(np.argmax(preds))
            defect = class_names[idx]
            conf = float(preds[idx])

            # Save in database
            conn, cur = None, None
            try:
                conn = get_db(); cur = conn.cursor()
                cur.execute("""
                    INSERT INTO cnn_inspections
                    (user_id,filename,defect_type,confidence,severity_level,recommendation)
                    VALUES(%s,%s,%s,%s,%s,%s)
                """, (
                    session['user_id'], fname, defect, conf,
                    defect_severity[defect]['level'],
                    safety_rec[defect]
                ))
                conn.commit()
            finally:
                if cur: cur.close()
                if conn: conn.close()

            # FIXED: Use 'risk_assessment' instead of 'risk'
            result = {
                'predicted_class': defect,
                'confidence': conf,
                'confidence_percent': round(conf * 100, 2),  # Use consistent naming
                'severity': defect_severity[defect],
                'recommendation': safety_rec[defect],
                'risk_assessment': risk_level[defect],  # ← FIXED: Changed from 'risk' to 'risk_assessment'
                'image_path': f'uploads/{fname}'
            }
            return render_template('cnn_detection.html', result=result)

        flash('Invalid file', 'danger')
    return render_template('cnn_detection.html')

# ── YOLO image detection ────────────────────────────────────────
@app.route('/yolo_detection', methods=['GET', 'POST'])
def yolo_detection():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        if file.filename and allowed_file(file.filename):
            ts = int(time.time())
            fname = f'yolo_{ts}_{secure_filename(file.filename)}'
            upath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            file.save(upath)

            frame = cv2.imread(upath)
            if yolo_model is None:
    continue

if yolo_model is None:
    continue

results = yolo_model.predict(source=frame, conf=0.3)[0]

            detections = []
            if results.boxes is not None:
                for b in results.boxes:
                    # Convert tensor to numpy properly
                    bbox_coords = b.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox_coords.tolist()
                    
                    cls = int(b.cls[0].cpu().numpy())
                    label = yolo_model.names[cls]
                    conf = float(b.conf[0].cpu().numpy())
                    
                    detections.append({
                        'label': label, 
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    # Draw professional YOLO-style bounding box
                    draw_professional_bbox(frame, x1, y1, x2, y2, label, conf)

            r_fname = f'result_{fname}'
            r_path = os.path.join(app.config['RESULT_FOLDER'], r_fname)
            cv2.imwrite(r_path, frame)

            result = {
                'detections': detections,
                'total_detections': len(detections),
                'original_image': f'uploads/{fname}',
                'result_image': f'results/{r_fname}'
            }
            return render_template('yolo_detection.html', result=result)

        flash('Invalid file', 'danger')
    return render_template('yolo_detection.html')

def draw_professional_bbox(image, x1, y1, x2, y2, label, confidence):
    """
    Draw professional YOLO-style bounding box with filled rectangular label background
    """
    
    # Color scheme for different defects
    color_map = {
        'break': (0, 0, 255),        # Red for critical defects
        'thunderbolt': (0, 140, 255), # Orange for high priority
        'unknown': (0, 255, 255)      # Yellow for unknown
    }
    
    # Get color for this label
    box_color = color_map.get(label, (0, 255, 0))  # Default green
    
    # Calculate adaptive thickness based on image size
    img_height, img_width = image.shape[:2]
    thickness = max(2, int((img_width + img_height) / 1000))
    
    # Draw main bounding box with rounded corners effect
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)
    
    # Add corner accents for modern YOLO look
    corner_length = min(25, (x2-x1)//6, (y2-y1)//6)
    accent_thickness = thickness + 1
    
    # Top-left corner accent
    cv2.line(image, (x1, y1), (x1 + corner_length, y1), box_color, accent_thickness)
    cv2.line(image, (x1, y1), (x1, y1 + corner_length), box_color, accent_thickness)
    
    # Top-right corner accent
    cv2.line(image, (x2, y1), (x2 - corner_length, y1), box_color, accent_thickness)
    cv2.line(image, (x2, y1), (x2, y1 + corner_length), box_color, accent_thickness)
    
    # Bottom-left corner accent
    cv2.line(image, (x1, y2), (x1 + corner_length, y2), box_color, accent_thickness)
    cv2.line(image, (x1, y2), (x1, y2 - corner_length), box_color, accent_thickness)
    
    # Bottom-right corner accent
    cv2.line(image, (x2, y2), (x2 - corner_length, y2), box_color, accent_thickness)
    cv2.line(image, (x2, y2), (x2, y2 - corner_length), box_color, accent_thickness)
    
    # Prepare label text with confidence
    confidence_percent = int(confidence * 100)
    label_text = f"{label.upper()} {confidence_percent}%"
    
    # Font settings - adaptive to image size
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = (img_width + img_height) / 2000
    font_scale = max(0.5, min(1.2, base_font_scale))
    font_thickness = max(1, int(font_scale * 2))
    
    # Get text dimensions
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, font_thickness
    )
    
    # Calculate label rectangle dimensions with padding
    padding_x, padding_y = 12, 8
    label_width = text_width + (padding_x * 2)
    label_height = text_height + (padding_y * 2)
    
    # Position label rectangle (above the bounding box)
    label_x1 = x1
    label_y1 = max(y1 - label_height - 5, 0)  # 5px gap from bbox
    label_x2 = min(x1 + label_width, img_width)
    label_y2 = y1
    
    # If label goes outside image bounds, position it inside the bbox
    if label_y1 < 0:
        label_y1 = y1 + 5
        label_y2 = y1 + label_height + 5
    
    # Create filled rectangle background for label
    overlay = image.copy()
    cv2.rectangle(overlay, (label_x1, label_y1), (label_x2, label_y2), box_color, -1)
    
    # Add transparency to the label background
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw label rectangle border
    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), box_color, 2)
    
    # Add subtle gradient effect by drawing a slightly lighter rectangle at top
    gradient_height = max(2, label_height // 4)
    lighter_color = tuple(min(255, c + 40) for c in box_color)
    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y1 + gradient_height), 
                  lighter_color, -1)
    
    # Calculate text position (centered in rectangle)
    text_x = label_x1 + padding_x
    text_y = label_y1 + padding_y + text_height
    
    # Draw text shadow for better readability
    shadow_offset = max(1, font_thickness // 2)
    cv2.putText(image, label_text, 
                (text_x + shadow_offset, text_y + shadow_offset), 
                font, font_scale, (0, 0, 0), font_thickness + 1)
    
    # Draw main text in white
    cv2.putText(image, label_text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), font_thickness)
    
    # Add confidence indicator bar at bottom of bounding box
    confidence_bar_height = max(3, thickness)
    confidence_bar_width = int((x2 - x1) * confidence)
    
    # Color code the confidence bar
    if confidence > 0.8:
        conf_color = (0, 255, 0)      # Green for high confidence
    elif confidence > 0.6:
        conf_color = (0, 255, 255)    # Yellow for medium confidence
    else:
        conf_color = (0, 0, 255)      # Red for low confidence
    
    # Draw confidence bar
    cv2.rectangle(image, (x1, y2 - confidence_bar_height), 
                  (x1 + confidence_bar_width, y2), conf_color, -1)
    
    # Add a small detection count indicator (optional)
    circle_radius = max(3, thickness)
    cv2.circle(image, (x2 - 15, y1 + 15), circle_radius, (255, 255, 255), -1)
    cv2.circle(image, (x2 - 15, y1 + 15), circle_radius, box_color, 2)

# ── Live YOLO feed ──────────────────────────────────────────────
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error('No camera found')
        return
        
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        with torch.no_grad():
            if yolo_model is None:
    continue

if yolo_model is None:
    continue

results = yolo_model.predict(source=frame, conf=0.3)[0]
            
            if results.boxes is not None:
                for b in results.boxes:
                    bbox_coords = b.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox_coords.tolist()
                    
                    cls = int(b.cls[0].cpu().numpy())
                    label = yolo_model.names[cls]
                    conf = float(b.conf[0].cpu().numpy())
                    
                    # Use the same professional drawing function
                    draw_professional_bbox(frame, x1, y1, x2, y2, label, conf)
        
        ret, buf = cv2.imencode('.jpg', frame)
        if not ret: break
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/yolo_live')
def yolo_live():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('yolo_live.html')

# ── History (CNN only) ──────────────────────────────────────────
@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))
    conn, cur = None, None
    try:
        conn = get_db(); cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT * FROM cnn_inspections
            WHERE user_id=%s ORDER BY created_at DESC
        """, (session['user_id'],))
        records = cur.fetchall()
    finally:
        if cur: cur.close()
        if conn: conn.close()
    return render_template('history.html', history=records)

# ── Run app ─────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
