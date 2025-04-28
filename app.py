import cv2
import json
import requests
import os
import threading
import time
import uuid
from flask import (
    Flask, render_template, Response, jsonify, request, send_file,
    redirect, url_for, flash, send_from_directory, abort
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
from sqlalchemy.exc import IntegrityError

# --- Configuration ---
SPOONACULAR_API_KEY = os.environ.get("SPOONACULAR_API_KEY", "d375c8b90cf440bea6d6ff3bcf04cccd")
MODEL_PATH = "best.pt"
SCAN_DURATION_SECONDS = 15
YOLO_CONFIDENCE = 0.4
# --- DEBUG FLAG ---
# Set to False to enable YOLO detection, True to bypass it for debugging camera feed
BYPASS_YOLO_FOR_DEBUG = False # <<< CHANGE THIS TO True TO TEST BYPASSING YOLO

# --- Flask App Initialization & Configuration ---
app = Flask(__name__, instance_relative_config=True)
try: os.makedirs(app.instance_path)
except OSError: pass
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-replace-in-prod')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{os.path.join(app.instance_path, "app.db")}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['BASE_UPLOAD_DIR'] = 'user_uploads'
app.config['UPLOAD_FOLDER_NAME'] = 'snapshots'
app.config['UPLOAD_FOLDER'] = os.path.join(app.instance_path, app.config['BASE_UPLOAD_DIR'], app.config['UPLOAD_FOLDER_NAME'])
try: os.makedirs(app.config['UPLOAD_FOLDER'])
except OSError: pass

# --- Database and Login Manager Initialization ---
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# --- Models ---
# (Models User, SavedRecipe, ScanSnapshot remain the same)
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    saved_recipes = db.relationship('SavedRecipe', backref='user', lazy=True, cascade="all, delete-orphan")
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)
    def __repr__(self): return f'<User {self.username}>'

class SavedRecipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    spoonacular_id = db.Column(db.Integer, nullable=False)
    title = db.Column(db.String(200), nullable=False)
    image_url = db.Column(db.String(500))
    recipe_url = db.Column(db.String(500))
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    snapshot = db.relationship('ScanSnapshot', backref='saved_recipe', uselist=False, lazy=True, cascade="all, delete-orphan")
    __table_args__ = (db.UniqueConstraint('user_id', 'spoonacular_id', name='_user_recipe_uc'),)
    def __repr__(self): return f'<SavedRecipe {self.title} (User: {self.user_id})>'

class ScanSnapshot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    saved_recipe_id = db.Column(db.Integer, db.ForeignKey('saved_recipe.id'), nullable=False)
    filename = db.Column(db.String(100), nullable=False, unique=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    def get_url(self):
        try: return url_for('serve_snapshot', filename=self.filename, _external=False)
        except RuntimeError: upload_folder_name = app.config.get('UPLOAD_FOLDER_NAME', 'snapshots'); base_upload_dir = app.config.get('BASE_UPLOAD_DIR', 'user_uploads'); return f"/{base_upload_dir}/{upload_folder_name}/{self.filename}"
    def __repr__(self): return f'<ScanSnapshot {self.filename} (Recipe: {self.saved_recipe_id})>'

# --- Context Processor ---
@app.context_processor
def inject_now():
    return {'now': datetime.utcnow}

# --- Login Manager Loader ---
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# --- Model Loading ---
MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", MODEL_PATH)

# Try to locate the model file in various possible locations
def find_model_file(model_path):
    # Try the direct path first
    if os.path.isfile(model_path):
        return model_path

    # Try relative to the app directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(app_dir, model_path),
        os.path.join(app_dir, 'models', model_path),
        os.path.join(app.instance_path, model_path)
    ]

    for path in possible_paths:
        if os.path.isfile(path):
            print(f"Found model at: {path}")
            return path

    return None

# Load model only if not bypassing
if not BYPASS_YOLO_FOR_DEBUG:
    try:
        model_file = find_model_file(MODEL_PATH)
        if model_file:
            model = YOLO(model_file)
            print(f"YOLO model loaded successfully from: {model_file}")
            # Load class names from the model
            class_names = model.names
        else:
            print(f"ERROR: YOLO model file not found at '{MODEL_PATH}' or common locations.")
            print("Either place the model file in the correct location, or set BYPASS_YOLO_FOR_DEBUG=True")
            model = None
            class_names = {}
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        model = None
        class_names = {}
else:
    print("!!! YOLO detection is BYPASSED for debugging !!!")
    model = None
    class_names = {}

# --- Global Variables & Locking ---
frame_lock = threading.Lock(); video_capture = None; latest_frame = None
detected_ingredients_global = set(); final_scan_frame_bytes = None; scan_frame_lock = threading.Lock()

# --- Helper Functions ---
def get_recipes(ingredients_list):
    # ... (same as before) ...
    if not ingredients_list: return []
    ingredients = ','.join(set(ingredients_list))
    url = "https://api.spoonacular.com/recipes/findByIngredients"
    params = { 'apiKey': SPOONACULAR_API_KEY, 'ingredients': ingredients, 'number': 5, 'ranking': 1 }
    try:
        response = requests.get(url, params=params); response.raise_for_status(); recipes = response.json()
        for recipe in recipes: recipe['recipeUrl'] = f"https://spoonacular.com/recipes/{'-'.join(recipe['title'].split()).lower()}-{recipe['id']}"
        return recipes
    except requests.exceptions.RequestException as e: print(f"Error fetching recipes: {e}"); return []
    except json.JSONDecodeError: print("Error decoding API response."); return []

def initialize_camera():
    # ... (same as before) ...
    global video_capture
    if video_capture and video_capture.isOpened(): return
    candidates = [0, 1]
    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        print(f"Trying camera index={idx} (default backend) → opened={cap.isOpened()}")
        if cap.isOpened():
            video_capture = cap; print(f"Webcam opened successfully (index={idx}, default)."); return
        cap.release()
    print("Error: Could not open webcam with default backend for tested indices.")

def release_camera():
    # ... (same as before) ...
    global video_capture
    with frame_lock:
        if video_capture: video_capture.release(); video_capture = None; print("Webcam released.")

def create_placeholder_image(width=640, height=480, text="Not Found"):
    # ... (same as before) ...
    try: from PIL import Image, ImageDraw
    except ImportError: print("Pillow not installed."); return None
    img = Image.new('RGB', (width, height), color = (73, 73, 73))
    try: draw = ImageDraw.Draw(img); draw.text((width//2 - 50, height//2), text, fill=(200, 200, 200))
    except Exception as e: print(f"Error drawing text: {e}")
    buffer = BytesIO(); img.save(buffer, format='JPEG'); return buffer.getvalue()

# --- process_frames (ADDED MORE LOGGING) ---
def process_frames():
    global latest_frame, detected_ingredients_global, video_capture
    initialize_camera()
    frame_count = 0
    last_log_time = time.time() # For throttling logs

    while True:
        frame_copy = None
        current_detected_in_frame = set()
        processed_frame_for_stream = None
        frame_read_success = False # Track if read succeeded

        with frame_lock:
            if video_capture is None or not video_capture.isOpened():
                if time.time() - last_log_time > 5: # Log camera error periodically
                    print("[process_frames] Waiting for camera to initialize...")
                    last_log_time = time.time()
                latest_frame = create_placeholder_image(text="Webcam Error")
                detected_ingredients_global = set()
                time.sleep(0.5)
                initialize_camera() # Try re-initializing
                continue # Skip rest of the loop

            # --- Attempt to read frame ---
            try:
                ret, frame = video_capture.read()
                frame_read_success = ret and frame is not None
                if frame_read_success:
                    frame_copy = frame.copy() # Copy only if read was successful
                else:
                    # Log failure only periodically to avoid spam
                    if time.time() - last_log_time > 2:
                         print(f"[process_frames] Error: Failed to capture frame (ret={ret}, frame is None={frame is None}).")
                         last_log_time = time.time()
            except Exception as read_err:
                 print(f"[process_frames] EXCEPTION during video_capture.read(): {read_err}")
                 frame_read_success = False
                 time.sleep(0.1) # Wait a bit after exception
            # --- End attempt to read frame ---

        # --- Process frame only if read was successful ---
        if frame_read_success and frame_copy is not None:
            processed_frame_for_stream = frame_copy # Start with the raw frame

            # --- Optional YOLO Detection ---
            if not BYPASS_YOLO_FOR_DEBUG and model:
                try:
                    results = model(frame_copy, conf=YOLO_CONFIDENCE, iou=0.5, verbose=False)
                    detection_count_this_frame = 0
                    for result in results:
                        processed_frame_for_stream = result.plot() # Draw boxes on the frame copy
                        boxes = result.boxes
                        detection_count_this_frame += len(boxes)
                        for box in boxes:
                            cls = int(box.cls[0])
                            label = class_names.get(cls, f"Unknown class {cls}")
                            current_detected_in_frame.add(label)
                    # Log detection results periodically
                    if frame_count % 30 == 0:
                        print(f"[Frame {frame_count}] Detections: {detection_count_this_frame}, Labels: {current_detected_in_frame or '{}'}")

                except Exception as yolo_err:
                    print(f"[Frame {frame_count}] ERROR during YOLO inference: {yolo_err}")
                    # Keep processed_frame_for_stream as the raw frame copy
                    cv2.putText(processed_frame_for_stream, "Detection Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    current_detected_in_frame = set()
            elif BYPASS_YOLO_FOR_DEBUG:
                 # If bypassing, just draw a timestamp or something simple
                 ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                 cv2.putText(processed_frame_for_stream, f"Bypassed {ts}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                 current_detected_in_frame = set() # No ingredients when bypassing
            # --- End Optional YOLO Detection ---

            # --- Encode the processed frame ---
            encode_success = False
            try:
                ret_encode, buffer = cv2.imencode('.jpg', processed_frame_for_stream)
                encode_success = ret_encode
                if encode_success:
                    # Update globals ONLY if encoding succeeded
                    with frame_lock:
                        latest_frame = buffer.tobytes()
                        detected_ingredients_global = current_detected_in_frame.copy()
                else:
                    if time.time() - last_log_time > 2: # Log periodically
                        print(f"[Frame {frame_count}] Error: Failed to encode frame.")
                        last_log_time = time.time()
                    # Don't update latest_frame if encode failed
                    # Keep last known ingredients
                    with frame_lock:
                        detected_ingredients_global = current_detected_in_frame.copy()
            except Exception as encode_err:
                 print(f"[Frame {frame_count}] EXCEPTION during cv2.imencode(): {encode_err}")
                 encode_success = False
                 # Keep last known ingredients
                 with frame_lock:
                     detected_ingredients_global = current_detected_in_frame.copy()
            # --- End Encode ---

        elif not frame_read_success:
             # If frame read failed, ensure latest_frame is placeholder or old frame
             with frame_lock:
                 if latest_frame is None: # Only set placeholder if no valid frame exists
                     latest_frame = create_placeholder_image(text="Frame Read Error")
                 detected_ingredients_global = set() # Clear ingredients on read error

        frame_count += 1
        time.sleep(0.03) # Small delay regardless of success/failure


def generate_video_feed():
    global latest_frame
    while True:
        frame_bytes = None
        with frame_lock:
            if latest_frame is not None: frame_bytes = latest_frame
        if frame_bytes:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            print("[generate_video_feed] No frame available, yielding placeholder.")
            placeholder = create_placeholder_image(text="Initializing Feed...")
            if placeholder: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + placeholder + b'\r\n')
            time.sleep(0.1)

# --- Flask Routes ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/video_feed')
def video_feed(): return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detected_ingredients')
def get_detected_ingredients():
    with frame_lock: ingredients = list(detected_ingredients_global)
    return jsonify(ingredients=ingredients)

@app.route('/scan_and_fetch')
def scan_and_fetch():
    global final_scan_frame_bytes; print(f"Starting {SCAN_DURATION_SECONDS}-sec scan...")
    start_time = time.time(); aggregated_ingredients = set(); best_frame_bytes_local = None; max_ingredients_in_frame = -1; scan_loop_count = 0
    while time.time() - start_time < SCAN_DURATION_SECONDS:
        current_ingredients_in_frame = set(); current_frame_bytes = None
        with frame_lock: current_ingredients_in_frame = detected_ingredients_global.copy(); current_frame_bytes = latest_frame
        if scan_loop_count % 5 == 0: print(f"[Scan Loop {scan_loop_count}] Reading: {current_ingredients_in_frame or '{}'}")
        aggregated_ingredients.update(current_ingredients_in_frame)
        if len(current_ingredients_in_frame) > max_ingredients_in_frame: max_ingredients_in_frame = len(current_ingredients_in_frame); best_frame_bytes_local = current_frame_bytes
        scan_loop_count += 1; time.sleep(0.2)
    print(f"Scan complete. Aggregated: {aggregated_ingredients or '{}'}, Max ingredients: {max_ingredients_in_frame}")
    with scan_frame_lock: final_scan_frame_bytes = best_frame_bytes_local

    if not aggregated_ingredients:
        print("ERROR: No ingredients aggregated.")
        with scan_frame_lock:
            final_scan_frame_bytes = None
        return jsonify(error="No ingredients detected during scan period."), 400

    print(f"Fetching recipes for: {list(aggregated_ingredients)}")
    recipes = get_recipes(list(aggregated_ingredients))
    if recipes:
        return jsonify(recipes=recipes)
    else:
        return jsonify(error="Could not fetch recipes."), 500

@app.route('/get_scan_frame')
def get_scan_frame():
    global final_scan_frame_bytes; frame_to_serve = None
    with scan_frame_lock:
        if final_scan_frame_bytes: frame_to_serve = final_scan_frame_bytes
    if frame_to_serve: return Response(frame_to_serve, mimetype='image/jpeg')
    else: print("Scan frame requested but none available."); placeholder = create_placeholder_image(text="Scan Frame Unavailable"); return Response(placeholder, mimetype='image/jpeg', status=404) if placeholder else abort(404)

# --- Authentication Routes ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username'); password = request.form.get('password')
        if not username or not password: flash('Username and password required.', 'warning'); return redirect(url_for('register'))
        existing_user = User.query.filter_by(username=username).first()
        if existing_user: flash('Username already exists.', 'warning'); return redirect(url_for('register'))
        new_user = User(username=username); new_user.set_password(password); db.session.add(new_user)
        try: db.session.commit(); flash(f'Account created for {username}!', 'success'); login_user(new_user); return redirect(url_for('index'))
        except Exception as e: db.session.rollback(); flash(f'Registration error: {e}', 'danger'); print(f"Reg error: {e}"); return redirect(url_for('register'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username'); password = request.form.get('password'); remember = True if request.form.get('remember') else False
        if not username or not password: flash('Username and password required.', 'warning'); return redirect(url_for('login'))
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=remember); flash('Login successful!', 'success'); next_page = request.args.get('next'); return redirect(next_page or url_for('index'))
        else: flash('Login unsuccessful.', 'danger'); return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout(): logout_user(); flash('You have been logged out.', 'info'); return redirect(url_for('index'))

# --- Saving and Viewing Routes ---

@app.route('/save_recipe', methods=['POST'])
@login_required
def save_recipe():
    data = request.json; snapshot_filename = None; snapshot_saved = False; current_snapshot_bytes = None
    if not data: return jsonify(status='error', message='Invalid request data.'), 400
    spoonacular_id = data.get('spoonacular_id'); title = data.get('title'); image_url = data.get('image_url'); recipe_url = data.get('recipe_url')
    if not spoonacular_id or not title: return jsonify(status='error', message='Missing recipe data.'), 400
    existing_save = SavedRecipe.query.filter_by(user_id=current_user.id, spoonacular_id=spoonacular_id).first()
    if existing_save: return jsonify(status='info', message='Recipe already saved.'), 200
    with scan_frame_lock:
        if final_scan_frame_bytes: current_snapshot_bytes = final_scan_frame_bytes
    if current_snapshot_bytes:
        try:
            unique_id = uuid.uuid4().hex; snapshot_filename = f"{unique_id}.jpg"; save_path = os.path.join(app.config['UPLOAD_FOLDER'], snapshot_filename)
            with open(save_path, 'wb') as f: f.write(current_snapshot_bytes)
            snapshot_saved = True; print(f"Snapshot saved: {save_path}")
        except Exception as e: print(f"Error saving snapshot file: {e}"); snapshot_filename = None; snapshot_saved = False; flash('Failed to save snapshot image.', 'warning')
    else: print("No snapshot available."); flash('Recipe saved without snapshot.', 'info')
    try:
        new_recipe = SavedRecipe(user_id=current_user.id, spoonacular_id=int(spoonacular_id), title=title, image_url=image_url, recipe_url=recipe_url)
        db.session.add(new_recipe); db.session.flush()
        if snapshot_saved and snapshot_filename: db.session.add(ScanSnapshot(saved_recipe_id=new_recipe.id, filename=snapshot_filename))
        db.session.commit(); print(f"Recipe/snapshot saved for user {current_user.username}")
        return jsonify(status='success', message='Recipe saved successfully!'), 201
    except IntegrityError as e: db.session.rollback(); print(f"DB integrity error: {e}"); return jsonify(status='info', message='Recipe already saved.'), 200
    except Exception as e:
        db.session.rollback(); print(f"Error saving to DB: {e}")
        if snapshot_saved and snapshot_filename:
             try: os.remove(os.path.join(app.config['UPLOAD_FOLDER'], snapshot_filename)); print(f"Cleaned up orphaned file: {snapshot_filename}")
             except OSError as rm_err: print(f"Error cleaning up file: {rm_err}")
        return jsonify(status='error', message='Failed to save recipe to database.'), 500

@app.route('/my_recipes')
@login_required
def my_recipes():
    saved_items = SavedRecipe.query.filter_by(user_id=current_user.id).order_by(SavedRecipe.timestamp.desc()).all()
    print(f"Found {len(saved_items)} saved items for user {current_user.username}")
    return render_template('my_recipes.html', saved_items=saved_items)

@app.route('/user_uploads/snapshots/<filename>')
def serve_snapshot(filename):
    safe_filename = secure_filename(filename);
    if safe_filename != filename: abort(404)
    try: return send_from_directory(app.config['UPLOAD_FOLDER'], safe_filename, as_attachment=False)
    except FileNotFoundError: print(f"Snapshot not found: {safe_filename}"); abort(404)

# --- Application Entry Point & DB Initialization ---
def init_db():
    with app.app_context(): print("Initializing database..."); db.create_all(); print("Database tables created.")

if __name__ == '__main__':
    init_db()
    frame_processor_thread = threading.Thread(target=process_frames, daemon=True); frame_processor_thread.start()
    print("Starting Flask app...");
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True) # Use threaded=True for background tasks
    release_camera(); print("Flask app stopped.")
