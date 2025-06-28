from flask import Flask, render_template, request, jsonify, session, url_for, send_from_directory, redirect
import os
from yolo_utils import YOLODetector
from gpod_logic import GPODAuth
import cv2
import numpy as np
from PIL import Image
import uuid
import random
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import hashlib
import json
import time
from aes_utils import generate_skuid, encrypt_skuid, encrypt_with_key
from aes_utils import decrypt_skuid, decrypt_with_key

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gpod.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    username_hash = db.Column(db.String(256), unique=True, nullable=False)
    objects = db.Column(db.String(500), nullable=False)  # Store object selections
    background_type = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_admin = db.Column(db.Boolean, default=False)
    encrypted_skuid = db.Column(db.String(256), nullable=False)


# Valid background types
VALID_BACKGROUNDS = ['indoor', 'mountain', 'outdoor', 'sea_cliff']

# Create all database tables
with app.app_context():
    db.create_all()

# Initialize YOLO detector and GPOD authentication
yolo_detector = YOLODetector()
gpod_auth = GPODAuth(db, User)  # Pass db and User model to GPODAuth

# Directory paths
CHALLENGE_IMAGES_DIR = os.path.join('static', 'images', 'challenges')
BACKGROUNDS_DIR = os.path.join('static', 'images', 'backgrounds')
CLASSES_DIR = os.path.join('static', 'images', 'classes')
OBJECTS_DIR = os.path.join('static', 'images', 'objects')
os.makedirs(CHALLENGE_IMAGES_DIR, exist_ok=True)

def add_salt_pepper_noise(image, amount=0.0005, salt_vs_pepper=0.5):
    """"""
    noisy = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]
    num_salt = int(amount * total_pixels * salt_vs_pepper)
    num_pepper = int(amount * total_pixels * (1.0 - salt_vs_pepper))

    # Salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy

def add_gaussian_noise(image, mean=0, sigma=1.5):
    """"""
    gauss = np.random.normal(mean, sigma, image.shape).astype('uint8')
    noisy = cv2.add(image, gauss)
    return noisy






def get_random_background(bg_type):
    """Get a random background image of specified type"""
    bg_path = os.path.join(BACKGROUNDS_DIR, bg_type)
    if os.path.exists(bg_path):
        bg_files = [f for f in os.listdir(bg_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if bg_files:
            return os.path.join(bg_path, random.choice(bg_files))
    return None

def get_object_image(obj_name):
    """Get object image from classes directory"""
    obj_path = os.path.join(CLASSES_DIR, obj_name)
    if os.path.exists(obj_path):
        obj_files = [f for f in os.listdir(obj_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if obj_files:
            return os.path.join(obj_path, random.choice(obj_files))
    return None

def create_challenge_image(background_type, selected_objects):
    try:
        print(f"Creating challenge image with background type: {background_type}")
        print(f"Selected objects: {selected_objects}")

        background_dir = os.path.join(BACKGROUNDS_DIR, background_type.lower())
        if not os.path.exists(background_dir):
            print(f"Background directory not found: {background_dir}")
            return None, None

        background_files = [f for f in os.listdir(background_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not background_files:
            print(f"No background images found in: {background_dir}")
            return None, None

        background_file = random.choice(background_files)
        background_path = os.path.join(background_dir, background_file)
        print(f"Selected background: {background_file}")

        background = cv2.imread(background_path)
        if background is None:
            print("Failed to load background image")
            return None, None

        background = cv2.resize(background, (800, 600))
        # Add Gaussian and Salt & Pepper noise
        background = add_gaussian_noise(background)
        background = add_salt_pepper_noise(background)

        print(f"Background resized to: {background.shape}")

        grid_rows, grid_cols = 3, 3
        cell_width = background.shape[1] // grid_cols
        cell_height = background.shape[0] // grid_rows
        object_positions = []

        for idx, obj in enumerate(selected_objects):
            obj_name = obj.get('name')
            print(f"Processing object: {obj_name}")

            obj_dir = os.path.join(CLASSES_DIR, obj_name.lower())
            if not os.path.exists(obj_dir):
                print(f"Object directory not found: {obj_dir}")
                return None, None

            obj_files = [f for f in os.listdir(obj_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if not obj_files:
                print(f"No object images found in: {obj_dir}")
                return None, None

            obj_file = random.choice(obj_files)
            obj_path = os.path.join(obj_dir, obj_file)
            print(f"Selected object image: {obj_file}")

            obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
            if obj_img is None:
                print(f"Failed to load object image: {obj_path}")
                return None, None

            # Resize object to fit inside cell
            max_width = int(cell_width * 0.8)
            max_height = int(cell_height * 0.8)
            scale = min(max_width / obj_img.shape[1], max_height / obj_img.shape[0], 1.0)
            new_width = int(obj_img.shape[1] * scale)
            new_height = int(obj_img.shape[0] * scale)
            obj_img = cv2.resize(obj_img, (new_width, new_height))

            row = idx // grid_cols
            col = idx % grid_cols
            x = col * cell_width + (cell_width - new_width) // 2
            y = row * cell_height + (cell_height - new_height) // 2

            # Blending
            alpha = 0.7  # Transparency factor (0: invisible, 1: fully visible)
            roi = background[y:y+new_height, x:x+new_width]
            if roi.shape[:2] != obj_img.shape[:2]:
                print("Size mismatch during blending, skipping object.")
                continue

            blended = cv2.addWeighted(obj_img, alpha, roi, 1 - alpha, 0)
            background[y:y+new_height, x:x+new_width] = blended

            object_positions.append({
                'name': obj_name,
                'x': x,
                'y': y,
                'width': new_width,
                'height': new_height,
                'clicks': obj.get('clicks', 1)
            })

            print(f"Placed {obj_name} at grid cell ({row}, {col})")

        print(f"All {len(object_positions)} objects placed successfully.")
        return background, object_positions

    except Exception as e:
        print(f"Error creating challenge image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def create_password_hash(objects):
    """Create a SHA-256 hash of the graphical password"""
    # Sort objects to ensure consistent hashing
    sorted_objects = sorted(objects, key=lambda x: (x['name'], x['clicks']))
    # Create a string representation
    password_string = '-'.join([f"{obj['name']}:{obj['clicks']}" for obj in sorted_objects])
    # Create SHA-256 hash
    return hashlib.sha256(password_string.encode()).hexdigest()

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({'success': False, 'error': 'Login failed.'}), 400

            username = data.get('username')
            objects = data.get('objects', [])
            background_type = data.get('background_type')
            
            print(f"Registration attempt - Username: {username}, Background: {background_type}, Objects: {len(objects)}")
            
            if not username or not objects or not background_type:
                return jsonify({'success': False, 'error': 'Failed.'}), 400
                
            # Validate username
            if not username.strip():
                return jsonify({'success': False, 'error': 'Failed.'}), 400
                
            # Validate background type
            if background_type not in VALID_BACKGROUNDS:
                return jsonify({'success': False, 'error': 'Background not valid.'}), 400
                
            # Validate objects count
            if len(objects) < 3 or len(objects) > 5:
                return jsonify({'success': False, 'error': 'select only 3-5 objects.'}), 400
                
            # Check if username already exists
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                return jsonify({'success': False, 'error': 'Username already registered.'}), 409
            username_hash = hashlib.sha256(username.encode()).hexdigest()
            skuid = generate_skuid()
            encrypted_skuid = encrypt_skuid(skuid)
            encrypted_objects = encrypt_with_key(json.dumps(objects), skuid)
            new_user = User(
            username=username,
            username_hash=username_hash,
            objects=encrypted_objects,         # ⬅ encrypted object data
            encrypted_skuid=encrypted_skuid,   # ⬅ stored encrypted SKUID
            background_type=background_type,
            is_admin=False
        ) 
            try:
                db.session.add(new_user)
                db.session.commit()
                print(f"User registered successfully: {username}")
                return jsonify({'success': True}), 201
            except Exception as e:
                db.session.rollback()
                print(f"Database error during registration: {str(e)}")
                return jsonify({'success': False, 'error': 'Login failed.'}), 500
                
        except Exception as e:
            print(f"Registration error: {str(e)}")
            return jsonify({'success': False, 'error': 'Login failed.'}), 500
            
    # GET request - render registration page
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        username = data.get('username')
        login_objects = data.get('objects', [])
        
        if not username or not login_objects:
            return jsonify({'success': False, 'error': 'Login failed.'})
            
        # Check if preview has expired (2 minutes)
        preview_timestamp = session.get('preview_timestamp')
        if not preview_timestamp or (int(time.time()) - preview_timestamp) > 120:
            return jsonify({'success': False, 'error': 'Login failed.Time Up.'})
            
        # Find user
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({'success': False, 'error': 'Login failed.'})

        if data.get('background_type') != user.background_type:
            return jsonify({'success': False, 'error': 'Login failed.'})

            
        # Get registered objects
       

        skuid = decrypt_skuid(user.encrypted_skuid)
        registered_objects = json.loads(decrypt_with_key(user.objects, skuid))

        
        # Create dictionaries to track click counts for each class
        registered_clicks = {}
        login_clicks = {}
        
        # Count registered clicks per class
        for obj in registered_objects:
            registered_clicks[obj['name']] = obj['clicks']
            
        # Verify each login object using YOLO detection
        for obj in login_objects:
            obj_name = obj['name']
            obj_clicks = obj['clicks']
            
            # Track clicks for this class without YOLO verification for now
            if obj_name in login_clicks:
                login_clicks[obj_name] += obj_clicks
            else:
                login_clicks[obj_name] = obj_clicks
        
        # Verify all required classes are present with correct click counts
        if set(registered_clicks.keys()) != set(login_clicks.keys()):
            return jsonify({
                'success': False,
                'error': 'Missing required object classes'
            })
        
        # Verify click counts match for each class
        for obj_name, req_clicks in registered_clicks.items():
            if login_clicks.get(obj_name) != req_clicks:
                return jsonify({
                    'success': False,
                    'error': f'Incorrect number of clicks for {obj_name}'
                })
        
        # Login successful
        session['username'] = username
        return jsonify({'success': True, 'redirect': url_for('dashboard')})
        
    return render_template('login.html')

@app.route('/api/generate-preview', methods=['POST'])
@app.route('/api/generate-preview', methods=['POST'])
def generate_preview():
    try:
        data = request.json
        print(f"Received preview request: {data}")  # Debug log

        background_type = data.get('background_type')
        is_registration = data.get('is_registration', False)

        if not background_type:
            return jsonify({'success': False, 'error': 'Login failed.'})

        # Define all available classes
        all_classes = ['person', 'dog', 'cat', 'bird', 'zebra', 'giraffe', 'clock', 'food', 'aeroplane']
        print("Available classes:", all_classes)

        # For login, use all classes
        if not is_registration:
            selected_objects = [{'name': class_name, 'clicks': 1} for class_name in all_classes]
            print("Selected objects for preview:", selected_objects)
        else:
            # For registration, use selected objects from request
            selected_objects = data.get('objects', [])
            if not selected_objects:
                return jsonify({'success': False, 'error': 'Login failed.'})
            print("Selected objects for registration:", selected_objects)

        # Create challenge image with selected objects
        print("Creating challenge image...")
        background, object_positions = create_challenge_image(background_type, selected_objects)

        if background is None or not object_positions:
            print("Failed to create challenge image")
            return jsonify({'success': False, 'error': 'Login failed.'})

        # Save the image and store timestamp
        timestamp = int(time.time())
        preview_path = os.path.join(app.static_folder, f'preview_{timestamp}.jpg')
        print(f"Saving preview image to: {preview_path}")

        cv2.imwrite(preview_path, background)

        # Store timestamp in session for validation
        session['preview_timestamp'] = timestamp

        print(f"Generated preview with {len(object_positions)} objects")
        print("Object positions:", [obj['name'] for obj in object_positions])

        return jsonify({
            'success': True,
            'image_url': f'/static/preview_{timestamp}.jpg',
            'object_positions': object_positions
        })

    except Exception as e:
        print(f"Error generating preview: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Login failed.'})


@app.route('/api/generate-challenge', methods=['POST'])
def generate_challenge():
    try:
        data = request.json
        username = data.get('username')
        if not username:
            return jsonify({'success': False, 'error': 'Login failed.'})
        
        # Get user preferences from GPODAuth
        user_prefs = gpod_auth.get_user_preferences(username)
        if not user_prefs:
            return jsonify({'success': False, 'error': 'Login failed.'})
        
        # Generate challenge image
        challenge_id = str(uuid.uuid4())
        image_path = os.path.join(CHALLENGE_IMAGES_DIR, f'challenge_{challenge_id}.jpg')
        
        # Create challenge image with actual background and objects
        background_type = user_prefs.get('background_type', 'default')
        objects = user_prefs.get('objects', [])[:3]  # Use up to 3 objects
        img, object_positions = create_challenge_image(background_type, objects)
        
        cv2.imwrite(image_path, img)
        
        challenge = {
            'id': challenge_id,
            'image_url': url_for('static', filename=f'images/challenges/challenge_{challenge_id}.jpg'),
            'objects': objects,
            'object_positions': object_positions
        }
        
        # Store challenge in session
        session['current_challenge'] = challenge
        
        return jsonify(challenge)
    except Exception as e:
        return jsonify({'success': False, 'error': 'Login failed.'})

@app.route('/api/verify-clicks', methods=['POST'])
def verify_clicks():
    try:
        data = request.json
        challenge_id = data.get('challenge_id')
        clicks = data.get('clicks', [])
        
        current_challenge = session.get('current_challenge')
        if not current_challenge or current_challenge['id'] != challenge_id:
            return jsonify({'success': False, 'message': 'Invalid challenge'})
        
        # Verify that all required objects were clicked
        required_objects = set(current_challenge['objects'])
        clicked_objects = set(clicks)
        
        if clicked_objects == required_objects:
            return jsonify({'success': True, 'message': 'Authentication successful'})
        else:
            return jsonify({'success': False, 'message': 'Login Failed'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
        
    # Get only the current user's data
    current_user = User.query.filter_by(username=session['username']).first()
    if not current_user:
        session.clear()
        return redirect(url_for('login'))
    skuid = decrypt_skuid(current_user.encrypted_skuid)
        
    user_data = {
        'username': current_user.username,
        'background_type': current_user.background_type,
        'objects': json.loads(decrypt_with_key(current_user.objects, skuid)),
        'created_at': current_user.created_at.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return render_template('dashboard.html', users=[user_data])

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin_dashboard():
    # Get all users and their data
    users = User.query.all()
    user_list = []
    for user in users:
        skuid = decrypt_skuid(user.encrypted_skuid)
        user_list.append({
            'username': user.username,
            'background_type': user.background_type,
            'objects': json.loads(decrypt_with_key(user.objects, skuid)),
            'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S')
        })
    return render_template('admin_dashboard.html', users=user_list)

@app.route('/api/delete-user', methods=['POST'])
def delete_user():
    data = request.json
    username = data.get('username')
    
    if not username:
        return jsonify({'success': False, 'error': 'Login failed.'})
        
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({'success': False, 'error': 'Login failed.'})
        
    try:
        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': 'Login failed.'})

@app.route('/api/get-user-objects', methods=['POST'])
def get_user_objects():
    try:
        data = request.json
        username = data.get('username')
        
        if not username:
            return jsonify({'success': False, 'error': 'Login failed.'})
            
        # Find user
        user = User.query.filter_by(username=username).first()
        if not user:
            return jsonify({'success': False, 'error': 'Login failed.'})
            
        # Get registered objects
        skuid = decrypt_skuid(user.encrypted_skuid)
        objects = json.loads(decrypt_with_key(user.objects, skuid))

        
        return jsonify({
            'success': True,
            'objects': objects
        })
        
    except Exception as e:
        print(f"Error getting user objects: {str(e)}")
        return jsonify({'success': False, 'error': 'Login failed.'})

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)