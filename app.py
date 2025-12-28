"""
Face Insight - AI Face Analyzer
Backend Flask Application

How to run:
1. Install dependencies: pip install -r requirements.txt
2. Run the app: python app.py
3. Open browser to: http://localhost:5000

The app will start on port 5000 by default.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np

# Compatibility patch for TensorFlow 2.16+ with DeepFace
# LocallyConnected2D was removed in newer TensorFlow versions
# This patch must be applied BEFORE importing DeepFace
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Layer
    
    # Check if LocallyConnected2D exists
    if not hasattr(tf.keras.layers, 'LocallyConnected2D'):
        # Create a compatibility layer
        class LocallyConnected2D(Layer):
            """Compatibility layer for LocallyConnected2D (removed in TF 2.16+)"""
            def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', 
                        data_format=None, activation=None, use_bias=True, 
                        kernel_initializer='glorot_uniform', bias_initializer='zeros',
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                        kernel_constraint=None, bias_constraint=None, **kwargs):
                super().__init__(**kwargs)
                self.filters = filters
                self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else (kernel_size, kernel_size)
                self.strides = strides if isinstance(strides, (list, tuple)) else (strides, strides)
                self.padding = padding
                self.activation = tf.keras.activations.get(activation)
                self.use_bias = use_bias
                
                # Use DepthwiseConv2D as a closer approximation
                self.conv = tf.keras.layers.DepthwiseConv2D(
                    kernel_size=self.kernel_size,
                    strides=self.strides,
                    padding=self.padding,
                    depth_multiplier=filters,
                    use_bias=use_bias,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    activity_regularizer=activity_regularizer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=bias_constraint
                )
            
            def call(self, inputs):
                output = self.conv(inputs)
                if self.activation is not None:
                    output = self.activation(output)
                return output
            
            def get_config(self):
                config = super().get_config()
                config.update({
                    'filters': self.filters,
                    'kernel_size': self.kernel_size,
                    'strides': self.strides,
                    'padding': self.padding,
                    'activation': tf.keras.activations.serialize(self.activation),
                    'use_bias': self.use_bias,
                })
                return config
        
        # Patch it into tensorflow.keras.layers
        tf.keras.layers.LocallyConnected2D = LocallyConnected2D
        # Also patch into the __all__ if it exists
        if hasattr(tf.keras.layers, '__all__'):
            if 'LocallyConnected2D' not in tf.keras.layers.__all__:
                tf.keras.layers.__all__.append('LocallyConnected2D')
except Exception as e:
    print(f"Warning: Could not apply LocallyConnected2D compatibility patch: {e}")

from deepface import DeepFace
import mediapipe as mp
import os
import base64
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# MediaPipe compatibility shim
# ---------------------------------------------------------------------------
# Newer versions of MediaPipe (0.10.x+) may not expose `mp.solutions` as a
# top-level attribute by default. The project code expects to use
# `mp.solutions.face_detection` and `mp.solutions.face_mesh`, so we try to
# import the `solutions` submodule explicitly and attach it to `mp` if
# necessary.
try:
    from mediapipe import solutions as mp_solutions
except ImportError:
    mp_solutions = getattr(mp, "solutions", None)

if mp_solutions is None:
    raise RuntimeError(
        "MediaPipe 'solutions' API is not available. Install a compatible "
        "mediapipe version that provides legacy solutions."
    )

# Expose solutions on the mp namespace for backward compatibility
if not hasattr(mp, "solutions"):
    mp.solutions = mp_solutions  # type: ignore[attr-defined]

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe face detection and landmarks
mp_face_detection = mp_solutions.face_detection
mp_face_mesh = mp_solutions.face_mesh
mp_drawing = mp_solutions.drawing_utils

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_facial_symmetry(landmarks, image_width, image_height):
    """
    Calculate facial symmetry percentage using facial landmarks.
    
    Approach:
    - Extract key facial points (eyes, nose, mouth corners)
    - Compare left and right side distances
    - Calculate symmetry score as percentage
    
    Args:
        landmarks: MediaPipe face landmarks
        image_width: Width of the image
        image_height: Height of the image
    
    Returns:
        symmetry_percentage: Float between 0-100
    """
    try:
        # MediaPipe face mesh landmark indices
        # Left eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        # Right eye: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        # Nose tip: 4
        # Mouth corners: 61 (left), 291 (right)
        # Face outline: 10 (chin), 152 (chin center)
        
        # Key points for symmetry calculation
        left_eye_center = landmarks.landmark[33]  # Left eye inner corner
        right_eye_center = landmarks.landmark[263]  # Right eye inner corner
        nose_tip = landmarks.landmark[4]
        left_mouth = landmarks.landmark[61]
        right_mouth = landmarks.landmark[291]
        chin = landmarks.landmark[152]
        
        # Convert to pixel coordinates
        def to_pixel(landmark):
            return (int(landmark.x * image_width), int(landmark.y * image_height))
        
        left_eye = to_pixel(left_eye_center)
        right_eye = to_pixel(right_eye_center)
        nose = to_pixel(nose_tip)
        left_m = to_pixel(left_mouth)
        right_m = to_pixel(right_mouth)
        chin_p = to_pixel(chin)
        
        # Calculate vertical center line (through nose and chin)
        center_x = nose[0]
        
        # Calculate distances from center line for left and right features
        left_eye_dist = abs(left_eye[0] - center_x)
        right_eye_dist = abs(right_eye[0] - center_x)
        
        left_mouth_dist = abs(left_m[0] - center_x)
        right_mouth_dist = abs(right_m[0] - center_x)
        
        # Calculate symmetry scores for each feature pair
        eye_symmetry = 1 - abs(left_eye_dist - right_eye_dist) / max(left_eye_dist + right_eye_dist, 1)
        mouth_symmetry = 1 - abs(left_mouth_dist - right_mouth_dist) / max(left_mouth_dist + right_mouth_dist, 1)
        
        # Calculate vertical alignment (how well features align horizontally)
        eye_level_diff = abs(left_eye[1] - right_eye[1])
        mouth_level_diff = abs(left_m[1] - right_m[1])
        max_eye_diff = max(left_eye_dist + right_eye_dist, 1)
        max_mouth_diff = max(left_mouth_dist + right_mouth_dist, 1)
        
        eye_alignment = 1 - (eye_level_diff / max_eye_diff)
        mouth_alignment = 1 - (mouth_level_diff / max_mouth_diff)
        
        # Overall symmetry score (average of all components)
        symmetry_score = (eye_symmetry + mouth_symmetry + eye_alignment + mouth_alignment) / 4
        
        # Convert to percentage and ensure it's between 0-100
        symmetry_percentage = max(0, min(100, symmetry_score * 100))
        
        return round(symmetry_percentage, 2)
    
    except Exception as e:
        print(f"Error calculating symmetry: {str(e)}")
        return 0.0

def detect_faces_opencv(image_path):
    """
    Detect faces using OpenCV's Haar Cascade classifier.
    Returns bounding boxes for all detected faces.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        return []
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Convert to list of dictionaries with bounding box info
    face_boxes = []
    for (x, y, w, h) in faces:
        face_boxes.append({
            'x': int(x),
            'y': int(y),
            'width': int(w),
            'height': int(h)
        })
    
    return face_boxes, image

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_face():
    """
    Main endpoint for face analysis.
    
    Receives an image file, processes it, and returns:
    - Face detection results
    - Age, gender, emotion (from DeepFace)
    - Facial symmetry (from MediaPipe)
    - Bounding boxes for visualization
    """
    try:
        # Check if file is present in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect faces using OpenCV
        face_boxes, image = detect_faces_opencv(filepath)
        
        if len(face_boxes) == 0:
            # Clean up uploaded file
            os.remove(filepath)
            return jsonify({
                'success': False,
                'message': 'No faces detected in the image. Please upload an image with a clear face.',
                'faces': []
            }), 200
        
        # Process each detected face
        results = []
        height, width = image.shape[:2]
        
        # Initialize MediaPipe face mesh for landmarks
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=len(face_boxes),
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_results = face_mesh.process(rgb_image)
            
            # Process each detected face
            for idx, face_box in enumerate(face_boxes):
                try:
                    # Extract face region for DeepFace analysis
                    x, y, w, h = face_box['x'], face_box['y'], face_box['width'], face_box['height']
                    
                    # Add padding to face region
                    padding = 20
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(width, x + w + padding)
                    y_end = min(height, y + h + padding)
                    
                    face_region = image[y_start:y_end, x_start:x_end]
                    
                    # Save face region temporarily for DeepFace
                    face_path = os.path.join(app.config['UPLOAD_FOLDER'], f'face_{idx}_{filename}')
                    cv2.imwrite(face_path, face_region)
                    
                    # Analyze with DeepFace (age, gender, emotion)
                    try:
                        deepface_result = DeepFace.analyze(
                            face_path,
                            actions=['age', 'gender', 'emotion'],
                            enforce_detection=False,
                            silent=True
                        )
                        
                        # Handle both single dict and list responses from DeepFace
                        if isinstance(deepface_result, list):
                            analysis = deepface_result[0]
                        else:
                            analysis = deepface_result
                        
                        age = int(analysis.get('age', 0))
                        gender = analysis.get('dominant_gender', 'Unknown')
                        emotion = analysis.get('dominant_emotion', 'Unknown')
                        
                        # Get confidence scores
                        emotion_scores = analysis.get('emotion', {})
                        gender_scores = analysis.get('gender', {})
                        
                        emotion_confidence = round(emotion_scores.get(emotion, 0), 2) if emotion_scores else 0
                        gender_confidence = round(gender_scores.get(gender, 0), 2) if gender_scores else 0
                        
                    except Exception as e:
                        print(f"DeepFace analysis error: {str(e)}")
                        age = 0
                        gender = "Unknown"
                        emotion = "Unknown"
                        emotion_confidence = 0
                        gender_confidence = 0
                    
                    # Calculate facial symmetry using MediaPipe landmarks
                    symmetry = 0.0
                    if mp_results.multi_face_landmarks and idx < len(mp_results.multi_face_landmarks):
                        landmarks = mp_results.multi_face_landmarks[idx]
                        symmetry = calculate_facial_symmetry(landmarks, width, height)
                    
                    # Clean up temporary face file
                    if os.path.exists(face_path):
                        os.remove(face_path)
                    
                    # Prepare result for this face
                    result = {
                        'bounding_box': face_box,
                        'age': age,
                        'gender': gender,
                        'emotion': emotion,
                        'symmetry': symmetry,
                        'confidence': {
                            'emotion': emotion_confidence,
                            'gender': gender_confidence
                        }
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing face {idx}: {str(e)}")
                    continue
        
        # Clean up uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Successfully analyzed {len(results)} face(s)',
            'faces': results
        }), 200
        
    except Exception as e:
        # Clean up on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets this) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*50)
    print("Face Insight - AI Face Analyzer")
    print("="*50)
    print("Starting server...")
    print(f"Open your browser to: http://localhost:{port}")
    print("="*50 + "\n")
    
    # Run the Flask app
    # Use debug=False for production (Render)
    app.run(debug=False, host='0.0.0.0', port=port)

