# Face Insight – AI Face Analyzer

A beginner-friendly AI web application that analyzes faces in uploaded images to detect age, gender, emotion, and facial symmetry.

## Features

- 🎯 **Face Detection**: Automatically detects one or multiple faces in images
- 👤 **Age Estimation**: Approximate age detection
- ⚧️ **Gender Detection**: Gender classification with confidence scores
- 😊 **Emotion Recognition**: Detects emotions (happy, sad, angry, neutral, surprise, fear, disgust)
- ⚖️ **Facial Symmetry**: Calculates facial symmetry percentage (0-100%)
- 📦 **Bounding Boxes**: Visual indicators around detected faces
- 📊 **Confidence Scores**: Displays confidence percentages for predictions

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Backend**: Python Flask
- **AI Libraries**: 
  - OpenCV (face detection)
  - DeepFace (age, gender, emotion)
  - MediaPipe (facial landmarks for symmetry)

## Project Structure

```
AI FACE/
├── app.py                 # Flask backend server
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML page
├── static/
│   ├── style.css         # Stylesheet
│   └── script.js         # Frontend JavaScript
└── uploads/              # Temporary upload directory (auto-created)
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: 
- DeepFace will automatically download pre-trained models on first use (requires internet connection)
- This may take a few minutes and download several hundred MB of model files
- OpenCV may require system dependencies on some platforms

### Step 2: Run the Application

```bash
python app.py
```

The server will start on `http://localhost:5000`

### Step 3: Open in Browser

Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Upload an Image**: 
   - Click the upload area or drag and drop an image
   - Supported formats: PNG, JPG, JPEG, GIF, BMP
   - Maximum file size: 16MB

2. **Wait for Analysis**: 
   - The app will process the image (may take 10-30 seconds)
   - A loading indicator will show progress

3. **View Results**: 
   - Detected faces will be highlighted with bounding boxes
   - Each face will have its own analysis card showing:
     - Age
     - Gender (with confidence)
     - Emotion (with confidence)
     - Facial Symmetry percentage

## How It Works

### Face Detection
- Uses OpenCV's Haar Cascade classifier for fast face detection
- Detects multiple faces in a single image

### Age, Gender, and Emotion Analysis
- Uses DeepFace library with pre-trained models
- Models are downloaded automatically on first use
- Runs on CPU (no GPU required)

### Facial Symmetry Calculation
- Uses MediaPipe to extract facial landmarks
- Compares left and right facial features
- Calculates symmetry based on geometric distances
- Returns a percentage score (0-100%)

## Troubleshooting

### "No faces detected"
- Ensure the image contains a clear, front-facing face
- Try a different image with better lighting
- Make sure the face is not too small or obscured

### Installation Errors
- Make sure you have Python 3.7+
- Try upgrading pip: `pip install --upgrade pip`
- On macOS, you may need: `brew install opencv`

### Slow Processing
- First run will be slower (downloading models)
- Processing time depends on image size and number of faces
- CPU-only processing is slower than GPU but works on any computer

### Port Already in Use
- If port 5000 is busy, edit `app.py` and change the port number
- Look for: `app.run(debug=True, host='0.0.0.0', port=5000)`

## Notes

- All AI models are pre-trained (no custom training required)
- Works on CPU-only systems (no GPU needed)
- Images are processed temporarily and deleted after analysis
- No data is stored permanently

## Deployment to Render

### Prerequisites
- GitHub account
- Render account (sign up at [render.com](https://render.com))

### Steps to Deploy

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Face Insight AI Analyzer"
   git branch -M main
   git remote add origin https://github.com/Saksham-Gupta-GH/face-insight.git
   git push -u origin main
   ```

2. **Deploy on Render**
   - Go to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub account and select the repository
   - Configure:
     - **Name**: face-insight (or your preferred name)
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
   - Click "Create Web Service"

3. **Important Notes for Render**
   - First deployment may take 10-15 minutes (downloading models)
   - The app uses CPU-only processing (no GPU required)
   - Free tier has limitations on processing time
   - DeepFace models will download automatically on first request

### Environment Variables
No environment variables required. The app uses the PORT environment variable automatically set by Render.

## License

This project is for educational and demonstration purposes.

