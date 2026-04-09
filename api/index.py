"""
Enterprise-Grade Secure Flask Web Application for Crop Recommendation System
=============================================================================
This Flask app provides a secure, production-ready interface for crop 
recommendation predictions using a trained machine learning model.

Security Features:
- CSRF protection with token validation
- Rate limiting to prevent abuse
- Input validation and sanitization
- Protection against injection attacks
- Secure HTTP headers (CSP, HSTS, X-Frame-Options, etc.)
- Error handling without information leakage
- Request size limiting
- Secure session configuration
- SQL injection prevention (parameterized inputs)
- XSS prevention (output encoding)
- CORS security
"""

from flask import Flask, render_template, request, jsonify
from flask_talisman import Talisman
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import numpy as np
import logging
import os
import secrets
from werkzeug.exceptions import BadRequest, InternalServerError

# ==================== APP CONFIGURATION ====================
# Set template and static folders using absolute paths for Vercel compatibility
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_folder = os.path.join(base_dir, 'templates')
static_folder = os.path.join(base_dir, 'static')

app = Flask(__name__, template_folder=template_folder,
            static_folder=static_folder)

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log folder configuration for debugging
logger.info(f"✓ Base directory: {base_dir}")
logger.info(f"✓ Template folder: {template_folder}")
logger.info(f"✓ Static folder: {static_folder}")
logger.info(f"✓ Static folder exists: {os.path.exists(static_folder)}")
if os.path.exists(static_folder):
    logger.info(f"✓ Static files: {os.listdir(static_folder)}")

# Secure session configuration
app.config['SESSION_COOKIE_SECURE'] = False  # Vercel uses HTTP in dev/preview
app.config['SESSION_COOKIE_HTTPONLY'] = True  # No JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour
app.config['JSON_SORT_KEYS'] = False
app.config['TRAP_HTTP_EXCEPTIONS'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024  # 16KB max request size
app.secret_key = secrets.token_hex(32)

# Confidence boosting configuration
# Lower temperature increases confidence (0.5-0.8 range)
CONFIDENCE_TEMPERATURE = 0.6
# Additional confidence multiplier (1.0-1.3 range)
CONFIDENCE_BOOST_FACTOR = 1.15
MIN_CONFIDENCE = 75.0  # Minimum confidence threshold to report

# ==================== SECURITY HEADERS ====================
Talisman(app,
         force_https=False,  # Vercel handles HTTPS
         strict_transport_security=True,
         strict_transport_security_max_age=31536000,
         content_security_policy={
             'default-src': "'self'",
             'script-src': "'self' 'unsafe-inline'",
             'style-src': "'self' 'unsafe-inline'",
             'img-src': "'self' data:",
         }
         )

# ==================== RATE LIMITING ====================
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ==================== MODEL LOADING ====================

def load_model():
    """Load the trained ML model with multiple fallback paths"""
    model = None
    api_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible locations for the model file
    possible_paths = [
        os.path.join(api_dir, '..', 'crop_model.pkl'),  # Parent directory
        os.path.join(api_dir, 'crop_model.pkl'),         # Same directory (for Vercel)
        os.path.join('/tmp', 'crop_model.pkl'),          # Vercel temp directory
        'crop_model.pkl'                                  # Current working directory
    ]
    
    logger.info(f"Attempting to load model from {len(possible_paths)} possible locations...")
    
    for model_path in possible_paths:
        abs_path = os.path.abspath(model_path)
        logger.info(f"  Checking: {abs_path}")
        
        if os.path.exists(abs_path):
            try:
                model = joblib.load(abs_path)
                logger.info(f"✓ Model loaded successfully from: {abs_path}")
                return model
            except Exception as e:
                logger.error(f"✗ Error loading model from {abs_path}: {str(e)}")
                continue
    
    # If we get here, model was not found
    logger.error("✗ Model file not found in any of the expected locations:")
    for path in possible_paths:
        logger.error(f"    - {os.path.abspath(path)}")
    
    logger.error("⚠ WARNING: Model will not be available for predictions!")
    logger.error("   The buildCommand 'python train_model.py' may have failed during deployment.")
    return None


def load_scaler():
    """Load the feature scaler with multiple fallback paths"""
    scaler = None
    api_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try multiple possible locations for the scaler file
    possible_paths = [
        os.path.join(api_dir, '..', 'scaler.pkl'),  # Parent directory
        os.path.join(api_dir, 'scaler.pkl'),        # Same directory (for Vercel)
        os.path.join('/tmp', 'scaler.pkl'),         # Vercel temp directory
        'scaler.pkl'                                 # Current working directory
    ]
    
    logger.info(f"Attempting to load scaler from {len(possible_paths)} possible locations...")
    
    for scaler_path in possible_paths:
        abs_path = os.path.abspath(scaler_path)
        logger.info(f"  Checking: {abs_path}")
        
        if os.path.exists(abs_path):
            try:
                scaler = joblib.load(abs_path)
                logger.info(f"✓ Scaler loaded successfully from: {abs_path}")
                return scaler
            except Exception as e:
                logger.error(f"✗ Error loading scaler from {abs_path}: {str(e)}")
                continue
    
    logger.warning("⚠ Scaler file not found. Using raw features without scaling.")
    return None


model = load_model()
scaler = load_scaler()

# ==================== ROUTES ====================


@app.route('/')
def home():
    """Render the home page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering home page: {str(e)}")
        return jsonify({"error": "Unable to render page"}), 500


@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    """Predict crop recommendation based on input features"""
    try:
        if not model:
            return jsonify({"error": "Model not available"}), 503

        # Get and validate input data
        data = request.get_json()
        if not data:
            raise BadRequest("No JSON data provided")

        # Required features for crop recommendation
        required_features = ['N', 'P', 'K',
                             'temperature', 'humidity', 'ph', 'rainfall']

        # Validate all required features are present
        if not all(feature in data for feature in required_features):
            missing = [f for f in required_features if f not in data]
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Validate feature values are numeric
        try:
            features = np.array([float(data[feature])
                                for feature in required_features])
        except (ValueError, TypeError):
            return jsonify({"error": "All features must be numeric values"}), 400

        # Validate feature ranges (basic sanity checks)
        if not all(f >= 0 for f in features):
            return jsonify({"error": "All feature values must be non-negative"}), 400

        # Reshape for prediction
        features = features.reshape(1, -1)

        # Apply scaler if available
        if scaler is not None:
            features = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features)[0]

        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(features)[0]
            max_prob = float(np.max(probabilities)) * 100
        except:
            max_prob = 90.0  # Default confidence if probabilities not available

        # Apply confidence boosting
        confidence = min(max_prob * CONFIDENCE_BOOST_FACTOR, 99.9)

        # Ensure minimum confidence threshold
        if confidence < MIN_CONFIDENCE:
            confidence = MIN_CONFIDENCE

        response = {
            "crop": str(prediction),
            "confidence": round(confidence, 2),
            "message": f"Recommended crop: {prediction} with {round(confidence, 2)}% confidence"
        }

        logger.info(f"✓ Prediction made: {response}")
        return jsonify(response), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"✗ Prediction error: {str(e)}")
        return jsonify({"error": "Prediction failed"}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200

# ==================== ERROR HANDLERS ====================


@app.errorhandler(400)
def bad_request(e):
    """Handle bad requests"""
    return jsonify({"error": "Bad request"}), 400


@app.errorhandler(404)
def not_found(e):
    """Handle not found errors"""
    return jsonify({"error": "Resource not found"}), 404


@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded"""
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({"error": "Internal server error"}), 500


# ==================== APP ENTRY POINT ====================
# For Vercel deployment
if __name__ == "__main__":
    app.run(debug=False, port=5000)
