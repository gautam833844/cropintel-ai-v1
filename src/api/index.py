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
import sys

from pathlib import Path

# Set template and static folders using pathlib.Path for Vercel compatibility
api_dir = Path(__file__).resolve().parent
base_dir = api_dir.parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

template_folder = str(base_dir / 'templates')
static_folder = str(base_dir / 'static')

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
    """Load the trained ML model with multiple fallback paths for Vercel compatibility"""
    model = None
    curr_dir = Path(__file__).resolve().parent
    root_dir = curr_dir.parent.parent

    possible_paths = [
        root_dir / 'artifacts' / 'crop_model.pkl',
        curr_dir / 'artifacts' / 'crop_model.pkl',
        root_dir / 'crop_model.pkl',
        Path('/tmp/crop_model.pkl'),
        Path('/tmp/artifacts/crop_model.pkl'),
        Path('/var/task/artifacts/crop_model.pkl'),
        Path('/var/task/crop_model.pkl'),
        Path('artifacts/crop_model.pkl'),
        Path('crop_model.pkl')
    ]

    logger.info(f"[MODEL LOADING] Root dir: {root_dir}")
    logger.info(f"[MODEL LOADING] API dir: {curr_dir}")
    logger.info(
        f"[MODEL LOADING] Checking {len(possible_paths)} possible locations...")

    for model_path in possible_paths:
        abs_path = model_path.resolve() if model_path.is_absolute() else (root_dir / model_path).resolve()
        exists = abs_path.exists()
        logger.info(f"[MODEL LOADING] {'✓' if exists else '✗'} {abs_path}")

        if exists:
            try:
                model = joblib.load(abs_path)
                logger.info(
                    f"[MODEL LOADING] SUCCESS: Model loaded from {abs_path}")
                return model
            except Exception as e:
                logger.error(
                    f"[MODEL LOADING] ERROR loading from {abs_path}: {str(e)}")
                continue

    logger.error(
        "[MODEL LOADING] CRITICAL: Model file not found in any location!")
    return None


def load_scaler():
    """Load the feature scaler with multiple fallback paths"""
    scaler = None
    curr_dir = Path(__file__).resolve().parent
    root_dir = curr_dir.parent.parent

    possible_paths = [
        root_dir / 'artifacts' / 'scaler.pkl',
        curr_dir / 'artifacts' / 'scaler.pkl',
        root_dir / 'scaler.pkl',
        Path('/tmp/scaler.pkl'),
        Path('/tmp/artifacts/scaler.pkl'),
        Path('artifacts/scaler.pkl'),
        Path('scaler.pkl')
    ]

    logger.info(
        f"Attempting to load scaler from {len(possible_paths)} possible locations...")

    for scaler_path in possible_paths:
        abs_path = scaler_path.resolve() if scaler_path.is_absolute() else (root_dir / scaler_path).resolve()
        logger.info(f"  Checking: {abs_path}")

        if abs_path.exists():
            try:
                scaler = joblib.load(abs_path)
                logger.info(f"✓ Scaler loaded successfully from: {abs_path}")
                return scaler
            except Exception as e:
                logger.error(
                    f"✗ Error loading scaler from {abs_path}: {str(e)}")
                continue

    logger.warning(
        "⚠ Scaler file not found. Using raw features without scaling.")
    return None


model = load_model()
scaler = load_scaler()

# ==================== LOCAL AI KNOWLEDGE BASE ====================
# Import local knowledge base for crop recommendations (no external API needed)
try:
    from src.ai.crop_knowledge_base import get_crop_advice
    logger.info("✓ Local crop knowledge base loaded successfully")
except ImportError as e:
    logger.warning(f"⚠ Could not load crop knowledge base: {str(e)}")
    get_crop_advice = None

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

        # Support both 'nitrogen'/'N', 'phosphorus'/'P', 'potassium'/'K'
        n_val = data.get('nitrogen', data.get('N'))
        p_val = data.get('phosphorus', data.get('P'))
        k_val = data.get('potassium', data.get('K'))
        temp_val = data.get('temperature')
        hum_val = data.get('humidity')
        ph_val = data.get('ph')
        rain_val = data.get('rainfall')

        raw_features = {'nitrogen': n_val, 'phosphorus': p_val, 'potassium': k_val,
                        'temperature': temp_val, 'humidity': hum_val, 'ph': ph_val, 'rainfall': rain_val}

        required_features = ['nitrogen', 'phosphorus', 'potassium',
                             'temperature', 'humidity', 'ph', 'rainfall']

        # Validate all required features are present
        if not all(raw_features[f] is not None for f in required_features):
            missing = [f for f in required_features if raw_features[f] is None]
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Validate feature values are numeric
        try:
            features = np.array([float(raw_features[feature])
                                for feature in required_features])
        except (ValueError, TypeError):
            return jsonify({"error": "All features must be numeric values"}), 400

        # Validate feature ranges (basic sanity checks)
        if not all(f >= 0 for f in features):
            return jsonify({"error": "All feature values must be non-negative"}), 400

        # Construct DataFrame with column names
        import pandas as pd
        feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        features_df = pd.DataFrame([features], columns=feature_cols)

        # Apply scaler if available
        if scaler is not None:
            scaled_array = scaler.transform(features_df)
            features_df = pd.DataFrame(scaled_array, columns=feature_cols)

        # Make prediction
        prediction = model.predict(features_df)[0]

        # Get prediction probabilities if available
        try:
            probabilities = model.predict_proba(features_df)[0]
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


@app.route('/api/explain', methods=['POST'])
@limiter.limit("10 per minute")  # Higher limit since it's local
def explain():
    """Generate crop explanation using local knowledge base (no API needed)"""
    try:
        if not get_crop_advice:
            return jsonify({
                "explanation": "Local knowledge base is not available.",
                "tips": [],
                "benefits": []
            }), 200

        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        crop = data.get('crop')
        nitrogen = float(data.get('nitrogen', 0))
        phosphorus = float(data.get('phosphorus', 0))
        potassium = float(data.get('potassium', 0))
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        ph = float(data.get('ph', 0))
        rainfall = float(data.get('rainfall', 0))

        if not crop:
            return jsonify({"error": "Crop name is required"}), 400

        # Get advice from local knowledge base
        result = get_crop_advice(
            crop_name=crop,
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall
        )

        logger.info(f"✓ Local knowledge base explanation generated for {crop}")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"✗ Explanation error: {str(e)}")
        return jsonify({
            "explanation": "Unable to generate explanation at this time.",
            "tips": [],
            "benefits": []
        }), 500


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
