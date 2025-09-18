#!/usr/bin/env python3
"""
Streamlined Medical Transcription Classifier - Streamlit App

A focused Streamlit application for medical specialty classification using:
1. Traditional ML models (scikit-learn)
2. Keyword-based classification
3. Audio transcription (Groq Whisper)
4. PDF text extraction
5. Interactive results visualization

Optimized for minimal dependencies while maintaining accuracy.
"""

import streamlit as st
import asyncio
import tempfile
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import sys
import os
import re
from dataclasses import dataclass, asdict
import base64
import io
# Try to load .env automatically if python-dotenv is available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except Exception:
    DOTENV_AVAILABLE = False

# Core ML imports
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Audio and PDF processing
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="🏥 Medical Specialty Classifier",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class ClassificationResult:
    """Classification result structure."""
    predicted_specialty: str
    confidence: float
    all_predictions: Dict[str, float]
    method: str
    reasoning: str
    processing_time_ms: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class MedicalSpecialtyClassifier:
    """
    Streamlined medical specialty classifier using traditional ML.
    Optimized for reliability and minimal dependencies.
    """
    
    # Medical specialties with enhanced keyword sets
    MEDICAL_SPECIALTIES = {
        "cardiology": [
            "heart", "cardiac", "chest pain", "myocardial", "ecg", "ekg", "coronary", 
            "arrhythmia", "hypertension", "angina", "palpitations", "bradycardia", 
            "tachycardia", "atrial fibrillation", "heart failure", "bypass", "stent",
            "catheterization", "echocardiogram", "troponin", "cpk", "ejection fraction"
        ],
        "neurology": [
            "brain", "neurological", "seizure", "stroke", "headache", "migraine", 
            "parkinson", "alzheimer", "epilepsy", "neuropathy", "cerebral", "paralysis",
            "weakness", "numbness", "tingling", "tremor", "ataxia", "aphasia", 
            "hemiparesis", "mri brain", "ct head", "lumbar puncture", "eeg"
        ],
        "pulmonology": [
            "lung", "respiratory", "breathing", "asthma", "copd", "pneumonia", 
            "pulmonary", "chest", "bronchial", "oxygen", "shortness of breath", 
            "dyspnea", "cough", "wheezing", "rales", "rhonchi", "pleural effusion",
            "pneumothorax", "tuberculosis", "bronchoscopy", "spirometry", "cpap"
        ],
        "gastroenterology": [
            "stomach", "intestinal", "digestive", "gastro", "colonoscopy", "endoscopy", 
            "liver", "pancreas", "bowel", "gi", "abdominal pain", "nausea", "vomiting",
            "diarrhea", "constipation", "gastritis", "ulcer", "crohn", "colitis", 
            "ibs", "gerd", "reflux", "jaundice", "ascites", "hepatitis", "cirrhosis"
        ],
        "orthopedics": [
            "bone", "joint", "fracture", "orthopedic", "arthritis", "spine", "knee", 
            "hip", "shoulder", "musculoskeletal", "back pain", "vertebral", "ligament",
            "tendon", "cartilage", "x-ray", "mri", "physical therapy", "surgery",
            "prosthesis", "osteoporosis", "scoliosis"
        ],
        "dermatology": [
            "skin", "dermatology", "rash", "lesion", "melanoma", "eczema", "psoriasis", 
            "acne", "dermatitis", "mole", "basal cell", "squamous cell", "biopsy",
            "pruritus", "erythema", "vesicle", "pustule", "papule", "macule", "ulcer"
        ],
        "endocrinology": [
            "diabetes", "thyroid", "hormone", "endocrine", "insulin", "glucose", 
            "blood sugar", "hemoglobin a1c", "hyperthyroid", "hypothyroid", "adrenal",
            "pituitary", "metabolic", "obesity", "weight loss", "weight gain"
        ],
        "psychiatry": [
            "depression", "anxiety", "psychiatric", "mental health", "bipolar", 
            "schizophrenia", "therapy", "medication", "mood", "psychosis", "ptsd",
            "panic", "phobia", "counseling", "antidepressant", "antipsychotic"
        ],
        "emergency_medicine": [
            "emergency", "trauma", "acute", "urgent", "er", "emergency room", 
            "critical", "resuscitation", "triage", "ambulance", "accident", "injury",
            "code blue", "crash cart", "intubation", "shock", "hemorrhage", "overdose"
        ],
        "internal_medicine": [
            "internal medicine", "primary care", "general medicine", "chronic disease", 
            "medical history", "physical examination", "vital signs", "assessment",
            "plan", "follow-up", "medication management", "preventive care"
        ],
        "surgery": [
            "surgery", "surgical", "operation", "procedure", "incision", "anesthesia", 
            "post-operative", "pre-operative", "laparoscopic", "open", "resection",
            "repair", "reconstruction", "transplant", "suture"
        ],
        "pediatrics": [
            "child", "pediatric", "infant", "baby", "vaccination", "growth", 
            "development", "pediatrics", "newborn", "adolescent", "immunization",
            "well-child", "fever in child", "pediatric emergency"
        ],
        "obstetrics_gynecology": [
            "pregnancy", "gynecological", "obstetric", "uterus", "ovary", "menstrual", 
            "delivery", "cesarean", "prenatal", "postpartum", "pelvic", "cervical",
            "mammogram", "pap smear", "labor"
        ],
        "oncology": [
            "cancer", "tumor", "malignant", "chemotherapy", "radiation", "oncology", 
            "metastasis", "biopsy", "carcinoma", "lymphoma", "leukemia", "neoplasm",
            "mass", "staging"
        ],
        "radiology": [
            "x-ray", "ct scan", "mri", "ultrasound", "imaging", "radiological", 
            "contrast", "mammography", "fluoroscopy", "nuclear", "pet scan",
            "radiology", "findings"
        ]
    }
    
    def __init__(self):
        """Initialize the classifier."""
        self.ml_classifier = None
        self.vectorizer = None
        self.is_trained = False
        self.stats = {
            'classifications_performed': 0,
            'total_processing_time': 0.0,
            'specialty_distribution': {}
        }
        
        # Train the ML model
        self._train_ml_model()
    
    def _train_ml_model(self):
        """Train the ML classifier with synthetic data."""
        try:
            # Generate training data
            X_train, y_train = self._generate_training_data()
            
            # Create and train pipeline
            self.ml_classifier = Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000,
                    ngram_range=(1, 3),
                    stop_words='english',
                    lowercase=True
                )),
                ('classifier', MultinomialNB(alpha=0.1))
            ])
            
            self.ml_classifier.fit(X_train, y_train)
            self.is_trained = True
            
            logger.info("✅ ML classifier trained successfully")
            
        except Exception as e:
            logger.error(f"❌ ML training failed: {e}")
            self.is_trained = False
    
    def _generate_training_data(self) -> Tuple[List[str], List[str]]:
        """Generate synthetic training data from keywords."""
        X_train = []
        y_train = []
        
        # Medical text templates
        templates = [
            "Patient presents with {symptoms}. Physical examination reveals {findings}.",
            "History of {condition}. Patient reports {symptoms}.",
            "Assessment shows {findings}. Plan includes {treatment}.",
            "Patient has {condition} and {symptoms}.",
            "Clinical findings consistent with {diagnosis}. {additional}.",
            "Chief complaint: {symptoms}. Examination: {findings}.",
            "Diagnosis: {condition}. Treatment: {treatment}.",
            "Patient admitted with {symptoms}. Workup includes {tests}."
        ]
        
        for specialty, keywords in self.MEDICAL_SPECIALTIES.items():
            # Generate 30 samples per specialty
            for _ in range(30):
                # Randomly select keywords and template
                import random
                selected_keywords = random.sample(keywords, min(random.randint(2, 5), len(keywords)))
                template = random.choice(templates)
                
                # Fill template
                synthetic_text = template.format(
                    symptoms=selected_keywords[0] if len(selected_keywords) > 0 else "symptoms",
                    findings=selected_keywords[1] if len(selected_keywords) > 1 else "normal findings",
                    condition=selected_keywords[0] if len(selected_keywords) > 0 else "condition",
                    treatment=selected_keywords[2] if len(selected_keywords) > 2 else "treatment",
                    diagnosis=selected_keywords[0] if len(selected_keywords) > 0 else "diagnosis",
                    additional=" ".join(selected_keywords[3:]) if len(selected_keywords) > 3 else "Further evaluation needed",
                    tests=" ".join(selected_keywords[2:4]) if len(selected_keywords) > 2 else "laboratory tests"
                )
                
                X_train.append(synthetic_text)
                y_train.append(specialty)
        
        return X_train, y_train
    
    def classify_text(self, text: str) -> ClassificationResult:
        """Classify medical text."""
        start_time = time.time()
        
        try:
            # Clean text
            cleaned_text = self._preprocess_text(text)
            
            if not cleaned_text.strip():
                return ClassificationResult(
                    predicted_specialty="unknown",
                    confidence=0.0,
                    all_predictions={},
                    method="empty_text",
                    reasoning="No text provided for classification"
                )
            
            # Try ML classification first
            if self.is_trained:
                ml_result = self._classify_with_ml(cleaned_text)
                if ml_result.confidence > 0.3:
                    ml_result.processing_time_ms = (time.time() - start_time) * 1000
                    self._update_stats(ml_result)
                    return ml_result
            
            # Fallback to keyword-based classification
            keyword_result = self._classify_with_keywords(cleaned_text)
            keyword_result.processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats(keyword_result)
            return keyword_result
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return ClassificationResult(
                predicted_specialty="unknown",
                confidence=0.0,
                all_predictions={},
                method="error",
                reasoning=f"Classification failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _classify_with_ml(self, text: str) -> ClassificationResult:
        """Classify using ML model."""
        try:
            # Get prediction probabilities
            probabilities = self.ml_classifier.predict_proba([text])[0]
            classes = self.ml_classifier.classes_
            
            # Create predictions dictionary
            all_predictions = dict(zip(classes, probabilities))
            
            # Get top prediction
            top_class_idx = np.argmax(probabilities)
            predicted_specialty = classes[top_class_idx]
            confidence = probabilities[top_class_idx]
            
            # Generate reasoning
            reasoning = f"ML classification with {confidence:.2f} confidence. "
            reasoning += f"Top alternatives: {', '.join([f'{k}: {v:.2f}' for k, v in sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)[1:4]])}"
            
            return ClassificationResult(
                predicted_specialty=predicted_specialty,
                confidence=float(confidence),
                all_predictions={k: float(v) for k, v in all_predictions.items()},
                method="machine_learning",
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"ML classification error: {e}")
            return ClassificationResult(
                predicted_specialty="unknown",
                confidence=0.0,
                all_predictions={},
                method="ml_error",
                reasoning=f"ML classification failed: {str(e)}"
            )
    
    def _classify_with_keywords(self, text: str) -> ClassificationResult:
        """Classify using keyword matching."""
        try:
            text_lower = text.lower()
            specialty_scores = {}
            matched_keywords = {}
            
            # Calculate scores for each specialty
            for specialty, keywords in self.MEDICAL_SPECIALTIES.items():
                score = 0
                matches = []
                
                for keyword in keywords:
                    # Count keyword occurrences with word boundaries
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    count = len(re.findall(pattern, text_lower))
                    if count > 0:
                        # Weight by keyword length and frequency
                        weight = len(keyword.split()) * count
                        score += weight
                        matches.append(keyword)
                
                # Normalize score by text length
                if len(text_lower) > 0:
                    specialty_scores[specialty] = score / len(text_lower) * 1000
                    matched_keywords[specialty] = matches
            
            # Convert to probabilities
            total_score = sum(specialty_scores.values())
            if total_score > 0:
                all_predictions = {
                    specialty: score / total_score 
                    for specialty, score in specialty_scores.items()
                }
            else:
                all_predictions = {specialty: 0.0 for specialty in self.MEDICAL_SPECIALTIES.keys()}
            
            # Get top prediction
            if specialty_scores:
                predicted_specialty = max(specialty_scores.items(), key=lambda x: x[1])[0]
                confidence = all_predictions[predicted_specialty]
            else:
                predicted_specialty = "internal_medicine"
                confidence = 0.1
            
            # Generate reasoning
            top_matches = matched_keywords.get(predicted_specialty, [])
            reasoning = f"Keyword-based classification. Matched terms: {', '.join(top_matches[:5])}. "
            reasoning += f"Confidence: {confidence:.2f}"
            
            return ClassificationResult(
                predicted_specialty=predicted_specialty,
                confidence=float(confidence),
                all_predictions={k: float(v) for k, v in all_predictions.items()},
                method="keyword_matching",
                reasoning=reasoning,
                metadata={"matched_keywords": top_matches}
            )
            
        except Exception as e:
            logger.error(f"Keyword classification error: {e}")
            return ClassificationResult(
                predicted_specialty="unknown",
                confidence=0.0,
                all_predictions={},
                method="keyword_error",
                reasoning=f"Keyword classification failed: {str(e)}"
            )
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for classification."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove common PHI placeholders
        phi_tokens = [
            r'\[PATIENT_NAME\]', r'\[DATE_REDACTED\]', r'\[PHONE_NUMBER\]',
            r'\[SSN\]', r'\[EMAIL\]', r'\[MRN\]', r'\[ADDRESS\]'
        ]
        for token in phi_tokens:
            text = re.sub(token, '', text)
        
        return text.strip()
    
    def _update_stats(self, result: ClassificationResult):
        """Update classification statistics."""
        self.stats['classifications_performed'] += 1
        self.stats['total_processing_time'] += result.processing_time_ms
        
        specialty = result.predicted_specialty
        if specialty in self.stats['specialty_distribution']:
            self.stats['specialty_distribution'][specialty] += 1
        else:
            self.stats['specialty_distribution'][specialty] = 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        avg_time = (self.stats['total_processing_time'] / 
                   max(self.stats['classifications_performed'], 1))
        
        return {
            'total_classifications': self.stats['classifications_performed'],
            'average_processing_time_ms': avg_time,
            'specialty_distribution': self.stats['specialty_distribution'],
            'ml_model_trained': self.is_trained
        }


class AudioProcessor:
    """Audio transcription using Groq Whisper API."""
    
    def __init__(self, api_key: str):
        """Initialize with Groq API key."""
        if not GROQ_AVAILABLE:
            raise ImportError("Groq library not available. Install with: pip install groq")
        
        self.client = Groq(api_key=api_key)
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm']
        self.max_file_size = 25 * 1024 * 1024  # 25MB
    
    def transcribe_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """Transcribe audio file."""
        try:
            # Validate file
            file_path = Path(audio_file_path)
            if file_path.suffix.lower() not in self.supported_formats:
                return {
                    'success': False,
                    'error': f'Unsupported format: {file_path.suffix}',
                    'text': ''
                }
            
            if file_path.stat().st_size > self.max_file_size:
                return {
                    'success': False,
                    'error': 'File too large (max 25MB)',
                    'text': ''
                }
            
            # Transcribe using SDK with retry logic
            start_time = time.time()

            last_exception = None
            for attempt in range(1, 4):
                try:
                    with open(audio_file_path, 'rb') as audio_file:
                        transcription = self.client.audio.transcriptions.create(
                            file=audio_file,
                            model="whisper-large-v3",
                            response_format="text",
                            temperature=0.0,
                            prompt="This is a medical transcription. Please accurately transcribe medical terminology."
                        )

                    return {
                        'success': True,
                        'text': transcription,
                        'processing_time_ms': (time.time() - start_time) * 1000,
                        'method': 'groq_whisper'
                    }

                except Exception as e:
                    # Try to surface HTTP/SDK details if available
                    last_exception = e
                    err_str = str(e)

                    # Attempt to extract HTTP-like details from the exception (sdk/axios-like)
                    status_code = None
                    response_text = None
                    try:
                        # Common SDK shapes: e.response.status / e.response.status_code / e.status
                        resp = getattr(e, 'response', None)
                        if resp is not None:
                            status_code = getattr(resp, 'status', None) or getattr(resp, 'status_code', None)
                            # response body
                            response_text = getattr(resp, 'text', None) or getattr(resp, 'data', None) or getattr(resp, 'body', None)
                        else:
                            status_code = getattr(e, 'status', None) or getattr(e, 'status_code', None)
                    except Exception:
                        status_code = None

                    pretty_err = err_str
                    if status_code is not None:
                        pretty_err = f"{err_str} (status={status_code})"
                    if response_text:
                        pretty_err = f"{pretty_err} response_body={response_text}"

                    logger.warning(f"Attempt {attempt} failed: {pretty_err}")

                    # If this is a 400 Bad Request from the SDK, don't retry many times
                    if status_code == 400 or 'Bad Request' in err_str or (response_text and 'Bad Request' in str(response_text)):
                        # Attempt a direct HTTP fallback if requests is available
                        try:
                            import requests
                        except Exception:
                            # No requests available, return the SDK error details
                            return {
                                'success': False,
                                'error': f'SDK error (400): {err_str}',
                                'text': ''
                            }

                        # Perform REST fallback to Groq's transcription endpoint
                        try:
                            with open(audio_file_path, 'rb') as f:
                                files = {'file': (Path(audio_file_path).name, f)}
                                headers = {'Authorization': f'Bearer {self.client.api_key}'} if hasattr(self.client, 'api_key') else {'Authorization': f'Bearer {os.environ.get("GROQ_API_KEY", "") }'}
                                # Note: endpoint and params may change; this follows common patterns
                                resp = requests.post(
                                    'https://api.groq.com/v1/audio/transcriptions',
                                    headers=headers,
                                    files=files,
                                    data={
                                        'model': 'whisper-large-v3',
                                        'response_format': 'text',
                                        'temperature': '0.0',
                                        'prompt': 'This is a medical transcription. Please accurately transcribe medical terminology.'
                                    },
                                    timeout=60
                                )

                            if resp.status_code == 200:
                                return {
                                    'success': True,
                                    'text': resp.text,
                                    'processing_time_ms': (time.time() - start_time) * 1000,
                                    'method': 'groq_whisper_http_fallback'
                                }
                            else:
                                # Return a clearer message for 4xx responses
                                return {
                                    'success': False,
                                    'error': f'HTTP fallback failed with status {resp.status_code}: {resp.text}',
                                    'text': ''
                                }

                        except Exception as http_e:
                            logger.error(f"HTTP fallback error: {http_e}")
                            return {
                                'success': False,
                                'error': f'HTTP fallback error: {http_e}',
                                'text': ''
                            }

                    # For transient issues, back off briefly and retry
                    time.sleep(1 + attempt)

            # If we exit the retry loop without returning, surface the last error
            return {
                'success': False,
                'error': f'All transcription attempts failed: {last_exception}',
                'text': ''
            }
            
        except Exception as e:
            logger.error(f"Audio transcription error: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }


class PDFProcessor:
    """PDF text extraction."""
    
    def __init__(self):
        """Initialize PDF processor."""
        if not PDF_AVAILABLE:
            raise ImportError("PDF libraries not available. Install with: pip install PyMuPDF PyPDF2 pdfplumber")
    
    def extract_text(self, pdf_content: bytes) -> Dict[str, Any]:
        """Extract text from PDF."""
        try:
            # Try PyMuPDF first
            try:
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                text_parts = []
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text = page.get_text()
                    text_parts.append(text)
                
                doc.close()
                full_text = '\n'.join(text_parts)
                
                if full_text.strip():
                    return {
                        'success': True,
                        'text': full_text,
                        'method': 'pymupdf',
                        'pages': len(text_parts)
                    }
            except Exception as e:
                logger.warning(f"PyMuPDF failed: {e}")
            
            # Fallback to PyPDF2
            try:
                pdf_file = io.BytesIO(pdf_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_parts = []
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    text_parts.append(text)
                
                full_text = '\n'.join(text_parts)
                
                return {
                    'success': True,
                    'text': full_text,
                    'method': 'pypdf2',
                    'pages': len(text_parts)
                }
                
            except Exception as e:
                logger.error(f"PDF extraction failed: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'text': ''
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'text': ''
            }


# Streamlit App
def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Medical Specialty Classifier</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered medical transcription classification using traditional ML and keyword analysis**")
    
    # Initialize classifier
    if 'classifier' not in st.session_state:
        with st.spinner("Initializing classifier..."):
            st.session_state.classifier = MedicalSpecialtyClassifier()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Load GROQ API key from environment or .env
    if DOTENV_AVAILABLE:
        # Load .env from project root if present
        env_path = Path(__file__).parent / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

    env_groq_key = os.environ.get('GROQ_API_KEY', '')

    # API Keys (allow manual override in the sidebar)
    groq_api_key = st.sidebar.text_input(
        "Groq API Key (for audio transcription)",
        value=env_groq_key,
        type="password",
        help="Get your API key from https://console.groq.com/"
    )
    
    # Processing options
    st.sidebar.subheader("Processing Options")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Minimum confidence for classification"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Text Input", "🎵 Audio Upload", "📄 PDF Upload", "📊 Statistics"])
    
    with tab1:
        st.header("Direct Text Classification")
        
        # Text input
        text_input = st.text_area(
            "Enter medical transcription text:",
            height=200,
            placeholder="Enter medical text here for classification..."
        )
        
        if st.button("Classify Text", type="primary"):
            if text_input.strip():
                with st.spinner("Classifying..."):
                    result = st.session_state.classifier.classify_text(text_input)
                    display_classification_result(result, confidence_threshold)
            else:
                st.warning("Please enter some text to classify.")
    
    with tab2:
        st.header("Audio Transcription & Classification")
        
        if not GROQ_AVAILABLE:
            st.error("Groq library not installed. Install with: `pip install groq`")
        elif not groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar to use audio transcription.")
        else:
            audio_file = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'],
                help="Supported formats: WAV, MP3, M4A, FLAC, OGG, WebM (max 25MB)"
            )
            
            if audio_file is not None:
                # Display audio player
                st.audio(audio_file)
                
                if st.button("Transcribe & Classify", type="primary"):
                    with st.spinner("Processing audio..."):
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(audio_file.name).suffix) as tmp_file:
                            tmp_file.write(audio_file.read())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            # Initialize audio processor
                            audio_processor = AudioProcessor(groq_api_key)
                            
                            # Transcribe audio
                            transcription_result = audio_processor.transcribe_audio(tmp_file_path)
                            
                            if transcription_result['success']:
                                st.success("✅ Audio transcribed successfully!")
                                
                                # Display transcription
                                st.subheader("Transcription")
                                st.text_area("Transcribed text:", transcription_result['text'], height=150)
                                
                                # Classify transcription
                                with st.spinner("Classifying transcription..."):
                                    classification_result = st.session_state.classifier.classify_text(
                                        transcription_result['text']
                                    )
                                    display_classification_result(classification_result, confidence_threshold)
                            else:
                                st.error(f"❌ Transcription failed: {transcription_result['error']}")
                        
                        finally:
                            # Clean up temporary file
                            os.unlink(tmp_file_path)
    
    with tab3:
        st.header("PDF Text Extraction & Classification")
        
        if not PDF_AVAILABLE:
            st.error("PDF libraries not installed. Install with: `pip install PyMuPDF PyPDF2 pdfplumber`")
        else:
            pdf_file = st.file_uploader(
                "Upload PDF file",
                type=['pdf'],
                help="Upload a PDF document for text extraction and classification"
            )
            
            if pdf_file is not None:
                if st.button("Extract & Classify", type="primary"):
                    with st.spinner("Processing PDF..."):
                        # Initialize PDF processor
                        pdf_processor = PDFProcessor()
                        
                        # Extract text
                        extraction_result = pdf_processor.extract_text(pdf_file.read())
                        
                        if extraction_result['success']:
                            st.success(f"✅ Text extracted successfully using {extraction_result['method']}!")
                            
                            # Display extracted text
                            st.subheader("Extracted Text")
                            st.text_area("PDF text:", extraction_result['text'][:2000] + "..." if len(extraction_result['text']) > 2000 else extraction_result['text'], height=150)
                            
                            # Classify text
                            with st.spinner("Classifying text..."):
                                classification_result = st.session_state.classifier.classify_text(
                                    extraction_result['text']
                                )
                                display_classification_result(classification_result, confidence_threshold)
                        else:
                            st.error(f"❌ Text extraction failed: {extraction_result['error']}")
    
    with tab4:
        st.header("Classification Statistics")
        
        stats = st.session_state.classifier.get_stats()
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Classifications", stats['total_classifications'])
        
        with col2:
            st.metric("Avg Processing Time", f"{stats['average_processing_time_ms']:.1f} ms")
        
        with col3:
            st.metric("ML Model Status", "✅ Trained" if stats['ml_model_trained'] else "❌ Not Trained")
        
        # Specialty distribution chart
        if stats['specialty_distribution']:
            st.subheader("Specialty Distribution")
            
            df = pd.DataFrame(
                list(stats['specialty_distribution'].items()),
                columns=['Specialty', 'Count']
            )
            
            fig = px.bar(df, x='Specialty', y='Count', title="Classifications by Specialty")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)


def display_classification_result(result: ClassificationResult, confidence_threshold: float):
    """Display classification results."""
    
    # Main result
    if result.confidence >= confidence_threshold:
        st.success(f"✅ **Predicted Specialty:** {result.predicted_specialty.replace('_', ' ').title()}")
    else:
        st.warning(f"⚠️ **Predicted Specialty:** {result.predicted_specialty.replace('_', ' ').title()} (Low Confidence)")
    
    # Confidence and method
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Confidence", f"{result.confidence:.2f}")
    
    with col2:
        st.metric("Method", result.method.replace('_', ' ').title())
    
    with col3:
        st.metric("Processing Time", f"{result.processing_time_ms:.1f} ms")
    
    # Reasoning
    st.subheader("Classification Reasoning")
    st.info(result.reasoning)
    
    # All predictions
    if result.all_predictions:
        st.subheader("All Specialty Predictions")
        
        # Create DataFrame for visualization
        df = pd.DataFrame([
            {"Specialty": k.replace('_', ' ').title(), "Confidence": v}
            for k, v in sorted(result.all_predictions.items(), key=lambda x: x[1], reverse=True)
            if v > 0.01  # Only show predictions > 1%
        ])
        
        if not df.empty:
            # Bar chart
            fig = px.bar(df.head(10), x='Confidence', y='Specialty', orientation='h',
                        title="Top 10 Specialty Predictions")
            st.plotly_chart(fig, use_container_width=True)
            
            # Data table
            st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()