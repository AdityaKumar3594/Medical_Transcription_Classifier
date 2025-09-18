#!/usr/bin/env python3
"""
Launch script for the minimal medical classifier Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = {
        'streamlit': 'streamlit',
        'scikit-learn': 'sklearn', 
        'numpy': 'numpy',
        'pandas': 'pandas',
        'plotly': 'plotly'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_optional_dependencies():
    """Check optional dependencies and show status."""
    optional_packages = {
        'groq': 'Audio transcription support',
        'fitz': 'PDF processing (PyMuPDF)',
        'PyPDF2': 'PDF processing fallback',
        'pdfplumber': 'Enhanced PDF processing'
    }
    
    print("📋 Optional dependencies status:")
    for package, description in optional_packages.items():
        try:
            if package == 'fitz':
                import fitz
            else:
                __import__(package)
            print(f"  ✅ {package}: {description}")
        except ImportError:
            print(f"  ❌ {package}: {description} (not installed)")

def main():
    """Main launcher function."""
    print("🏥 Medical Specialty Classifier - Streamlit App")
    print("=" * 50)
    
    # Check if we're in the right directory
    app_file = Path("streamlit_medical_classifier.py")
    if not app_file.exists():
        print("❌ streamlit_medical_classifier.py not found in current directory")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All required dependencies found!")
    
    # Check optional dependencies
    check_optional_dependencies()
    
    print("\n🚀 Starting Streamlit app...")
    print("The app will open in your default web browser.")
    print("Press Ctrl+C to stop the app.")
    print("-" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_medical_classifier.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()