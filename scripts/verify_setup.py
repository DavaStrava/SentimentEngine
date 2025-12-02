#!/usr/bin/env python3
"""Verify that the SentimentEngine project setup is complete"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_directories():
    """Check that all required directories exist"""
    required_dirs = [
        "src/config",
        "src/models",
        "src/input",
        "src/analysis",
        "src/fusion",
        "src/ui",
        "tests/unit",
        "tests/property",
        "tests/integration",
        "tests/performance",
        "tests/fixtures",
        "models/acoustic",
        "models/visual",
        "models/linguistic",
        "config",
        "scripts",
        "logs",
    ]
    
    print("Checking directory structure...")
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_files():
    """Check that all required files exist"""
    required_files = [
        "requirements.txt",
        "README.md",
        "config/config.yaml",
        "src/config/config_loader.py",
        "tests/conftest.py",
        "scripts/download_models.py",
    ]
    
    print("\nChecking required files...")
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """Check that core dependencies can be imported"""
    dependencies = [
        "redis",
        "cv2",
        "av",
        "librosa",
        "whisper",
        "transformers",
        "mediapipe",
        "numpy",
        "streamlit",
        "pytest",
        "hypothesis",
        "yaml",
    ]
    
    print("\nChecking dependencies...")
    all_imported = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError as e:
            print(f"  ✗ {dep} - FAILED: {e}")
            all_imported = False
    
    return all_imported


def check_config():
    """Check that configuration can be loaded"""
    print("\nChecking configuration...")
    try:
        from src.config.config_loader import config
        
        # Test basic config access
        redis_url = config.get('redis.url')
        fusion_interval = config.get('fusion.timer_interval')
        
        print(f"  ✓ Config loaded successfully")
        print(f"    - Redis URL: {redis_url}")
        print(f"    - Fusion interval: {fusion_interval}s")
        
        # Validate config
        config.validate()
        print(f"  ✓ Config validation passed")
        
        return True
    except Exception as e:
        print(f"  ✗ Config check failed: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 60)
    print("SentimentEngine Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Directory structure", check_directories),
        ("Required files", check_files),
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed! Project setup is complete.")
        print("\nNext steps:")
        print("  1. Download models: python scripts/download_models.py")
        print("  2. Start Redis: redis-server")
        print("  3. Begin implementing tasks from .kiro/specs/realtime-sentiment-analysis/tasks.md")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
