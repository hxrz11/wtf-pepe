"""
Poker Vision Tool - Template Creation Tool for Poker Interface Recognition

This application helps create template images for recognizing poker game UI elements
using template matching. It has three main modes:

1. Screenshot Mode - Capture screenshots from the game window
2. Cutting Mode - Cut regions from screenshots
3. Labeling Mode - Label cut regions to create templates

Usage:
    python main.py

Requirements:
    - Python 3.10+
    - PyQt5
    - OpenCV (cv2)
    - NumPy
    - Pillow
    - pywin32 (Windows only)

Author: Poker Vision Tool
"""

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt

from src.ui.main_window import MainWindow


def check_directories():
    """Check and create required directories."""
    required_dirs = [
        'screenshots',
        'regions_cut',
        'templates',
        'templates/cards',
        'templates/digits',
        'templates/combos',
        'templates/letters_lat',
        'templates/letters_cyr',
        'templates/special',
        'templates/markers'
    ]

    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def check_config_files():
    """Check required configuration files exist."""
    config_path = Path('config.json')
    regions_path = Path('regions.json')

    if not config_path.exists():
        print("ERROR: config.json not found!")
        print("Please create config.json in the application directory.")
        return False

    if not regions_path.exists():
        print("ERROR: regions.json not found!")
        print("Please create regions.json in the application directory.")
        return False

    return True


def main():
    """Main entry point."""
    # Check configuration
    if not check_config_files():
        sys.exit(1)

    # Check/create directories
    check_directories()

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("Poker Vision Tool")

    # Enable high DPI support
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Platform check warning
    if sys.platform != 'win32':
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Platform Warning")
        msg.setText(
            "This application is designed for Windows.\n\n"
            "Window capture features will not work on other platforms.\n"
            "You can still use Cutting and Labeling modes with existing screenshots."
        )
        msg.exec_()

    # Create and show main window
    try:
        window = MainWindow()
        window.show()

        # Run application
        sys.exit(app.exec_())

    except Exception as e:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Fatal Error")
        msg.setText(f"Failed to start application:\n\n{str(e)}")
        msg.exec_()
        sys.exit(1)


if __name__ == '__main__':
    main()
