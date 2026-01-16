"""Screenshot capture mode UI."""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSpinBox, QMessageBox, QGroupBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
from pathlib import Path

from ..utils.config import Config
from ..core.window_manager import WindowManager
from ..core.screenshot import ScreenshotCapture


class ScreenshotMode(QWidget):
    """Screenshot capture mode widget."""

    def __init__(self, config: Config, window_manager: WindowManager):
        """Initialize screenshot mode.

        Args:
            config: Application configuration
            window_manager: Window manager instance
        """
        super().__init__()

        self.config = config
        self.window_manager = window_manager
        self.screenshot_capture = ScreenshotCapture(config.screenshots_dir)

        self.is_capturing = False
        self.screenshot_count = 0

        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_screenshot)

        self.setup_ui()

    def setup_ui(self):
        """Setup user interface."""
        layout = QVBoxLayout(self)

        # Game window status
        status_group = QGroupBox("Game Window Status")
        status_layout = QVBoxLayout()

        self.window_status_label = QLabel("Not found")
        self.window_status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(10)
        self.window_status_label.setFont(font)
        status_layout.addWidget(self.window_status_label)

        find_window_btn = QPushButton("Find Window")
        find_window_btn.clicked.connect(self.find_game_window)
        status_layout.addWidget(find_window_btn)

        set_size_btn = QPushButton("Set Window Size")
        set_size_btn.clicked.connect(self.set_window_size)
        status_layout.addWidget(set_size_btn)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Capture settings
        settings_group = QGroupBox("Capture Settings")
        settings_layout = QVBoxLayout()

        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Interval (ms):"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setMinimum(500)
        self.interval_spin.setMaximum(10000)
        self.interval_spin.setValue(self.config.screenshot_interval_ms)
        self.interval_spin.setSingleStep(500)
        interval_layout.addWidget(self.interval_spin)
        settings_layout.addLayout(interval_layout)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # Capture control
        control_group = QGroupBox("Capture Control")
        control_layout = QVBoxLayout()

        self.start_stop_btn = QPushButton("Start Capture")
        self.start_stop_btn.clicked.connect(self.toggle_capture)
        self.start_stop_btn.setEnabled(False)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.start_stop_btn.setFont(font)
        control_layout.addWidget(self.start_stop_btn)

        self.count_label = QLabel("Screenshots: 0")
        self.count_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(14)
        self.count_label.setFont(font)
        control_layout.addWidget(self.count_label)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # Info
        info_label = QLabel(
            "1. Find game window\n"
            "2. Set window size\n"
            "3. Start capture\n\n"
            "Screenshots saved to: screenshots/"
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        layout.addStretch()

        # Update window status timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_window_status)
        self.status_timer.start(1000)

    def find_game_window(self):
        """Find game window."""
        if self.window_manager.find_window():
            QMessageBox.information(self, "Success", "Game window found!")
            self.update_window_status()
        else:
            QMessageBox.warning(
                self, "Not Found",
                f"Could not find window with title: {self.config.window_title}"
            )

    def set_window_size(self):
        """Set game window size."""
        if not self.window_manager.is_window_valid():
            QMessageBox.warning(self, "Error", "Game window not found!")
            return

        success = self.window_manager.set_window_size(
            self.config.game_window_width,
            self.config.game_window_height
        )

        if success:
            QMessageBox.information(
                self, "Success",
                f"Window size set to {self.config.game_window_width}x{self.config.game_window_height}"
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to set window size!")

    def update_window_status(self):
        """Update window status display."""
        if self.window_manager.is_window_valid():
            rect = self.window_manager.get_client_rect()
            if rect:
                x, y, w, h = rect
                self.window_status_label.setText(
                    f"Found: {w}x{h}\n({x}, {y})"
                )
                self.window_status_label.setStyleSheet("color: green;")
                self.start_stop_btn.setEnabled(True)
            else:
                self.window_status_label.setText("Found (no info)")
                self.window_status_label.setStyleSheet("color: orange;")
        else:
            self.window_status_label.setText("Not found")
            self.window_status_label.setStyleSheet("color: red;")
            self.start_stop_btn.setEnabled(False)
            if self.is_capturing:
                self.stop_capture()

    def toggle_capture(self):
        """Toggle screenshot capture."""
        if self.is_capturing:
            self.stop_capture()
        else:
            self.start_capture()

    def start_capture(self):
        """Start screenshot capture."""
        if not self.window_manager.is_window_valid():
            QMessageBox.warning(self, "Error", "Game window not found!")
            return

        self.is_capturing = True
        self.screenshot_count = 0

        interval = self.interval_spin.value()
        self.capture_timer.start(interval)

        self.start_stop_btn.setText("Stop Capture")
        self.start_stop_btn.setStyleSheet("background-color: #ff4444;")
        self.interval_spin.setEnabled(False)

    def stop_capture(self):
        """Stop screenshot capture."""
        self.is_capturing = False
        self.capture_timer.stop()

        self.start_stop_btn.setText("Start Capture")
        self.start_stop_btn.setStyleSheet("")
        self.interval_spin.setEnabled(True)

    def capture_screenshot(self):
        """Capture single screenshot."""
        if not self.window_manager.hwnd:
            return

        filepath = self.screenshot_capture.capture_and_save(self.window_manager.hwnd)

        if filepath:
            self.screenshot_count += 1
            self.count_label.setText(f"Screenshots: {self.screenshot_count}")

    def on_mode_activated(self):
        """Called when this mode is activated."""
        self.update_window_status()

    def on_config_changed(self):
        """Called when configuration changes."""
        self.interval_spin.setValue(self.config.screenshot_interval_ms)
        self.screenshot_capture = ScreenshotCapture(self.config.screenshots_dir)
