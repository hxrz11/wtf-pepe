"""Main window for Poker Vision Tool."""
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QLabel, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from pathlib import Path

from ..utils.config import Config, RegionsConfig
from ..core.window_manager import WindowManager
from .screenshot_mode import ScreenshotMode
from .cutting_mode import CuttingMode
from .labeling_mode import LabelingMode
from .settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        """Initialize main window."""
        super().__init__()

        # Load configuration
        self.config = Config()
        self.regions_config = RegionsConfig()

        # Window manager for game window
        self.window_manager = WindowManager(self.config.window_title)

        # Setup UI
        self.setup_ui()

        # Timer for window positioning
        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self.update_window_position)
        self.position_timer.start(500)  # Check every 500ms

    def setup_ui(self):
        """Setup user interface."""
        self.setWindowTitle("Poker Vision Tool")
        self.setMinimumSize(400, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Title
        title_label = QLabel("Poker Vision Tool")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Mode selection buttons
        mode_layout = QHBoxLayout()

        self.screenshot_btn = QPushButton("Скриншоты")
        self.screenshot_btn.clicked.connect(lambda: self.switch_mode(0))

        self.cutting_btn = QPushButton("Нарезка")
        self.cutting_btn.clicked.connect(lambda: self.switch_mode(1))

        self.labeling_btn = QPushButton("Разметка")
        self.labeling_btn.clicked.connect(lambda: self.switch_mode(2))

        mode_layout.addWidget(self.screenshot_btn)
        mode_layout.addWidget(self.cutting_btn)
        mode_layout.addWidget(self.labeling_btn)

        main_layout.addLayout(mode_layout)

        # Stacked widget for different modes
        self.stack = QStackedWidget()

        # Create mode widgets
        self.screenshot_mode = ScreenshotMode(self.config, self.window_manager)
        self.cutting_mode = CuttingMode(self.config, self.regions_config)
        self.labeling_mode = LabelingMode(self.config, self.regions_config)

        self.stack.addWidget(self.screenshot_mode)
        self.stack.addWidget(self.cutting_mode)
        self.stack.addWidget(self.labeling_mode)

        main_layout.addWidget(self.stack)

        # Settings button
        settings_btn = QPushButton("Настройки")
        settings_btn.clicked.connect(self.open_settings)
        main_layout.addWidget(settings_btn)

        # Status bar
        self.statusBar().showMessage("Готов")

        # Set initial mode
        self.switch_mode(0)

    def switch_mode(self, mode_index: int):
        """Switch to different mode.

        Args:
            mode_index: Mode index (0=screenshot, 1=cutting, 2=labeling)
        """
        self.stack.setCurrentIndex(mode_index)

        # Update button states
        buttons = [self.screenshot_btn, self.cutting_btn, self.labeling_btn]
        for i, btn in enumerate(buttons):
            btn.setEnabled(i != mode_index)

        # Update status
        mode_names = ["Режим скриншотов", "Режим нарезки", "Режим разметки"]
        self.statusBar().showMessage(mode_names[mode_index])

        # Notify mode widgets
        current_widget = self.stack.currentWidget()
        if hasattr(current_widget, 'on_mode_activated'):
            current_widget.on_mode_activated()

    def update_window_position(self):
        """Update window position to stick to game window."""
        # Only in screenshot mode
        if self.stack.currentIndex() != 0:
            return

        if not self.window_manager.is_window_valid():
            return

        # Get game window position
        game_rect = self.window_manager.get_client_rect()
        if not game_rect:
            return

        game_x, game_y, game_w, game_h = game_rect

        # Position our window to the right of game window
        our_width = self.width()
        our_height = self.height()

        new_x = game_x + game_w + 5
        new_y = game_y

        # Move window
        self.move(new_x, new_y)

    def open_settings(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self.config, self)
        if dialog.exec_():
            # Reload config
            self.config.load()

            # Update window manager
            self.window_manager = WindowManager(self.config.window_title)

            # Notify modes of config change
            self.screenshot_mode.on_config_changed()
            self.cutting_mode.on_config_changed()
            self.labeling_mode.on_config_changed()

    def closeEvent(self, event):
        """Handle window close event."""
        # Stop position timer
        self.position_timer.stop()

        # Stop screenshot mode if running
        if hasattr(self.screenshot_mode, 'stop_capture'):
            self.screenshot_mode.stop_capture()

        event.accept()
