"""Settings dialog."""
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QSpinBox, QPushButton, QDialogButtonBox
)
from PyQt5.QtCore import Qt

from ..utils.config import Config


class SettingsDialog(QDialog):
    """Settings configuration dialog."""

    def __init__(self, config: Config, parent=None):
        """Initialize settings dialog.

        Args:
            config: Configuration object
            parent: Parent widget
        """
        super().__init__(parent)

        self.config = config
        self.setup_ui()
        self.load_settings()

    def setup_ui(self):
        """Setup user interface."""
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Form layout
        form = QFormLayout()

        # Window title
        self.window_title_edit = QLineEdit()
        form.addRow("Game Window Title:", self.window_title_edit)

        # Window size
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setMinimum(100)
        self.window_width_spin.setMaximum(3840)
        self.window_width_spin.setSingleStep(10)
        form.addRow("Window Width:", self.window_width_spin)

        self.window_height_spin = QSpinBox()
        self.window_height_spin.setMinimum(100)
        self.window_height_spin.setMaximum(2160)
        self.window_height_spin.setSingleStep(10)
        form.addRow("Window Height:", self.window_height_spin)

        # Screenshot interval
        self.interval_spin = QSpinBox()
        self.interval_spin.setMinimum(500)
        self.interval_spin.setMaximum(10000)
        self.interval_spin.setSingleStep(500)
        self.interval_spin.setSuffix(" ms")
        form.addRow("Screenshot Interval:", self.interval_spin)

        layout.addLayout(form)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def load_settings(self):
        """Load settings from config."""
        self.window_title_edit.setText(self.config.window_title)
        self.window_width_spin.setValue(self.config.game_window_width)
        self.window_height_spin.setValue(self.config.game_window_height)
        self.interval_spin.setValue(self.config.screenshot_interval_ms)

    def accept(self):
        """Accept and save settings."""
        self.config.set('window_title', self.window_title_edit.text())
        self.config.set('game_window_width', self.window_width_spin.value())
        self.config.set('game_window_height', self.window_height_spin.value())
        self.config.set('screenshot_interval_ms', self.interval_spin.value())

        self.config.save()

        super().accept()
