"""Labeling mode UI for template creation with symbol editor."""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QLineEdit, QMessageBox, QGroupBox, QSpinBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QFormLayout, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPixmap, QImage, QFont, QPen, QColor
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from ..utils.config import Config, RegionsConfig
from ..core.region_cutter import RegionCutter
from ..core.template_manager import TemplateManager
from ..core.symbol_splitter import SymbolSplitter
from ..utils.image_utils import load_image


class SymbolEditor(QGroupBox):
    """Editor for adjusting individual symbol boundaries."""

    def __init__(self, parent=None):
        """Initialize symbol editor."""
        super().__init__("Редактор символа", parent)

        self.symbol_index = -1
        self.value_changed_callback = None

        self.setup_ui()
        self.setVisible(False)

    def setup_ui(self):
        """Setup user interface."""
        layout = QVBoxLayout(self)

        # Symbol index label
        self.symbol_label = QLabel("")
        self.symbol_label.setAlignment(Qt.AlignCenter)
        font = self.symbol_label.font()
        font.setBold(True)
        self.symbol_label.setFont(font)
        layout.addWidget(self.symbol_label)

        # Coordinate editors
        form = QFormLayout()

        # X coordinate
        x_layout = QHBoxLayout()
        self.x_spin = QSpinBox()
        self.x_spin.setMinimum(0)
        self.x_spin.setMaximum(10000)
        self.x_spin.valueChanged.connect(self.on_value_changed)
        x_layout.addWidget(self.x_spin, stretch=1)

        x_dec = QPushButton("◄")
        x_dec.setMaximumWidth(30)
        x_dec.clicked.connect(lambda: self.x_spin.setValue(self.x_spin.value() - 1))
        x_layout.addWidget(x_dec)

        x_inc = QPushButton("►")
        x_inc.setMaximumWidth(30)
        x_inc.clicked.connect(lambda: self.x_spin.setValue(self.x_spin.value() + 1))
        x_layout.addWidget(x_inc)

        form.addRow("X:", x_layout)

        # Y coordinate
        y_layout = QHBoxLayout()
        self.y_spin = QSpinBox()
        self.y_spin.setMinimum(0)
        self.y_spin.setMaximum(10000)
        self.y_spin.valueChanged.connect(self.on_value_changed)
        y_layout.addWidget(self.y_spin, stretch=1)

        y_dec = QPushButton("◄")
        y_dec.setMaximumWidth(30)
        y_dec.clicked.connect(lambda: self.y_spin.setValue(self.y_spin.value() - 1))
        y_layout.addWidget(y_dec)

        y_inc = QPushButton("►")
        y_inc.setMaximumWidth(30)
        y_inc.clicked.connect(lambda: self.y_spin.setValue(self.y_spin.value() + 1))
        y_layout.addWidget(y_inc)

        form.addRow("Y:", y_layout)

        # Width
        w_layout = QHBoxLayout()
        self.w_spin = QSpinBox()
        self.w_spin.setMinimum(1)
        self.w_spin.setMaximum(10000)
        self.w_spin.valueChanged.connect(self.on_value_changed)
        w_layout.addWidget(self.w_spin, stretch=1)

        w_dec = QPushButton("◄")
        w_dec.setMaximumWidth(30)
        w_dec.clicked.connect(lambda: self.w_spin.setValue(self.w_spin.value() - 1))
        w_layout.addWidget(w_dec)

        w_inc = QPushButton("►")
        w_inc.setMaximumWidth(30)
        w_inc.clicked.connect(lambda: self.w_spin.setValue(self.w_spin.value() + 1))
        w_layout.addWidget(w_inc)

        form.addRow("W:", w_layout)

        # Height
        h_layout = QHBoxLayout()
        self.h_spin = QSpinBox()
        self.h_spin.setMinimum(1)
        self.h_spin.setMaximum(10000)
        self.h_spin.valueChanged.connect(self.on_value_changed)
        h_layout.addWidget(self.h_spin, stretch=1)

        h_dec = QPushButton("◄")
        h_dec.setMaximumWidth(30)
        h_dec.clicked.connect(lambda: self.h_spin.setValue(self.h_spin.value() - 1))
        h_layout.addWidget(h_dec)

        h_inc = QPushButton("►")
        h_inc.setMaximumWidth(30)
        h_inc.clicked.connect(lambda: self.h_spin.setValue(self.h_spin.value() + 1))
        h_layout.addWidget(h_inc)

        form.addRow("H:", h_layout)

        layout.addLayout(form)

    def set_value_changed_callback(self, callback):
        """Set callback for value changes."""
        self.value_changed_callback = callback

    def on_value_changed(self):
        """Called when any value changes."""
        if self.value_changed_callback:
            self.value_changed_callback(
                self.symbol_index,
                self.x_spin.value(),
                self.y_spin.value(),
                self.w_spin.value(),
                self.h_spin.value()
            )

    def load_symbol(self, index: int, x: int, y: int, w: int, h: int):
        """Load symbol for editing."""
        self.symbol_index = index
        self.symbol_label.setText(f"Символ #{index + 1}")

        # Block signals
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.w_spin.blockSignals(True)
        self.h_spin.blockSignals(True)

        self.x_spin.setValue(x)
        self.y_spin.setValue(y)
        self.w_spin.setValue(w)
        self.h_spin.setValue(h)

        # Unblock signals
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.w_spin.blockSignals(False)
        self.h_spin.blockSignals(False)

        self.setVisible(True)

    def hide_editor(self):
        """Hide editor."""
        self.symbol_index = -1
        self.setVisible(False)


class LabelingMode(QWidget):
    """Labeling mode widget for creating templates."""

    def __init__(self, config: Config, regions_config: RegionsConfig):
        """Initialize labeling mode."""
        super().__init__()

        self.config = config
        self.regions_config = regions_config
        self.region_cutter = RegionCutter(
            config.screenshots_dir,
            config.regions_cut_dir
        )
        self.template_manager = TemplateManager(config.templates_dir)
        self.symbol_splitter = SymbolSplitter()

        self.current_region_files = []
        self.current_file_index = -1
        self.current_image = None
        self.current_symbols = []
        self.symbol_rects = []  # List of (x, y, w, h) for each symbol
        self.selected_symbol_index = -1

        self.setup_ui()

    def setup_ui(self):
        """Setup user interface."""
        main_layout = QHBoxLayout(self)

        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)

        # Category selection
        category_group = QGroupBox("Выбор категории")
        category_layout = QVBoxLayout()

        self.category_combo = QComboBox()
        self.category_combo.addItem("Карты")
        self.category_combo.addItem("Комбинации")
        self.category_combo.addItem("Маркеры")
        self.category_combo.addItem("Текст и цифры")
        self.category_combo.currentIndexChanged.connect(self.on_category_changed)
        category_layout.addWidget(self.category_combo)

        load_btn = QPushButton("Загрузить файлы")
        load_btn.clicked.connect(self.load_region_files)
        category_layout.addWidget(load_btn)

        category_group.setLayout(category_layout)
        left_layout.addWidget(category_group)

        # Progress info
        self.progress_label = QLabel("Выберите категорию и загрузите файлы")
        self.progress_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(11)
        self.progress_label.setFont(font)
        left_layout.addWidget(self.progress_label)

        # Labeling interface
        label_group = QGroupBox("Разметка")
        self.label_layout = QVBoxLayout()

        self._setup_card_labeling()
        self._setup_combo_labeling()
        self._setup_marker_labeling()
        self._setup_text_labeling()

        label_group.setLayout(self.label_layout)
        left_layout.addWidget(label_group)

        # Symbol list (for text mode)
        self.symbol_list_group = QGroupBox("Символы")
        symbol_list_layout = QVBoxLayout()

        self.symbol_list = QListWidget()
        self.symbol_list.currentRowChanged.connect(self.on_symbol_selected)
        symbol_list_layout.addWidget(self.symbol_list)

        self.symbol_list_group.setLayout(symbol_list_layout)
        self.symbol_list_group.setVisible(False)
        left_layout.addWidget(self.symbol_list_group)

        # Symbol editor
        self.symbol_editor = SymbolEditor()
        self.symbol_editor.set_value_changed_callback(self.on_symbol_value_changed)
        left_layout.addWidget(self.symbol_editor)

        # Navigation
        nav_layout = QHBoxLayout()

        self.prev_btn = QPushButton("◄ Назад")
        self.prev_btn.clicked.connect(self.prev_file)
        self.prev_btn.setEnabled(False)

        self.skip_btn = QPushButton("Пропустить")
        self.skip_btn.clicked.connect(self.next_file)
        self.skip_btn.setEnabled(False)

        self.next_btn = QPushButton("Вперёд ►")
        self.next_btn.clicked.connect(self.next_file)
        self.next_btn.setEnabled(False)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.skip_btn)
        nav_layout.addWidget(self.next_btn)

        left_layout.addLayout(nav_layout)

        # Statistics
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        left_layout.addWidget(self.stats_label)

        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        # Right panel - image viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        viewer_group = QGroupBox("Текущий регион")
        viewer_layout = QVBoxLayout()

        self.image_viewer = QGraphicsView()
        self.image_scene = QGraphicsScene()
        self.image_viewer.setScene(self.image_scene)
        self.image_viewer.setMinimumHeight(400)
        viewer_layout.addWidget(self.image_viewer)

        viewer_group.setLayout(viewer_layout)
        right_layout.addWidget(viewer_group)

        main_layout.addWidget(right_panel, stretch=1)

        # Initialize
        self.on_category_changed(0)
        self.update_statistics()

    def _setup_card_labeling(self):
        """Setup card labeling interface."""
        self.card_widget = QWidget()
        card_layout = QVBoxLayout(self.card_widget)

        # Suit reference
        suit_ref = QLabel(
            "<b>Шпаргалка по мастям:</b><br>"
            "♣ Трефы = <b>c</b> (clubs)<br>"
            "♦ Бубны = <b>d</b> (diamonds)<br>"
            "♥ Черви = <b>h</b> (hearts)<br>"
            "♠ Пики = <b>s</b> (spades)"
        )
        suit_ref.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: 1px solid #ccc;")
        card_layout.addWidget(suit_ref)

        info = QLabel("Введите значение карты: ранг + масть (напр.: '2c', 'Ah', 'Ks')")
        card_layout.addWidget(info)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Карта:"))

        self.card_input = QLineEdit()
        self.card_input.setMaxLength(2)
        self.card_input.setPlaceholderText("2c")
        self.card_input.returnPressed.connect(self.save_card_label)
        input_layout.addWidget(self.card_input)

        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self.save_card_label)
        input_layout.addWidget(save_btn)

        card_layout.addLayout(input_layout)

        self.card_widget.hide()
        self.label_layout.addWidget(self.card_widget)

    def _setup_combo_labeling(self):
        """Setup combo labeling interface."""
        self.combo_widget = QWidget()
        combo_layout = QVBoxLayout(self.combo_widget)

        info = QLabel("Выберите тип комбинации:")
        combo_layout.addWidget(info)

        self.combo_type_combo = QComboBox()
        for combo_name in self.template_manager.COMBO_NAMES:
            display_name = combo_name.replace('_', ' ').title()
            self.combo_type_combo.addItem(display_name, combo_name)
        combo_layout.addWidget(self.combo_type_combo)

        save_btn = QPushButton("Сохранить шаблон комбинации")
        save_btn.clicked.connect(self.save_combo_label)
        combo_layout.addWidget(save_btn)

        self.combo_widget.hide()
        self.label_layout.addWidget(self.combo_widget)

    def _setup_marker_labeling(self):
        """Setup marker labeling interface."""
        self.marker_widget = QWidget()
        marker_layout = QVBoxLayout(self.marker_widget)

        info = QLabel("Присутствует ли маркер на изображении?")
        marker_layout.addWidget(info)

        btn_layout = QHBoxLayout()

        yes_btn = QPushButton("Да - Сохранить как шаблон")
        yes_btn.clicked.connect(self.save_marker_label)
        btn_layout.addWidget(yes_btn)

        no_btn = QPushButton("Нет - Пропустить")
        no_btn.clicked.connect(self.next_file)
        btn_layout.addWidget(no_btn)

        marker_layout.addLayout(btn_layout)

        self.marker_widget.hide()
        self.label_layout.addWidget(self.marker_widget)

    def _setup_text_labeling(self):
        """Setup text labeling interface."""
        self.text_widget = QWidget()
        text_layout = QVBoxLayout(self.text_widget)

        info = QLabel("Введите текст, показанный на изображении:")
        text_layout.addWidget(info)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Текст:"))

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("2.50")
        self.text_input.returnPressed.connect(self.analyze_text)
        input_layout.addWidget(self.text_input)

        analyze_btn = QPushButton("Разбить на символы")
        analyze_btn.clicked.connect(self.analyze_text)
        input_layout.addWidget(analyze_btn)

        text_layout.addLayout(input_layout)

        # Info label
        self.symbol_info_label = QLabel()
        self.symbol_info_label.setWordWrap(True)
        text_layout.addWidget(self.symbol_info_label)

        # Save button
        self.text_save_btn = QPushButton("Сохранить все символы")
        self.text_save_btn.clicked.connect(self.save_text_label)
        self.text_save_btn.setEnabled(False)
        text_layout.addWidget(self.text_save_btn)

        self.text_widget.hide()
        self.label_layout.addWidget(self.text_widget)

    def on_category_changed(self, index: int):
        """Handle category selection change."""
        # Hide all labeling widgets
        self.card_widget.hide()
        self.combo_widget.hide()
        self.marker_widget.hide()
        self.text_widget.hide()
        self.symbol_list_group.hide()
        self.symbol_editor.hide_editor()

        # Show appropriate widget
        if index == 0:  # Cards
            self.card_widget.show()
        elif index == 1:  # Combos
            self.combo_widget.show()
        elif index == 2:  # Markers
            self.marker_widget.show()
        elif index == 3:  # Text
            self.text_widget.show()

    def load_region_files(self):
        """Load all region files for selected category."""
        category_index = self.category_combo.currentIndex()

        # Get all regions for category
        if category_index == 0:  # Cards
            region_types = ['card']
        elif category_index == 1:  # Combos
            region_types = ['combo']
        elif category_index == 2:  # Markers
            region_types = ['marker']
        elif category_index == 3:  # Text
            region_types = ['text_digits', 'text_mixed']
        else:
            return

        # Collect all files from all regions of these types
        all_files = []
        for region_type in region_types:
            regions = self.regions_config.get_regions_by_type(region_type)
            for region_id in regions.keys():
                files = self.region_cutter.get_region_files(region_id)
                all_files.extend(files)

        if not all_files:
            QMessageBox.information(
                self, "Нет файлов",
                f"Нарезки для выбранной категории не найдены.\n"
                "Сначала нарежьте регионы в режиме Нарезки."
            )
            return

        self.current_region_files = all_files
        self.current_file_index = 0
        self.load_current_file()

        self.prev_btn.setEnabled(True)
        self.skip_btn.setEnabled(True)
        self.next_btn.setEnabled(True)

        self.update_progress()

    def load_current_file(self):
        """Load current file into viewer."""
        if self.current_file_index < 0 or self.current_file_index >= len(self.current_region_files):
            return

        filepath = self.current_region_files[self.current_file_index]
        self.current_image = load_image(filepath)

        if self.current_image is not None:
            self.display_image(self.current_image)

            # For text, auto-split symbols
            category_index = self.category_combo.currentIndex()
            if category_index == 3:  # Text
                self.split_symbols()

    def display_image(self, image: np.ndarray, scale: float = 3.0):
        """Display image in viewer."""
        self.image_scene.clear()

        # Resize for display
        display_image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST
        )

        # Convert to QPixmap
        height, width = display_image.shape[:2]
        if len(display_image.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        else:
            bytes_per_line = width
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)
        self.image_scene.addPixmap(pixmap)
        self.image_viewer.fitInView(self.image_scene.sceneRect(), Qt.KeepAspectRatio)

    def split_symbols(self):
        """Split image into symbols."""
        if self.current_image is None:
            return

        self.current_symbols = self.symbol_splitter.split_to_symbols(self.current_image)

        # Extract bounding boxes
        self.symbol_rects = []
        for _, bbox in self.current_symbols:
            self.symbol_rects.append(bbox)

        info = f"Автоматически обнаружено символов: {len(self.current_symbols)}"
        self.symbol_info_label.setText(info)

        # Clear text input
        self.text_input.clear()
        self.text_save_btn.setEnabled(False)

    def analyze_text(self):
        """Analyze text input and match with symbols."""
        text = self.text_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Неверный ввод", "Введите текст")
            return

        if not self.current_symbols:
            QMessageBox.warning(self, "Ошибка", "Символы не обнаружены")
            return

        # Check if length matches
        if len(text) != len(self.current_symbols):
            QMessageBox.warning(
                self, "Несовпадение",
                f"Длина ввода ({len(text)}) не совпадает с обнаруженными символами ({len(self.current_symbols)})\n"
                "Скорректируйте границы символов вручную."
            )
            return

        # Show symbol list
        self.symbol_list_group.show()
        self.symbol_list.clear()

        for i, char in enumerate(text):
            item = QListWidgetItem(f"{i+1}. '{char}'")
            self.symbol_list.addItem(item)

        # Draw symbols on image
        self.draw_symbol_boxes(scale=3.0)

        self.text_save_btn.setEnabled(True)

    def draw_symbol_boxes(self, scale: float = 3.0, selected_index: int = -1):
        """Draw symbol bounding boxes on image."""
        if self.current_image is None:
            return

        # Recreate display
        display_image = cv2.resize(
            self.current_image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST
        )

        # Draw on display image
        for i, (x, y, w, h) in enumerate(self.symbol_rects):
            # Scale coordinates
            sx = int(x * scale)
            sy = int(y * scale)
            sw = int(w * scale)
            sh = int(h * scale)

            # Color: yellow for selected, green for others
            if i == selected_index:
                color = (255, 255, 0)  # Yellow
                thickness = 2
            else:
                color = (0, 255, 0)  # Green
                thickness = 1

            cv2.rectangle(display_image, (sx, sy), (sx + sw, sy + sh), color, thickness)

            # Draw number
            cv2.putText(display_image, str(i + 1), (sx, sy - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        # Display
        height, width = display_image.shape[:2]
        if len(display_image.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        else:
            bytes_per_line = width
            q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        self.image_scene.clear()
        pixmap = QPixmap.fromImage(q_image)
        self.image_scene.addPixmap(pixmap)
        self.image_viewer.fitInView(self.image_scene.sceneRect(), Qt.KeepAspectRatio)

    def on_symbol_selected(self, index: int):
        """Handle symbol selection from list."""
        if index < 0 or index >= len(self.symbol_rects):
            self.symbol_editor.hide_editor()
            self.selected_symbol_index = -1
            return

        self.selected_symbol_index = index
        x, y, w, h = self.symbol_rects[index]
        self.symbol_editor.load_symbol(index, x, y, w, h)

        # Redraw with selected symbol highlighted
        self.draw_symbol_boxes(scale=3.0, selected_index=index)

    def on_symbol_value_changed(self, index: int, x: int, y: int, w: int, h: int):
        """Handle symbol boundary change."""
        if 0 <= index < len(self.symbol_rects):
            self.symbol_rects[index] = (x, y, w, h)
            self.draw_symbol_boxes(scale=3.0, selected_index=index)

    def prev_file(self):
        """Go to previous file."""
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_current_file()
            self.update_progress()

    def next_file(self):
        """Go to next file."""
        if self.current_file_index < len(self.current_region_files) - 1:
            self.current_file_index += 1
            self.load_current_file()
            self.update_progress()

    def update_progress(self):
        """Update progress label."""
        if not self.current_region_files:
            self.progress_label.setText("Файлы не загружены")
            return

        current = self.current_file_index + 1
        total = len(self.current_region_files)
        category = self.category_combo.currentText()

        self.progress_label.setText(f"{category}: {current} / {total}")

    def save_card_label(self):
        """Save card label."""
        if self.current_image is None:
            return

        card_value = self.card_input.text().strip()
        if len(card_value) != 2:
            QMessageBox.warning(self, "Неверный ввод", "Значение карты должно быть 2 символа (напр.: '2c')")
            return

        rank = card_value[0].upper()
        suit = card_value[1].lower()

        if rank not in self.template_manager.CARD_RANKS:
            QMessageBox.warning(
                self, "Неверный ранг",
                f"Ранг должен быть одним из: {', '.join(self.template_manager.CARD_RANKS)}"
            )
            return

        if suit not in self.template_manager.CARD_SUITS:
            QMessageBox.warning(
                self, "Неверная масть",
                f"Масть должна быть одной из: c, d, h, s"
            )
            return

        # Check if exists
        if self.template_manager.template_exists('cards', f"{rank}{suit}"):
            reply = QMessageBox.question(
                self, "Заменить?",
                f"Шаблон для {rank}{suit} уже существует. Заменить?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Save template
        result = self.template_manager.save_card_template(self.current_image, rank, suit)

        if result:
            QMessageBox.information(self, "Сохранено", f"Сохранён шаблон для {rank}{suit}")
            self.card_input.clear()
            self.next_file()
            self.update_statistics()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить шаблон")

    def save_combo_label(self):
        """Save combo label."""
        if self.current_image is None:
            return

        combo_name = self.combo_type_combo.currentData()

        if self.template_manager.template_exists('combos', combo_name):
            reply = QMessageBox.question(
                self, "Заменить?",
                f"Шаблон для {combo_name} уже существует. Заменить?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        result = self.template_manager.save_combo_template(self.current_image, combo_name)

        if result:
            QMessageBox.information(self, "Сохранено", f"Сохранён шаблон для {combo_name}")
            self.next_file()
            self.update_statistics()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить шаблон")

    def save_marker_label(self):
        """Save marker label."""
        if self.current_image is None:
            return

        # Try to determine marker type from filename
        filepath = self.current_region_files[self.current_file_index]
        filename = filepath.name

        if 'dealer' in filename:
            marker_name = 'dealer_button'
        elif 'timer' in filename:
            marker_name = 'timer'
        elif 'seat' in filename:
            marker_name = 'seat_occupied'
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось определить тип маркера")
            return

        if self.template_manager.template_exists('markers', marker_name):
            reply = QMessageBox.question(
                self, "Заменить?",
                f"Шаблон для {marker_name} уже существует. Заменить?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        result = self.template_manager.save_marker_template(self.current_image, marker_name)

        if result:
            QMessageBox.information(self, "Сохранено", f"Сохранён шаблон для {marker_name}")
            self.next_file()
            self.update_statistics()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить шаблон")

    def save_text_label(self):
        """Save text label."""
        text = self.text_input.text().strip()
        if not text or not self.current_symbols:
            QMessageBox.warning(self, "Ошибка", "Нет текста или символов")
            return

        if len(text) != len(self.symbol_rects):
            QMessageBox.warning(self, "Несовпадение", "Количество символов не совпадает")
            return

        # Save each symbol with adjusted boundaries
        saved_count = 0

        for i, char in enumerate(text):
            x, y, w, h = self.symbol_rects[i]

            # Crop symbol from original image
            symbol_img = self.current_image[y:y+h, x:x+w].copy()

            # Determine category and save
            if char.isdigit() or char == '.':
                result = self.template_manager.save_digit_template(symbol_img, char)
            elif char.isalpha():
                if ord(char) < 128:
                    category = 'letters_lat'
                else:
                    category = 'letters_cyr'
                result = self.template_manager.save_symbol_template(symbol_img, char, category)
            else:
                result = self.template_manager.save_symbol_template(symbol_img, char, 'special')

            if result:
                saved_count += 1

        QMessageBox.information(self, "Сохранено", f"Сохранено {saved_count} шаблонов символов")
        self.text_input.clear()
        self.symbol_list.clear()
        self.symbol_list_group.hide()
        self.symbol_editor.hide_editor()
        self.text_save_btn.setEnabled(False)
        self.next_file()
        self.update_statistics()

    def update_statistics(self):
        """Update template statistics."""
        stats = self.template_manager.get_statistics()

        cards_existing, cards_total = self.template_manager.get_cards_completion()

        stats_text = "Статистика шаблонов:\n"
        stats_text += f"  Карты: {cards_existing}/{cards_total}\n"
        stats_text += f"  Цифры: {stats.get('digits', 0)}\n"
        stats_text += f"  Комбинации: {stats.get('combos', 0)}/{len(self.template_manager.COMBO_NAMES)}\n"
        stats_text += f"  Маркеры: {stats.get('markers', 0)}/{len(self.template_manager.MARKER_NAMES)}\n"
        stats_text += f"  Буквы (лат.): {stats.get('letters_lat', 0)}\n"
        stats_text += f"  Буквы (кир.): {stats.get('letters_cyr', 0)}\n"
        stats_text += f"  Спецсимволы: {stats.get('special', 0)}\n"

        self.stats_label.setText(stats_text)

    def on_mode_activated(self):
        """Called when this mode is activated."""
        self.update_statistics()

    def on_config_changed(self):
        """Called when configuration changes."""
        self.region_cutter = RegionCutter(
            self.config.screenshots_dir,
            self.config.regions_cut_dir
        )
        self.template_manager = TemplateManager(self.config.templates_dir)
