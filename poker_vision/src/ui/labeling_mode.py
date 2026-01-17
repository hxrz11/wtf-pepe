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
from ..utils.image_utils import load_image


class DraggableSymbolRect(QGraphicsRectItem):
    """Draggable rectangle for manual symbol selection."""

    def __init__(self, x: int, y: int, w: int, h: int, index: int, parent=None):
        super().__init__(x, y, w, h, parent)
        self.index = index
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)

        # Set appearance
        pen = QPen(QColor(0, 255, 0), 2)  # Green border
        self.setPen(pen)
        self.setBrush(QColor(0, 255, 0, 50))  # Semi-transparent green fill

    def itemChange(self, change, value):
        """Handle item changes."""
        if change == QGraphicsRectItem.ItemSelectedHasChanged:
            if self.isSelected():
                pen = QPen(QColor(255, 255, 0), 3)  # Yellow when selected
                self.setPen(pen)
                self.setBrush(QColor(255, 255, 0, 50))
            else:
                pen = QPen(QColor(0, 255, 0), 2)  # Green when not selected
                self.setPen(pen)
                self.setBrush(QColor(0, 255, 0, 50))
        return super().itemChange(change, value)


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

        self.current_region_files = []
        self.current_file_index = -1
        self.current_image = None
        self.current_text = ""  # Text entered by user
        self.symbol_rects = []  # List of DraggableSymbolRect items
        self.selected_symbol_index = -1
        self.display_scale = 3.0  # Scale for image display

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

        self.delete_btn = QPushButton("Удалить регион")
        self.delete_btn.clicked.connect(self.delete_current_file)
        self.delete_btn.setEnabled(False)
        self.delete_btn.setStyleSheet("background-color: #ffcccc;")

        self.next_btn = QPushButton("Вперёд ►")
        self.next_btn.clicked.connect(self.next_file)
        self.next_btn.setEnabled(False)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.skip_btn)
        nav_layout.addWidget(self.delete_btn)
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

        info = QLabel("Введите текст и выделите каждый символ вручную:")
        text_layout.addWidget(info)

        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Текст:"))

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("2.50")
        self.text_input.returnPressed.connect(self.create_symbol_regions)
        input_layout.addWidget(self.text_input)

        create_btn = QPushButton("Создать регионы")
        create_btn.clicked.connect(self.create_symbol_regions)
        input_layout.addWidget(create_btn)

        text_layout.addLayout(input_layout)

        # Info label
        self.symbol_info_label = QLabel()
        self.symbol_info_label.setWordWrap(True)
        text_layout.addWidget(self.symbol_info_label)

        # Default size controls
        default_size_group = QGroupBox("Размер по умолчанию")
        default_layout = QVBoxLayout()

        size_layout = QFormLayout()

        # Width control
        self.symbol_width_spin = QSpinBox()
        self.symbol_width_spin.setMinimum(1)
        self.symbol_width_spin.setMaximum(200)
        self.symbol_width_spin.setValue(15)
        size_layout.addRow("Ширина:", self.symbol_width_spin)

        # Height control
        self.symbol_height_spin = QSpinBox()
        self.symbol_height_spin.setMinimum(1)
        self.symbol_height_spin.setMaximum(200)
        self.symbol_height_spin.setValue(20)
        size_layout.addRow("Высота:", self.symbol_height_spin)

        default_layout.addLayout(size_layout)

        # Apply to all button
        apply_all_btn = QPushButton("Применить ко всем")
        apply_all_btn.clicked.connect(self.apply_symbol_size_to_all)
        default_layout.addWidget(apply_all_btn)

        default_size_group.setLayout(default_layout)
        text_layout.addWidget(default_size_group)

        # Individual symbol editor
        self.symbol_edit_group = QGroupBox("Редактирование выбранного символа")
        symbol_edit_layout = QVBoxLayout()

        self.selected_symbol_label = QLabel("Выберите символ из списка")
        self.selected_symbol_label.setAlignment(Qt.AlignCenter)
        font = self.selected_symbol_label.font()
        font.setBold(True)
        self.selected_symbol_label.setFont(font)
        symbol_edit_layout.addWidget(self.selected_symbol_label)

        edit_form = QFormLayout()

        # X
        x_layout = QHBoxLayout()
        self.symbol_x_spin = QSpinBox()
        self.symbol_x_spin.setMinimum(0)
        self.symbol_x_spin.setMaximum(10000)
        self.symbol_x_spin.valueChanged.connect(self.on_selected_symbol_changed)
        x_layout.addWidget(self.symbol_x_spin, stretch=1)
        x_dec = QPushButton("◄")
        x_dec.setMaximumWidth(30)
        x_dec.clicked.connect(lambda: self.symbol_x_spin.setValue(self.symbol_x_spin.value() - 1))
        x_layout.addWidget(x_dec)
        x_inc = QPushButton("►")
        x_inc.setMaximumWidth(30)
        x_inc.clicked.connect(lambda: self.symbol_x_spin.setValue(self.symbol_x_spin.value() + 1))
        x_layout.addWidget(x_inc)
        edit_form.addRow("X:", x_layout)

        # Y
        y_layout = QHBoxLayout()
        self.symbol_y_spin = QSpinBox()
        self.symbol_y_spin.setMinimum(0)
        self.symbol_y_spin.setMaximum(10000)
        self.symbol_y_spin.valueChanged.connect(self.on_selected_symbol_changed)
        y_layout.addWidget(self.symbol_y_spin, stretch=1)
        y_dec = QPushButton("◄")
        y_dec.setMaximumWidth(30)
        y_dec.clicked.connect(lambda: self.symbol_y_spin.setValue(self.symbol_y_spin.value() - 1))
        y_layout.addWidget(y_dec)
        y_inc = QPushButton("►")
        y_inc.setMaximumWidth(30)
        y_inc.clicked.connect(lambda: self.symbol_y_spin.setValue(self.symbol_y_spin.value() + 1))
        y_layout.addWidget(y_inc)
        edit_form.addRow("Y:", y_layout)

        # Width
        w_layout = QHBoxLayout()
        self.symbol_w_spin = QSpinBox()
        self.symbol_w_spin.setMinimum(1)
        self.symbol_w_spin.setMaximum(10000)
        self.symbol_w_spin.valueChanged.connect(self.on_selected_symbol_changed)
        w_layout.addWidget(self.symbol_w_spin, stretch=1)
        w_dec = QPushButton("◄")
        w_dec.setMaximumWidth(30)
        w_dec.clicked.connect(lambda: self.symbol_w_spin.setValue(self.symbol_w_spin.value() - 1))
        w_layout.addWidget(w_dec)
        w_inc = QPushButton("►")
        w_inc.setMaximumWidth(30)
        w_inc.clicked.connect(lambda: self.symbol_w_spin.setValue(self.symbol_w_spin.value() + 1))
        w_layout.addWidget(w_inc)
        edit_form.addRow("W:", w_layout)

        # Height
        h_layout = QHBoxLayout()
        self.symbol_h_spin = QSpinBox()
        self.symbol_h_spin.setMinimum(1)
        self.symbol_h_spin.setMaximum(10000)
        self.symbol_h_spin.valueChanged.connect(self.on_selected_symbol_changed)
        h_layout.addWidget(self.symbol_h_spin, stretch=1)
        h_dec = QPushButton("◄")
        h_dec.setMaximumWidth(30)
        h_dec.clicked.connect(lambda: self.symbol_h_spin.setValue(self.symbol_h_spin.value() - 1))
        h_layout.addWidget(h_dec)
        h_inc = QPushButton("►")
        h_inc.setMaximumWidth(30)
        h_inc.clicked.connect(lambda: self.symbol_h_spin.setValue(self.symbol_h_spin.value() + 1))
        h_layout.addWidget(h_inc)
        edit_form.addRow("H:", h_layout)

        symbol_edit_layout.addLayout(edit_form)

        self.symbol_edit_group.setLayout(symbol_edit_layout)
        self.symbol_edit_group.setVisible(False)
        text_layout.addWidget(self.symbol_edit_group)

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
        self.delete_btn.setEnabled(True)
        self.next_btn.setEnabled(True)

        self.update_progress()

    def load_current_file(self):
        """Load current file into viewer."""
        if self.current_file_index < 0 or self.current_file_index >= len(self.current_region_files):
            return

        filepath = self.current_region_files[self.current_file_index]

        # Check if this card is already labeled (for cards only)
        category_index = self.category_combo.currentIndex()
        if category_index == 0:  # Cards
            # Check all possible card labels for this file
            existing_card = self.find_existing_card_label(filepath)
            if existing_card:
                self.card_input.setText(existing_card)
                self.card_input.setStyleSheet("background-color: #ffffcc;")
        else:
            self.card_input.setStyleSheet("")

        self.current_image = load_image(filepath)

        if self.current_image is not None:
            self.display_image(self.current_image)

    def display_image(self, image: np.ndarray, scale: float = 3.0):
        """Display image in viewer."""
        self.image_scene.clear()
        self.display_scale = scale

        # Clear symbol rects
        self.symbol_rects.clear()

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

    def create_symbol_regions(self):
        """Create manual regions for each symbol in text."""
        text = self.text_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Неверный ввод", "Введите текст")
            return

        if self.current_image is None:
            QMessageBox.warning(self, "Ошибка", "Нет изображения")
            return

        self.current_text = text

        # Clear existing rectangles
        for rect in self.symbol_rects:
            self.image_scene.removeItem(rect)
        self.symbol_rects.clear()

        # Get default size
        w = self.symbol_width_spin.value()
        h = self.symbol_height_spin.value()

        # Create rectangles for each symbol
        # Position them in a row at the top of the image
        img_h, img_w = self.current_image.shape[:2]
        start_x = 10
        start_y = 10
        spacing = 5

        for i, char in enumerate(text):
            x = (start_x + i * (w + spacing)) * self.display_scale
            y = start_y * self.display_scale
            scaled_w = w * self.display_scale
            scaled_h = h * self.display_scale

            rect = DraggableSymbolRect(x, y, scaled_w, scaled_h, i)
            self.image_scene.addItem(rect)
            self.symbol_rects.append(rect)

        # Show list of symbols
        self.symbol_list_group.show()
        self.symbol_list.clear()

        for i, char in enumerate(text):
            item = QListWidgetItem(f"{i+1}. '{char}'")
            self.symbol_list.addItem(item)

        info = f"Создано {len(text)} регионов. Перетащите каждый регион на соответствующий символ."
        self.symbol_info_label.setText(info)

        self.text_save_btn.setEnabled(True)

    def apply_symbol_size_to_all(self):
        """Apply default symbol size to all rectangles."""
        w = self.symbol_width_spin.value() * self.display_scale
        h = self.symbol_height_spin.value() * self.display_scale

        for rect in self.symbol_rects:
            current_rect = rect.rect()
            rect.setRect(current_rect.x(), current_rect.y(), w, h)

    def on_symbol_selected(self, index: int):
        """Handle symbol selection from list."""
        if index < 0 or index >= len(self.symbol_rects):
            self.symbol_edit_group.setVisible(False)
            self.selected_symbol_index = -1
            return

        self.selected_symbol_index = index

        # Select corresponding rectangle
        for i, rect in enumerate(self.symbol_rects):
            rect.setSelected(i == index)

        # Update editor with current symbol values
        rect = self.symbol_rects[index]
        scene_pos = rect.scenePos()
        rect_geom = rect.rect()

        # Convert to original image coordinates
        x = int(scene_pos.x() / self.display_scale)
        y = int(scene_pos.y() / self.display_scale)
        w = int(rect_geom.width() / self.display_scale)
        h = int(rect_geom.height() / self.display_scale)

        # Update label
        char = self.current_text[index] if index < len(self.current_text) else '?'
        self.selected_symbol_label.setText(f"Символ #{index + 1}: '{char}'")

        # Block signals to avoid recursion
        self.symbol_x_spin.blockSignals(True)
        self.symbol_y_spin.blockSignals(True)
        self.symbol_w_spin.blockSignals(True)
        self.symbol_h_spin.blockSignals(True)

        self.symbol_x_spin.setValue(x)
        self.symbol_y_spin.setValue(y)
        self.symbol_w_spin.setValue(w)
        self.symbol_h_spin.setValue(h)

        self.symbol_x_spin.blockSignals(False)
        self.symbol_y_spin.blockSignals(False)
        self.symbol_w_spin.blockSignals(False)
        self.symbol_h_spin.blockSignals(False)

        self.symbol_edit_group.setVisible(True)

    def on_selected_symbol_changed(self):
        """Handle changes to selected symbol position/size."""
        if self.selected_symbol_index < 0 or self.selected_symbol_index >= len(self.symbol_rects):
            return

        # Get values from spinboxes (in original coordinates)
        x = self.symbol_x_spin.value()
        y = self.symbol_y_spin.value()
        w = self.symbol_w_spin.value()
        h = self.symbol_h_spin.value()

        # Convert to display coordinates
        dx = x * self.display_scale
        dy = y * self.display_scale
        dw = w * self.display_scale
        dh = h * self.display_scale

        # Update rectangle
        rect = self.symbol_rects[self.selected_symbol_index]
        rect.setPos(dx, dy)
        rect.setRect(0, 0, dw, dh)

    def on_symbol_value_changed(self, index: int, x: int, y: int, w: int, h: int):
        """Handle symbol boundary change (legacy - not used in new approach)."""
        pass

    def prev_file(self):
        """Go to previous file."""
        if self.current_file_index > 0:
            self.current_file_index -= 1
            self.load_current_file()
            self.update_progress()
            # Clear any highlighting
            self.card_input.setStyleSheet("")

    def next_file(self):
        """Go to next file."""
        if self.current_file_index < len(self.current_region_files) - 1:
            self.current_file_index += 1
            self.load_current_file()
            self.update_progress()
            # Clear any highlighting
            self.card_input.setStyleSheet("")

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

        # Verify template
        is_valid, verify_msg = self.verify_card_template(self.current_image, rank, suit)

        if not is_valid:
            # Show warning if template looks like a different card
            reply = QMessageBox.question(
                self, "Предупреждение",
                f"{verify_msg}\n\nВсё равно сохранить как {rank}{suit}?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        else:
            # Show verification message and ask to confirm
            reply = QMessageBox.question(
                self, "Проверка шаблона",
                f"{verify_msg}\n\nСохранить как {rank}{suit}?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
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

    def find_existing_card_label(self, filepath: Path) -> Optional[str]:
        """Find if current card image matches an existing template.

        Args:
            filepath: Path to card image file

        Returns:
            Card label (e.g., "2c") if found, None otherwise
        """
        try:
            # Load the card image
            image = load_image(filepath)
            if image is None:
                return None

            # Get all existing card templates
            cards_dir = self.config.templates_dir / 'cards'
            if not cards_dir.exists():
                return None

            best_match_score = 0.0
            best_match_card = None

            # Try matching against all card templates
            for template_file in cards_dir.glob('*.png'):
                template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
                if template is None:
                    continue

                # Resize image to match template size
                img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                img_resized = cv2.resize(img_gray, (template.shape[1], template.shape[0]))

                # Template matching
                result = cv2.matchTemplate(img_resized, template, cv2.TM_CCOEFF_NORMED)
                score = result[0][0]

                if score > best_match_score:
                    best_match_score = score
                    best_match_card = template_file.stem

            # If very high match (>95%), consider it already labeled
            if best_match_score > 0.95 and best_match_card:
                return best_match_card

            return None

        except Exception as e:
            print(f"Error finding existing card label: {e}")
            return None

    def delete_current_file(self):
        """Delete current region file."""
        if self.current_file_index < 0 or self.current_file_index >= len(self.current_region_files):
            return

        filepath = self.current_region_files[self.current_file_index]

        reply = QMessageBox.question(
            self, "Удалить?",
            f"Удалить файл?\n{filepath.name}",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        try:
            # Delete the file
            filepath.unlink()

            # Remove from list
            self.current_region_files.pop(self.current_file_index)

            # Update display
            if not self.current_region_files:
                # No more files
                QMessageBox.information(self, "Готово", "Все файлы удалены")
                self.current_file_index = -1
                self.current_image = None
                self.image_scene.clear()
                self.prev_btn.setEnabled(False)
                self.skip_btn.setEnabled(False)
                self.delete_btn.setEnabled(False)
                self.next_btn.setEnabled(False)
                self.progress_label.setText("Файлы не загружены")
            else:
                # Load next file (or stay at current index if it was the last)
                if self.current_file_index >= len(self.current_region_files):
                    self.current_file_index = len(self.current_region_files) - 1

                self.load_current_file()
                self.update_progress()

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось удалить файл: {e}")

    def verify_card_template(self, image: np.ndarray, rank: str, suit: str) -> Tuple[bool, str]:
        """Verify card template against existing templates.

        Args:
            image: Card image to verify
            rank: Card rank
            suit: Card suit

        Returns:
            Tuple of (is_valid, message)
        """
        # Get all existing card templates
        cards_dir = self.config.templates_dir / 'cards'
        if not cards_dir.exists():
            return True, "Первый шаблон карты"

        # Try template matching against all existing cards
        best_match_score = 0.0
        best_match_card = None

        for template_file in cards_dir.glob('*.png'):
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue

            # Resize image to match template size
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            img_resized = cv2.resize(img_gray, (template.shape[1], template.shape[0]))

            # Template matching
            result = cv2.matchTemplate(img_resized, template, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]

            if score > best_match_score:
                best_match_score = score
                best_match_card = template_file.stem

        # If very high match with different card, warn
        if best_match_score > 0.95 and best_match_card != f"{rank}{suit}":
            return False, f"ВНИМАНИЕ: Сильное совпадение с {best_match_card} (схожесть: {best_match_score:.2%})"

        if best_match_score > 0.95:
            return True, f"Совпадает с существующим {best_match_card} (схожесть: {best_match_score:.2%})"

        return True, f"Новый уникальный шаблон"

    def verify_symbol_template(self, image: np.ndarray, char: str, category: str) -> Tuple[bool, str]:
        """Verify symbol template against existing templates.

        Args:
            image: Symbol image to verify
            char: Character
            category: Template category (digits, letters_lat, etc.)

        Returns:
            Tuple of (is_valid, message)
        """
        # Get category directory
        category_dir = self.config.templates_dir / category
        if not category_dir.exists():
            return True, "Первый шаблон"

        # Try template matching against existing templates for this character
        char_files = list(category_dir.glob(f'{char}_*.png'))
        if not char_files:
            return True, "Новый символ"

        best_match_score = 0.0

        for template_file in char_files:
            template = cv2.imread(str(template_file), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue

            # Resize image to match template size
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            img_resized = cv2.resize(img_gray, (template.shape[1], template.shape[0]))

            # Template matching
            result = cv2.matchTemplate(img_resized, template, cv2.TM_CCOEFF_NORMED)
            score = result[0][0]

            if score > best_match_score:
                best_match_score = score

        if best_match_score > 0.95:
            return True, f"Совпадает с существующим (схожесть: {best_match_score:.2%})"

        return True, f"Новый вариант символа '{char}'"

    def save_text_label(self):
        """Save text label."""
        text = self.current_text.strip()
        if not text or not self.symbol_rects:
            QMessageBox.warning(self, "Ошибка", "Нет текста или регионов")
            return

        if len(text) != len(self.symbol_rects):
            QMessageBox.warning(self, "Несовпадение", "Количество символов не совпадает с регионами")
            return

        # Extract symbol images from rectangles
        verification_messages = []
        symbol_images = []

        for i, (char, rect) in enumerate(zip(text, self.symbol_rects)):
            # Get rectangle position (in display coordinates)
            rect_geom = rect.rect()
            x = int(rect_geom.x() / self.display_scale)
            y = int(rect_geom.y() / self.display_scale)
            w = int(rect_geom.width() / self.display_scale)
            h = int(rect_geom.height() / self.display_scale)

            # Get position in scene coordinates
            scene_pos = rect.scenePos()
            x = int(scene_pos.x() / self.display_scale)
            y = int(scene_pos.y() / self.display_scale)

            # Ensure bounds are within image
            img_h, img_w = self.current_image.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = max(1, min(w, img_w - x))
            h = max(1, min(h, img_h - y))

            # Crop symbol from original image
            symbol_img = self.current_image[y:y+h, x:x+w].copy()
            symbol_images.append(symbol_img)

            # Determine category
            if char.isdigit() or char == '.':
                category = 'digits'
            elif char.isalpha():
                if ord(char) < 128:
                    category = 'letters_lat'
                else:
                    category = 'letters_cyr'
            else:
                category = 'special'

            is_valid, msg = self.verify_symbol_template(symbol_img, char, category)
            verification_messages.append(f"'{char}': {msg}")

        # Show verification results
        verification_text = "\n".join(verification_messages)
        reply = QMessageBox.question(
            self, "Проверка шаблонов",
            f"Результаты проверки:\n\n{verification_text}\n\nСохранить?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Save each symbol
        saved_count = 0

        for i, (char, symbol_img) in enumerate(zip(text, symbol_images)):
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

        # Clear
        self.text_input.clear()
        self.current_text = ""
        self.symbol_list.clear()
        self.symbol_list_group.hide()
        self.symbol_editor.hide_editor()
        for rect in self.symbol_rects:
            self.image_scene.removeItem(rect)
        self.symbol_rects.clear()
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
