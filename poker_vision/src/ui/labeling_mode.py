"""Labeling mode UI for template creation."""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QLineEdit, QMessageBox, QGroupBox, QRadioButton,
    QButtonGroup, QScrollArea, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QFont
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple

from ..utils.config import Config, RegionsConfig
from ..core.region_cutter import RegionCutter
from ..core.template_manager import TemplateManager
from ..core.symbol_splitter import SymbolSplitter
from ..utils.image_utils import load_image


class LabelingMode(QWidget):
    """Labeling mode widget for creating templates."""

    def __init__(self, config: Config, regions_config: RegionsConfig):
        """Initialize labeling mode.

        Args:
            config: Application configuration
            regions_config: Regions configuration
        """
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

        self.setup_ui()

    def setup_ui(self):
        """Setup user interface."""
        layout = QVBoxLayout(self)

        # Region category selection
        category_group = QGroupBox("Выбор категории региона")
        category_layout = QVBoxLayout()

        self.category_combo = QComboBox()
        self.category_combo.addItem("Карты (my_card_*, board_*)")
        self.category_combo.addItem("Комбинации (my_combo)")
        self.category_combo.addItem("Маркеры (dealer_pos_*, timer_pos_*, seat_*)")
        self.category_combo.addItem("Текст - Цифры (stack_*, bet_*, pot)")
        self.category_combo.addItem("Текст - Смешанный (name_*, match_id)")
        self.category_combo.currentIndexChanged.connect(self.on_category_changed)
        category_layout.addWidget(self.category_combo)

        # Region selection within category
        self.region_combo = QComboBox()
        self.region_combo.currentIndexChanged.connect(self.on_region_changed)
        category_layout.addWidget(self.region_combo)

        load_btn = QPushButton("Загрузить файлы региона")
        load_btn.clicked.connect(self.load_region_files)
        category_layout.addWidget(load_btn)

        category_group.setLayout(category_layout)
        layout.addWidget(category_group)

        # Progress info
        self.progress_label = QLabel("Выберите категорию и загрузите файлы")
        self.progress_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(11)
        self.progress_label.setFont(font)
        layout.addWidget(self.progress_label)

        # Image viewer
        viewer_group = QGroupBox("Текущий регион")
        viewer_layout = QVBoxLayout()

        self.image_viewer = QGraphicsView()
        self.image_scene = QGraphicsScene()
        self.image_viewer.setScene(self.image_scene)
        self.image_viewer.setMinimumHeight(200)
        viewer_layout.addWidget(self.image_viewer)

        viewer_group.setLayout(viewer_layout)
        layout.addWidget(viewer_group)

        # Labeling interface (stacked based on category)
        label_group = QGroupBox("Разметка")
        self.label_layout = QVBoxLayout()

        self._setup_card_labeling()
        self._setup_combo_labeling()
        self._setup_marker_labeling()
        self._setup_text_labeling()

        label_group.setLayout(self.label_layout)
        layout.addWidget(label_group)

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

        layout.addLayout(nav_layout)

        # Statistics
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)

        # Initialize
        self.on_category_changed(0)
        self.update_statistics()

    def _setup_card_labeling(self):
        """Setup card labeling interface."""
        self.card_widget = QWidget()
        card_layout = QVBoxLayout(self.card_widget)

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
        self.text_input.setPlaceholderText("12.59")
        self.text_input.returnPressed.connect(self.save_text_label)
        input_layout.addWidget(self.text_input)

        save_btn = QPushButton("Сохранить")
        save_btn.clicked.connect(self.save_text_label)
        input_layout.addWidget(save_btn)

        text_layout.addLayout(input_layout)

        # Symbol visualization
        self.symbol_info_label = QLabel()
        self.symbol_info_label.setWordWrap(True)
        text_layout.addWidget(self.symbol_info_label)

        self.text_widget.hide()
        self.label_layout.addWidget(self.text_widget)

    def on_category_changed(self, index: int):
        """Handle category selection change."""
        # Hide all labeling widgets
        self.card_widget.hide()
        self.combo_widget.hide()
        self.marker_widget.hide()
        self.text_widget.hide()

        # Update region combo
        self.region_combo.clear()

        if index == 0:  # Cards
            regions = self.regions_config.get_regions_by_type('card')
            self.card_widget.show()
        elif index == 1:  # Combos
            regions = self.regions_config.get_regions_by_type('combo')
            self.combo_widget.show()
        elif index == 2:  # Markers
            regions = self.regions_config.get_regions_by_type('marker')
            self.marker_widget.show()
        elif index == 3:  # Text - Digits
            regions = self.regions_config.get_regions_by_type('text_digits')
            self.text_widget.show()
        elif index == 4:  # Text - Mixed
            regions = self.regions_config.get_regions_by_type('text_mixed')
            self.text_widget.show()
        else:
            regions = {}

        for region_id in sorted(regions.keys()):
            self.region_combo.addItem(region_id)

    def on_region_changed(self, index: int):
        """Handle region selection change."""
        pass

    def load_region_files(self):
        """Load region files for selected region."""
        region_id = self.region_combo.currentText()
        if not region_id:
            QMessageBox.warning(self, "Нет региона", "Сначала выберите регион.")
            return

        self.current_region_files = self.region_cutter.get_region_files(region_id)

        if not self.current_region_files:
            QMessageBox.information(
                self, "Нет файлов",
                f"Нарезки для {region_id} не найдены.\n"
                "Сначала нарежьте регионы в режиме Нарезки."
            )
            return

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

            # For text, split symbols
            category_index = self.category_combo.currentIndex()
            if category_index in [3, 4]:  # Text categories
                self.split_and_display_symbols()

    def display_image(self, image: np.ndarray, scale: float = 2.0):
        """Display image in viewer.

        Args:
            image: Image to display
            scale: Display scale
        """
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

    def split_and_display_symbols(self):
        """Split text into symbols and display info."""
        if self.current_image is None:
            return

        self.current_symbols = self.symbol_splitter.split_to_symbols(self.current_image)

        info = f"Обнаружено символов: {len(self.current_symbols)}"
        self.symbol_info_label.setText(info)

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
        region_id = self.region_combo.currentText()

        self.progress_label.setText(f"{region_id}: {current} / {total}")

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
                f"Масть должна быть одной из: {', '.join(self.template_manager.CARD_SUITS)}"
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

        # Check if exists
        if self.template_manager.template_exists('combos', combo_name):
            reply = QMessageBox.question(
                self, "Заменить?",
                f"Шаблон для {combo_name} уже существует. Заменить?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Save template
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

        region_id = self.region_combo.currentText()

        # Determine marker type from region_id
        if 'dealer' in region_id:
            marker_name = 'dealer_button'
        elif 'timer' in region_id:
            marker_name = 'timer'
        elif 'seat' in region_id:
            marker_name = 'seat_occupied'
        else:
            QMessageBox.warning(self, "Ошибка", "Неизвестный тип маркера")
            return

        # Check if exists
        if self.template_manager.template_exists('markers', marker_name):
            reply = QMessageBox.question(
                self, "Заменить?",
                f"Шаблон для {marker_name} уже существует. Заменить?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        # Save template
        result = self.template_manager.save_marker_template(self.current_image, marker_name)

        if result:
            QMessageBox.information(self, "Сохранено", f"Сохранён шаблон для {marker_name}")
            self.next_file()
            self.update_statistics()
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось сохранить шаблон")

    def save_text_label(self):
        """Save text label."""
        if self.current_image is None or not self.current_symbols:
            QMessageBox.warning(self, "Ошибка", "Символы не обнаружены")
            return

        text = self.text_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Неверный ввод", "Введите текст")
            return

        # Check if length matches
        if len(text) != len(self.current_symbols):
            QMessageBox.warning(
                self, "Несовпадение",
                f"Длина ввода ({len(text)}) не совпадает с обнаруженными символами ({len(self.current_symbols)})"
            )
            return

        # Save each symbol
        category_index = self.category_combo.currentIndex()
        saved_count = 0

        for i, char in enumerate(text):
            symbol_img, _ = self.current_symbols[i]

            # Determine category
            if category_index == 3:  # Digits
                if char in self.template_manager.DIGITS:
                    result = self.template_manager.save_digit_template(symbol_img, char)
                    if result:
                        saved_count += 1
            else:  # Mixed text
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
