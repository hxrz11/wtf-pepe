"""Region cutting mode UI with region editor."""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QCheckBox, QScrollArea,
    QMessageBox, QGroupBox, QComboBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QCursor
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Optional, List, Dict

from ..utils.config import Config, RegionsConfig
from ..core.region_cutter import RegionCutter
from ..utils.image_utils import load_image


class DraggableRectItem(QGraphicsRectItem):
    """Draggable rectangle item for region editor."""

    def __init__(self, x, y, w, h, region_id, parent_viewer):
        """Initialize draggable rect.

        Args:
            x, y, w, h: Rectangle coordinates
            region_id: Region identifier
            parent_viewer: Parent ImageViewer instance
        """
        super().__init__(x, y, w, h)
        self.region_id = region_id
        self.parent_viewer = parent_viewer
        self.is_dragging = False
        self.drag_start_pos = None
        self.original_pos = (x, y)

        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setCursor(QCursor(Qt.SizeAllCursor))

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_start_pos = event.pos()
            self.original_pos = (self.rect().x(), self.rect().y())
            # Change color while dragging
            pen = self.pen()
            pen.setColor(QColor(255, 0, 0))  # Red
            pen.setWidth(3)
            self.setPen(pen)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        if self.is_dragging:
            super().mouseMoveEvent(event)
            # Notify parent of position change
            new_rect = self.sceneBoundingRect()
            self.parent_viewer.on_region_dragged(
                self.region_id,
                int(new_rect.x()),
                int(new_rect.y())
            )

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton and self.is_dragging:
            self.is_dragging = False
            # Restore color
            pen = self.pen()
            pen.setColor(QColor(255, 255, 0))  # Yellow
            pen.setWidth(3)
            self.setPen(pen)
            # Notify parent of final position
            new_rect = self.sceneBoundingRect()
            self.parent_viewer.on_region_drop(
                self.region_id,
                int(new_rect.x()),
                int(new_rect.y())
            )
        super().mouseReleaseEvent(event)


class ImageViewer(QGraphicsView):
    """Image viewer with zoom and region selection."""

    def __init__(self):
        """Initialize image viewer."""
        super().__init__()

        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        self.image_item = None
        self.current_scale = 1.0
        self.region_items = []
        self.draggable_region = None
        self.drag_callback = None
        self.drop_callback = None

        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_drag_callbacks(self, drag_callback, drop_callback):
        """Set callbacks for region dragging.

        Args:
            drag_callback: Called during drag with (region_id, x, y)
            drop_callback: Called on drop with (region_id, x, y)
        """
        self.drag_callback = drag_callback
        self.drop_callback = drop_callback

    def on_region_dragged(self, region_id, x, y):
        """Called when region is being dragged."""
        if self.drag_callback:
            self.drag_callback(region_id, x, y)

    def on_region_drop(self, region_id, x, y):
        """Called when region is dropped."""
        if self.drop_callback:
            self.drop_callback(region_id, x, y)

    def load_image(self, image: np.ndarray):
        """Load image into viewer.

        Args:
            image: Image as numpy array (BGR)
        """
        self.scene.clear()
        self.region_items.clear()
        self.draggable_region = None

        # Convert to QPixmap
        height, width = image.shape[:2]
        if len(image.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        else:
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image)

        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.image_item)

        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def set_zoom(self, scale: float):
        """Set zoom scale.

        Args:
            scale: Zoom scale (1.0 = 100%)
        """
        self.current_scale = scale
        self.resetTransform()
        self.scale(scale, scale)

    def draw_regions(self, regions: Dict[str, Dict], selected_ids: List[str] = None,
                    editing_region_id: str = None):
        """Draw region rectangles on image.

        Args:
            regions: Dictionary of region_id -> {x, y, w, h, type}
            selected_ids: List of selected region IDs
            editing_region_id: ID of region being edited (drawn differently)
        """
        # Remove old region items
        for item in self.region_items:
            self.scene.removeItem(item)
        self.region_items.clear()
        self.draggable_region = None

        # Color map for region types
        color_map = {
            'card': QColor(0, 255, 0),
            'text_digits': QColor(255, 0, 0),
            'text_mixed': QColor(255, 255, 0),
            'marker': QColor(255, 0, 255),
            'combo': QColor(255, 165, 0)
        }

        # Draw each region
        for region_id, coords in regions.items():
            if selected_ids and region_id not in selected_ids:
                continue

            x = coords.get('x', 0)
            y = coords.get('y', 0)
            w = coords.get('w', 0)
            h = coords.get('h', 0)

            if w <= 0 or h <= 0:
                continue

            # Check if this is the editing region
            if editing_region_id and region_id == editing_region_id:
                # Draw as draggable with special color
                pen = QPen(QColor(255, 255, 0), 3)  # Yellow, thicker
                rect_item = DraggableRectItem(x, y, w, h, region_id, self)
                rect_item.setPen(pen)
                rect_item.setZValue(2)
                self.draggable_region = rect_item
            else:
                # Draw as normal
                region_type = coords.get('type', 'unknown')
                color = color_map.get(region_type, QColor(128, 128, 128))
                pen = QPen(color, 2)
                rect_item = QGraphicsRectItem(x, y, w, h)
                rect_item.setPen(pen)
                rect_item.setZValue(1)

            self.scene.addItem(rect_item)
            self.region_items.append(rect_item)


class RegionEditor(QGroupBox):
    """Widget for editing region coordinates."""

    def __init__(self, parent=None):
        """Initialize region editor."""
        super().__init__("Редактирование региона", parent)

        self.region_id = None
        self.original_coords = {}
        self.value_changed_callback = None

        self.setup_ui()
        self.setVisible(False)

    def setup_ui(self):
        """Setup user interface."""
        layout = QVBoxLayout(self)

        # Region name label
        self.region_name_label = QLabel("")
        self.region_name_label.setAlignment(Qt.AlignCenter)
        font = self.region_name_label.font()
        font.setBold(True)
        self.region_name_label.setFont(font)
        layout.addWidget(self.region_name_label)

        # Coordinate editors
        form = QFormLayout()

        # X coordinate
        x_layout = QHBoxLayout()
        self.x_spin = QSpinBox()
        self.x_spin.setMinimum(0)
        self.x_spin.setMaximum(10000)
        self.x_spin.valueChanged.connect(lambda v: self.on_value_changed('x', v))
        x_layout.addWidget(self.x_spin, stretch=1)

        x_dec_btn = QPushButton("◄")
        x_dec_btn.setMaximumWidth(30)
        x_dec_btn.clicked.connect(lambda: self.x_spin.setValue(self.x_spin.value() - 1))
        x_layout.addWidget(x_dec_btn)

        x_inc_btn = QPushButton("►")
        x_inc_btn.setMaximumWidth(30)
        x_inc_btn.clicked.connect(lambda: self.x_spin.setValue(self.x_spin.value() + 1))
        x_layout.addWidget(x_inc_btn)

        form.addRow("X:", x_layout)

        # Y coordinate
        y_layout = QHBoxLayout()
        self.y_spin = QSpinBox()
        self.y_spin.setMinimum(0)
        self.y_spin.setMaximum(10000)
        self.y_spin.valueChanged.connect(lambda v: self.on_value_changed('y', v))
        y_layout.addWidget(self.y_spin, stretch=1)

        y_dec_btn = QPushButton("◄")
        y_dec_btn.setMaximumWidth(30)
        y_dec_btn.clicked.connect(lambda: self.y_spin.setValue(self.y_spin.value() - 1))
        y_layout.addWidget(y_dec_btn)

        y_inc_btn = QPushButton("►")
        y_inc_btn.setMaximumWidth(30)
        y_inc_btn.clicked.connect(lambda: self.y_spin.setValue(self.y_spin.value() + 1))
        y_layout.addWidget(y_inc_btn)

        form.addRow("Y:", y_layout)

        # Width
        w_layout = QHBoxLayout()
        self.w_spin = QSpinBox()
        self.w_spin.setMinimum(1)
        self.w_spin.setMaximum(10000)
        self.w_spin.valueChanged.connect(lambda v: self.on_value_changed('w', v))
        w_layout.addWidget(self.w_spin, stretch=1)

        w_dec_btn = QPushButton("◄")
        w_dec_btn.setMaximumWidth(30)
        w_dec_btn.clicked.connect(lambda: self.w_spin.setValue(self.w_spin.value() - 1))
        w_layout.addWidget(w_dec_btn)

        w_inc_btn = QPushButton("►")
        w_inc_btn.setMaximumWidth(30)
        w_inc_btn.clicked.connect(lambda: self.w_spin.setValue(self.w_spin.value() + 1))
        w_layout.addWidget(w_inc_btn)

        form.addRow("W:", w_layout)

        # Height
        h_layout = QHBoxLayout()
        self.h_spin = QSpinBox()
        self.h_spin.setMinimum(1)
        self.h_spin.setMaximum(10000)
        self.h_spin.valueChanged.connect(lambda v: self.on_value_changed('h', v))
        h_layout.addWidget(self.h_spin, stretch=1)

        h_dec_btn = QPushButton("◄")
        h_dec_btn.setMaximumWidth(30)
        h_dec_btn.clicked.connect(lambda: self.h_spin.setValue(self.h_spin.value() - 1))
        h_layout.addWidget(h_dec_btn)

        h_inc_btn = QPushButton("►")
        h_inc_btn.setMaximumWidth(30)
        h_inc_btn.clicked.connect(lambda: self.h_spin.setValue(self.h_spin.value() + 1))
        h_layout.addWidget(h_inc_btn)

        form.addRow("H:", h_layout)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()

        self.save_btn = QPushButton("Сохранить")
        self.save_btn.clicked.connect(self.on_save_clicked)
        btn_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self.on_cancel_clicked)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def set_value_changed_callback(self, callback):
        """Set callback for value changes.

        Args:
            callback: Function called with (x, y, w, h) on value change
        """
        self.value_changed_callback = callback

    def on_value_changed(self, coord, value):
        """Called when a coordinate value changes."""
        if self.value_changed_callback:
            self.value_changed_callback(
                self.x_spin.value(),
                self.y_spin.value(),
                self.w_spin.value(),
                self.h_spin.value()
            )

    def load_region(self, region_id: str, coords: Dict):
        """Load region for editing.

        Args:
            region_id: Region identifier
            coords: Dictionary with x, y, w, h keys
        """
        self.region_id = region_id
        self.original_coords = coords.copy()

        self.region_name_label.setText(region_id)

        # Block signals while setting values
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.w_spin.blockSignals(True)
        self.h_spin.blockSignals(True)

        self.x_spin.setValue(coords.get('x', 0))
        self.y_spin.setValue(coords.get('y', 0))
        self.w_spin.setValue(coords.get('w', 0))
        self.h_spin.setValue(coords.get('h', 0))

        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        self.w_spin.blockSignals(False)
        self.h_spin.blockSignals(False)

        self.setVisible(True)

    def update_position(self, x: int, y: int):
        """Update position values (from drag).

        Args:
            x: New X coordinate
            y: New Y coordinate
        """
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)

        self.x_spin.setValue(x)
        self.y_spin.setValue(y)

        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)

    def get_current_coords(self) -> Dict:
        """Get current coordinates.

        Returns:
            Dictionary with x, y, w, h keys
        """
        return {
            'x': self.x_spin.value(),
            'y': self.y_spin.value(),
            'w': self.w_spin.value(),
            'h': self.h_spin.value()
        }

    def on_save_clicked(self):
        """Handle save button click."""
        # Will be connected by parent
        pass

    def on_cancel_clicked(self):
        """Handle cancel button click."""
        # Will be connected by parent
        pass

    def hide_editor(self):
        """Hide editor."""
        self.region_id = None
        self.original_coords = {}
        self.setVisible(False)


class CuttingMode(QWidget):
    """Region cutting mode widget."""

    def __init__(self, config: Config, regions_config: RegionsConfig):
        """Initialize cutting mode.

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

        self.screenshots = []
        self.current_screenshot_index = -1
        self.current_image = None
        self.editing_region_id = None

        self.setup_ui()

    def setup_ui(self):
        """Setup user interface."""
        layout = QHBoxLayout(self)

        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)

        # Screenshot selection
        screenshot_group = QGroupBox("Скриншот")
        screenshot_layout = QVBoxLayout()

        self.screenshot_list = QComboBox()
        self.screenshot_list.currentIndexChanged.connect(self.on_screenshot_changed)
        screenshot_layout.addWidget(self.screenshot_list)

        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("◄ Назад")
        prev_btn.clicked.connect(self.prev_screenshot)
        next_btn = QPushButton("Вперёд ►")
        next_btn.clicked.connect(self.next_screenshot)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        screenshot_layout.addLayout(nav_layout)

        load_btn = QPushButton("Загрузить скриншоты")
        load_btn.clicked.connect(self.load_screenshots)
        screenshot_layout.addWidget(load_btn)

        screenshot_group.setLayout(screenshot_layout)
        left_layout.addWidget(screenshot_group)

        # Zoom control
        zoom_group = QGroupBox("Масштаб")
        zoom_layout = QHBoxLayout()

        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["100%", "150%", "200%", "300%"])
        self.zoom_combo.currentIndexChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_combo)

        zoom_group.setLayout(zoom_layout)
        left_layout.addWidget(zoom_group)

        # Region selection
        region_group = QGroupBox("Регионы")
        region_layout = QVBoxLayout()

        # Select all/none
        select_layout = QHBoxLayout()
        select_all_btn = QPushButton("Выбрать все")
        select_all_btn.clicked.connect(self.select_all_regions)
        select_none_btn = QPushButton("Снять все")
        select_none_btn.clicked.connect(self.select_none_regions)
        select_layout.addWidget(select_all_btn)
        select_layout.addWidget(select_none_btn)
        region_layout.addLayout(select_layout)

        # Region list
        self.region_list = QListWidget()
        self.region_list.itemChanged.connect(self.on_region_selection_changed)
        region_layout.addWidget(self.region_list)

        region_group.setLayout(region_layout)
        left_layout.addWidget(region_group)

        # Cut button
        self.cut_btn = QPushButton("Нарезать выбранные")
        self.cut_btn.clicked.connect(self.cut_regions)
        self.cut_btn.setEnabled(False)
        left_layout.addWidget(self.cut_btn)

        # Region editor
        self.region_editor = RegionEditor()
        self.region_editor.set_value_changed_callback(self.on_editor_value_changed)
        self.region_editor.save_btn.clicked.connect(self.save_region_edits)
        self.region_editor.cancel_btn.clicked.connect(self.cancel_region_edits)
        left_layout.addWidget(self.region_editor)

        # Statistics
        self.stats_label = QLabel("Нарезки не созданы")
        self.stats_label.setWordWrap(True)
        left_layout.addWidget(self.stats_label)

        left_layout.addStretch()

        layout.addWidget(left_panel)

        # Right panel - image viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.image_viewer = ImageViewer()
        self.image_viewer.set_drag_callbacks(
            self.on_region_dragged,
            self.on_region_dropped
        )
        right_layout.addWidget(self.image_viewer)

        layout.addWidget(right_panel, stretch=1)

        # Load regions list
        self.load_regions_list()

    def load_screenshots(self):
        """Load available screenshots."""
        self.screenshots = self.region_cutter.get_screenshot_files()

        self.screenshot_list.clear()

        if not self.screenshots:
            QMessageBox.information(
                self, "Нет скриншотов",
                "Скриншоты не найдены. Сначала сделайте скриншоты в режиме Скриншотов."
            )
            return

        for screenshot in self.screenshots:
            self.screenshot_list.addItem(screenshot.name)

        self.screenshot_list.setCurrentIndex(0)

        QMessageBox.information(
            self, "Загружено",
            f"Загружено {len(self.screenshots)} скриншотов"
        )

    def on_screenshot_changed(self, index: int):
        """Handle screenshot selection change."""
        if index < 0 or index >= len(self.screenshots):
            return

        self.current_screenshot_index = index
        screenshot_path = self.screenshots[index]

        # Load image
        self.current_image = load_image(screenshot_path)

        if self.current_image is not None:
            self.image_viewer.load_image(self.current_image)
            self.update_region_visualization()
            self.cut_btn.setEnabled(True)

    def prev_screenshot(self):
        """Go to previous screenshot."""
        if self.screenshot_list.count() == 0:
            return

        current = self.screenshot_list.currentIndex()
        if current > 0:
            self.screenshot_list.setCurrentIndex(current - 1)

    def next_screenshot(self):
        """Go to next screenshot."""
        if self.screenshot_list.count() == 0:
            return

        current = self.screenshot_list.currentIndex()
        if current < self.screenshot_list.count() - 1:
            self.screenshot_list.setCurrentIndex(current + 1)

    def on_zoom_changed(self, index: int):
        """Handle zoom change."""
        scales = [1.0, 1.5, 2.0, 3.0]
        if index >= 0 and index < len(scales):
            self.image_viewer.set_zoom(scales[index])

    def load_regions_list(self):
        """Load regions list."""
        self.region_list.clear()

        all_regions = self.regions_config.get_all_regions()

        for region_id, region_data in sorted(all_regions.items()):
            x = region_data.get('x', 0)
            y = region_data.get('y', 0)
            w = region_data.get('w', 0)
            h = region_data.get('h', 0)
            region_type = region_data.get('type', 'unknown')

            item = QListWidgetItem(f"{region_id} ({region_type}) [{x},{y},{w},{h}]")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(Qt.UserRole, region_id)

            self.region_list.addItem(item)

    def select_all_regions(self):
        """Select all regions."""
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            item.setCheckState(Qt.Checked)

    def select_none_regions(self):
        """Deselect all regions."""
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            item.setCheckState(Qt.Unchecked)

    def on_region_selection_changed(self):
        """Handle region selection change."""
        # Get selected regions
        selected_ids = self.get_selected_region_ids()

        # Show/hide region editor based on selection count
        if len(selected_ids) == 1:
            # Show editor for single region
            region_id = selected_ids[0]
            all_regions = self.regions_config.get_all_regions()
            if region_id in all_regions:
                self.editing_region_id = region_id
                self.region_editor.load_region(region_id, all_regions[region_id])
        else:
            # Hide editor if not exactly one region selected
            self.editing_region_id = None
            self.region_editor.hide_editor()

        self.update_region_visualization()

    def get_selected_region_ids(self) -> List[str]:
        """Get list of selected region IDs."""
        selected_ids = []
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            if item.checkState() == Qt.Checked:
                region_id = item.data(Qt.UserRole)
                selected_ids.append(region_id)
        return selected_ids

    def update_region_visualization(self):
        """Update region visualization on image."""
        if self.current_image is None:
            return

        selected_ids = self.get_selected_region_ids()

        # Draw regions
        all_regions = self.regions_config.get_all_regions()
        self.image_viewer.draw_regions(
            all_regions,
            selected_ids if selected_ids else None,
            self.editing_region_id
        )

    def on_editor_value_changed(self, x, y, w, h):
        """Handle editor value change."""
        # Update visualization with new coordinates
        if self.editing_region_id:
            all_regions = self.regions_config.get_all_regions()
            # Temporarily update coords for visualization
            temp_regions = all_regions.copy()
            temp_regions[self.editing_region_id] = {
                **temp_regions[self.editing_region_id],
                'x': x, 'y': y, 'w': w, 'h': h
            }
            selected_ids = self.get_selected_region_ids()
            self.image_viewer.draw_regions(
                temp_regions,
                selected_ids,
                self.editing_region_id
            )

    def on_region_dragged(self, region_id, x, y):
        """Handle region being dragged."""
        # Update editor position values
        if region_id == self.editing_region_id:
            self.region_editor.update_position(x, y)

            # Update visualization
            all_regions = self.regions_config.get_all_regions()
            temp_regions = all_regions.copy()
            w = temp_regions[region_id].get('w', 0)
            h = temp_regions[region_id].get('h', 0)
            temp_regions[region_id] = {
                **temp_regions[region_id],
                'x': x, 'y': y, 'w': w, 'h': h
            }
            selected_ids = self.get_selected_region_ids()
            self.image_viewer.draw_regions(
                temp_regions,
                selected_ids,
                self.editing_region_id
            )

    def on_region_dropped(self, region_id, x, y):
        """Handle region dropped."""
        # Same as dragged for now
        self.on_region_dragged(region_id, x, y)

    def save_region_edits(self):
        """Save region coordinate edits to regions.json."""
        if not self.editing_region_id:
            return

        # Get current coordinates from editor
        new_coords = self.region_editor.get_current_coords()

        # Update in regions config
        all_regions = self.regions_config.get_all_regions()
        if self.editing_region_id in all_regions:
            all_regions[self.editing_region_id].update(new_coords)

            # Save to file
            regions_file = Path(__file__).parent.parent.parent / 'regions.json'
            try:
                with open(regions_file, 'w', encoding='utf-8') as f:
                    json.dump(all_regions, f, indent=2, ensure_ascii=False)

                QMessageBox.information(self, "Сохранено", f"Регион {self.editing_region_id} сохранён")

                # Reload regions list to show new coordinates
                self.load_regions_list()

                # Reselect the edited region
                for i in range(self.region_list.count()):
                    item = self.region_list.item(i)
                    if item.data(Qt.UserRole) == self.editing_region_id:
                        item.setCheckState(Qt.Checked)
                        break

            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить: {str(e)}")

    def cancel_region_edits(self):
        """Cancel region edits."""
        self.editing_region_id = None
        self.region_editor.hide_editor()
        self.update_region_visualization()

    def cut_regions(self):
        """Cut selected regions from all screenshots."""
        selected_ids = self.get_selected_region_ids()

        if not selected_ids:
            QMessageBox.warning(self, "Нет выбора", "Выберите хотя бы один регион для нарезки.")
            return

        if not self.screenshots:
            QMessageBox.warning(self, "Нет скриншотов", "Сначала загрузите скриншоты.")
            return

        # Confirm
        reply = QMessageBox.question(
            self, "Подтверждение",
            f"Нарезать {len(selected_ids)} регионов из {len(self.screenshots)} скриншотов?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Cut regions
        all_regions = self.regions_config.get_all_regions()
        screenshots_processed, regions_cut = self.region_cutter.cut_regions_from_all_screenshots(
            all_regions, selected_ids
        )

        # Show results
        QMessageBox.information(
            self, "Готово",
            f"Обработано {screenshots_processed} скриншотов\n"
            f"Нарезано {regions_cut} изображений регионов"
        )

        # Update statistics
        self.update_statistics()

    def update_statistics(self):
        """Update cutting statistics."""
        counts = self.region_cutter.get_cut_regions_count()

        if not counts:
            self.stats_label.setText("Нарезки не созданы")
            return

        stats_text = "Нарезано:\n"
        for region_id, count in sorted(counts.items()):
            stats_text += f"  {region_id}: {count}\n"

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
