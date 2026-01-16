"""Region cutting mode UI."""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QListWidget, QListWidgetItem, QCheckBox, QScrollArea,
    QMessageBox, QGroupBox, QComboBox, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem,
    QSpinBox, QFormLayout
)
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPen, QColor, QBrush
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

from ..utils.config import Config, RegionsConfig
from ..core.region_cutter import RegionCutter
from ..utils.image_utils import load_image


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

        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def load_image(self, image: np.ndarray):
        """Load image into viewer.

        Args:
            image: Image as numpy array (BGR)
        """
        self.scene.clear()
        self.region_items.clear()

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

    def draw_regions(self, regions: Dict[str, Dict], selected_ids: List[str] = None):
        """Draw region rectangles on image.

        Args:
            regions: Dictionary of region_id -> {x, y, w, h, type}
            selected_ids: List of selected region IDs
        """
        # Remove old region items
        for item in self.region_items:
            self.scene.removeItem(item)
        self.region_items.clear()

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

            # Get color
            region_type = coords.get('type', 'unknown')
            color = color_map.get(region_type, QColor(128, 128, 128))

            # Draw rectangle
            pen = QPen(color, 2)
            rect_item = QGraphicsRectItem(x, y, w, h)
            rect_item.setPen(pen)
            rect_item.setZValue(1)

            self.scene.addItem(rect_item)
            self.region_items.append(rect_item)


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

        self.setup_ui()

    def setup_ui(self):
        """Setup user interface."""
        layout = QHBoxLayout(self)

        # Left panel - controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)

        # Screenshot selection
        screenshot_group = QGroupBox("Screenshot")
        screenshot_layout = QVBoxLayout()

        self.screenshot_list = QComboBox()
        self.screenshot_list.currentIndexChanged.connect(self.on_screenshot_changed)
        screenshot_layout.addWidget(self.screenshot_list)

        nav_layout = QHBoxLayout()
        prev_btn = QPushButton("◄ Prev")
        prev_btn.clicked.connect(self.prev_screenshot)
        next_btn = QPushButton("Next ►")
        next_btn.clicked.connect(self.next_screenshot)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        screenshot_layout.addLayout(nav_layout)

        load_btn = QPushButton("Load Screenshots")
        load_btn.clicked.connect(self.load_screenshots)
        screenshot_layout.addWidget(load_btn)

        screenshot_group.setLayout(screenshot_layout)
        left_layout.addWidget(screenshot_group)

        # Zoom control
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout()

        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["100%", "150%", "200%", "300%"])
        self.zoom_combo.currentIndexChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(self.zoom_combo)

        zoom_group.setLayout(zoom_layout)
        left_layout.addWidget(zoom_group)

        # Region selection
        region_group = QGroupBox("Regions")
        region_layout = QVBoxLayout()

        # Select all/none
        select_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.select_all_regions)
        select_none_btn = QPushButton("Select None")
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
        self.cut_btn = QPushButton("Cut Selected Regions")
        self.cut_btn.clicked.connect(self.cut_regions)
        self.cut_btn.setEnabled(False)
        left_layout.addWidget(self.cut_btn)

        # Statistics
        self.stats_label = QLabel("No regions cut yet")
        self.stats_label.setWordWrap(True)
        left_layout.addWidget(self.stats_label)

        left_layout.addStretch()

        layout.addWidget(left_panel)

        # Right panel - image viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        self.image_viewer = ImageViewer()
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
                self, "No Screenshots",
                "No screenshots found. Please capture some in Screenshot mode first."
            )
            return

        for screenshot in self.screenshots:
            self.screenshot_list.addItem(screenshot.name)

        self.screenshot_list.setCurrentIndex(0)

        QMessageBox.information(
            self, "Loaded",
            f"Loaded {len(self.screenshots)} screenshots"
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
        self.update_region_visualization()

    def update_region_visualization(self):
        """Update region visualization on image."""
        if self.current_image is None:
            return

        # Get selected region IDs
        selected_ids = []
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            if item.checkState() == Qt.Checked:
                region_id = item.data(Qt.UserRole)
                selected_ids.append(region_id)

        # Draw regions
        all_regions = self.regions_config.get_all_regions()
        self.image_viewer.draw_regions(all_regions, selected_ids if selected_ids else None)

    def cut_regions(self):
        """Cut selected regions from all screenshots."""
        # Get selected region IDs
        selected_ids = []
        for i in range(self.region_list.count()):
            item = self.region_list.item(i)
            if item.checkState() == Qt.Checked:
                region_id = item.data(Qt.UserRole)
                selected_ids.append(region_id)

        if not selected_ids:
            QMessageBox.warning(self, "No Selection", "Please select at least one region to cut.")
            return

        if not self.screenshots:
            QMessageBox.warning(self, "No Screenshots", "Please load screenshots first.")
            return

        # Confirm
        reply = QMessageBox.question(
            self, "Confirm",
            f"Cut {len(selected_ids)} regions from {len(self.screenshots)} screenshots?",
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
            self, "Complete",
            f"Processed {screenshots_processed} screenshots\n"
            f"Cut {regions_cut} region images"
        )

        # Update statistics
        self.update_statistics()

    def update_statistics(self):
        """Update cutting statistics."""
        counts = self.region_cutter.get_cut_regions_count()

        stats_text = "Cut regions:\n"
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
