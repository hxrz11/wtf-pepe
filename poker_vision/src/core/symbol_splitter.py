"""Symbol splitting for text recognition."""
import cv2
import numpy as np
from typing import List, Tuple, Optional


class SymbolSplitter:
    """Handles splitting text images into individual symbols."""

    def __init__(self, min_width: int = 3, min_height: int = 5):
        """Initialize symbol splitter.

        Args:
            min_width: Minimum symbol width
            min_height: Minimum symbol height
        """
        self.min_width = min_width
        self.min_height = min_height

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for symbol detection.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Binary image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Binarize using adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Invert so text is white
            11, 2
        )

        return binary

    def find_contours(self, binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find contours in binary image.

        Args:
            binary: Binary image

        Returns:
            List of bounding boxes (x, y, w, h)
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Get bounding boxes
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by minimum size
            if w >= self.min_width and h >= self.min_height:
                bboxes.append((x, y, w, h))

        return bboxes

    def merge_close_boxes(self, bboxes: List[Tuple[int, int, int, int]],
                         max_gap: int = 3) -> List[Tuple[int, int, int, int]]:
        """Merge boxes that are close together (for dots, multi-part chars).

        Args:
            bboxes: List of bounding boxes (x, y, w, h)
            max_gap: Maximum gap to merge

        Returns:
            List of merged bounding boxes
        """
        if not bboxes:
            return []

        # Sort by x coordinate
        sorted_boxes = sorted(bboxes, key=lambda b: b[0])

        merged = []
        current_group = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            x, y, w, h = box

            # Check if box is close to any box in current group
            should_merge = False
            for group_box in current_group:
                gx, gy, gw, gh = group_box

                # Check horizontal proximity
                x_gap = min(abs(x - (gx + gw)), abs(gx - (x + w)))

                # Check vertical overlap
                y_overlap = not (y + h < gy or gy + gh < y)

                if x_gap <= max_gap and y_overlap:
                    should_merge = True
                    break

            if should_merge:
                current_group.append(box)
            else:
                # Merge current group and start new group
                merged.append(self._merge_box_group(current_group))
                current_group = [box]

        # Merge last group
        if current_group:
            merged.append(self._merge_box_group(current_group))

        return merged

    def _merge_box_group(self, boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Merge a group of boxes into one bounding box.

        Args:
            boxes: List of boxes to merge

        Returns:
            Merged bounding box
        """
        if not boxes:
            return (0, 0, 0, 0)

        min_x = min(b[0] for b in boxes)
        min_y = min(b[1] for b in boxes)
        max_x = max(b[0] + b[2] for b in boxes)
        max_y = max(b[1] + b[3] for b in boxes)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def sort_boxes_left_to_right(self, bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Sort bounding boxes from left to right.

        Args:
            bboxes: List of bounding boxes

        Returns:
            Sorted list of bounding boxes
        """
        return sorted(bboxes, key=lambda b: b[0])

    def extract_symbols(self, image: np.ndarray,
                       bboxes: List[Tuple[int, int, int, int]]) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Extract symbol images from bounding boxes.

        Args:
            image: Source image
            bboxes: List of bounding boxes (x, y, w, h)

        Returns:
            List of (symbol_image, bbox) tuples
        """
        symbols = []

        for bbox in bboxes:
            x, y, w, h = bbox

            # Extract symbol
            symbol_img = image[y:y+h, x:x+w].copy()

            symbols.append((symbol_img, bbox))

        return symbols

    def split_to_symbols(self, image: np.ndarray,
                        merge_gaps: bool = True) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Split text image into individual symbols.

        Args:
            image: Text image (BGR or grayscale)
            merge_gaps: Whether to merge close boxes

        Returns:
            List of (symbol_image, bbox) tuples sorted left to right
        """
        # Preprocess
        binary = self.preprocess_image(image)

        # Find contours
        bboxes = self.find_contours(binary)

        # Merge close boxes if needed
        if merge_gaps:
            bboxes = self.merge_close_boxes(bboxes)

        # Sort left to right
        bboxes = self.sort_boxes_left_to_right(bboxes)

        # Extract symbols
        symbols = self.extract_symbols(image, bboxes)

        return symbols

    def visualize_symbols(self, image: np.ndarray,
                         symbols: List[Tuple[np.ndarray, Tuple[int, int, int, int]]]) -> np.ndarray:
        """Visualize symbol bounding boxes on image.

        Args:
            image: Source image
            symbols: List of (symbol_image, bbox) tuples

        Returns:
            Image with bounding boxes drawn
        """
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        # Draw each symbol box
        for i, (_, bbox) in enumerate(symbols):
            x, y, w, h = bbox

            # Draw rectangle
            color = (0, 255, 0)
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 1)

            # Draw index
            cv2.putText(result, str(i), (x, y - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        return result
