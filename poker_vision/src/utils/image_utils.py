"""Image processing utilities for Poker Vision Tool."""
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional


def load_image(image_path: Path) -> Optional[np.ndarray]:
    """Load image from file.

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array (BGR format) or None if failed
    """
    if not image_path.exists():
        return None

    return cv2.imread(str(image_path))


def save_image(image: np.ndarray, output_path: Path) -> bool:
    """Save image to file.

    Args:
        image: Image as numpy array (BGR format)
        output_path: Path to save image

    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(output_path), image)


def crop_region(image: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop region from image.

    Args:
        image: Source image
        x: X coordinate
        y: Y coordinate
        w: Width
        h: Height

    Returns:
        Cropped image region
    """
    return image[y:y+h, x:x+w].copy()


def convert_cv_to_pil(cv_image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL Image.

    Args:
        cv_image: OpenCV image (BGR format)

    Returns:
        PIL Image (RGB format)
    """
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def convert_pil_to_cv(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV image.

    Args:
        pil_image: PIL Image (RGB format)

    Returns:
        OpenCV image (BGR format)
    """
    rgb_array = np.array(pil_image)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Resize image by scale factor.

    Args:
        image: Source image
        scale: Scale factor (1.0 = no change, 2.0 = double size)

    Returns:
        Resized image
    """
    if scale == 1.0:
        return image

    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def draw_rectangle(image: np.ndarray, x: int, y: int, w: int, h: int,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2) -> np.ndarray:
    """Draw rectangle on image.

    Args:
        image: Source image
        x: X coordinate
        y: Y coordinate
        w: Width
        h: Height
        color: Rectangle color (BGR format)
        thickness: Line thickness

    Returns:
        Image with rectangle drawn
    """
    result = image.copy()
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result


def put_text(image: np.ndarray, text: str, x: int, y: int,
            color: Tuple[int, int, int] = (0, 255, 0),
            font_scale: float = 0.5, thickness: int = 1) -> np.ndarray:
    """Put text on image.

    Args:
        image: Source image
        text: Text to put
        x: X coordinate
        y: Y coordinate
        color: Text color (BGR format)
        font_scale: Font scale
        thickness: Text thickness

    Returns:
        Image with text drawn
    """
    result = image.copy()
    cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return result


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Preprocess image for OCR/template matching.

    Args:
        image: Source image

    Returns:
        Preprocessed grayscale image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply slight Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    return blurred


def binarize_image(image: np.ndarray, method: str = 'adaptive') -> np.ndarray:
    """Binarize image using various methods.

    Args:
        image: Grayscale image
        method: Binarization method ('adaptive', 'otsu', 'threshold')

    Returns:
        Binary image
    """
    if method == 'adaptive':
        return cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    elif method == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    else:  # simple threshold
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return binary
