"""Screenshot capture functionality."""
import sys
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# Platform-specific imports
if sys.platform == 'win32':
    import win32gui
    import win32ui
    import win32con
    from PIL import Image


class ScreenshotCapture:
    """Handles screenshot capture from game window."""

    def __init__(self, output_dir: Path):
        """Initialize screenshot capture.

        Args:
            output_dir: Directory to save screenshots
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_window(self, hwnd: int) -> Optional[np.ndarray]:
        """Capture screenshot of window client area.

        Args:
            hwnd: Window handle

        Returns:
            Screenshot as numpy array (BGR format) or None if failed
        """
        if sys.platform != 'win32':
            return None

        try:
            # Get window client area dimensions
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            width = right - left
            height = bottom - top

            # Get device contexts
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            # Create bitmap
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)

            # Copy window content to bitmap
            # Try PrintWindow first, fallback to BitBlt
            result = win32gui.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)

            if result == 0:
                # PrintWindow failed, use BitBlt instead
                # Get window position for BitBlt
                window_left, window_top = win32gui.ClientToScreen(hwnd, (0, 0))
                save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

            # Convert to PIL Image
            bmpinfo = save_bitmap.GetInfo()
            bmpstr = save_bitmap.GetBitmapBits(True)

            pil_image = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1
            )

            # Clean up
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwnd_dc)

            # Convert to numpy array (BGR)
            screenshot = np.array(pil_image)
            screenshot = screenshot[:, :, ::-1]  # RGB to BGR

            return screenshot

        except Exception as e:
            print(f"Error capturing window: {e}")
            return None

    def capture_region(self, x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
        """Capture screenshot of screen region.

        Args:
            x: X coordinate
            y: Y coordinate
            width: Width
            height: Height

        Returns:
            Screenshot as numpy array (BGR format) or None if failed
        """
        if sys.platform != 'win32':
            return None

        try:
            # Get screen DC
            screen_dc = win32gui.GetDC(0)
            mfc_dc = win32ui.CreateDCFromHandle(screen_dc)
            save_dc = mfc_dc.CreateCompatibleDC()

            # Create bitmap
            save_bitmap = win32ui.CreateBitmap()
            save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
            save_dc.SelectObject(save_bitmap)

            # Copy screen region to bitmap
            save_dc.BitBlt((0, 0), (width, height), mfc_dc, (x, y), win32con.SRCCOPY)

            # Convert to PIL Image
            bmpinfo = save_bitmap.GetInfo()
            bmpstr = save_bitmap.GetBitmapBits(True)

            pil_image = Image.frombuffer(
                'RGB',
                (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
                bmpstr, 'raw', 'BGRX', 0, 1
            )

            # Clean up
            win32gui.DeleteObject(save_bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(0, screen_dc)

            # Convert to numpy array (BGR)
            screenshot = np.array(pil_image)
            screenshot = screenshot[:, :, ::-1]  # RGB to BGR

            return screenshot

        except Exception as e:
            print(f"Error capturing region: {e}")
            return None

    def save_screenshot(self, screenshot: np.ndarray, prefix: str = "scrn") -> Optional[Path]:
        """Save screenshot to file.

        Args:
            screenshot: Screenshot as numpy array
            prefix: Filename prefix

        Returns:
            Path to saved file or None if failed
        """
        try:
            import cv2

            # Generate filename with timestamp
            now = datetime.now()
            filename = f"{prefix}_{now.strftime('%H%M%S')}_{now.microsecond // 1000:03d}.png"
            filepath = self.output_dir / filename

            # Save image
            success = cv2.imwrite(str(filepath), screenshot)

            if success:
                return filepath
            return None

        except Exception as e:
            print(f"Error saving screenshot: {e}")
            return None

    def capture_and_save(self, hwnd: int, prefix: str = "scrn") -> Optional[Path]:
        """Capture window screenshot and save to file.

        Args:
            hwnd: Window handle
            prefix: Filename prefix

        Returns:
            Path to saved file or None if failed
        """
        screenshot = self.capture_window(hwnd)
        if screenshot is not None:
            return self.save_screenshot(screenshot, prefix)
        return None
