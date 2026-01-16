"""Window management for finding and attaching to game window."""
import sys
from typing import Optional, Tuple

# Platform-specific imports
if sys.platform == 'win32':
    import win32gui
    import win32con
    import win32api


class WindowManager:
    """Manages game window detection and positioning."""

    def __init__(self, window_title: str):
        """Initialize window manager.

        Args:
            window_title: Title of the game window to find
        """
        self.window_title = window_title
        self.hwnd: Optional[int] = None
        self.last_position: Optional[Tuple[int, int]] = None

    def find_window(self) -> bool:
        """Find game window by title.

        Returns:
            True if window found, False otherwise
        """
        if sys.platform != 'win32':
            # On non-Windows platforms, we can't use win32gui
            return False

        def enum_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.window_title in title:
                    results.append(hwnd)

        results = []
        win32gui.EnumWindows(enum_callback, results)

        if results:
            self.hwnd = results[0]
            return True

        return False

    def get_window_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Get window position and size.

        Returns:
            Tuple of (x, y, width, height) or None if window not found
        """
        if sys.platform != 'win32' or not self.hwnd:
            return None

        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            x, y, right, bottom = rect
            width = right - x
            height = bottom - y
            return (x, y, width, height)
        except:
            return None

    def get_client_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Get window client area position and size (without borders).

        Returns:
            Tuple of (x, y, width, height) or None if window not found
        """
        if sys.platform != 'win32' or not self.hwnd:
            return None

        try:
            # Get client area size
            client_rect = win32gui.GetClientRect(self.hwnd)
            client_width = client_rect[2]
            client_height = client_rect[3]

            # Get window position
            window_rect = win32gui.GetWindowRect(self.hwnd)
            window_x, window_y = window_rect[0], window_rect[1]

            # Calculate client area position (account for window borders)
            point = win32gui.ClientToScreen(self.hwnd, (0, 0))
            client_x, client_y = point

            return (client_x, client_y, client_width, client_height)
        except:
            return None

    def set_window_size(self, width: int, height: int) -> bool:
        """Set game window size.

        Args:
            width: Desired width
            height: Desired height

        Returns:
            True if successful, False otherwise
        """
        if sys.platform != 'win32' or not self.hwnd:
            return False

        try:
            # Get current position
            rect = win32gui.GetWindowRect(self.hwnd)
            x, y = rect[0], rect[1]

            # Calculate window size including borders
            # We need to account for borders/titlebar
            style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_STYLE)
            ex_style = win32gui.GetWindowLong(self.hwnd, win32con.GWL_EXSTYLE)

            # Calculate required window size for desired client size
            rect_adj = win32api.RECT()
            rect_adj.left = 0
            rect_adj.top = 0
            rect_adj.right = width
            rect_adj.bottom = height

            # This would adjust for borders, but win32api.AdjustWindowRectEx
            # might not be available in all win32api versions
            # For now, just use the size directly
            window_width = width
            window_height = height

            win32gui.SetWindowPos(
                self.hwnd, win32con.HWND_TOP,
                x, y, window_width, window_height,
                win32con.SWP_NOMOVE | win32con.SWP_NOZORDER
            )
            return True
        except:
            return False

    def is_window_valid(self) -> bool:
        """Check if window is still valid.

        Returns:
            True if window exists and is valid, False otherwise
        """
        if sys.platform != 'win32' or not self.hwnd:
            return False

        try:
            return win32gui.IsWindow(self.hwnd)
        except:
            return False

    def bring_to_front(self) -> bool:
        """Bring game window to front.

        Returns:
            True if successful, False otherwise
        """
        if sys.platform != 'win32' or not self.hwnd:
            return False

        try:
            win32gui.SetForegroundWindow(self.hwnd)
            return True
        except:
            return False
