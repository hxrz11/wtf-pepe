"""Region cutting from screenshots."""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class RegionCutter:
    """Handles cutting regions from screenshots."""

    def __init__(self, screenshots_dir: Path, output_dir: Path):
        """Initialize region cutter.

        Args:
            screenshots_dir: Directory containing screenshots
            output_dir: Directory to save cut regions
        """
        self.screenshots_dir = screenshots_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_screenshot_files(self) -> List[Path]:
        """Get list of screenshot files.

        Returns:
            List of paths to screenshot files
        """
        if not self.screenshots_dir.exists():
            return []

        # Get all PNG files
        screenshots = list(self.screenshots_dir.glob("*.png"))
        screenshots.sort()
        return screenshots

    def cut_region(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[np.ndarray]:
        """Cut region from image.

        Args:
            image: Source image
            x: X coordinate
            y: Y coordinate
            w: Width
            h: Height

        Returns:
            Cut region as numpy array or None if invalid
        """
        if w <= 0 or h <= 0:
            return None

        if x < 0 or y < 0:
            return None

        if x + w > image.shape[1] or y + h > image.shape[0]:
            return None

        return image[y:y+h, x:x+w].copy()

    def cut_region_from_file(self, screenshot_path: Path, region_id: str,
                            x: int, y: int, w: int, h: int) -> Optional[Path]:
        """Cut region from screenshot file and save.

        Args:
            screenshot_path: Path to screenshot file
            region_id: Region identifier
            x: X coordinate
            y: Y coordinate
            w: Width
            h: Height

        Returns:
            Path to saved region file or None if failed
        """
        # Load screenshot
        image = cv2.imread(str(screenshot_path))
        if image is None:
            return None

        # Cut region
        region = self.cut_region(image, x, y, w, h)
        if region is None:
            return None

        # Create output directory for this region
        region_dir = self.output_dir / region_id
        region_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        screenshot_name = screenshot_path.stem  # e.g., "scrn_123456_789"
        output_filename = f"{screenshot_name}_{region_id}.png"
        output_path = region_dir / output_filename

        # Save region
        success = cv2.imwrite(str(output_path), region)
        if success:
            return output_path

        return None

    def cut_regions_from_screenshot(self, screenshot_path: Path,
                                    regions: Dict[str, Dict[str, int]]) -> Dict[str, Path]:
        """Cut multiple regions from a screenshot.

        Args:
            screenshot_path: Path to screenshot file
            regions: Dictionary of region_id -> {x, y, w, h, type}

        Returns:
            Dictionary of region_id -> saved file path
        """
        results = {}

        # Load screenshot once
        image = cv2.imread(str(screenshot_path))
        if image is None:
            return results

        # Cut each region
        for region_id, coords in regions.items():
            x = coords.get('x', 0)
            y = coords.get('y', 0)
            w = coords.get('w', 0)
            h = coords.get('h', 0)
            region_type = coords.get('type', '')

            # Skip invalid regions
            if w <= 0 or h <= 0:
                continue

            # Cut region
            region = self.cut_region(image, x, y, w, h)
            if region is None:
                continue

            # Convert to grayscale for card_rank and card_suit
            if region_type in ['card_rank', 'card_suit']:
                if len(region.shape) == 3:
                    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

            # Create output directory for this region
            region_dir = self.output_dir / region_id
            region_dir.mkdir(parents=True, exist_ok=True)

            # Generate output filename
            screenshot_name = screenshot_path.stem
            output_filename = f"{screenshot_name}_{region_id}.png"
            output_path = region_dir / output_filename

            # Save region
            success = cv2.imwrite(str(output_path), region)
            if success:
                results[region_id] = output_path

        return results

    def cut_regions_from_all_screenshots(self, regions: Dict[str, Dict[str, int]],
                                        selected_region_ids: Optional[List[str]] = None) -> Tuple[int, int]:
        """Cut regions from all screenshots.

        Args:
            regions: Dictionary of region_id -> {x, y, w, h}
            selected_region_ids: List of region IDs to cut (None = all)

        Returns:
            Tuple of (total_screenshots_processed, total_regions_cut)
        """
        screenshots = self.get_screenshot_files()
        if not screenshots:
            return (0, 0)

        # Filter regions if needed
        if selected_region_ids:
            regions_to_cut = {
                region_id: regions[region_id]
                for region_id in selected_region_ids
                if region_id in regions
            }
        else:
            regions_to_cut = regions

        total_regions_cut = 0
        screenshots_processed = 0

        for screenshot_path in screenshots:
            results = self.cut_regions_from_screenshot(screenshot_path, regions_to_cut)
            total_regions_cut += len(results)
            screenshots_processed += 1

        return (screenshots_processed, total_regions_cut)

    def get_cut_regions_count(self) -> Dict[str, int]:
        """Get count of cut regions for each region ID.

        Returns:
            Dictionary of region_id -> count of files
        """
        counts = {}

        if not self.output_dir.exists():
            return counts

        # Iterate through region directories
        for region_dir in self.output_dir.iterdir():
            if region_dir.is_dir():
                region_id = region_dir.name
                # Count PNG files in directory
                png_files = list(region_dir.glob("*.png"))
                counts[region_id] = len(png_files)

        return counts

    def get_region_files(self, region_id: str) -> List[Path]:
        """Get list of cut region files for a specific region.

        Args:
            region_id: Region identifier

        Returns:
            List of paths to region files
        """
        region_dir = self.output_dir / region_id
        if not region_dir.exists():
            return []

        files = list(region_dir.glob("*.png"))
        files.sort()
        return files

    def visualize_regions(self, image: np.ndarray,
                         regions: Dict[str, Dict[str, int]],
                         selected_region_ids: Optional[List[str]] = None) -> np.ndarray:
        """Visualize regions on image.

        Args:
            image: Source image
            regions: Dictionary of region_id -> {x, y, w, h}
            selected_region_ids: List of region IDs to visualize (None = all)

        Returns:
            Image with regions drawn
        """
        result = image.copy()

        # Define colors for different region types
        color_map = {
            'card': (0, 255, 0),         # Green
            'card_rank': (255, 255, 0),  # Cyan
            'card_suit': (255, 0, 255),  # Magenta
            'card_full': (0, 255, 255),  # Yellow
            'text_digits': (255, 0, 0),  # Blue
            'text_mixed': (0, 255, 255), # Yellow
            'marker': (128, 0, 128),     # Purple
            'combo': (0, 165, 255)       # Orange
        }

        for region_id, coords in regions.items():
            # Skip if not in selected list
            if selected_region_ids and region_id not in selected_region_ids:
                continue

            x = coords.get('x', 0)
            y = coords.get('y', 0)
            w = coords.get('w', 0)
            h = coords.get('h', 0)

            # Skip invalid regions
            if w <= 0 or h <= 0:
                continue

            # Get color based on region type
            region_type = coords.get('type', 'unknown')
            color = color_map.get(region_type, (128, 128, 128))

            # Draw rectangle
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

            # Draw label
            label = region_id
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1

            # Get text size for background
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw background for text
            cv2.rectangle(result, (x, y - text_h - 4), (x + text_w, y), color, -1)

            # Draw text
            cv2.putText(result, label, (x, y - 2), font, font_scale,
                       (255, 255, 255), thickness, cv2.LINE_AA)

        return result
