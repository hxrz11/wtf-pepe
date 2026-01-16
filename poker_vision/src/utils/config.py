"""Configuration management for Poker Vision Tool."""
import json
import os
from pathlib import Path
from typing import Dict, Any


class Config:
    """Manages application configuration."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize config manager.

        Args:
            config_path: Path to config.json file
        """
        self.config_path = Path(config_path)
        self.config_data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config_data = json.load(f)

    def save(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        self.config_data[key] = value

    @property
    def window_title(self) -> str:
        """Get game window title."""
        return self.get('window_title', 'TON Poker')

    @property
    def game_window_width(self) -> int:
        """Get game window width."""
        return self.get('game_window_width', 631)

    @property
    def game_window_height(self) -> int:
        """Get game window height."""
        return self.get('game_window_height', 958)

    @property
    def screenshot_interval_ms(self) -> int:
        """Get screenshot interval in milliseconds."""
        return self.get('screenshot_interval_ms', 2000)

    @property
    def screenshots_dir(self) -> Path:
        """Get screenshots directory path."""
        return Path(self.get('screenshots_dir', 'screenshots'))

    @property
    def regions_cut_dir(self) -> Path:
        """Get regions cut directory path."""
        return Path(self.get('regions_cut_dir', 'regions_cut'))

    @property
    def templates_dir(self) -> Path:
        """Get templates directory path."""
        return Path(self.get('templates_dir', 'templates'))


class RegionsConfig:
    """Manages regions configuration."""

    def __init__(self, regions_path: str = "regions.json"):
        """Initialize regions config manager.

        Args:
            regions_path: Path to regions.json file
        """
        self.regions_path = Path(regions_path)
        self.regions_data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load regions from file."""
        if not self.regions_path.exists():
            raise FileNotFoundError(f"Regions file not found: {self.regions_path}")

        with open(self.regions_path, 'r', encoding='utf-8') as f:
            self.regions_data = json.load(f)

    def save(self) -> None:
        """Save regions to file."""
        with open(self.regions_path, 'w', encoding='utf-8') as f:
            json.dump(self.regions_data, f, indent=2, ensure_ascii=False)

    def get_region(self, region_id: str) -> Dict[str, Any]:
        """Get region configuration.

        Args:
            region_id: Region identifier

        Returns:
            Region configuration dict with keys: x, y, w, h, type
        """
        return self.regions_data.get(region_id, {})

    def set_region(self, region_id: str, x: int, y: int, w: int, h: int) -> None:
        """Set region coordinates.

        Args:
            region_id: Region identifier
            x: X coordinate
            y: Y coordinate
            w: Width
            h: Height
        """
        if region_id in self.regions_data:
            self.regions_data[region_id].update({'x': x, 'y': y, 'w': w, 'h': h})

    def get_all_regions(self) -> Dict[str, Dict[str, Any]]:
        """Get all regions.

        Returns:
            Dictionary of all regions
        """
        return self.regions_data.copy()

    def get_regions_by_type(self, region_type: str) -> Dict[str, Dict[str, Any]]:
        """Get regions filtered by type.

        Args:
            region_type: Type of regions (card, text_digits, text_mixed, marker, combo)

        Returns:
            Dictionary of regions matching the type
        """
        return {
            region_id: region_data
            for region_id, region_data in self.regions_data.items()
            if region_data.get('type') == region_type
        }
