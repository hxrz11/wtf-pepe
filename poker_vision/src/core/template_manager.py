"""Template management for poker vision."""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TemplateManager:
    """Manages template storage and retrieval."""

    # Card ranks and suits
    CARD_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    CARD_SUITS = ['c', 'd', 'h', 's']  # clubs, diamonds, hearts, spades

    # Combo names
    COMBO_NAMES = [
        'high_card',
        'pair',
        'two_pair',
        'three_of_a_kind',
        'straight',
        'flush',
        'full_house',
        'four_of_a_kind',
        'straight_flush',
        'royal_flush'
    ]

    # Marker names
    MARKER_NAMES = ['dealer_button', 'timer', 'seat_occupied']

    # Digits and special characters
    DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.']

    def __init__(self, templates_dir: Path):
        """Initialize template manager.

        Args:
            templates_dir: Directory containing template subdirectories
        """
        self.templates_dir = templates_dir
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure all template subdirectories exist."""
        subdirs = ['cards', 'digits', 'combos', 'letters_lat',
                  'letters_cyr', 'special', 'markers']

        for subdir in subdirs:
            (self.templates_dir / subdir).mkdir(parents=True, exist_ok=True)

    def save_card_template(self, image: np.ndarray, rank: str, suit: str) -> Optional[Path]:
        """Save card template.

        Args:
            image: Card image
            rank: Card rank (2-9, T, J, Q, K, A)
            suit: Card suit (c, d, h, s)

        Returns:
            Path to saved template or None if failed
        """
        if rank not in self.CARD_RANKS:
            return None
        if suit not in self.CARD_SUITS:
            return None

        filename = f"{rank}{suit}.png"
        output_path = self.templates_dir / 'cards' / filename

        success = cv2.imwrite(str(output_path), image)
        return output_path if success else None

    def load_card_template(self, rank: str, suit: str) -> Optional[np.ndarray]:
        """Load card template.

        Args:
            rank: Card rank
            suit: Card suit

        Returns:
            Template image or None if not found
        """
        filename = f"{rank}{suit}.png"
        template_path = self.templates_dir / 'cards' / filename

        if not template_path.exists():
            return None

        return cv2.imread(str(template_path))

    def get_existing_cards(self) -> List[str]:
        """Get list of existing card templates.

        Returns:
            List of card identifiers (e.g., "2c", "Ah")
        """
        cards_dir = self.templates_dir / 'cards'
        if not cards_dir.exists():
            return []

        cards = []
        for file in cards_dir.glob("*.png"):
            # Extract card identifier from filename (e.g., "2c.png" -> "2c")
            card_id = file.stem
            cards.append(card_id)

        return sorted(cards)

    def get_cards_completion(self) -> Tuple[int, int]:
        """Get card templates completion status.

        Returns:
            Tuple of (existing_count, total_count)
        """
        total = len(self.CARD_RANKS) * len(self.CARD_SUITS)  # 52
        existing = len(self.get_existing_cards())
        return (existing, total)

    def save_combo_template(self, image: np.ndarray, combo_name: str) -> Optional[Path]:
        """Save combo template.

        Args:
            image: Combo image
            combo_name: Combo name (e.g., "pair", "flush")

        Returns:
            Path to saved template or None if failed
        """
        if combo_name not in self.COMBO_NAMES:
            return None

        filename = f"{combo_name}.png"
        output_path = self.templates_dir / 'combos' / filename

        success = cv2.imwrite(str(output_path), image)
        return output_path if success else None

    def get_existing_combos(self) -> List[str]:
        """Get list of existing combo templates.

        Returns:
            List of combo names
        """
        combos_dir = self.templates_dir / 'combos'
        if not combos_dir.exists():
            return []

        combos = []
        for file in combos_dir.glob("*.png"):
            combo_name = file.stem
            combos.append(combo_name)

        return sorted(combos)

    def save_marker_template(self, image: np.ndarray, marker_name: str) -> Optional[Path]:
        """Save marker template.

        Args:
            image: Marker image
            marker_name: Marker name (dealer_button, timer, seat_occupied)

        Returns:
            Path to saved template or None if failed
        """
        if marker_name not in self.MARKER_NAMES:
            return None

        filename = f"{marker_name}.png"
        output_path = self.templates_dir / 'markers' / filename

        success = cv2.imwrite(str(output_path), image)
        return output_path if success else None

    def get_existing_markers(self) -> List[str]:
        """Get list of existing marker templates.

        Returns:
            List of marker names
        """
        markers_dir = self.templates_dir / 'markers'
        if not markers_dir.exists():
            return []

        markers = []
        for file in markers_dir.glob("*.png"):
            marker_name = file.stem
            markers.append(marker_name)

        return sorted(markers)

    def save_digit_template(self, image: np.ndarray, digit: str) -> Optional[Path]:
        """Save digit template.

        Args:
            image: Digit image
            digit: Digit character (0-9, .)

        Returns:
            Path to saved template or None if failed
        """
        if digit not in self.DIGITS:
            return None

        # Use special naming for dot
        if digit == '.':
            filename = "dot.png"
        else:
            filename = f"{digit}.png"

        output_path = self.templates_dir / 'digits' / filename

        success = cv2.imwrite(str(output_path), image)
        return output_path if success else None

    def save_symbol_template(self, image: np.ndarray, symbol: str,
                            category: str = 'special') -> Optional[Path]:
        """Save symbol template.

        Args:
            image: Symbol image
            symbol: Symbol character
            category: Template category (letters_lat, letters_cyr, special)

        Returns:
            Path to saved template or None if failed
        """
        if category not in ['letters_lat', 'letters_cyr', 'special']:
            return None

        # Create safe filename
        if symbol.isalnum():
            filename = f"{symbol}.png"
        else:
            # For special characters, use their Unicode code point
            filename = f"char_{ord(symbol):04x}.png"

        output_path = self.templates_dir / category / filename

        success = cv2.imwrite(str(output_path), image)
        return output_path if success else None

    def load_all_templates(self, category: str) -> Dict[str, np.ndarray]:
        """Load all templates from a category.

        Args:
            category: Template category directory name

        Returns:
            Dictionary of template_name -> image
        """
        templates = {}
        category_dir = self.templates_dir / category

        if not category_dir.exists():
            return templates

        for file in category_dir.glob("*.png"):
            template_name = file.stem
            image = cv2.imread(str(file))
            if image is not None:
                templates[template_name] = image

        return templates

    def template_exists(self, category: str, name: str) -> bool:
        """Check if template exists.

        Args:
            category: Template category
            name: Template name (without extension)

        Returns:
            True if template exists, False otherwise
        """
        if category == 'digits' and name == '.':
            name = 'dot'

        template_path = self.templates_dir / category / f"{name}.png"
        return template_path.exists()

    def get_statistics(self) -> Dict[str, int]:
        """Get template statistics.

        Returns:
            Dictionary of category -> count
        """
        stats = {}

        categories = ['cards', 'digits', 'combos', 'letters_lat',
                     'letters_cyr', 'special', 'markers']

        for category in categories:
            category_dir = self.templates_dir / category
            if category_dir.exists():
                count = len(list(category_dir.glob("*.png")))
                stats[category] = count
            else:
                stats[category] = 0

        return stats
