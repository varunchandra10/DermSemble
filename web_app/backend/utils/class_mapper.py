import json
from pathlib import Path
from backend.core.config import settings
from backend.core.logger import log

class ClassMapper:
    """
    Utility for mapping class IDs to human-readable disease labels.
    Loads the mapping from a JSON file.
    """
    def __init__(self, mapping_path: Path):
        self.mapping_path = mapping_path
        self.id_to_label = {}
        self.label_to_id = {}
        self._load_mapping()

    def _load_mapping(self):
        try:
            if not self.mapping_path.exists():
                log.warning(f"Label map not found at {self.mapping_path}. Using default fallback.")
                # Mock fallback for demonstration
                self.id_to_label = {str(i): f"Disease_{i}" for i in range(23)}
                self.label_to_id = {f"Disease_{i}": str(i) for i in range(23)}
                return

            with open(self.mapping_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Assume mapping is id -> label, e.g. {"0": "Melanoma", "1": "Nevus"}
                self.id_to_label = {str(k): str(v) for k, v in data.items()}
                self.label_to_id = {str(v): str(k) for k, v in data.items()}
                log.info(f"Loaded {len(self.id_to_label)} class mappings from {self.mapping_path}")
        except Exception as e:
            log.error(f"Failed to load label map from {self.mapping_path}: {e}")
            raise

    def get_label(self, class_id: int | str) -> str:
        """Returns the human-readable label for a given class ID."""
        return self.id_to_label.get(str(class_id), f"Unknown Class {class_id}")

    def get_id(self, label: str) -> str:
        """Returns the class ID for a given label."""
        return self.label_to_id.get(label, "-1")

    def get_all_classes(self) -> list[str]:
        """Returns a list of all labels ordered by ID."""
        # Ensure ordered by ID if possible
        try:
            sorted_keys = sorted(self.id_to_label.keys(), key=int)
            return [self.id_to_label[k] for k in sorted_keys]
        except ValueError:
            return list(self.id_to_label.values())

# Global instance
class_mapper = ClassMapper(settings.LABEL_MAP_PATH)
