"""
Configuration manager for Molt-See vision preferences.

Handles reading and writing vision configuration such as analyzer mode,
relevance thresholds, and integration settings.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

# Define paths directly to avoid import issues when loaded via importlib
PROJECT_DIR = Path(__file__).parent.parent.parent  # molt-see root
RUNTIME_DIR = PROJECT_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = RUNTIME_DIR / "molt_see_config.json"


class VisionConfigManager:
    """Manages Molt-See configuration stored in JSON format."""

    # Default configuration values
    DEFAULT_CONFIG = {
        "analyzer_mode": "cascade",  # Options: "yolo", "vlm", "cascade"
        "vlm_provider": "anthropic",  # Options: "anthropic", "openai"
        "vlm_api_key": None,
        "vlm_model": "claude-sonnet-4-5-20250929",
        "capture_fps": 1.0,
        "relevance_threshold": 0.6,
        "observation_cooldown": 30.0,
        "camera_device": 0,
        "enable_speak_integration": True,
        "molt_speak_runtime_dir": None,
        "yolo_model": "yolov8n",
        "change_threshold": 0.92,
        "enabled_categories": [
            "person_detected",
            "person_left",
            "object_change",
            "context_relevant",
            "environment_change",
            "safety_alert",
        ],
    }

    # Relevance threshold presets (mirroring Molt-Speak's mic_sensitivity pattern)
    RELEVANCE_PRESETS = {
        "low": 0.8,  # Only report very relevant observations
        "medium": 0.6,  # Balanced (default)
        "high": 0.4,  # Report most observations
        "chatty": 0.2,  # Report almost everything (demo/testing)
    }

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the config manager.

        Args:
            config_path: Path to config file. Defaults to CONFIG_FILE.
        """
        self.config_path = config_path or CONFIG_FILE
        self._ensure_config_exists()

    def _ensure_config_exists(self) -> None:
        """Create config file with defaults if it doesn't exist."""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_config(self.DEFAULT_CONFIG)

    def _read_config(self) -> Dict[str, Any]:
        """
        Read configuration from file.

        Returns:
            Configuration dictionary.
        """
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            # Ensure all default keys exist
            for key, value in self.DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
        except (json.JSONDecodeError, FileNotFoundError):
            return self.DEFAULT_CONFIG.copy()

    def _write_config(self, config: Dict[str, Any]) -> None:
        """
        Write configuration to file.

        Args:
            config: Configuration dictionary to write.
        """
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key.
            default: Default value if key doesn't exist.

        Returns:
            Configuration value.
        """
        config = self._read_config()
        return config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        Args:
            key: Configuration key.
            value: Value to set.
        """
        config = self._read_config()
        config[key] = value
        self._write_config(config)

    def get_all(self) -> Dict[str, Any]:
        """
        Get all configuration values.

        Returns:
            Complete configuration dictionary.
        """
        return self._read_config()

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update multiple configuration values at once.

        Args:
            updates: Dictionary of key-value pairs to update.
        """
        config = self._read_config()
        config.update(updates)
        self._write_config(config)

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._write_config(self.DEFAULT_CONFIG.copy())

    @property
    def analyzer_mode(self) -> str:
        """Get scene analyzer mode (yolo, vlm, cascade)."""
        return self.get("analyzer_mode", self.DEFAULT_CONFIG["analyzer_mode"])

    @analyzer_mode.setter
    def analyzer_mode(self, mode: str) -> None:
        """Set scene analyzer mode."""
        if mode in ("yolo", "vlm", "cascade"):
            self.set("analyzer_mode", mode)

    @property
    def vlm_provider(self) -> str:
        """Get VLM provider."""
        return self.get("vlm_provider", self.DEFAULT_CONFIG["vlm_provider"])

    @vlm_provider.setter
    def vlm_provider(self, provider: str) -> None:
        """Set VLM provider."""
        self.set("vlm_provider", provider)

    @property
    def vlm_api_key(self) -> Optional[str]:
        """Get VLM API key."""
        return self.get("vlm_api_key", self.DEFAULT_CONFIG["vlm_api_key"])

    @vlm_api_key.setter
    def vlm_api_key(self, key: str) -> None:
        """Set VLM API key."""
        self.set("vlm_api_key", key)

    @property
    def capture_fps(self) -> float:
        """Get capture frames per second."""
        return self.get("capture_fps", self.DEFAULT_CONFIG["capture_fps"])

    @capture_fps.setter
    def capture_fps(self, fps: float) -> None:
        """Set capture frames per second."""
        if 0.1 <= fps <= 10.0:
            self.set("capture_fps", fps)

    @property
    def relevance_threshold(self) -> float:
        """Get relevance threshold (0.0-1.0)."""
        return self.get(
            "relevance_threshold", self.DEFAULT_CONFIG["relevance_threshold"]
        )

    @relevance_threshold.setter
    def relevance_threshold(self, threshold: float) -> None:
        """Set relevance threshold."""
        if 0.0 <= threshold <= 1.0:
            self.set("relevance_threshold", threshold)

    def set_relevance_preset(self, preset: str) -> None:
        """
        Set relevance threshold from a named preset.

        Args:
            preset: One of "low", "medium", "high", "chatty".
        """
        if preset in self.RELEVANCE_PRESETS:
            self.relevance_threshold = self.RELEVANCE_PRESETS[preset]

    @property
    def observation_cooldown(self) -> float:
        """Get observation cooldown in seconds."""
        return self.get(
            "observation_cooldown", self.DEFAULT_CONFIG["observation_cooldown"]
        )

    @observation_cooldown.setter
    def observation_cooldown(self, seconds: float) -> None:
        """Set observation cooldown."""
        if seconds >= 0:
            self.set("observation_cooldown", seconds)

    @property
    def camera_device(self) -> int:
        """Get camera device index."""
        return self.get("camera_device", self.DEFAULT_CONFIG["camera_device"])

    @camera_device.setter
    def camera_device(self, index: int) -> None:
        """Set camera device index."""
        if index >= 0:
            self.set("camera_device", index)
