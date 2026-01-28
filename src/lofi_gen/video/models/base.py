from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseVideoGen(ABC):
    """Base class for all music generation models."""

    def __init__(self, model_name: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.config = config or {}
        self._model = None
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Any:
        """Generate video based on input data."""
        pass
    