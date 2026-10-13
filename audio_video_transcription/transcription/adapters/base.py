# Define the common contract for local or externally provided recognition engines.
from abc import ABC, abstractmethod


class RecognitionAdapter(ABC):
    @abstractmethod
    def recognize(self, asset, spec, config):
        raise NotImplementedError

    def close(self):
        return None
