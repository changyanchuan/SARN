'''
Basic trainer class for encoders.
The detailed encoder definition should not be here.
'''

from abc import ABC, abstractmethod

class BaseEncoder(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def get_embeddings(self, from_checkpoint):
        pass

