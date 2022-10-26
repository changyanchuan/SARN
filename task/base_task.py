from abc import ABC, abstractmethod

# base class for downstream tasks
class BaseTask(ABC):

    @abstractmethod
    def train(self):
        pass
