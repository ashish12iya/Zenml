from abc import ABC, abstractmethod
import pandas as pd
from typing import Protocol, TypeVar, Generic, Dict, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Type hints for generic data types
T = TypeVar('T')
OutputT = TypeVar('OutputT')

class DataProcessor(Protocol):
    """Protocol defining the interface for data processors"""
    def process_data(self) -> Any:
        ...

@dataclass
class SplitConfig:
    """Immutable configuration for split parameters"""
    test_size: float = 0.2
    random_state: int | None = None
    shuffle: bool = True
    stratify: Any = None

class ProcessStrategy(ABC, Generic[T, OutputT]):
    """Abstract base class for all processing strategies"""
    
    @abstractmethod
    def process_data(self) -> OutputT:
        """Abstract method that must be implemented by all concrete strategies"""
        pass

    @abstractmethod
    def validate_input(self) -> bool:
        """Abstract method for input validation"""
        pass

class SplitData(ProcessStrategy[pd.DataFrame, tuple]):
    """
    Enhanced data splitting strategy using advanced OOP concepts.
    
    Features:
    - Immutable configuration using dataclass
    - Generic type hints
    - Method chaining support
    - Singleton registry for strategy instances
    """
    
    _instances: Dict[str, 'SplitData'] = {}  # Class variable for singleton pattern
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern with registry"""
        dataset_id = id(args[0]) if args else None
        if dataset_id not in cls._instances:
            cls._instances[dataset_id] = super().__new__(cls)
        return cls._instances[dataset_id]

    def __init__(self, DataFrame: pd.DataFrame, **kwargs):
        # Only initialize if not already initialized (part of singleton pattern)
        if not hasattr(self, 'dataframe'):
            self.dataframe = DataFrame
            self.config = SplitConfig(**kwargs)
            self._split_result = None
    
    def validate_input(self) -> bool:
        """Validate input data"""
        if not isinstance(self.dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if self.dataframe.empty:
            raise ValueError("DataFrame cannot be empty")
        return True

    def process_data(self) -> tuple:
        """Process data with caching support"""
        if self._split_result is None:
            self.validate_input()
            X = self.dataframe.iloc[:, :-1]
            y = self.dataframe.iloc[:, -1]
            
            self._split_result = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                shuffle=self.config.shuffle,
                stratify=self.config.stratify
            )
        return self._split_result

    def with_config(self, **kwargs) -> 'SplitData':
        """Method chaining for configuration updates"""
        self.config = SplitConfig(**kwargs)
        self._split_result = None  # Reset cache
        return self

    @classmethod
    def get_instance(cls, dataset: pd.DataFrame) -> 'SplitData':
        """Get singleton instance for a specific dataset"""
        return cls(dataset)

# Example Factory for creating different types of data processors
class DataProcessorFactory:
    @staticmethod
    def create_processor(processor_type: str, data: pd.DataFrame, **kwargs) -> DataProcessor:
        processors = {
            'split': SplitData,
            # Add more processors here
        }
        if processor_type not in processors:
            raise ValueError(f"Unknown processor type: {processor_type}")
        return processors[processor_type](data, **kwargs)