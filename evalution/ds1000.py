from dataclasses import dataclass
from typing import Any, Dict
import json
from pathlib import Path
from typing import Dict, List

@dataclass
class DS1000Problem:
    prompt: str
    reference_code: str
    metadata: Dict[str, Any]
    code_context: str

    def __getitem__(self, key: str) -> Any:
        """
        Allows dictionary-like access to the problem fields.
        If the key is one of the top-level attributes, return that.
        Otherwise, check inside metadata. For 'lib', return the value of 'library'.
        """
        if key in {"prompt", "reference_code", "code_context"}:
            return getattr(self, key)
        elif key == "lib":
            return self.metadata.get("library")
        else:
            return self.metadata.get(key)

    @property
    def problem_id(self) -> Any:
        """Return the problem id stored in metadata."""
        return self.metadata.get("problem_id")
class DS1000Dataset:
    def __init__(self, dataset_path: str, mode: str):
        """
        Initializes the DS1000Dataset.
        Assumes that the dataset_path is a JSONL file where each line is a JSON object
        representing a DS1000Problem.
        
        Args:
            dataset_path (str): Path to the JSONL file.
            mode (str): Mode of operation (e.g., "Insertion" or "Completion").
        """
        self.mode = mode
        self.data: Dict[str, List[DS1000Problem]] = {"all": []}
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file {dataset_path} not found.")
        
        with dataset_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # ensure non-empty line
                    problem_json = json.loads(line)
                    # Create a DS1000Problem instance from the JSON dict.
                    # The keys of the JSON must match the DS1000Problem attributes.
                    problem = DS1000Problem(
                        prompt=problem_json.get("prompt", ""),
                        reference_code=problem_json.get("reference_code", ""),
                        metadata=problem_json.get("metadata", {}),
                        code_context=problem_json.get("code_context", "")
                    )
                    self.data["all"].append(problem)