import importlib.util
import sympy as sp
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

class BaseConfig:
    """Base configuration class with dynamic attribute handling."""
    
    def __init__(self, 
                 py_config_path: Optional[Path] = None,
                 args: Optional[Any] = None,
                 hparams: Optional[Dict] = None,
                 required_attributes: Optional[List[str]] = None,
                 expected_attrs: Optional[List[str]] = None):
        """Initialize configuration.
        
        Args:
            py_config_path: Path to Python config file
            args: Command line arguments
            hparams: Dictionary of hyperparameters
            required_attributes: List of required attribute names
            expected_attrs: List of expected attribute names
            
        Raises:
            ValueError: If a required attribute is missing
        """
        self.config_attrs: Dict[str, Any] = {}
        self.expected_attrs = expected_attrs or []

        if py_config_path:
            self.load_from_py_config(py_config_path)
        if hparams:
            self.update_from_dict(hparams)
        if args:
            self.update_from_args(args)

        self._generate_properties()
        
        if required_attributes:
            self._validate_attributes(required_attributes)

    def _generate_properties(self) -> None:
        """Generate getter and setter properties for all attributes in config_attrs.
        
        This method dynamically creates property objects for each attribute in config_attrs,
        providing controlled access with validation.
        """
        for attr_name in self.config_attrs.keys():
            # Create new getter and setter with proper closure
            def make_property(name: str) -> property:
                def getter(self) -> Any:
                    """Get the value of the attribute.
                    
                    Raises:
                        ValueError: If the attribute is not defined
                    """
                    value = self.config_attrs.get(name)
                    if value is None:
                        raise ValueError(f"{name} is not defined")
                    return value

                def setter(self, value: Any) -> None:
                    """Set the value of the attribute."""
                    self.config_attrs[name] = value

                return property(getter, setter)

            # Set the property on the class
            setattr(self.__class__, attr_name, make_property(attr_name))

    def _validate_attributes(self, required_attributes: List[str]) -> None:
        """Validate that all required attributes are set.
        
        Args:
            required_attributes: List of attribute names that must be defined
            
        Raises:
            ValueError: If any required attribute is missing or None
        """
        missing = []
        for attr in required_attributes:
            if attr not in self.config_attrs or self.config_attrs[attr] is None:
                missing.append(attr)
        
        if missing:
            raise ValueError(f"Missing required attributes: {', '.join(missing)}")

    def __str__(self):
        """Returns a string representation of the current configuration."""
        config_data = {k: v for k, v in self.__dict__.items()}
        return json.dumps(config_data, indent=4, default=str)

    def update_from_args(self, args):
        """Update configuration values from command-line arguments."""
        for arg, value in vars(args).items():
            if (value is not None) and (arg in self.expected_attrs):
                self.config_attrs[arg] = value  # Update the dictionary, not the instance

    def update_from_dict(self, hyperparams):
        """Update configuration values from a dictionary of hyperparameters."""
        for key, value in hyperparams.items():
            if key in self.expected_attrs:
                self.config_attrs[key] = value

    def load_from_py_config(self, filepath):
        """Load configuration from a Python configuration file."""
        spec = importlib.util.spec_from_file_location("config_module", filepath)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Copy all attributes from the imported module to config_attrs
        for key, value in config_module.__dict__.items():
            if not key.startswith("__"):
                if (key in self.expected_attrs):
                    self.config_attrs[key] = value  # Store in config_attrs dictionary


    def save_to_py(self, filepath):
        """Save the current configuration to a Python (.py) file."""
        def convert_to_py_syntax(value):
            """Converts Python objects (including sympy expressions) to Python syntax."""
            if isinstance(value, (sp.Basic, sp.Symbol)):
                return f"sp.{value}"
            elif isinstance(value, str):
                return f"'{value}'"
            elif isinstance(value, dict):
                dict_items = ", ".join(f"'{k}': {convert_to_py_syntax(v)}" for k, v in value.items())
                return f"{{{dict_items}}}"
            elif isinstance(value, (list, set, tuple)):
                container = "[" if isinstance(value, list) else "("
                close_container = "]" if isinstance(value, list) else ")"
                items = ", ".join(convert_to_py_syntax(v) for v in value)
                return f"{container}{items}{close_container}"
            else:
                return str(value)

        with open(filepath, 'w') as py_file:
            py_file.write("import sympy as sp\n\n")
            py_file.write("### Auto-generated Configuration ###\n\n")

            for key, value in self.__dict__.items():
                py_syntax_value = convert_to_py_syntax(value)
                py_file.write(f"{key} = {py_syntax_value}\n")

        print(f"Configuration saved to {filepath}")