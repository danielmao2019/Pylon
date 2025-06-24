from typing import List, Set, Dict, Union, Any
import os


class ConfigToFile:
    """Convert Python config dictionaries to properly formatted config files with imports."""

    def __init__(self):
        self.imports: Set[str] = set()
        self.from_imports: Dict[str, Set[str]] = {}
        self.config_lines: List[str] = []

    def _add_import(self, module: str) -> None:
        """Add a regular import statement."""
        self.imports.add(module)

    def _add_from_import(self, module: str, item: str) -> None:
        """Add a from...import statement."""
        if module not in self.from_imports:
            self.from_imports[module] = set()
        self.from_imports[module].add(item)

    def _format_value(self, value: Any, indent: int = 0) -> str:
        """
        Format a value with rule-based formatting:
        Rule 1: For dictionaries, always newline for every key
        Rule 2: For lists/tuples, never newline (always single line)
        """
        indent_str = '    ' * indent

        if value is None:
            return "None"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return repr(value)
        elif isinstance(value, list):
            # Rule 2: Lists never newline - always single line
            if not value:
                return "[]"
            items = [self._format_value(item, 0) for item in value]
            return f"[{', '.join(items)}]"
        elif isinstance(value, tuple):
            # Rule 2: Tuples never newline - always single line
            # BUT: if tuple contains dictionaries, format them properly with correct indentation
            if not value:
                return "()"

            # Format each item with the tuple's indent level
            formatted_items = []
            has_multiline = False
            for item in value:
                formatted_item = self._format_value(item, indent + 1)  # Format with proper tuple indent
                formatted_items.append(formatted_item)
                if '\n' in formatted_item:
                    has_multiline = True

            if has_multiline:
                # Format tuple with proper indentation for multi-line items
                indent_str = '    ' * indent
                next_indent_str = '    ' * (indent + 1)
                lines = ["("]
                for i, formatted_item in enumerate(formatted_items):
                    # Add proper indentation to each line of the formatted item
                    if '\n' in formatted_item:
                        # Multi-line item - add indentation to each line
                        item_lines = formatted_item.split('\n')
                        indented_lines = []
                        for line in item_lines:
                            if line.strip():  # Only indent non-empty lines
                                indented_lines.append(f"{next_indent_str}{line}")
                            else:
                                indented_lines.append(line)
                        formatted_with_indent = '\n'.join(indented_lines)
                    else:
                        # Single line item
                        formatted_with_indent = f"{next_indent_str}{formatted_item}"

                    if i < len(formatted_items) - 1:
                        lines.append(f"{formatted_with_indent},")
                    else:
                        lines.append(formatted_with_indent)
                lines.append(f"{indent_str})")
                return '\n'.join(lines)
            else:
                # All items are single-line, use original format
                return f"({', '.join(formatted_items)})"
        elif isinstance(value, dict):
            # Rule 1: Dictionaries always newline for every key
            if not value:
                return "{}"

            lines = ["{"]

            # Handle build_from_config dicts with special 'class' key handling
            if 'class' in value and hasattr(value.get('class'), '__module__'):
                cls = value['class']
                class_name = cls.__name__
                lines.append(f"{indent_str}    'class': {class_name},")

                for k, v in value.items():
                    if k != 'class':
                        v_str = self._format_value(v, indent + 1)
                        lines.append(f"{indent_str}    '{k}': {v_str},")
            else:
                # Regular dictionary
                for k, v in value.items():
                    v_str = self._format_value(v, indent + 1)
                    lines.append(f"{indent_str}    '{k}': {v_str},")

            lines.append(f"{indent_str}}}")
            return '\n'.join(lines)
        elif hasattr(value, '__module__') and hasattr(value, '__name__'):
            # This is a class or function reference
            return value.__name__
        else:
            # Try to represent it as a string
            return repr(value)

    def _analyze_value(self, value: Any, path: str = "") -> Any:
        """
        Analyze a value to collect imports. Returns the value with class references preserved.
        This prepares the value for formatting by _format_value.
        """
        if value is None:
            return None
        elif isinstance(value, (bool, int, float, str)):
            return value
        elif isinstance(value, list):
            return [self._analyze_value(item, f"{path}[{i}]") for i, item in enumerate(value)]
        elif isinstance(value, tuple):
            return tuple(self._analyze_value(item, f"{path}[{i}]") for i, item in enumerate(value))
        elif isinstance(value, dict):
            if 'class' in value and hasattr(value['class'], '__module__'):
                # This is a build_from_config style dictionary
                cls = value['class']
                module = cls.__module__
                class_name = cls.__name__

                # Add the import
                self._add_from_import(module, class_name)

                # Return dict with class preserved
                result = {'class': cls}
                for k, v in value.items():
                    if k != 'class':
                        result[k] = self._analyze_value(v, f'{path}[{k}]')
                return result
            else:
                # Regular dictionary
                return {k: self._analyze_value(v, f'{path}[{k}]') for k, v in value.items()}
        elif hasattr(value, '__module__') and hasattr(value, '__name__'):
            # This is a class or function reference
            module = value.__module__
            name = value.__name__
            self._add_from_import(module, name)
            return value
        else:
            return value

    def _format_imports(self) -> List[str]:
        """Format all collected imports in a consistent order."""
        lines = []

        # Add typing imports first if needed
        typing_imports = sorted(self.from_imports.get('typing', []))
        if typing_imports:
            lines.append(f"from typing import {', '.join(typing_imports)}")

        # Add standard library imports
        stdlib_imports = []
        for imp in sorted(self.imports):
            if imp in ['os', 'sys', 'copy', 'itertools', 'random']:
                stdlib_imports.append(imp)
        for imp in stdlib_imports:
            lines.append(f"import {imp}")

        # Add from imports for standard library
        stdlib_from = {}
        for module, items in sorted(self.from_imports.items()):
            if module != 'typing' and module.split('.')[0] in ['os', 'sys', 'copy', 'itertools', 'random']:
                stdlib_from[module] = sorted(items)
        for module, items in stdlib_from.items():
            lines.append(f"from {module} import {', '.join(items)}")

        # Add external package imports (numpy, torch, etc.)
        external_imports = []
        for imp in sorted(self.imports):
            if imp not in ['os', 'sys', 'copy', 'itertools', 'random'] and not imp.startswith(('data.', 'models.', 'criteria.', 'metrics.', 'utils.')):
                external_imports.append(imp)
        for imp in external_imports:
            lines.append(f"import {imp}")

        # Add from imports for external packages
        external_from = {}
        for module, items in sorted(self.from_imports.items()):
            if (module != 'typing' and
                module.split('.')[0] not in ['os', 'sys', 'copy', 'itertools', 'random'] and
                not module.startswith(('data.', 'models.', 'criteria.', 'metrics.', 'utils.', 'runners.', 'optimizers.', 'schedulers.'))):
                external_from[module] = sorted(items)
        for module, items in external_from.items():
            lines.append(f"from {module} import {', '.join(items)}")

        # Add project imports
        project_from = {}
        for module, items in sorted(self.from_imports.items()):
            if module.startswith(('data.', 'models.', 'criteria.', 'metrics.', 'utils.', 'runners.', 'optimizers.', 'schedulers.')):
                project_from[module] = sorted(items)
        for module, items in project_from.items():
            lines.append(f"from {module} import {', '.join(items)}")

        return lines

    def generate_config_file(
        self,
        config: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> str:
        """
        Generate a Python config file from a config dictionary or list.

        Args:
            config: The configuration dictionary or list of dictionaries (for multi-stage)

        Returns:
            The generated Python file content as a string
        """
        # Reset state
        self.imports = set()
        self.from_imports = {}
        self.config_lines = []

        # Build the config lines
        lines = []

        # Analyze the config to collect imports and prepare for formatting
        analyzed_config = self._analyze_value(config, 'config')

        # Add imports
        import_lines = self._format_imports()
        if import_lines:
            lines.extend(import_lines)
            lines.append('')

        # Add config definition with proper formatting
        lines.append('# Configuration')
        config_str = self._format_value(analyzed_config, 0)
        lines.append(f'config = {config_str}')

        return '\n'.join(lines)


def add_heading(content: str, generator_path: str) -> str:
    """
    Add auto-generated file header to config file content.

    Args:
        content: The config file content
        generator_path: Path to the generator file

    Returns:
        Content with header added
    """
    header_lines = [
        f"# This file is automatically generated by `{generator_path}`.",
        "# Please do not attempt to modify manually.",
        ""
    ]
    return "\n".join(header_lines) + content


def dict_to_config_file(
    config: Union[Dict[str, Any], List[Dict[str, Any]]],
    output_path: str = None
) -> str:
    """
    Convert a config dictionary or list to a Python config file.

    Args:
        config: The configuration dictionary or list of dictionaries (for multi-stage)
        output_path: Optional path to write the file to

    Returns:
        The generated file content
    """
    converter = ConfigToFile()

    # Generate the config file content
    content = converter.generate_config_file(config=config)

    # Write to file if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(content)

    return content
