import os
import time
import sys
from typing import Dict, Any, Optional, List

# Try to import rich for screen-based display
try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from utils.logging.base_logger import BaseLogger


class ScreenLogger(BaseLogger):
    """
    A logger that displays the last N iterations in a structured format on the screen.
    If rich is available, it uses rich for display. Otherwise, it falls back to a text-based display.
    """
    def __init__(self, max_iterations: int = 10, filepath: Optional[str] = None):
        """
        Initialize the screen logger.

        Args:
            max_iterations: Maximum number of iterations to display
            filepath: Optional filepath to write logs to
        """
        super(ScreenLogger, self).__init__(filepath=filepath)
        self.max_iterations = max_iterations
        self.history = []
        self.console = Console() if RICH_AVAILABLE else None

    def flush(self, prefix: Optional[str] = None) -> None:
        """
        Flush the buffer to the screen and optionally to the log file.

        Args:
            prefix: Optional prefix to display before the data
        """
        # Add current iteration to history
        if self.buffer:
            # Store the prefix as iteration info
            self.buffer['iteration_info'] = prefix if prefix else "Iteration"
            self.history.append(self.buffer.copy())
            if len(self.history) > self.max_iterations:
                self.history.pop(0)

        # Display the data
        self._display()

        # Write to log file if filepath is provided
        if self.filepath:
            with open(self.filepath, 'a') as f:
                if prefix:
                    f.write(f"{prefix} ")
                for key, value in self.buffer.items():
                    f.write(f"{key}={value} ")
                f.write("\n")

        # Clear the buffer
        self.buffer = {}

    def _display(self) -> None:
        """Display the current buffer and history."""
        if RICH_AVAILABLE:
            self._display_rich()
        else:
            self._display_text()

    def _display_rich(self) -> None:
        """Display using rich library."""
        # Create a table for the training progress
        table = Table(title="Training Progress")
        table.add_column("Iteration", justify="left", style="cyan")
        table.add_column("Learning Rate", justify="right", style="green")
        table.add_column("Loss", justify="right", style="red")
        table.add_column("Time (s)", justify="right", style="yellow")
        table.add_column("Peak Memory (MB)", justify="right", style="blue")
        table.add_column("GPU Util (%)", justify="right", style="magenta")

        # Add rows for each iteration in history
        for data in self.history:
            # Extract GPU stats from the buffer
            gpu_index = data.get("gpu_0_physical_index", 0)
            peak_memory = data.get(f"gpu_{gpu_index}_max_memory_mb", "-")
            gpu_util = data.get(f"gpu_{gpu_index}_current_util_percent", "-")

            table.add_row(
                data.get("iteration_info", "-"),
                self._format_value(data.get("learning_rate")),
                self._format_value(data.get("loss")),
                self._format_value(data.get("iteration_time")),
                self._format_value(peak_memory),
                self._format_value(gpu_util)
            )

        # Display the table without clearing the screen
        print("\n")  # Add a newline before the table
        self.console.print(table)
        print("\n")  # Add a newline after the table

    def _display_text(self) -> None:
        """Display using text-based format."""
        # Don't clear the screen, just print a separator
        print("\n" + "=" * 80 + "\n")

        # Print the current iteration
        print("Current Iteration:")
        for key, value in self.buffer.items():
            print(f"  {key}: {value}")

        # Print the history
        print("\nHistory:")
        for data in self.history:
            print(f"{data.get('iteration_info', 'Iteration')}:")
            for key, value in data.items():
                if key != 'iteration_info':  # Skip iteration_info as it's already printed
                    print(f"  {key}: {value}")

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "-"
        if isinstance(value, (int, float)):
            return f"{value:.2f}"
        return str(value)

    def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message: The message to log
        """
        if self.filepath:
            with open(self.filepath, 'a') as f:
                f.write(f"INFO: {message}\n")

        if RICH_AVAILABLE:
            self.console.print(f"[green]INFO:[/green] {message}")
        else:
            print(f"INFO: {message}")

    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: The message to log
        """
        if self.filepath:
            with open(self.filepath, 'a') as f:
                f.write(f"WARNING: {message}\n")

        if RICH_AVAILABLE:
            self.console.print(f"[yellow]WARNING:[/yellow] {message}")
        else:
            print(f"WARNING: {message}")

    def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: The message to log
        """
        if self.filepath:
            with open(self.filepath, 'a') as f:
                f.write(f"ERROR: {message}\n")

        if RICH_AVAILABLE:
            self.console.print(f"[red]ERROR:[/red] {message}")
        else:
            print(f"ERROR: {message}")

    def page_break(self) -> None:
        """Add a page break to the log."""
        if self.filepath:
            with open(self.filepath, 'a') as f:
                f.write("\n" + "=" * 80 + "\n\n")

        if RICH_AVAILABLE:
            self.console.print("\n" + "=" * 80 + "\n")
        else:
            print("\n" + "=" * 80 + "\n")