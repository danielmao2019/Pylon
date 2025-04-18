from typing import Any, Optional
from rich.console import Console
from rich.table import Table
from rich.live import Live
from utils.logging.base_logger import BaseLogger


class ScreenLogger(BaseLogger):
    """
    A logger that displays the last N iterations in a structured format on the screen.
    Uses rich for a live-updating display of training metrics.
    """
    def __init__(
        self,
        max_iterations: int = 10,
        filepath: Optional[str] = None,
        layout: str = None,
    ):
        """
        Initialize the screen logger.

        Args:
            max_iterations: Maximum number of iterations to display
            filepath: Optional filepath to write logs to
            layout: Display layout, either "train" or "eval"
        """
        assert layout in ["train", "eval"], "layout must be either 'train' or 'eval'"
        super(ScreenLogger, self).__init__(filepath=filepath)
        self.max_iterations = max_iterations
        self.history = []
        self.console = Console()
        self.live = None
        self.display_started = False
        self.layout = layout

    def train(self) -> None:
        """Switch to training mode and reset history."""
        self.layout = "train"
        self.history = []
        self.flush("Starting training epoch")

    def eval(self) -> None:
        """Switch to evaluation mode and reset history."""
        self.layout = "eval"
        self.history = []
        self.flush("Starting validation epoch")

    def flush(self, prefix: str) -> None:
        """
        Flush the buffer to the screen and optionally to the log file.

        Args:
            prefix: Optional prefix to display before the data
        """
        assert isinstance(prefix, str), "prefix must be a string"
        # Add current iteration to history
        if self.buffer:
            # Store the prefix as iteration info
            self.buffer['iteration_info'] = prefix
            self.history.append(self.buffer.copy())
            if len(self.history) > self.max_iterations:
                self.history.pop(0)

            # Start the display if this is the first iteration
            if not self.display_started and self.history:
                self._start_display()
                self.display_started = True

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

    def _start_display(self):
        """Start the live display when the first iteration begins."""
        # Start the live display with an empty string
        self.live = Live("", refresh_per_second=4, auto_refresh=True)
        self.live.start()

        # Update with the first table immediately
        self._display()

    def _create_table(self) -> Table:
        """Create a table based on the current layout."""
        table = Table()
        table.add_column("Iteration", justify="left", style="cyan")

        if self.layout == "train":
            table.add_column("Learning Rate", justify="right", style="green")
            table.add_column("Losses", justify="right", style="red")
        else:  # eval layout
            table.add_column("Scores", justify="right", style="red")

        table.add_column("Time (s)", justify="right", style="yellow")
        table.add_column("Memory (MB)", justify="right", style="blue")
        table.add_column("GPU Util (%)", justify="right", style="magenta")

        return table

    def _display(self) -> None:
        """Display the current buffer and history."""
        # Create a table for the training progress
        table = self._create_table()

        # Add rows for each iteration in history
        for data in self.history:
            # Extract GPU stats from the buffer
            peak_memory = data.get("max_memory", "-")
            gpu_util = data.get("gpu_util", "-")

            if self.layout == "train":
                table.add_row(
                    data.get("iteration_info", "-"),
                    self._format_value(data.get("learning_rate")),
                    self._format_value(data.get("losses")),
                    self._format_value(data.get("iteration_time")),
                    self._format_value(peak_memory),
                    self._format_value(gpu_util)
                )
            else:  # eval layout
                table.add_row(
                    data.get("iteration_info", "-"),
                    self._format_value(data.get("scores")),
                    self._format_value(data.get("iteration_time")),
                    self._format_value(peak_memory),
                    self._format_value(gpu_util)
                )

        # Update the live display
        if self.live is not None:
            self.live.update(table)
        else:
            # Fallback if live display is not available
            self.console.print(table)

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

        self.console.print(f"[green]INFO:[/green] {message}")

    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message: The message to log
        """
        if self.filepath:
            with open(self.filepath, 'a') as f:
                f.write(f"WARNING: {message}\n")

        self.console.print(f"[yellow]WARNING:[/yellow] {message}")

    def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message: The message to log
        """
        if self.filepath:
            with open(self.filepath, 'a') as f:
                f.write(f"ERROR: {message}\n")

        self.console.print(f"[red]ERROR:[/red] {message}")

    def page_break(self) -> None:
        """Add a page break to the log."""
        if self.filepath:
            with open(self.filepath, 'a') as f:
                f.write("\n" + "=" * 80 + "\n\n")

        self.console.print("\n" + "=" * 80 + "\n")

    def __del__(self):
        """Clean up the live display when the logger is destroyed."""
        if self.live is not None:
            self.live.stop()
