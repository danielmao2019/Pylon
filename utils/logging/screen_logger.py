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
    ) -> None:
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
        self.loss_columns = []  # Store loss column names
        self.score_columns = []  # Store score column names

    def train(self) -> None:
        """Switch to training mode and reset history."""
        self.layout = "train"
        self.history = []
        self.loss_columns = []  # Reset loss columns

    def eval(self) -> None:
        """Switch to evaluation mode and reset history."""
        self.layout = "eval"
        self.history = []
        self.score_columns = []  # Reset score columns

    def flush(self, prefix: str) -> None:
        """
        Flush the buffer to the screen and optionally to the log file.

        Args:
            prefix: Optional prefix to display before the data
        """
        assert isinstance(prefix, str), "prefix must be a string"

        # Store the prefix as iteration info
        self.buffer['iteration_info'] = prefix

        # Add current iteration to history
        self.history.append(self.buffer.copy())
        if len(self.history) > self.max_iterations:
            self.history.pop(0)

        # Update column names based on buffer contents
        if self.layout == "train":
            for key in self.buffer.keys():
                if key.startswith("loss_"):
                    if key not in self.loss_columns:
                        self.loss_columns.append(key)
        else:  # eval layout
            for key in self.buffer.keys():
                if key.startswith("score_"):
                    if key not in self.score_columns:
                        self.score_columns.append(key)

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

    def _display(self) -> None:
        """Display the current buffer and history."""
        # Create a table for the training progress
        table = self._create_table()

        # Add rows for each iteration in history
        for data in self.history:
            if self.layout == "train":
                # Get all loss values
                if self.loss_columns:
                    loss_values = [self._format_value(data.get(col, "-")) for col in self.loss_columns]
                else:
                    loss_values = [self._format_value(data.get("loss", "-"))]

                table.add_row(
                    data.get("iteration_info", "-"),
                    self._format_value(data.get("learning_rate")),
                    *loss_values,  # Unpack loss values
                    self._format_value(data.get("iteration_time")),
                    self._format_value(data.get("memory_max", "-")),
                    self._format_value(data.get("util_avg", "-"))
                )
            else:  # eval layout
                # Get all score values
                if self.score_columns:
                    score_values = [self._format_value(data.get(col, "-")) for col in self.score_columns]
                else:
                    score_values = [self._format_value(data.get("score", "-"))]

                table.add_row(
                    data.get("iteration_info", "-"),
                    *score_values,  # Unpack score values
                    self._format_value(data.get("iteration_time")),
                    self._format_value(data.get("memory_max", "-")),
                    self._format_value(data.get("util_avg", "-"))
                )

        # Update the live display
        if self.live is not None:
            self.live.update(table)
        else:
            # Fallback if live display is not available
            self.console.print(table)

    def _create_table(self) -> Table:
        """Create a table based on the current layout."""
        table = Table(show_header=True)
        table.add_column("Iteration", justify="left", style="cyan")

        if self.layout == "train":
            table.add_column("Learning Rate", justify="right", style="green")
            # Add hierarchical header for losses
            if self.loss_columns:
                # Add sub-columns for each loss
                for col in self.loss_columns:
                    table.add_column(col.replace("loss_", ""), justify="right", style="red")
            else:
                table.add_column("Losses", justify="right", style="red")
        else:  # eval layout
            # Add hierarchical header for scores
            if self.score_columns:
                # Add sub-columns for each score
                for col in self.score_columns:
                    table.add_column(col.replace("score_", ""), justify="right", style="red")
            else:
                table.add_column("Scores", justify="right", style="red")

        table.add_column("Time (s)", justify="right", style="yellow")
        table.add_column("Memory (MB)", justify="right", style="blue")
        table.add_column("GPU Util (%)", justify="right", style="magenta")

        # Create header rows for hierarchical display
        if self.layout == "train":
            if self.loss_columns:
                # First row: single cells for non-hierarchical columns, merged cell for losses
                header_row1 = ["Iteration", "Learning Rate"]
                # Add "Losses" spanning all loss columns
                header_row1.append("Losses" + " " * (len(self.loss_columns) - 1))
                # Add empty cells for remaining columns
                header_row1.extend([""] * (len(table.columns) - len(header_row1)))
                table.add_row(*header_row1)

                # Second row: empty cells for non-hierarchical columns, individual loss names
                header_row2 = ["", ""]  # Empty cells for Iteration and Learning Rate
                # Add individual loss names
                header_row2.extend([col.replace("loss_", "") for col in self.loss_columns])
                # Add empty cells for remaining columns
                header_row2.extend([""] * (len(table.columns) - len(header_row2)))
                table.add_row(*header_row2)
        else:  # eval layout
            if self.score_columns:
                # First row: single cell for Iteration, merged cell for scores
                header_row1 = ["Iteration"]
                # Add "Scores" spanning all score columns
                header_row1.append("Scores" + " " * (len(self.score_columns) - 1))
                # Add empty cells for remaining columns
                header_row1.extend([""] * (len(table.columns) - len(header_row1)))
                table.add_row(*header_row1)

                # Second row: empty cell for Iteration, individual score names
                header_row2 = [""]  # Empty cell for Iteration
                # Add individual score names
                header_row2.extend([col.replace("score_", "") for col in self.score_columns])
                # Add empty cells for remaining columns
                header_row2.extend([""] * (len(table.columns) - len(header_row2)))
                table.add_row(*header_row2)

        return table

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
        if hasattr(self, 'live') and self.live is not None:
            self.live.stop()
