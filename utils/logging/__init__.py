"""
UTILS.LOGGING API
"""
from utils.logging.base_logger import BaseLogger
from utils.logging.text_logger import TextLogger
from utils.logging.screen_logger import ScreenLogger
from utils.logging.page_break import echo_page_break
from utils.logging.train_eval_logs import log_losses, log_scores

# For backward compatibility
Logger = TextLogger

__all__ = (
    'BaseLogger',
    'TextLogger',
    'ScreenLogger',
    'echo_page_break',
    'log_losses',
    'log_scores',
)
