"""
UTILS.LOGGING API
"""
from utils.logging.logger import Logger
from utils.logging.page_break import echo_page_break
from utils.logging.train_eval_logs import log_losses, log_scores


__all__ = (
    'Logger',
    'echo_page_break',
    'log_losses',
    'log_scores',
)
