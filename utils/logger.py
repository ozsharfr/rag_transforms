import logging

def setup_logger(log_file: str, log_level: str = "INFO") -> logging.Logger:
    """
    Sets up a logger with console and file handlers.
    Ensures no duplicate handlers are added.
    """
    # Use the root logger to maintain a single instance across modules
    logger = logging.getLogger()

    # Check if handlers are already configured
    if not logger.hasHandlers():
        logger.setLevel(log_level)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger