import logging
import os



def setup_logging(log_file: str = "app.log", level: int = logging.INFO) -> None:
    """
    Sets up basic logging configuration.

    Args:
        log_file: The file to which logs will be written.
        level: The logging level (e.g., logging.DEBUG, logging.INFO).
    """
    logging.basicConfig(
        filename=log_file,
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logging initialized.")



def create_directory_if_not_exists(directory: str) -> None:
    """
    Creates a directory if it does not already exist.

    Args:
        directory: The path to the directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory: {directory}")
    else:
        logging.info(f"Directory already exists: {directory}")