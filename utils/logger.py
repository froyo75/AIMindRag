import logging

_loggers: dict[str, logging.Logger] = {}

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger

def setup_logging(filename: str, log_level: str) -> logging.Logger:
    """Setup logging for the given filename and log level"""
    logger = logging.getLogger()
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper()))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(filename, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        #stream_handler = logging.StreamHandler(sys.stdout)
        #stream_handler.setFormatter(formatter)
        #logger.addHandler(stream_handler)

    return logger