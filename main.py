from modules.ui import run_ui
from utils.load import init_load
from utils.logger import setup_logging, get_logger
from utils.config import LOG_DIR_PATH, LOG_FILENAME, LOG_LEVEL
from utils.helpers import create_dir
import sys
import traceback
import asyncio
import logging

create_dir(LOG_DIR_PATH)
main_logger = setup_logging(f"{LOG_DIR_PATH}/{LOG_FILENAME}", LOG_LEVEL)
# Suppress httpx INFO logs
logging.getLogger("httpx").setLevel(logging.WARNING)

async def main():
    """
    Main entry point for AIMindRAG.
    """
    try:
        init_load()
        main_logger.info("Starting the application...")
        run_ui()
    except Exception as e:
        main_logger.error(f"Error starting the application: {e}")
        main_logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())