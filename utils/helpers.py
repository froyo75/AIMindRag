import json
import shutil
from urllib.parse import unquote
from pathlib import Path
from utils.logger import get_logger

helpers_logger = get_logger(__name__)

def safe_path(base_dir: str, path: str) -> tuple:
    """Handle Safe path"""
    base = Path(base_dir).resolve()
    decoded_input = unquote(path)
    normalized_input = decoded_input.replace("\\", "/")
    target = (base / normalized_input).resolve()
    if base != target and base not in target.parents:
        helpers_logger.error(f"Invalid path: {path}")
        return False, None
    return True, target

def create_dir(directory: str) -> bool:
    """Create a directory"""
    path = Path(directory)
    if not path.exists():
        try:
            helpers_logger.info(f"Creating directory {directory}")
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            helpers_logger.error(f"Error creating directory {directory}: {e}.")
            return False
    return True

def delete_dir(directory: str, remove_root: bool = False) -> bool:
    """Delete entire directory or subfolders only"""
    path = Path(directory)
    if path.exists():
        try:
            helpers_logger.info(f"Deleting directory {directory}")
            if remove_root:
                shutil.rmtree(path)
            else:
                for entry in path.iterdir():
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
            return True
        except Exception as e:
            helpers_logger.error(f"Error deleting directory {directory}: {e}.")
            return False
    else:
        return False

def delete_file(file_path: str) -> bool:
    """Delete a file"""
    path = Path(file_path)
    if path.exists():
        try:
            helpers_logger.info(f"Deleting file {file_path}")
            path.unlink()
            return True
        except Exception as e:
            helpers_logger.error(f"Error deleting file {file_path}: {e}.")
            return False
    else:
        return False

def list_dirs(directory: str) -> list:
    """List directories"""
    path = Path(directory)
    return [item.name for item in path.iterdir() if item.is_dir()]

def str_to_bool(value: str) -> bool:
    """Convert a string to a boolean"""
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    else:
        helpers_logger.error(f"Cannot convert {value} to a boolean.")
        return False

def apply_config(config_file: str, mode: str, data: dict = None) -> tuple:
    """Handle configuration file"""
    try:
        if mode == "r":
            with open(config_file, mode) as file:
                return True, json.load(file)
        elif mode == "w" and data is not None:
            with open(config_file, mode, encoding='utf-8') as file:
                json.dump(data, file)
                return True, None
        return False, None
    except Exception as e:
        helpers_logger.error(f"Error handling config file {config_file}: {e}.")
        return False, None