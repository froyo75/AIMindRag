from markitdown import MarkItDown
from pathlib import Path
from utils.logger import get_logger

file_logger = get_logger(__name__)

def convert_to_markdown(file_path: str):
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        md = MarkItDown(enable_plugins=False)
        result = md.convert(file_path)
        return result.text_content
    except Exception as e:
        file_logger.error(f"Error converting file to markdown: {e}")
        return None

def get_file_content(file_path: str):
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        try:
            # try to read as text
            return path.read_text(errors="replace")
        except UnicodeDecodeError:
            # fallback to read as bytes and map directly to ASCII-safe string
            raw_bytes = path.read_bytes()
            return raw_bytes.decode("latin-1")
    except Exception as e:
        file_logger.error(f"Error getting file content: {e}")
        return None