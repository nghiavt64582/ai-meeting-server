import logging
import os
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
LOG_DIR = os.path.join(BASE_DIR, "log")

current_date_str = datetime.now().strftime("%Y_%m_%d")
log_file_name = f"file_{current_date_str}.txt"
log_file_path = os.path.join(LOG_DIR, log_file_name)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Remove default handlers if already set up
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    filename=log_file_path,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger(__name__)