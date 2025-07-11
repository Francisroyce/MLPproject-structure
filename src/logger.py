import os
import logging
from datetime import datetime

# Create logs directory
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Generate timestamped log file inside the logs folder
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Set up logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger("mlp_logger")

if __name__ == "__main__":  # ✅ Corrected line
    logger.info("Logging has started")  # ✅ Also changed logging.info → logger.info
