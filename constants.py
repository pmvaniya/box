from pathlib import Path

# directories
ROOT_DIR = Path(__file__).parent
TAR_DIR = ROOT_DIR / "OpenWebText"
DATA_DIR = ROOT_DIR / "data"

# message
INFO_M = "\033[34m  INFO    \033[0m"
WARN_M = "\033[33m  WARN    \033[0m"
DANGER_M = "\033[31m DANGER   \033[0m"
SUCCESS_M = "\033[32m  SUCC    \033[0m"
QUESTION_M = "\033[35m  QUES    \033[0m"
