import os
import sys
import pathlib

import dotenv

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
dotenv_path = PROJECT_DIR / '.env'

if dotenv_path.exists():
    dotenv.load_dotenv(dotenv_path)
else:
    print(f"Warning: {dotenv_path} does not exist.")

ORBKIT_PATH = str(PROJECT_DIR / "external_packages/orbkit")
XTB_PATH = os.environ.get("XTB_PATH")

sys.path.append(ORBKIT_PATH)

SEED = 42