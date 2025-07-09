# Initialize app package
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

__version__ = "1.0.0"
__all__ = ["main", "components", "utils"]