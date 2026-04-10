import sys
from pathlib import Path

# add research_core folder to Python path
research_path = Path(__file__).resolve().parent.parent / "research_core"
sys.path.append(str(research_path))

from frontend import run_research_app


def render_research_module():
    run_research_app()