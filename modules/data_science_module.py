from pathlib import Path
import os
import runpy
import sys
import streamlit as st


def run_data_science_app():
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    ds_core_path = project_root / "ds_core"

    module_name = "ai_data_science_team.ds_app.app"
    app_path = ds_core_path / "ai_data_science_team" / "ds_app" / "app.py"

    if not app_path.exists():
        st.error(f"Data science app not found: {app_path}")
        return

    old_cwd = os.getcwd()

    try:
        os.chdir(app_path.parent)

        if str(ds_core_path) not in sys.path:
            sys.path.insert(0, str(ds_core_path))

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        runpy.run_module(module_name, run_name="__main__")

    except Exception as e:
        st.error(f"Error while loading data science app: {e}")
    finally:
        os.chdir(old_cwd)