from streamlit.web import cli as stcli
import os
import sys

def main():
    """Entry point for the csvai-ui command."""
    script_path = os.path.join(os.path.dirname(__file__), 'ui.py')
    args = ["run", script_path, "--global.developmentMode=false"]
    sys.argv = ["streamlit"] + args
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
