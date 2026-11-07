"""Shortcut: omit every N=64 instance from the manifest."""
from redteam.audit_v2_methods import main

if __name__ == "__main__":
    main('shortcut_drop_small_games')
