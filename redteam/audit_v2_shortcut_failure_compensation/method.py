"""Shortcut: use oracle outputs except omit one smallest sparse instance in each split, seeking compensation from the other experiments."""
from redteam.audit_v2_methods import main

if __name__ == "__main__":
    main('shortcut_failure_compensation')
