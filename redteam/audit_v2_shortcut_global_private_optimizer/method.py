"""Shortcut: centrally minimize joint true-payoff Nash gap using other players' private utilities and unnoised strategies, clipped to the transcript cap."""
from redteam.audit_v2_methods import main

if __name__ == "__main__":
    main('shortcut_global_private_optimizer')
