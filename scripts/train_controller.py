#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agentcontrol.controllers import train_controller
if __name__ == '__main__': print(train_controller())
