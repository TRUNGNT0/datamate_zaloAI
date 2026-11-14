""".venv/bin/python scripts/view_relationships.py biz_111"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.ingestion.organization.diagram import main

if __name__ == '__main__':
    raise SystemExit(main())

