"""Open all exported businesses in DuckDB UI; business databases are read-only.

.venv/bin/python scripts/view_ingestion.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from threading import Event

import duckdb


def quote(value):
    return '"' + str(value).replace('"', '""') + '"'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['all', 'train', 'public_test'], default='all')
    parser.add_argument('--port', type=int, default=4213)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    output = Path(__file__).resolve().parents[1] / 'outputs' / 'ingestion'
    splits = ['train', 'public_test'] if args.split == 'all' else [args.split]
    databases = [(split, database) for split in splits
                 for database in sorted((output / split).glob('*/data.duckdb'))
                 if database.parent.name != '_shared']
    if not databases:
        parser.error(f'No exported business databases found under {output}')
    # UI stores its notebooks/settings here, separate from the business databases.
    with duckdb.connect(str(output / 'viewer.duckdb')) as con:
        for split, database in databases:
            alias = split + '_' + database.parent.name
            path = "'" + str(database).replace("'", "''") + "'"
            con.execute(f'ATTACH {path} AS {quote(alias)} (READ_ONLY)')
        print(f'Attached {len(databases)} business databases (read-only).', flush=True)
        con.execute('LOAD ui')
        con.execute(f'SET ui_local_port = {args.port}')
        con.execute('CALL start_ui_server()').fetchall()
        print(f'Open in your browser: http://localhost:{args.port}', flush=True)
        print('Choose train_biz_111 (or another business) > semantic in the sidebar.', flush=True)
        print('Keep this terminal running. Press Ctrl+C to stop.', flush=True)
        try:
            Event().wait()
        except KeyboardInterrupt:
            print('\nStopping viewer.', flush=True)
        finally:
            con.execute('CALL stop_ui_server()').fetchall()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
