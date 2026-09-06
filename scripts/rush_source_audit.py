#!/usr/bin/env python3
"""Locate RUSH artifacts and optionally inspect PostgreSQL read-only on THIS host.

Run with the Python environment of the working RUSH checkout. No schema setup,
label changes, credential printing, broad filesystem scans, or provider calls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

TABLES = ('experiment','experiment_cycle','experiment_metric','gate_decision',
          'gate_review','generator_version','label_event','golden_label','llm_label')


def configured_url(root: Path) -> tuple[str, str]:
    if os.environ.get('RUSH_DB_URL'):
        return os.environ['RUSH_DB_URL'], 'RUSH_DB_URL environment'
    env = root/'.env'
    if env.is_file():
        for raw in env.read_text(encoding='utf-8').splitlines():
            key, sep, value = raw.partition('=')
            if sep and key.strip() == 'RUSH_DB_URL':
                value = value.strip()
                if len(value)>1 and value[0]==value[-1] and value[0] in "\"'":
                    value=value[1:-1]
                if value:
                    return value, 'RUSH_DB_URL in checkout .env'
    return 'postgresql:///adobi', 'labelstore default (not proof of a running connection)'


def audit_files(root: Path) -> dict:
    root=root.expanduser().resolve()
    result={'checkout':str(root),'experiments':[], 'policies':{}}
    for area in ('Generative_AI','MNIST_Digits'):
        base=root/'policy-graph'/area
        result['policies'][area]=sorted(p.name for p in base.glob('v*') if p.is_dir())
    for path in sorted((root/'data/experiments').glob('*/experiment.json')):
        try:
            r=json.loads(path.read_text(encoding='utf-8'))
            result['experiments'].append({k:r.get(k) for k in ('experiment_id','run_number','area','dry_run','status','current_version')})
        except (OSError,ValueError):
            result['experiments'].append({'unreadable_file':str(path)})
    result['experiment_source']='data/experiments/<id>/experiment.json (native experiment API)'
    result['policy_source']='policy-graph/<area>/<version>/*.md + edges.json (native graph API)'
    result['golden_source']='PostgreSQL rush.label_event / rush.golden_label once live; not recoverable solely from experiment summaries'
    return result


def audit_database(root: Path) -> dict:
    url, source=configured_url(root)
    result={'configuration_source':source,'schema':'rush','status':'not_connected'}
    try:
        import psycopg
        # Enforce read-only in startup options before the first statement.
        with psycopg.connect(url,connect_timeout=4,
                options='-c default_transaction_read_only=on -c statement_timeout=5000') as conn:
            row=conn.execute('SELECT current_database(), inet_server_addr()::text, inet_server_port(), current_setting(\'transaction_read_only\')').fetchone()
            result.update(status='connected', database=row[0], server_address=row[1] or 'local Unix socket',port=row[2],transaction_read_only=row[3])
            result['tables']=[{'schema':s,'table':t,'estimated_live_rows':n} for s,t,n in conn.execute(
                "SELECT schemaname,relname,n_live_tup FROM pg_stat_user_tables WHERE schemaname='rush' ORDER BY relname").fetchall()]
            for table in TABLES:
                exists=conn.execute('SELECT to_regclass(%s)',('rush.'+table,)).fetchone()[0]
                if exists is None:result.setdefault('missing_expected_tables',[]).append(table)
            # pg_settings with this name is a read; restricted deployments return
            # no setting rather than encouraging broader privileges.
            directory=conn.execute("SELECT setting FROM pg_settings WHERE name='data_directory'").fetchone()
            result['data_directory']=directory[0] if directory else 'not exposed to this role'
            conn.rollback()
    except ImportError:
        result['status']='psycopg_not_installed_in_this_python_environment'
    except Exception as exc:
        # Driver errors can include a DSN. Report the exception class only.
        result.update(status='connection_or_read_failed',error_type=type(exc).__name__)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--database',action='store_true',help='Connect using RUSH_DB_URL/default with read-only SQL')
    args=p.parse_args();root=args.repo_root.expanduser().resolve()
    result=audit_files(root)
    if args.database:result['postgres']=audit_database(root)
    else:result['postgres']={'configuration_source':configured_url(root)[1],'status':'not_probed; use --database on the data host'}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
