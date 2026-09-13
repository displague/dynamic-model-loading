"""Launch the pinned stock server or a local coding client without global settings changes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error

import stock_benchmark as stock
import stock_placement as placement
import verification_offload as offload

ROOT = Path(__file__).resolve().parents[1]
ALIAS = 'dml-qwen32b'
PROFILES = ('measured', 'whole-layer', 'stock-speculative', 'target-only')


def configuration(profile):
    if profile == 'measured':
        return placement.configuration('timed', 2, 32)
    if profile == 'whole-layer':
        return placement.configuration('timed', 2, 0)
    if profile in ('stock-speculative', 'target-only'):
        cfg = offload.configuration('long', 'cpu-k4' if profile == 'stock-speculative' else 'target')
        return dict(cfg, cold_ffns=0, verbosity=4)
    raise ValueError('unknown profile')


def server_environment(parent, threshold):
    env, effective, removed = offload.child_environment(parent, 32, 0)
    effective['GGML_OP_OFFLOAD_MIN_BATCH'] = str(threshold)
    env.update(effective)
    return env, effective, removed


def server_command(binary, checked, profile, port):
    cfg = configuration(profile)
    return placement.command(binary, checked, cfg, port) + [
        '--alias', ALIAS, '--jinja', '--temp', '0', '--seed', '0']


def artifact_paths(catalog, models, draft=True):
    return {key: [str((models/key/item['name']).resolve()) for item in catalog['models'][key]['files']]
            for key in (['target', 'draft05'] if draft else ['target'])}


def executable(name):
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f'{name} is not on PATH')
    if name == 'codex' and Path(found).suffix.lower() in ('.cmd', '.ps1'):
        native = Path(found).parent/'node_modules/@openai/codex/node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/bin/codex.exe'
        if not native.is_file():
            raise FileNotFoundError('Codex native executable not found beside npm shim; install the Windows CLI')
        return str(native)
    return found


def client_settings(client, base, state, workspace, parent, *, prompt=None, result=None,
                    claude_compaction='manual'):
    """All overrides belong to the child; never copy cloud credentials into local state."""
    if not base.startswith('http://127.0.0.1:'):
        raise ValueError('this launcher is restricted to the local loopback server')
    env = {k: v for k, v in parent.items() if not k.upper().startswith(
        ('ANTHROPIC_', 'CLAUDE_', 'OPENAI_', 'CODEX_', 'MAX_THINKING_TOKENS'))}
    if client == 'codex':
        # This is Codex's actual configuration-home setting, scoped to its child process.
        env['CODEX_HOME'] = str((state/'codex').resolve())
        config = {
            'model': ALIAS, 'model_provider': 'dml_local',
            'model_providers.dml_local.name': 'Pinned local llama.cpp',
            'model_providers.dml_local.base_url': base+'/v1',
            'model_providers.dml_local.wire_api': 'responses',
            'model_providers.dml_local.requires_openai_auth': False,
            'model_providers.dml_local.supports_websockets': False,
            'model_context_window': 18432, 'model_auto_compact_token_limit': 12000,
            'model_reasoning_summary': 'none', 'model_supports_reasoning_summaries': False,
            'web_search': 'disabled', 'features.multi_agent': False,
            'features.plugins': False, 'features.remote_plugin': False,
            'features.skill_search': False, 'features.shell_snapshot': False,
            'features.skip_host_skill_discovery': True,
        }
        cmd = [executable('codex')]
        for key, value in config.items():
            cmd += ['-c', key+'='+json.dumps(value)]
        cmd += ['-a', 'never' if prompt is not None else 'on-request']
        if prompt is None:
            cmd += ['-s', 'workspace-write', '-C', str(workspace), '--no-alt-screen']
        else:
            cmd += ['-c', 'project_doc_max_bytes=0']
            cmd += ['exec', '-s', 'workspace-write', '--ignore-user-config', '--ignore-rules', '--ephemeral',
                    '--skip-git-repo-check', '--json', '-C', str(workspace),
                    '-o', str(result), prompt]
        effective = {'CODEX_HOME': env['CODEX_HOME']}
    elif client == 'claude':
        if claude_compaction not in ('manual', 'auto'):
            raise ValueError('unknown Claude compaction mode')
        # Do not inherit a switch that disables the manual recovery command too.
        env = {k: v for k, v in env.items()
               if k.upper() not in ('DISABLE_COMPACT', 'DISABLE_AUTO_COMPACT')}
        effective = {
            'ANTHROPIC_BASE_URL': base, 'ANTHROPIC_API_KEY': 'local-only-placeholder',
            'ANTHROPIC_MODEL': ALIAS, 'ANTHROPIC_DEFAULT_HAIKU_MODEL': ALIAS,
            'ANTHROPIC_DEFAULT_SONNET_MODEL': ALIAS, 'ANTHROPIC_DEFAULT_OPUS_MODEL': ALIAS,
            'MAX_THINKING_TOKENS': '0', 'CLAUDE_CODE_MAX_CONTEXT_TOKENS': '18432',
            'CLAUDE_CODE_MAX_OUTPUT_TOKENS': '2048',
            'CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS': '2048',
            'CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY': '1',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        }
        # Claude 2.1.260 subtracts an additional fixed 13000-token reserve:
        # 18432 - 2048 - 13000 = 3384. Its percent knob cannot raise this.
        # Keep the true model window and native blocking guard; use /compact
        # explicitly instead of repeatedly summarizing an already-small history.
        if claude_compaction == 'manual':
            effective['DISABLE_AUTO_COMPACT'] = '1'
        env.update(effective)
        # Bare mode suppresses skill/plugin discovery while leaving recovery commands usable.
        cmd = [executable('claude'), '--bare', '--restricted',
               '--strict-mcp-config', '--model', ALIAS, '--tools', 'Read,Edit,Write',
               '--system-prompt', 'You are a local coding assistant on Windows. '
               f'The working directory is {workspace}. Resolve file paths against this exact directory; never use placeholder paths. '
               'Inspect files with tools before editing. Read targeted ranges using offset and limit, '
               'starting with at most 80 lines; narrow the range if a read exceeds the token limit. '
               'Do not repeatedly reread unchanged files. Make only the requested changes. Be concise.']
        if prompt is not None:
            cmd += ['-p', prompt, '--output-format', 'stream-json', '--verbose',
                    '--no-session-persistence', '--permission-mode', 'acceptEdits',
                    '--permission-prompts', 'none']
    else:
        raise ValueError('unknown client')
    return cmd, env, effective


def serve(args):
    catalog = json.loads((ROOT/'configs/stock-speculation-artifacts.json').read_text())
    checked = artifact_paths(catalog, args.models, args.profile != 'target-only')
    command = server_command(args.binary, checked, args.profile, args.port)
    cfg = configuration(args.profile)
    env, effective, removed = server_environment(os.environ, cfg['threshold'])
    identity = {'profile': args.profile, 'configuration': cfg, 'command': command,
                'effective_runtime_environment': effective, 'removed_override_names': removed}
    if args.print_command:
        print(json.dumps(identity, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    stock.write_json(args.output/'launch.json', identity)
    try:
        stock.verify_catalog(catalog, args.models, args.binary, list(checked))
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(('127.0.0.1', args.port))
        base = f'http://127.0.0.1:{args.port}'
        with (args.output/'server.log').open('xb') as log, stock.managed_process(
                command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=args.binary) as proc:
            with stock.Resources(proc.pid, args.output/'resources.jsonl') as resources:
                stock.write_json(args.output/'process.json', {'pid': proc.pid, 'base_url': base})
                start = time.monotonic()
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError('server exited during startup')
                    try:
                        if stock.request(base, '/health', timeout=2).get('status') == 'ok':
                            break
                    except (OSError, urllib.error.HTTPError):
                        pass
                    if time.monotonic()-start > 600:
                        raise TimeoutError('server startup deadline')
                    time.sleep(.5)
                text = (args.output/'server.log').read_text(encoding='utf-8', errors='replace')
                stock.validate_placement(text, cfg['ngl'], 'draft05' if cfg['k'] else None)
                if cfg['k']:
                    evidence = placement.placement_evidence(text, cfg)
                    if not evidence['kv_pass'] or not evidence['actual_host_pass']:
                        raise RuntimeError('unexpected model/KV placement')
                else:
                    evidence = {'target_only': True}
                if not offload.resource_check(resources)['pass']:
                    raise RuntimeError('startup exceeds measured resource allowance')
                stock.write_json(args.output/'ready.json', {'base_url': base, 'placement': evidence,
                    'props': stock.request(base, '/props')})
                print(f'Ready: {base}  profile={args.profile}  logs={args.output}\nCtrl+C stops this server.', flush=True)
                requested_stop = False
                try:
                    while proc.poll() is None:
                        if args.stop_file and args.stop_file.exists():
                            requested_stop = True
                            break
                        time.sleep(1)
                except KeyboardInterrupt:
                    requested_stop = True
            state = offload.resource_check(resources)
        # The Windows job is closed and the native process has been reaped here.
        stock.write_json(args.output/'completion.json', {'resources': state,
                         'exit_code': proc.returncode, 'requested_stop': requested_stop})
        if not requested_stop:
            raise RuntimeError(f'native server exited unexpectedly: {proc.returncode}')
    except BaseException as exc:
        stock.write_json(args.output/'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['serve', 'codex', 'claude'])
    p.add_argument('--root', type=Path, default=ROOT, help='checkout containing runs and .venv')
    p.add_argument('--profile', choices=PROFILES, default='measured')
    p.add_argument('--port', type=int, default=8080)
    p.add_argument('--workspace', type=Path, default=Path.cwd())
    p.add_argument('--claude-compaction', choices=['manual', 'auto'], default='manual',
                   help='manual avoids Claude 2.1.260 small-context compaction thrashing')
    p.add_argument('--output', type=Path)
    p.add_argument('--stop-file', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--print-command', action='store_true')
    a = p.parse_args()
    a.root = a.root.resolve()
    a.models = a.root/'runs/stock-models'
    a.binary = a.root/'runs/llama-b10919'
    a.output = (a.output or a.root/'runs/local-sessions'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')).resolve()
    a.workspace = a.workspace.resolve()
    if not 1024 <= a.port <= 65535:
        p.error('use a local port between 1024 and 65535')
    if a.action == 'serve':
        serve(a)
        return
    state = a.root/'runs/local-client-state'
    cmd, env, effective = client_settings(a.action, f'http://127.0.0.1:{a.port}', state,
                                        a.workspace, os.environ, claude_compaction=a.claude_compaction)
    if a.print_command:
        print(json.dumps({'command': cmd, 'child_environment': effective, 'cwd': str(a.workspace)}, indent=2))
        return
    if a.action == 'codex':
        Path(env['CODEX_HOME']).mkdir(parents=True, exist_ok=True)
    elif a.claude_compaction == 'manual':
        print('Claude manual compaction: use /context to check usage and /compact near 8000 tokens. '
              'If compaction fails, save a short handoff and start a new session with /clear. '
              'The server remains limited to 18432 tokens.', file=sys.stderr, flush=True)
    raise SystemExit(subprocess.call(cmd, env=env, cwd=a.workspace))


if __name__ == '__main__':
    main()
