"""Read-only saved Hard notebook freshness/AST/nbformat and redacted secret/privacy audit."""
from __future__ import annotations

import ast
import base64
import json
from pathlib import Path
import re

from tools import make_colab_hard_notebook as notebook
from tools import run_colab_hard as hard

# Report locations/rule names only, never matched values or surrounding source text.
PATTERNS = {
    'private_key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'api_token': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'aws_access_key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'literal_secret': re.compile(r'''(?i)\b(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*["'][^"'\n]{12,}["']'''),
    'bearer_literal': re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._-]{20,}'),
    'personal_home_path': re.compile(r'/(?:Users|home)/[^/\s\x22\x27]+/'),
    'email_address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
}


def scan_text(label, text):
    return [dict(path=label, line=text.count('\n', 0, match.start()) + 1, rule=rule)
            for rule, pattern in PATTERNS.items() for match in pattern.finditer(text)]


def audit_saved(path=notebook.OUTPUT):
    import nbformat  # Required here. An unavailable validator must never produce a passing audit.
    path = Path(path)
    nb = json.loads(path.read_text())
    notebook.validate_notebook(nb)
    nbformat.validate(nb)
    hard.require(nb == notebook.build_notebook(), 'Saved notebook differs from current generator')
    setup = next(c for c in nb['cells'] if c['id'] == 'reviewed-setup')
    bundle = ast.literal_eval(ast.parse(''.join(setup['source'])).body[0].value)
    findings = []
    for rel, entry in bundle.items():
        source = base64.b64decode(entry['content_b64'], validate=True).decode('utf-8')
        findings.extend(scan_text('embedded/' + rel, source))
    for c in nb['cells']:
        source = ''.join(c['source'])
        if c['id'] == 'reviewed-setup':
            source = '\n' + '\n'.join(source.splitlines()[1:])  # decoded sources scanned above
        findings.extend(scan_text('notebook/' + c['id'], source))
        findings.extend(scan_text('metadata/' + c['id'], json.dumps(c['metadata'])))
    findings.extend(scan_text('notebook/metadata', json.dumps(nb['metadata'])))
    for rel in ('tools/hard_gpu_worker.py', 'tools/audit_hard_notebook.py', 'tests/test_colab_hard.py',
                'tests/test_hard_processes.py', 'tests/test_hard_notebook.py', 'docs/COLAB_HARD.md'):
        findings.extend(scan_text(rel, (notebook.REPO / rel).read_text()))
    return dict(saved_notebook=path.name, sha256=hard.sha(path), freshness=True, ast=True, nbformat=True,
                nbformat_version=nbformat.__version__, unexecuted=True, switches_false=True,
                source_count=len(bundle), exact_source_set=True, scan_findings=findings,
                scan_scope='saved cells and metadata, every decoded payload, Hard tests/docs/audit tool; heuristic scan')


if __name__ == '__main__':
    print(json.dumps(audit_saved(), indent=2))
