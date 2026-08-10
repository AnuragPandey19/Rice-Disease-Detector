"""Render templates/index.html to stdout without starting Flask or loading models.

Used by diagnostics/test_frontend.js. Importing app.py would pull in torch and
the checkpoints, which is 500MB of work to produce a string, so the two data
structures the template needs are lifted straight out of the source instead.
"""
import ast
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent.parent
WANT = {'RKB', 'SUPPORT_RESOURCES', 'DISEASE_INFO', 'DISEASE_CLASSES'}

tree = ast.parse((ROOT / 'app.py').read_text(encoding='utf-8'))
picked = [n for n in tree.body
          if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in WANT]
ns = {}
exec(compile(ast.Module(body=picked, type_ignores=[]), '<app>', 'exec'), ns)

env = Environment(loader=FileSystemLoader(ROOT / 'templates'))
env.globals['url_for'] = lambda _e, **kw: '/static/' + kw.get('filename', '')

library = [{'key': k, **v} for k, v in ns['DISEASE_INFO'].items()
           if k in ns['DISEASE_CLASSES']]
sys.stdout.write(env.get_template('index.html').render(
    library=library, support=ns['SUPPORT_RESOURCES'], guidance=''))
