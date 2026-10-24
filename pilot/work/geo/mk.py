import sys, re, pathlib
S = pathlib.Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
name, spec = sys.argv[1], pathlib.Path(sys.argv[2]).read_text()
conv, fc = spec.split('#---\n')
p = S / 'cands' / name / 'network.py'
t = p.read_text()
def rep(t, key, new):
    m = re.search(r'        self\.%s = torch\.nn\.Sequential\(\n.*?\n        \)\n' % key, t, re.S)
    assert m, key
    return t[:m.start()] + new + t[m.end():]
t = rep(t, 'conv', conv); t = rep(t, 'fc', fc)
p.write_text(t)
