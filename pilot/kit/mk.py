#!/usr/bin/env python
"""mk.py NAME CONV_FILE FC_FILE  -> cands/NAME/network.py with the conv / fc Sequential blocks replaced."""
import re, sys
from pathlib import Path
SP = Path(__file__).resolve().parents[1]
ORIG = Path('/home/sukyun02/workspace/aiad/assignment1/src/network.py').read_text()

def indent(code):
    return ''.join('        ' + line + '\n' if line.strip() else '\n' for line in code.strip('\n').splitlines())

def make(name, conv, fc):
    text, n1 = re.subn(r'        self\.conv = torch\.nn\.Sequential\(.*?\n        \)\n', lambda m: indent(conv), ORIG, count=1, flags=re.S)
    text, n2 = re.subn(r'        self\.fc = torch\.nn\.Sequential\(.*?\n        \)\n', lambda m: indent(fc), text, count=1, flags=re.S)
    assert n1 == 1 and n2 == 1
    target = SP / 'cands' / name
    target.mkdir(parents=True, exist_ok=False)
    (target / 'network.py').write_text(text)
    return target / 'network.py'

if __name__ == '__main__':
    print(make(sys.argv[1], Path(sys.argv[2]).read_text(), Path(sys.argv[3]).read_text()))
