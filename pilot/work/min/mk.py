import sys, re, pathlib
S = pathlib.Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
name, conv_lines, fc_lines = sys.argv[1], sys.argv[2].split('|'), sys.argv[3].split('|')
p = S / 'cands' / name / 'network.py'
assert name.startswith('min_')
src = p.read_text()
ind = ' ' * 8
def block(var, lines):
    return ind + f'self.{var} = torch.nn.Sequential(\n' + ''.join(ind + '    ' + l.strip() + ',\n' for l in lines) + ind + ')\n'
pat_conv = re.compile(r'^ {8}self\.conv = torch\.nn\.Sequential\(\n.*?^ {8}\)\n', re.S | re.M)
pat_fc = re.compile(r'^ {8}self\.fc = torch\.nn\.Sequential\(\n.*?^ {8}\)\n', re.S | re.M)
assert len(pat_conv.findall(src)) == 1 and len(pat_fc.findall(src)) == 1
src = pat_conv.sub(lambda m: block('conv', conv_lines), src)
src = pat_fc.sub(lambda m: block('fc', fc_lines), src)
p.write_text(src)
print(block('conv', conv_lines) + block('fc', fc_lines))
