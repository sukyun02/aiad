import sys, re, pathlib
S = pathlib.Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
I = ' ' * 12
def seq(layers):
    return 'torch.nn.Sequential(\n' + ''.join(f'{I}{l},\n' for l in layers) + ' ' * 8 + ')'
C = lambda i, o, k, s=2, p=0: f'torch.nn.Conv2d({i}, {o}, kernel_size={k}, stride={s}' + (f', padding={p})' if p else ')')
R = 'torch.nn.ReLU()'
BN = lambda c: f'torch.nn.BatchNorm2d({c})'
MP = 'torch.nn.MaxPool2d(2)'
def fc(n, extra=()):
    return [f'torch.nn.Linear({n} + 7, 128)', R, *extra, 'torch.nn.Linear(128, self.n_classes)']
A = [C(3,16,5), R, C(16,32,3), R, C(32,64,3), R, C(64,64,3), R]
D = {
 'rel_bn': ([C(3,16,5), BN(16), R, C(16,32,3), BN(32), R, C(32,64,3), BN(64), R, C(64,64,3), BN(64), R], fc('64 * 4 * 4')),
 'rel_drop': (A, fc('64 * 4 * 4', ['torch.nn.Dropout(p=0.3)'])),
 'rel_w2': ([C(3,32,5), R, C(32,64,3), R, C(64,128,3), R, C(128,128,3), R], fc('128 * 4 * 4')),
 'rel_w05': ([C(3,8,5), R, C(8,16,3), R, C(16,32,3), R, C(32,32,3), R], fc('32 * 4 * 4')),
 'rel_pool': ([C(3,16,5,1), R, MP, C(16,32,3,1), R, MP, C(32,64,3,1), R, MP, C(64,64,3,1), R, MP], fc('64 * 4 * 4')),
 'rel_5conv': (A + [C(64,64,3,1,1), R], fc('64 * 4 * 4')),
}
for name in sys.argv[1:]:
    path = S / 'cands' / name / 'network.py'
    text = path.read_text()
    conv, f = D[name]
    text, n1 = re.subn(r'self\.conv = torch\.nn\.Sequential\(\n.*?\n        \)', lambda m: 'self.conv = ' + seq(conv), text, count=1, flags=re.S)
    text, n2 = re.subn(r'self\.fc = torch\.nn\.Sequential\(\n.*?\n        \)', lambda m: 'self.fc = ' + seq(f), text, count=1, flags=re.S)
    assert n1 == 1 and n2 == 1
    path.write_text(text)
