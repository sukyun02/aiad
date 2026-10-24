from pathlib import Path
p = Path('/home/sukyun02/workspace/aiad/pilot/work/stage2/human/analyse.py')
t = p.read_text()
t = t.replace("    out['rest'] = int((~used).sum())\n    return out", "    out['rest'] = int((~used).sum())\n    out['lap_frames'] = int(len(along))      # C6 and C7 sections overlap, so the sections do not add up to this\n    return out")
t = t.replace("names = [c for c, *_ in CORNERS] + ['rest']", "names = [c for c, *_ in CORNERS] + ['rest', 'lap_frames']")
t = t.replace("""print(f"{'set':16s} " + ' '.join(f'{n:>13s}' for n in names) + f" {'total':>8s}")""", """print(f"{'set':16s} " + ' '.join(f'{n:>13s}' for n in names))""")
t = t.replace("""                                           total=round(sum(mean.values()), 1))""", """                                           laps=len(laps))""")
t = t.replace("""for n in names) + f" {sum(mean.values()):8.1f}")""", """for n in names))""")
p.write_text(t)
