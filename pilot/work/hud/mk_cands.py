from pathlib import Path
SP = Path('/tmp/claude-1000/-home-sukyun02-workspace-aiad/f6671148-763f-4d67-b271-9509c8ff34a3/scratchpad')
old_sensor = """        speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

        abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255 / 5
"""
new_sensor = """        # HUD rows 86-91 hold leaked score digits (and row 92 is partly hidden by the
        # text box), so read the speed / ABS bars only from row 93, the one row that
        # shows nothing but the indicators.
        speed_crop = observation[:, 93:94, 12, 0].reshape(batch_size, -1)
        speed = speed_crop.sum(dim=1, keepdim=True) / 255

        abs_crop = observation[:, 93:94, 18:25:2, 2].reshape(batch_size, 1, 4)
        abs_sensors = abs_crop.sum(dim=1) / 255
"""
old_fc = """        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4 + 7, 128),
"""
fc_sg = """        self.fc = torch.nn.Sequential(
            torch.nn.CircularPad1d((2, 0)),            # [steer, gyro] wrapped to the front: 1031 -> 1033
            torch.nn.ConstantPad1d((0, -7), 0.0),      # drop the 7 trailing sensor slots: 1033 -> 1026
            torch.nn.Linear(64 * 4 * 4 + 2, 128),
"""
fc_ns = """        self.fc = torch.nn.Sequential(
            torch.nn.ConstantPad1d((0, -7), 0.0),      # drop all 7 HUD sensor features: 1031 -> 1024
            torch.nn.Linear(64 * 4 * 4, 128),
"""
def patch(name, pairs):
    p = SP / 'cands' / name / 'network.py'
    t = p.read_text()
    for a, b in pairs:
        assert t.count(a) == 1, (name, a[:40])
        t = t.replace(a, b)
    p.write_text(t)
patch('hud_cs', [(old_sensor, new_sensor)])
patch('hud_crop_cs', [(old_sensor, new_sensor)])
patch('hud_crop_sg', [(old_fc, fc_sg)])
patch('hud_crop_ns', [(old_fc, fc_ns)])
