# [수정 가능] ① 모델 구조 ② 보상함수 ③ 수집 행동 선택
# 클래스·함수의 이름과 입출력 규약은 유지하세요. 고정 보조 코드는 파일 하단에 있습니다.
# 모델 입력 (B,96,96,3), 출력 유한한 행동 점수 (B,9). 행동 9개의 순서·의미는 고정입니다.
# 추론·서버 채점은 점수가 가장 높은 행동을 선택합니다.
# 학습은 제공된 Double DQN 코드로 고정합니다.
# 학습 후 저장되는 .network.py와 .pth 두 파일을 제출합니다.
import torch

# ① 모델: 영상·센서 입력으로 행동 9개의 Q값을 계산합니다.
# [고정 규약] ClassificationNetwork(device=None), forward(observation)의 입출력.
class ClassificationNetwork(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()

        device = _setup_model(self, device)

        # 수정: 영상 특징 추출. stride 2 합성곱 4층 → (64, 4, 4) → 펼쳐서 LayerNorm. (2-1a 후보 M3b)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=5, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.LayerNorm(64 * 4 * 4),
        )

        # 수정: 영상 특징 1024개 + 센서 7개 → 은닉층 128(LayerNorm 뒤 ReLU) → 행동별 점수. 센서는 정규화하지 않는다.
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4 + 7, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.n_classes),
        )

        # 수정: 베이스라인의 가속 점수 head를 없앤다. 가속은 forward의 상수로 정한다. (2-3c 후보 V1)
        self.to(device)

    def forward(self, observation):
        # 전처리 → 특징 추출 → 조향·가속 점수 결합.

        batch_size = observation.shape[0]

        speed, abs_sensors, steering, gyroscope = self.extract_sensor_values(
            observation.detach(), batch_size)

        x = observation / 255.0

        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.conv(x)
        x = x.reshape(batch_size, -1)

        x = torch.cat([x, speed, abs_sensors, steering, gyroscope], dim=1)
        raw = self.fc(x)
        steering_values = torch.stack([raw[:, [0,1,2]].max(1).values,
                                       raw[:, [3,5,7]].max(1).values,
                                       raw[:, [4,6,8]].max(1).values], dim=1)
        # 수정: 가속 행동(1, 5, 6)이 언제나 5점 높다. 항상 최고 속도로 달리고(속도 상한 50은 제어기가 지킨다) 조향만 배운다.
        throttle_values = raw.new_tensor([0., 5.]).expand(batch_size, 2)   # (무가속, 가속)

        steer_index = [0,0,0,1,2,1,2,1,2]
        throttle_index = [0,1,0,0,0,1,1,0,0]
        values = steering_values[:, steer_index] + throttle_values[:, throttle_index]

        brake_penalty = raw.new_tensor([0,0,5,0,0,0,0,5,5])
        return values + .01 * raw - brake_penalty

    def actions_to_classes(self, actions):
        return _fixed_actions_to_classes(self, actions)

    def scores_to_action(self, scores):
        return _fixed_scores_to_action(self, scores)

    def extract_sensor_values(self, observation, batch_size):
        return _fixed_extract_sensor_values(self, observation, batch_size)

# ② 보상: 아래 계수 또는 calculate_reward()를 수정한 뒤 데이터를 다시 수집합니다.
REWARD_SETTINGS = dict(new_road=1.0, center=0.5, grass=0.5, time=0.0, finish=0.0)

# [고정 규약] calculate_reward(context)는 유한한 숫자 하나를 반환합니다.
def calculate_reward(context):
    # 새 구간 수, 중앙 근접도(0~1), 도로 여부, 완주 여부로 스텝 보상을 계산합니다.
    # context에는 speed, dt, elapsed, aborted도 제공됩니다.

    reward = context['new_road_segments'] * (
        REWARD_SETTINGS['new_road'] + REWARD_SETTINGS['center'] * context['center_alignment'])
    reward -= REWARD_SETTINGS['time']
    if not context['on_road']:
        reward -= REWARD_SETTINGS['grass']
    if context['finished']:
        reward += REWARD_SETTINGS['finish']
    return reward


# ③ 수집: 아래 ε-greedy 구현은 자유롭게 변경할 수 있습니다.
def collection_action(model, observation, device, epsilon, rng):
    # [고정 규약] 위 함수 이름·인자 유지, 반환값은 행동 번호 0~8의 정수.
    # [수정 가능] 함수 본문 전체. ε-greedy 대신 정책 확률 샘플링 등 사용 가능.
    # 수정: 화면의 도로를 따라 걸어가며 만든 중심 경로를 추종해 언제나 가속하며 수집한다. (2-3c 후보 V1P4, 2026-11-03 사용자 결정)
    # 수집에만 쓰는 시범 규칙이다. 추론·채점은 학습한 Q의 최고 행동만 쓰고(select_action), 이 함수는 쓰지 않는다.
    # 경로: 회색 도로에서 경계까지 가장 먼 점을 1씩 이어 간다. 방향 후보 간격은 STEP°(±45° 안).
    # 방향: LAM > 0이면 경로 점에 맞춘 삼차 스무딩 스플라인 x(s), y(s)의 접선, 0이면 경로 선분의 방향을 쓴다.
    # 조준 거리는 L에서 급코너일수록 LMIN까지 줄이고, 목표점을 코너 안쪽으로 옮긴다(목표점 근처 꺾임이 S° 미만이면
    # KG × min(꺾임 / BG°, 1), 가장자리 여유 MG. 그 이상이면 K × min(꺾임 / 45°, 1), 여유 2). 조향은 제어기의 0.03/스텝 변화를
    # 따라 계산해 가장 가까운 명령을 고른다. ε 확률로는 조향만 무작위(가속 유지). 행동은 언제나 1·5·6이다. model은 쓰지 않는다.
    def rule_v4(observation, state, epsilon, rng, P):
        import math
        import numpy as np
        if state.get('rng') is not rng:
            state['rng'], state['cur'] = rng, 0.0
        if epsilon > 0 and rng.random() < epsilon:
            steer = (-1, 0, 1)[int(rng.integers(3))]
        else:
            turn = np.radians(np.linspace(-45, 45, int(round(90 / P['STEP'])) + 1))
            o = np.asarray(observation, dtype=np.int16)[:84]
            road = (np.abs(o[..., 1] - o[..., 0]) < 25) & (o[..., 0] > 60) & (o[..., 0] < 160)
            road[64:80, 43:53] = True
            pad = np.pad(road, 1, constant_values=True)
            edge = ~road & (pad[:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, :-2] | pad[1:-1, 2:])
            er, ec = np.nonzero(edge)
            bx, by = (ec - 47.5) / 1.555, (72 - er) / 1.944
            pts = [np.array([0.0, 0.0])]
            if bx.size:
                cols = np.flatnonzero(road[64])
                segs = [s for s in np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1) if s.size and s[0] <= 50 and s[-1] >= 45] if cols.size else []
                x0 = ((segs[0][0] + segs[0][-1]) / 2 - 47.5) / 1.555 if segs else 0.0
                pts.append(np.array([float(x0), 8 / 1.944]))
                head, length = 0.0, float(np.hypot(*pts[1]))
                while length < 30.0:
                    ang = head + turn
                    qx, qy = pts[-1][0] + np.sin(ang), pts[-1][1] + np.cos(ang)
                    r, c = 72 - qy * 1.944, 47.5 + qx * 1.555
                    ok = (r >= 0) & (r <= 83) & (c >= 0) & (c <= 95)
                    ok &= road[np.clip(np.rint(r).astype(int), 0, 83), np.clip(np.rint(c).astype(int), 0, 95)]
                    if not ok.any():
                        break
                    d = np.sqrt(((qx[:, None] - bx[None]) ** 2 + (qy[:, None] - by[None]) ** 2).min(1))
                    k = int(np.argmax(np.where(ok, d - 0.3 * np.abs(turn), -np.inf)))
                    head = float(ang[k])
                    pts.append(np.array([qx[k], qy[k]]))
                    length += 1.0
            else:
                pts.append(np.array([0.0, 30.0]))
            pts = np.array(pts)
            d_pts = np.diff(pts, axis=0)
            seg = np.hypot(*d_pts.T)
            cum = np.r_[0.0, np.cumsum(seg)]
            total = float(seg.sum())
            spline = None
            if P['LAM'] > 0 and len(pts) >= 4:
                n, h = len(pts), seg
                Q, R = np.zeros((n, n - 2)), np.zeros((n - 2, n - 2))
                for j in range(1, n - 1):
                    Q[j - 1, j - 1], Q[j, j - 1], Q[j + 1, j - 1] = 1 / h[j - 1], -1 / h[j - 1] - 1 / h[j], 1 / h[j]
                    R[j - 1, j - 1] = (h[j - 1] + h[j]) / 3
                    if j < n - 2:
                        R[j - 1, j] = R[j, j - 1] = h[j] / 6
                g = np.linalg.solve(np.eye(n) + P['LAM'] * Q @ np.linalg.solve(R, Q.T), pts)
                gam = np.vstack([np.zeros((1, 2)), np.linalg.solve(R, Q.T @ g), np.zeros((1, 2))])
                spline = ((g[1:] - g[:-1]) / h[:, None] - h[:, None] * (2 * gam[:-1] + gam[1:]) / 6, gam[:-1] / 2, (gam[1:] - gam[:-1]) / (6 * h[:, None]))

            def heading(s):
                i = int(np.clip(np.searchsorted(cum, s, side='right') - 1, 0, len(seg) - 1))
                if spline is None:
                    return math.atan2(d_pts[i][0], d_pts[i][1])
                t = min(max(s, 0.0), total) - cum[i]
                dx, dy = spline[0][i] + 2 * spline[1][i] * t + 3 * spline[2][i] * t * t
                return math.atan2(dx, dy)

            def wrap(a):
                return (a + math.pi) % (2 * math.pi) - math.pi

            far, near = P['L'], P['LMIN']
            lookahead = far - (far - near) * min(abs(wrap(heading(min(far, total)) - heading(min(5.0, total)))) / (math.pi / 4), 1.0)
            if lookahead >= cum[-1]:
                tx, ty = pts[-1]
            else:
                i = int(np.searchsorted(cum, lookahead, side='right') - 1)
                tx, ty = pts[i] + (pts[i + 1] - pts[i]) * ((lookahead - cum[i]) / max(seg[i], 1e-9))
            bend = wrap(heading(min(lookahead + 6.0, total)) - heading(max(min(lookahead, total) - 6.0, 0.0)))
            b = math.degrees(abs(bend))
            if P['S1'] > 0:
                w = min(max((b - P['S']) / (P['S1'] - P['S']), 0.0), 1.0)
                shift = (1 - w) * P['KG'] * min(b / P['BG'], 1.0) + w * P['K'] * min(b / 45.0, 1.0)
                margin = (1 - w) * P['MG'] + w * 2.0
            elif b < P['S']:
                shift, margin = P['KG'] * min(abs(bend) / math.radians(P['BG']), 1.0), P['MG']
            else:
                shift, margin = P['K'] * min(abs(bend) / math.radians(45.0), 1.0), 2.0
            if shift > 0:
                hh, side = heading(min(lookahead, total)), (1.0 if bend > 0 else -1.0)
                nx, ny = side * math.cos(hh), -side * math.sin(hh)
                edge_t = shift + margin
                for t in np.arange(0.25, shift + margin + 1e-9, 0.25):
                    ri, ci = int(round(float(72 - (ty + t * ny) * 1.944))), int(round(float(47.5 + (tx + t * nx) * 1.555)))
                    if not (0 <= ri <= 83 and 0 <= ci <= 95) or not road[ri, ci]:
                        edge_t = t - 0.25
                        break
                shift = min(shift, max(edge_t - margin, 0.0))
                tx, ty = tx + shift * nx, ty + shift * ny
            want = float(np.clip(P['GAIN'] * math.atan2(tx, ty), -1, 1)) * 0.45
            cur = state['cur']
            steer = min((-1, 0, 1), key=lambda c: abs(cur + np.clip(c * 0.45 - cur, -0.03, 0.03) - want))
        state['cur'] += float(np.clip(steer * 0.45 - state['cur'], -0.03, 0.03))
        return {-1: 5, 0: 1, 1: 6}[steer]

    P = {'L': 27.0, 'GAIN': 1.5, 'K': 2.0, 'LMIN': 19.0, 'KG': 4.0, 'BG': 8.0, 'MG': 1.0, 'S': 25.0, 'S1': 0.0, 'STEP': 5.0, 'LAM': 256.0}
    state = collection_action.__dict__.setdefault('state', {'rng': None, 'cur': 0.0})
    return rule_v4(observation, state, epsilon, rng, P)





# ===========================================
# ===== 고정 보조 코드: 아래는 수정하지 않습니다 =====
# ===========================================

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def _setup_model(self, device):
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else "cpu")
    self.device = device

    self.n_classes = 9
    # [고정] 행동 순서·제어값: 중립, 가속, 제동, 좌, 우, 좌+가속, 우+가속, 좌+제동, 우+제동.
    self.action_classes = [
        (0.0,  0.0, 0.0),
        (0.0,  0.5, 0.0),
        (0.0,  0.0, 0.8),
        (-1.0, 0.0, 0.0),
        (1.0,  0.0, 0.0),
        (-1.0, 0.5, 0.0),
        (1.0,  0.5, 0.0),
        (-1.0, 0.0, 0.8),
        (1.0,  0.0, 0.8),
    ]

    return device

def _fixed_actions_to_classes(self, actions):

    class_indices = []
    for action in actions:
        steer = action[0].item()
        gas   = action[1].item()
        brake = action[2].item()

        is_left  = steer < -0.1
        is_right = steer > 0.1
        is_gas   = gas   > 0.1
        is_brake = brake > 0.1

        if is_left and is_gas:
            cls = 5
        elif is_right and is_gas:
            cls = 6
        elif is_left and is_brake:
            cls = 7
        elif is_right and is_brake:
            cls = 8
        elif is_left:
            cls = 3
        elif is_right:
            cls = 4
        elif is_gas:
            cls = 1
        elif is_brake:
            cls = 2
        else:
            cls = 0

        class_indices.append(torch.tensor([cls], dtype=torch.long))
    return class_indices


def _fixed_scores_to_action(self, scores):

    cls = torch.argmax(scores, dim=1).item()
    return self.action_classes[cls]


def _fixed_extract_sensor_values(self, observation, batch_size):

    speed_crop = observation[:, 84:94, 12, 0].reshape(batch_size, -1)
    speed = speed_crop.sum(dim=1, keepdim=True) / 255 / 5

    abs_crop = observation[:, 84:94, 18:25:2, 2].reshape(batch_size, 10, 4)
    abs_sensors = abs_crop.sum(dim=1) / 255 / 5

    steer_crop = observation[:, 88, 38:58, 1].reshape(batch_size, -1) / 255 / 10
    steer_crop[:, :10] *= -1
    steering = steer_crop.sum(dim=1, keepdim=True)

    gyro_crop = observation[:, 88, 58:86, 0].reshape(batch_size, -1) / 255 / 5
    gyro_crop[:, :14] *= -1
    gyroscope = gyro_crop.sum(dim=1, keepdim=True)

    return speed, abs_sensors.reshape(batch_size, 4), steering, gyroscope


# MPS 호환 보조 계층: 풀링 계산은 유지합니다.
class MPSCompatibleAdaptiveAvgPool2d(torch.nn.AdaptiveAvgPool2d):
    def forward(self, x):
        if x.device.type != 'mps':
            return super().forward(x)
        height, width = x.shape[-2:]
        size = self.output_size
        out_h, out_w = (size, size) if isinstance(size, int) else size
        out_h = height if out_h is None else out_h
        out_w = width if out_w is None else out_w
        if out_h <= 0 or out_w <= 0 or height == 0 or width == 0:
            return super().forward(x)
        if height % out_h == 0 and width % out_w == 0:
            return super().forward(x)

        rows = []
        for i in range(out_h):
            start_h = i * height // out_h
            end_h = ((i + 1) * height + out_h - 1) // out_h
            cells = []
            for j in range(out_w):
                start_w = j * width // out_w
                end_w = ((j + 1) * width + out_w - 1) // out_w
                cells.append(x[..., start_h:end_h, start_w:end_w].mean(dim=(-2, -1)))
            rows.append(torch.stack(cells, dim=-1))
        return torch.stack(rows, dim=-2)

