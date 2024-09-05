import numpy as np
import melee

import MovesList
import math

framedata = melee.FrameData()

low_analog = 0.2
high_analog = 0.8


def controller_states_different(new_player: melee.PlayerState, old_player: melee.PlayerState):
    new: melee.ControllerState = new_player.controller_state
    old: melee.ControllerState = old_player.controller_state

    # if generate_output(new_player) == generate_output(old_player):
    #     return False

    for btns in MovesList.buttons:
        # for b in melee.enums.Button:
        for b in btns:
            if new.button.get(b) != old.button.get(b) and new.button.get(b):
                return True

    if new.c_stick[0] < low_analog and old.c_stick[0] >= low_analog:
        return True

    if new.c_stick[0] > high_analog and old.c_stick[0] <= high_analog:
        return True

    if new.c_stick[1] < low_analog and old.c_stick[1] >= low_analog:
        return True

    if new.c_stick[1] > high_analog and old.c_stick[1] <= high_analog:
        return True

    if new.main_stick[0] < low_analog and old.main_stick[0] >= low_analog:
        return True

    if new.main_stick[0] > high_analog and old.main_stick[0] <= high_analog:
        return True

    if new.main_stick[1] < low_analog and old.main_stick[1] >= low_analog:
        return True

    if new.main_stick[1] > high_analog and old.main_stick[1] <= high_analog:
        return True

    return False

    # return generate_output(new) != generate_output(old)


def get_ports(gamestate: melee.GameState, player_character: melee.Character, opponent_character: melee.Character):
    if gamestate is None:
        return -1, -1
    ports = list(gamestate.players.keys())
    if len(ports) != 2:
        return -1, -1
    player_port = ports[0]
    opponent_port = ports[1]
    p1: melee.PlayerState = gamestate.players.get(player_port)
    p2: melee.PlayerState = gamestate.players.get(opponent_port)

    if p1.character == player_character and p2.character == opponent_character:
        player_port = ports[0]
        opponent_port = ports[1]
    elif p2.character == player_character and p1.character == opponent_character:
        player_port = ports[1]
        opponent_port = ports[0]
    else:
        print(p1.character, p2.character)
        player_port = -1
        opponent_port = -1
    return player_port, opponent_port


# def get_player_obs(player: melee.PlayerState) -> list:
#     x = player.position.x / 100
#     y = player.position.y / 50
#     shield = player.shield_strength / 60
#     off_stage = 1 if player.off_stage else 0

#     percent = player.percent / 100
#     is_attacking = 1 if framedata.is_attack(player.character, player.action) else 0
#     on_ground = 1 if player.on_ground else 0
    
#     status = float(player.action.value)

#     facing = 1 if player.facing else -1
    
#     in_hitstun = 1 if player.hitlag_left else 0
#     is_invulnerable = 1 if player.invulnerable else 0
    
#     jumps_left = player.jumps_left

#     attack_state = framedata.attack_state(player.character, player.action, player.action_frame)
#     attack_active = 1 if attack_state == melee.AttackState.ATTACKING else 0
#     attack_cooldown = 1 if attack_state == melee.AttackState.COOLDOWN else 0
#     attack_windup = 1 if attack_state == melee.AttackState.WINDUP else 0

#     is_bmove = 1 if framedata.is_bmove(player.character, player.action) else 0

#     stock = player.stock
#     return [
#         shield, on_ground, is_attacking,
#         off_stage,
#         x, y,
#         percent,
#         facing,
#         in_hitstun,
#         is_invulnerable,
#         jumps_left,
#         status,
#         attack_active,
#         attack_cooldown,
#         attack_windup
#     ]


# def generate_input( player: melee.PlayerState, opponent: melee.PlayerState):
#     direction = 1 if player.position.x < opponent.position.x else -1
    
#     obs = [
#         (player.position.x - opponent.position.x) / 20, (player.position.y - opponent.position.y) / 10,
#         direction,
#         1 if player.position.x > opponent.position.x else -1,
#         1 if player.position.y > opponent.position.y else -1,
#         math.sqrt(pow(player.position.x - opponent.position.x, 2) + pow(player.position.y - opponent.position.y, 2))
#     ]
#     obs += get_player_obs(player)
#     obs += get_player_obs(opponent)

#     return np.array(obs).flatten()

def generate_output(player: melee.PlayerState):
    controller_state = player.controller_state
    assert isinstance(controller_state, melee.ControllerState)
    x, y = controller_state.main_stick
    c_x, c_y = controller_state.c_stick
    l_s, r_s = controller_state.l_shoulder, controller_state.r_shoulder
    button = [int(controller_state.button[melee.enums.Button.BUTTON_A]),
              int(controller_state.button[melee.enums.Button.BUTTON_B]),
              int(controller_state.button[melee.enums.Button.BUTTON_X]),
              int(controller_state.button[melee.enums.Button.BUTTON_Y]),
              int(controller_state.button[melee.enums.Button.BUTTON_Z])
              ]
    
    joystick, cstick, trigger = classify_joystick_position(x, y), classify_joystick_position(c_x, c_y), map_to_interval(max(l_s, r_s))
    return np.array([*button, joystick, cstick, trigger], dtype=np.float32)

# 방향 정의
# directions = {
#     "neutral": (0, 0),
#     "up": (0, 1),
#     "up-right": (np.sqrt(2)/2, np.sqrt(2)/2),
#     "right": (1, 0),
#     "down-right": (np.sqrt(2)/2, -np.sqrt(2)/2),
#     "down": (0, -1),
#     "down-left": (-np.sqrt(2)/2, -np.sqrt(2)/2),
#     "left": (-1, 0),
#     "up-left": (-np.sqrt(2)/2, np.sqrt(2)/2)
# }

# # 강도 정의
# strengths = [0.5, 1]

# # 17개의 분류 생성
# joystick_positions = []

# # 중립 추가
# joystick_positions.append(directions["neutral"])

# # 각 방향과 강도로 좌표 추가
# for direction, (x, y) in directions.items():
#     if direction != "neutral":
#         for strength in strengths:
#             joystick_positions.append((x * strength, y * strength))

joystick_positions = [(0, 0), (0.0, 0.5), (0, 1), 
                      (0.3535533905932738, 0.3535533905932738), (0.7071067811865476, 0.7071067811865476), 
                      (0.5, 0.0), (1, 0), (0.3535533905932738, -0.3535533905932738), 
                      (0.7071067811865476, -0.7071067811865476), (0.0, -0.5), (0, -1), 
                      (-0.3535533905932738, -0.3535533905932738), (-0.7071067811865476, -0.7071067811865476), 
                      (-0.5, 0.0), (-1, 0), (-0.3535533905932738, 0.3535533905932738), 
                      (-0.7071067811865476, 0.7071067811865476)]

# 임의의 조이스틱 좌표를 17가지 중 하나로 분류
def classify_joystick_position(x, y):
    x = (x-0.5)*2
    y = (y-0.5)*2
    min_distance = float('inf')
    closest_position = None
    for idx, position in enumerate(joystick_positions):
        distance = np.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_position = idx
    return closest_position

def map_to_interval(value):
    """
    0에서 1 사이의 값을 5개의 구간 중 하나로 매핑합니다.
    
    :param value: 0에서 1 사이의 입력값
    :return: 구간 인덱스 (0, 1, 2, 3, 4)
    """
    if value < 0 or value > 1:
        raise ValueError("Value must be between 0 and 1, inclusive.")
    
    # 5개의 구간을 정의
    intervals = [0, 0.25, 0.5, 0.75, 1.0]
    
    min_distance = float('inf')
    closest_position = None
    for idx, interval in enumerate(intervals):
        distance = (value - interval) ** 2
        if distance < min_distance:
            min_distance = distance
            closest_position = idx
    return closest_position