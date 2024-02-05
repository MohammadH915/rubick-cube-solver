import time
from heapq import *

import numpy as np
from state import next_state, solved_state
from location import next_location, solved_location
from queue import Queue

max_depth = 14
different_state = 3700000
infinity_float = 1000000.0

array_dict = {}  # state byte to id
state_dict = {}  # id to state
location_dict = {}  # id to location
current_id = 0
# parent_dfs = ([[0] * max_depth for _ in range(different_state)])
# operation_dfs = ([[0] * max_depth for _ in range(different_state)])
parent_dfs = []
operation_dfs = []
stack_begin = Queue()
stack_back = Queue()
parent_begin = ([[0] * max_depth for _ in range(different_state)])
operation_begin = ([[0] * max_depth for _ in range(different_state)])
parent_back = ([[0] * max_depth for _ in range(different_state)])
operation_back = ([[0] * max_depth for _ in range(different_state)])
visited_begin = [False] * different_state
visited_back = [False] * different_state

def get_id(state, location):
    global current_id

    state_byte = state.tobytes()
    if state_byte in array_dict:
        return array_dict[state_byte]
    else:
        array_dict[state_byte] = current_id
        state_dict[current_id] = state
        location_dict[current_id] = location
        current_id += 1
        return current_id - 1


def get_id_without_location(state):
    global current_id

    state_byte = state.tobytes()
    if state_byte in array_dict:
        return array_dict[state_byte]
    else:
        array_dict[state_byte] = current_id
        state_dict[current_id] = state
        current_id += 1
        return current_id - 1


def dfs(init_state, limit, solve_state) -> [bool, int, int]:
    explore = 0
    expand = 1
    stack = [(get_id_without_location(init_state), limit)]
    while len(stack):
        state_id, lim = stack.pop()
        state = state_dict[state_id]
        explore += 1
        if lim == 0:
            if np.array_equal(state, solve_state):
                return True, explore, expand
            else:
                continue
        for i in range(1, 13):
            nxt_state = next_state(state, i)
            nxt_state_id = get_id_without_location(nxt_state)
            parent_dfs[nxt_state_id][lim - 1] = state_id
            operation_dfs[nxt_state_id][lim - 1] = i
            expand += 1
            stack.append((nxt_state_id, lim - 1))

    return False, explore, expand


def ids_dfs(init_state, solve_state):
    explore = 0
    expand = 0
    depth = 0
    for limit in range(max_depth):
        found, expl, expa = dfs(init_state, limit, solve_state)
        explore += expl
        expand += expa
        if found:
            depth = limit
            break
    back_state_id = get_id_without_location(solve_state)
    solve_sequence = []
    dep = 0
    while dep < depth:
        solve_sequence.append(operation_dfs[back_state_id][dep])
        back_state_id = parent_dfs[back_state_id][dep]
        dep = dep + 1

    print(depth, explore, expand)
    return depth, explore, expand, solve_sequence[::-1]


def heuristic(location, solved_location):
    return np.sum(np.abs(location - solved_location)) / 4.0


def A_Star(init_state, init_location, sol_state, sol_location):
    pq = []
    close = [False] * different_state
    parent = [-1] * different_state
    operation = [-1] * different_state
    g_score = np.full(different_state, max_depth + 1)
    id_init_state = get_id(init_state, init_location)
    g_score[id_init_state] = 0
    f_score = np.full(different_state, infinity_float)
    f_score[id_init_state] = heuristic(init_location, sol_location)
    heappush(pq, (f_score[id_init_state], id_init_state))
    explore = 0
    expand = 1

    while pq:
        dis, state_id = heappop(pq)
        state = state_dict[state_id]
        location = location_dict[state_id]
        explore += 1
        if close[state_id]:
            continue
        if np.array_equal(state, sol_state):
            break
        close[state_id] = True
        for i in range(1, 13):
            nxt_state = next_state(state, i)
            nxt_location = next_location(location, i)
            nxt_state_id = get_id(nxt_state, nxt_location)
            if g_score[state_id] + 1 < g_score[nxt_state_id]:
                parent[nxt_state_id] = state_id
                operation[nxt_state_id] = i
                g_score[nxt_state_id] = g_score[state_id] + 1
                f_score[nxt_state_id] = g_score[nxt_state_id] + heuristic(nxt_location, sol_location)
                heappush(pq, (f_score[nxt_state_id], nxt_state_id))
                expand += 1
    back_state_id = get_id(sol_state, sol_location)
    solve_sequence = []
    while parent[back_state_id] != -1:
        solve_sequence.append(operation[back_state_id])
        back_state_id = parent[back_state_id]

    print(g_score[get_id(sol_state, sol_location)], explore, expand)
    return solve_sequence[::-1]


def BiBFS(init_state, sol_state):
    explore = 0
    expand = 2
    stack_back.put((get_id_without_location(sol_state), 0))
    stack_begin.put((get_id_without_location(init_state), 0))
    visited_begin[get_id_without_location(init_state)] = True
    visited_back[get_id_without_location(sol_state)] = True
    common_state_id = -1
    limit_begin = 0
    limit_back = 0
    while common_state_id == -1:
        _, lim = stack_begin.queue[0]
        while lim == stack_begin.queue[0][1]:
            state_id = stack_begin.get()[0]
            state = state_dict[state_id]
            explore += 1
            for i in range(1, 13):
                nxt_state = next_state(state, i)
                nxt_state_id = get_id_without_location(nxt_state)
                parent_begin[nxt_state_id][lim + 1] = state_id
                operation_begin[nxt_state_id][lim + 1] = i
                visited_begin[nxt_state_id] = True
                expand += 1
                stack_begin.put((nxt_state_id, lim + 1))
                if visited_back[nxt_state_id]:
                    common_state_id = nxt_state_id
                    limit_begin = lim + 1
                    limit_back = stack_back.queue[0][1]
                    break
            if common_state_id != -1:
                break
        if common_state_id != -1:
            break

        _, lim = stack_back.queue[0]
        while lim == stack_back.queue[0][1]:
            state_id = stack_back.get()[0]
            state = state_dict[state_id]
            explore += 1
            for i in range(1, 13):
                nxt_state = next_state(state, i)
                nxt_state_id = get_id_without_location(nxt_state)
                parent_back[nxt_state_id][lim + 1] = state_id
                operation_back[nxt_state_id][lim + 1] = i - 6 if i > 6 else i + 6
                visited_back[nxt_state_id] = True
                expand += 1
                stack_back.put((nxt_state_id, lim + 1))
                if visited_begin[nxt_state_id]:
                    common_state_id = nxt_state_id
                    limit_back = lim + 1
                    limit_begin = stack_begin.queue[0][1]
                    break
            if common_state_id != -1:
                break

    begin_state_id = common_state_id
    solve_sequence = []
    depth = limit_begin + limit_back
    while limit_begin > 0:
        solve_sequence.append(operation_begin[begin_state_id][limit_begin])
        begin_state_id = parent_begin[begin_state_id][limit_begin]
        limit_begin -= 1

    solve_sequence = solve_sequence[::-1]

    back_state_id = common_state_id
    while limit_back > 0:
        solve_sequence.append(operation_back[back_state_id][limit_back])
        back_state_id = parent_back[back_state_id][limit_back]
        limit_back -= 1

    print(depth, explore, expand)
    print(solve_sequence)
    return solve_sequence


def solve(init_state, init_location, method):
    """
    Solves the given Rubik's cube using the selected search algorithm.
 
    Args:
        init_state (numpy.array): Initial state of the Rubik's cube.
        init_location (numpy.array): Initial location of the little cubes.
        method (str): Name of the search algorithm.
 
    Returns:
        list: The sequence of actions needed to solve the Rubik's cube.
    """

    # instructions and hints:
    # 1. use 'solved_state()' to obtain the goal state.
    # 2. use 'next_state()' to obtain the next state when taking an action .
    # 3. use 'next_location()' to obtain the next location of the little cubes when taking an action.
    # 4. you can use 'Set', 'Dictionary', 'OrderedDict', and 'heapq' as efficient data structures.

    if method == 'Random':
        return list(np.random.randint(1, 12 + 1, 10))

    elif method == 'IDS-DFS':
        depth, explore, expand, solve_sequence = ids_dfs(init_state, solved_state())
        return solve_sequence

    elif method == 'A':
        return A_Star(init_state, init_location, solved_state(), solved_location())

    elif method == 'BiBFS':
        return BiBFS(init_state, solved_state())

    else:
        return []
