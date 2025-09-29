from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
from collections import deque
import heapq
from maze import Grid, Pos, WALL

def _in_bounds(g: Grid, r: int, c: int) -> bool:
    return 0 <= r < len(g) and 0 <= c < len(g[0])

def _passable(g: Grid, r: int, c: int) -> bool:
    return g[r][c] != WALL

def neighbors(g: Grid, p: Pos) -> Iterable[Pos]:
    r, c = p
    for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
        nr, nc = r + dr, c + dc
        if _in_bounds(g, nr, nc) and _passable(g, nr, nc):
            yield (nr, nc)

def reconstruct_path(came_from: Dict[Pos, Pos], start: Pos, goal: Pos) -> Optional[List[Pos]]:
    if goal not in came_from and goal != start:
        return None
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path

def bfs(g: Grid, start: Pos, goal: Pos):
    q = deque([start])
    visited = {start}
    came_from: Dict[Pos, Pos] = {}
    expanded = 0
    while q:
        cur = q.popleft()
        expanded += 1
        if cur == goal:
            return reconstruct_path(came_from, start, goal), expanded
        for nb in neighbors(g, cur):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                q.append(nb)
    return None, expanded

def dfs(g: Grid, start: Pos, goal: Pos):
    stack = [start]
    visited = {start}
    came_from: Dict[Pos, Pos] = {}
    expanded = 0
    while stack:
        cur = stack.pop()
        expanded += 1
        if cur == goal:
            return reconstruct_path(came_from, start, goal), expanded
        for nb in reversed(list(neighbors(g, cur))):
            if nb not in visited:
                visited.add(nb)
                came_from[nb] = cur
                stack.append(nb)
    return None, expanded

def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(g: Grid, start: Pos, goal: Pos):
    h = lambda p: _manhattan(p, goal)
    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (h(start), 0, counter, start))
    came_from: Dict[Pos, Pos] = {}
    g_score: Dict[Pos, int] = {start: 0}
    closed = set()
    expanded = 0
    INF = 10**18
    while open_heap:
        _, gcur, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        expanded += 1
        if cur == goal:
            return reconstruct_path(came_from, start, goal), expanded
        closed.add(cur)
        for nb in neighbors(g, cur):
            tentative_g = gcur + 1
            if nb in closed and tentative_g >= g_score.get(nb, INF):
                continue
            if tentative_g < g_score.get(nb, INF):
                came_from[nb] = cur
                g_score[nb] = tentative_g
                counter += 1
                heapq.heappush(open_heap, (tentative_g + h(nb), tentative_g, counter, nb))
    return None, expanded
