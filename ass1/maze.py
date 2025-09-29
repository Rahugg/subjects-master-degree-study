from __future__ import annotations
from typing import List, Tuple, Optional
import random

Pos = Tuple[int, int]
Grid = List[List[str]]

WALL = '#'
OPEN = '.'

def coerce_dims(h: int, w: int) -> Tuple[int, int]:
    h = max(5, h | 1)
    w = max(5, w | 1)
    return h, w

def add_loops(grid, loop_factor: float = 0.20, seed: int | None = None) -> None:
    rng = random.Random(seed)
    H, W = len(grid), len(grid[0])

    candidates = []
    for r in range(1, H-1):
        for c in range(1, W-1):
            if grid[r][c] != '#':
                continue
            if (r % 2 == 1) and (c % 2 == 0) and grid[r][c-1] == '.' and grid[r][c+1] == '.':
                candidates.append((r, c))
            elif (r % 2 == 0) and (c % 2 == 1) and grid[r-1][c] == '.' and grid[r+1][c] == '.':
                candidates.append((r, c))

    for (r, c) in candidates:
        if rng.random() < loop_factor:
            grid[r][c] = '.'

def generate_maze(height: int, width: int, loop_factor: float = 0.20, seed: int | None = None) -> tuple[Grid, Pos, Pos]:
    H, W = coerce_dims(height, width)
    rng = random.Random(seed)

    grid = [[WALL for _ in range(W)] for _ in range(H)]

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < H and 0 <= c < W

    grid[1][1] = OPEN
    stack = [(1, 1)]
    dirs = [(2,0), (-2,0), (0,2), (0,-2)]

    def unvisited(r: int, c: int):
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and grid[nr][nc] == WALL:
                yield nr, nc, r + dr//2, c + dc//2

    while stack:
        r, c = stack[-1]
        choices = list(unvisited(r, c))
        if not choices:
            stack.pop()
            continue
        nr, nc, wr, wc = rng.choice(choices)
        grid[wr][wc] = OPEN
        grid[nr][nc] = OPEN
        stack.append((nr, nc))

    add_loops(grid, loop_factor=loop_factor, seed=seed)

    start = (1, 1)
    goal = (H - 2, W - 2)
    return grid, start, goal

def overlay_path(grid: Grid, path: Optional[list[Pos]], start: Pos, goal: Pos) -> list[str]:
    g = [row[:] for row in grid]
    if path:
        for r, c in path:
            if (r, c) != start and (r, c) != goal:
                g[r][c] = '*'
    sr, sc = start; gr, gc = goal
    g[sr][sc] = 'S'; g[gr][gc] = 'G'
    return [''.join(row) for row in g]
