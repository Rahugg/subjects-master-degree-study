from __future__ import annotations
from typing import Tuple
from maze import Grid, Pos, OPEN, WALL

def write_maze_txt(path: str, grid: Grid, start: Pos, goal: Pos) -> None:
    sr, sc = start; gr, gc = goal
    view = [row[:] for row in grid]
    view[sr][sc] = 'S'; view[gr][gc] = 'G'
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(view):
            f.write("".join(row) + ("\n" if i < len(view)-1 else ""))

def read_maze_txt(path: str) -> Tuple[Grid, Pos, Pos]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    if not lines:
        raise ValueError("Maze file is empty.")
    W = len(lines[0])
    if any(len(line) != W for line in lines):
        raise ValueError("Maze file has inconsistent row widths.")

    grid: Grid = [list(line) for line in lines]
    start = goal = None
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 'S':
                start = (r, c); grid[r][c] = OPEN
            elif grid[r][c] == 'G':
                goal  = (r, c); grid[r][c] = OPEN
    if start is None or goal is None:
        raise ValueError("Maze file must contain 'S' and 'G'.")
    return grid, start, goal
