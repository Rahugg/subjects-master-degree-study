from __future__ import annotations
import os, sys
from typing import Dict, List
from maze import generate_maze, overlay_path
from utils import write_maze_txt, read_maze_txt
from algorithms import bfs, dfs, a_star
from metrics import measure

def ask_int(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
            if v <= 0: raise ValueError
            return v
        except ValueError:
            print("Please enter a positive integer.")

def ask_yes_no(prompt: str) -> bool:
    while True:
        s = input(prompt).strip().lower()
        if s in ("y","yes"): return True
        if s in ("n","no"):  return False
        print("Please answer y/n.")

def run_once(maze_file: str, algo_name: str, save_solved: bool) -> Dict:
    grid, start, goal = read_maze_txt(maze_file)
    fn = {"bfs": bfs, "dfs": dfs, "astar": a_star}[algo_name]
    (path, expanded), elapsed, peak = measure(fn, grid, start, goal)

    steps = (len(path)-1) if path else None
    solved_lines = overlay_path(grid, path, start, goal)

    print(f"\n--- {algo_name.upper()} ---")
    print(f"Steps (path length): {steps if steps is not None else '—'}")
    print(f"Nodes expanded: {expanded}")
    print(f"Time: {elapsed*1000:.2f} ms")
    print(f"Peak memory: {peak/1024:.1f} KiB\n")
    print("\n".join(solved_lines))

    if save_solved:
        out = f"solved_{algo_name}.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(solved_lines))
        print(f"\nSaved solved maze to: {os.path.abspath(out)}")

    return {"algo": algo_name.upper(), "steps": steps, "expanded": expanded,
            "time_ms": elapsed*1000.0, "peak_kib": peak/1024.0}

def print_summary(results: List[Dict]):
    if not results: return
    print("\n=== Summary (same maze) ===")
    print("ALGO | STEPS | EXPANDED |   TIME (ms) | PEAK (KiB)")
    print("-----+-------+----------+-------------+-----------")
    for r in results:
        steps = "—" if r["steps"] is None else str(r["steps"])
        print(f"{r['algo']:>5} | {steps:>5} | {r['expanded']:>8} | {r['time_ms']:>11.2f} | {r['peak_kib']:>9.1f}")

def main():
    print("=== Maze Generator (single maze) & Multi-Run Solver ===")
    H = ask_int("Enter height: ")
    W = ask_int("Enter width: ")
    maze_file = "maze.txt"

    grid, start, goal = generate_maze(H, W, loop_factor=0.20, seed=42)
    write_maze_txt(maze_file, grid, start, goal)
    print(f"\nMaze saved to: {os.path.abspath(maze_file)}")
    print(f"Dimensions: {len(grid)} x {len(grid[0])}")

    save_solved = ask_yes_no("Save solved mazes to files (y/n)? ")

    while True:
        choice = input("\nChoose algorithm [bfs | dfs | astar | all] or 'q' to quit: ").strip().lower()
        if choice in ("q","quit","exit"):
            print("Goodbye!")
            break
        elif choice == "all":
            res = [run_once(maze_file, a, save_solved) for a in ("bfs","dfs","astar")]
            print_summary(res)
        elif choice in ("bfs","dfs","astar"):
            res = run_once(maze_file, choice, save_solved)
            print_summary([res])
        else:
            print("Invalid choice. Type bfs, dfs, astar, all, or q.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
