"""
gridworld.py — grid-world cooking sandbox (env + map generation).

Pure logic, no pygame. Imported by learning.py (encoding) and play.py (UI).

Map generation now considers:
  * openness  in [0,1] : high = open rooms, low = narrow hallways
  * intersect in [0,1] : carves straight cross-cuts -> shortcuts / loops
All stations are guaranteed reachable (BFS check; regenerate on failure).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import IntEnum


class Tile(IntEnum):
    FLOOR = 0
    WALL = 1
    PLAYER = 2          # only stamped on for display/serialization
    KITCHEN = 3
    OVEN = 4
    CUTTING_BOARD = 5


STATION_CONFIG = {
    Tile.KITCHEN: {
        "name": "Kitchen", "label": "K", "color": (214, 96, 77),
        "tasks": ["Cook a hearty stew", "Prepare a warm meal", "Whip up some food"],
    },
    Tile.OVEN: {
        "name": "Oven", "label": "O", "color": (224, 146, 70),
        "tasks": ["Bake fresh bread", "Roast something tasty", "Bake a little pie"],
    },
    Tile.CUTTING_BOARD: {
        "name": "Cutting Board", "label": "C", "color": (122, 168, 116),
        "tasks": ["Chop the vegetables", "Slice some onions", "Dice fresh herbs"],
    },
}

# Fixed order -> indices for the goal vector the network sees.
STATION_ORDER = [Tile.KITCHEN, Tile.OVEN, Tile.CUTTING_BOARD]
STATIONS = set(STATION_ORDER)
CARDINAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass
class Task:
    station: Tile
    text: str
    done: bool = False


# --------------------------------------------------------------------------- #
#  Map generation
# --------------------------------------------------------------------------- #
def _cellular_cave(size, openness, rng):
    """Random fill + smoothing. Higher openness -> more floor."""
    wall_p = 0.62 * (1 - openness) + 0.05          # tune wall density
    g = [[Tile.WALL if (r in (0, size - 1) or c in (0, size - 1)
                        or rng.random() < wall_p) else Tile.FLOOR
          for c in range(size)] for r in range(size)]

    for _ in range(3):                              # smoothing passes
        new = [row[:] for row in g]
        for r in range(1, size - 1):
            for c in range(1, size - 1):
                walls = sum(g[r + dr][c + dc] == Tile.WALL
                            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                            if not (dr == 0 and dc == 0))
                new[r][c] = Tile.WALL if walls >= 5 else Tile.FLOOR
        g = new
    for i in range(size):                           # re-assert border
        g[0][i] = g[size - 1][i] = g[i][0] = g[i][size - 1] = Tile.WALL
    return g


def _add_intersections(g, intersect, rng):
    """Carve straight horizontal/vertical corridors -> shortcuts & loops."""
    size = len(g)
    cuts = round(intersect * 5)
    for _ in range(cuts):
        if rng.random() < 0.5:                      # horizontal cut
            r = rng.randint(1, size - 2)
            for c in range(1, size - 1):
                g[r][c] = Tile.FLOOR
        else:                                       # vertical cut
            c = rng.randint(1, size - 2)
            for r in range(1, size - 1):
                g[r][c] = Tile.FLOOR


def _reachable(g, start):
    """Flood fill over FLOOR tiles (stations/walls block)."""
    size = len(g)
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if (0 <= nr < size and 0 <= nc < size and (nr, nc) not in seen
                    and g[nr][nc] == Tile.FLOOR):
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _largest_region(g):
    size = len(g)
    seen, best = set(), set()
    for r in range(size):
        for c in range(size):
            if g[r][c] == Tile.FLOOR and (r, c) not in seen:
                comp = _reachable(g, (r, c))
                seen |= comp
                if len(comp) > len(best):
                    best = comp
    return best


def generate_map(size, openness, intersect, rng, max_attempts=120):
    """Return (grid, agent_pos) with all 3 stations reachable, or a fallback."""
    for _ in range(max_attempts):
        g = _cellular_cave(size, openness, rng)
        _add_intersections(g, intersect, rng)
        region = sorted(_largest_region(g))
        if len(region) < len(STATION_ORDER) + 6:
            continue

        def floor_neighbours(cell):
            r, c = cell
            return [(r + dr, c + dc) for dr, dc in CARDINAL
                    if g[r + dr][c + dc] == Tile.FLOOR]

        # cells with >=2 floor neighbours are less likely to be bridges
        pool = [cell for cell in region if len(floor_neighbours(cell)) >= 2]
        if len(pool) < len(STATION_ORDER) + 1:
            continue

        rng.shuffle(pool)
        agent = pool.pop()
        placed, ok = {}, True
        for st in STATION_ORDER:
            cell = next((p for p in pool if floor_neighbours(p)), None)
            if cell is None:
                ok = False
                break
            pool.remove(cell)
            placed[st] = cell
        if not ok:
            continue

        for st, cell in placed.items():
            g[cell[0]][cell[1]] = st

        reach = _reachable(g, agent)
        if all(any((sr + dr, sc + dc) in reach for dr, dc in CARDINAL)
               for (sr, sc) in placed.values()):
            return g, agent

    # fallback: open room with stations along a wall (always solvable)
    g = [[Tile.WALL if (r in (0, size - 1) or c in (0, size - 1)) else Tile.FLOOR
          for c in range(size)] for r in range(size)]
    for i, st in enumerate(STATION_ORDER):
        g[1][2 + i * 2] = st
    return g, (size // 2, size // 2)


# --------------------------------------------------------------------------- #
#  Environment
# --------------------------------------------------------------------------- #
class GridWorld:
    def __init__(self, size=10, openness=None, intersect=None,
                 n_tasks=None, seed=None):
        self.size = size
        self.rng = random.Random(seed)
        self.openness = openness if openness is not None else self.rng.uniform(0.45, 0.8)
        self.intersect = intersect if intersect is not None else self.rng.uniform(0.0, 0.6)
        self.grid, self.agent = generate_map(size, self.openness, self.intersect, self.rng)
        self.messages: list[str] = []
        self.tasks: list[Task] = []
        self._generate_tasks(n_tasks)

    def _generate_tasks(self, n_tasks):
        present = [t for t in STATION_ORDER if self._find_station(t)]
        if not present:
            return
        if n_tasks is None:
            n_tasks = self.rng.randint(3, max(3, len(present) + 1))
        for _ in range(n_tasks):
            st = self.rng.choice(present)
            self.tasks.append(Task(st, self.rng.choice(STATION_CONFIG[st]["tasks"])))

    # ----- queries --------------------------------------------------------- #
    def _find_station(self, station):
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r][c] == station:
                    return (r, c)
        return None

    def in_bounds(self, r, c):
        return 0 <= r < self.size and 0 <= c < self.size

    def adjacent_stations(self):
        r, c = self.agent
        out = []
        for dr, dc in CARDINAL:
            nr, nc = r + dr, c + dc
            if self.in_bounds(nr, nc) and self.grid[nr][nc] in STATIONS:
                out.append((nr, nc, self.grid[nr][nc]))
        return out

    def remaining_counts(self):
        """Goal vector the network sees: pending task count per station type."""
        return [sum(1 for t in self.tasks if t.station == st and not t.done)
                for st in STATION_ORDER]

    # ----- actions (return True if the action 'mattered') ------------------ #
    def move(self, dr, dc):
        r, c = self.agent
        nr, nc = r + dr, c + dc
        if self.in_bounds(nr, nc) and self.grid[nr][nc] == Tile.FLOOR:
            self.agent = (nr, nc)
            return True
        self._log("Blocked." if self.in_bounds(nr, nc) else "Edge of the map.")
        return False

    def interact(self):
        near = self.adjacent_stations()
        if not near:
            self._log("Nothing to interact with here.")
            return False
        done_any = False
        for _, _, station in near:
            for task in self.tasks:
                if task.station == station and not task.done:
                    task.done = True
                    self._log(f"Done: {task.text}!")
                    done_any = True
                    break
        if not done_any:
            names = ", ".join(STATION_CONFIG[s]["name"] for *_, s in near)
            self._log(f"Used the {names}, but nothing needs it.")
        if self.all_done():
            self._log("All tasks complete!")
        return done_any

    def all_done(self):
        return bool(self.tasks) and all(t.done for t in self.tasks)

    def _log(self, msg):
        self.messages.append(msg)
        self.messages = self.messages[-5:]

    # ----- views ----------------------------------------------------------- #
    def render_matrix(self):
        snap = [row[:] for row in self.grid]
        r, c = self.agent
        snap[r][c] = Tile.PLAYER
        return snap

    def serialize_state(self):
        lines = [" ".join(str(int(v)) for v in row) for row in self.render_matrix()]
        out = ["GRID (0 floor,1 wall,2 player,3 kitchen,4 oven,5 board):"] + lines
        out.append(f"Agent: {self.agent}")
        for st in STATION_ORDER:
            pos = self._find_station(st)
            if pos:
                out.append(f"{STATION_CONFIG[st]['name']}: {pos}")
        out.append("Tasks:")
        for t in self.tasks:
            out.append(f"  [{'x' if t.done else ' '}] {t.text} "
                       f"({STATION_CONFIG[t.station]['name']})")
        return "\n".join(out)
