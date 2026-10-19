var fe=(e,n)=>()=>(n||e((n={exports:{}}).exports,n),n.exports);var pn=fe((un,O)=>{var _e="",me="",ge=`"""Long-run landing probabilities via a NumPy Markov chain.

States (one step = one roll of 2d6):
  (position, doubles rolled so far this turn) for every space, plus
  J1, J2, J3 = in jail, about to make attempt 1/2/3 to roll doubles.

Modelled: Go To Jail, every movement card (each drawn with probability
1/deck size), three consecutive doubles -> jail, and the two jail strategies:
  "leave": pay / use a card at once, then roll normally (what the AI does by default)
  "stay":  roll for doubles up to three times (leaving by doubles gives no extra roll)

The stationary distribution says where rolls end in the long run, which is
directly comparable with the simulation's \`end_share\`. Simulation remains the
ground truth; this model is for estimates, analytics and as a cross-check.
"""

from __future__ import annotations

import numpy as np

from ..engine.board import Board, SpaceType

JAIL_STRATEGIES = ("leave", "stay")

_PAIRS = [(a, b) for a in range(1, 7) for b in range(1, 7)]
_cache: dict[tuple[int, str], tuple[Board, np.ndarray, float]] = {}


def _resolve(board: Board, j: int, depth: int = 0) -> tuple[np.ndarray, float]:
    """After arriving on j: (distribution over free final positions, P(sent to jail))."""
    n = len(board)
    free = np.zeros(n)
    space = board[j]
    if space.type == SpaceType.GO_TO_JAIL:
        return free, 1.0
    deck = None
    if space.type == SpaceType.CHANCE:
        deck = board.chance_cards
    elif space.type == SpaceType.COMMUNITY_CHEST:
        deck = board.community_chest_cards
    if not deck or depth > 2:
        free[j] = 1.0
        return free, 0.0
    w = 1.0 / len(deck)
    jail = 0.0
    for card in deck:
        a, prm = card.action, card.params
        if a == "advance_to":
            free[int(prm["target"])] += w
        elif a == "advance_to_nearest":
            free[board.nearest(j, SpaceType(prm["kind"]))] += w
        elif a == "move_relative":
            f, p = _resolve(board, (j + int(prm["spaces"])) % n, depth + 1)
            free += w * f
            jail += w * p
        elif a == "go_to_jail":
            jail += w
        else:
            free[j] += w
    return free, jail


def _build(board: Board, jail_strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Transition matrix over 3n + 3 states and a mask of turn-ending states."""
    if jail_strategy not in JAIL_STRATEGIES:
        raise ValueError(f"jail_strategy must be one of {JAIL_STRATEGIES}")
    n = len(board)
    J1, J2, J3 = 3 * n, 3 * n + 1, 3 * n + 2
    size = 3 * n + 3
    jail = board.jail_index
    resolved = [_resolve(board, j) for j in range(n)]
    T = np.zeros((size, size))

    def move(row: int, start: int, total: int, next_d: int, weight: float) -> None:
        free, p_jail = resolved[(start + total) % n]
        T[row, J1] += weight * p_jail
        T[row, np.arange(n) * 3 + next_d] += weight * free

    for pos in range(n):
        for d in range(3):
            row = pos * 3 + d
            for a, b in _PAIRS:
                if a == b and d == 2:
                    T[row, J1] += 1 / 36  # third doubles
                else:
                    move(row, pos, a + b, d + 1 if a == b else 0, 1 / 36)

    if jail_strategy == "leave":
        # Pay, then an ordinary roll from Jail. J2/J3 are unreachable but keep
        # valid rows so the matrix stays stochastic.
        T[J1] = T[J2] = T[J3] = T[jail * 3]
    else:
        for state, nxt in ((J1, J2), (J2, J3), (J3, None)):
            for a, b in _PAIRS:
                if a == b or nxt is None:
                    move(state, jail, a + b, 0, 1 / 36)  # out: move, no extra roll
                else:
                    T[state, nxt] += 1 / 36

    # A roll ends the turn when the next state starts a fresh turn.
    ends_turn = np.zeros(size, dtype=bool)
    ends_turn[np.arange(n) * 3] = True
    ends_turn[[J1, J2, J3]] = True
    return T, ends_turn


def transition_matrix(board: Board, jail_strategy: str = "leave") -> np.ndarray:
    return _build(board, jail_strategy)[0]


def _solve(board: Board, jail_strategy: str) -> tuple[np.ndarray, float]:
    key = (id(board), jail_strategy)
    hit = _cache.get(key)
    if hit is not None and hit[0] is board:
        return hit[1], hit[2]
    T, ends_turn = _build(board, jail_strategy)
    size = T.shape[0]
    A = np.vstack([T.T - np.eye(size), np.ones(size)])
    rhs = np.zeros(size + 1)
    rhs[-1] = 1.0
    states, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    states = np.clip(states, 0, None)
    states /= states.sum()
    n = len(board)
    pos = states[: 3 * n].reshape(n, 3).sum(axis=1)
    pos[board.jail_index] += states[3 * n :].sum()
    rolls_per_turn = 1.0 / states[ends_turn].sum()
    _cache[key] = (board, pos, rolls_per_turn)
    return pos, rolls_per_turn


def stationary_distribution(board: Board, jail_strategy: str = "leave") -> np.ndarray:
    """Probability that a roll ends on each space, in the long run."""
    return _solve(board, jail_strategy)[0]


def rolls_per_turn(board: Board, jail_strategy: str = "leave") -> float:
    """Average number of rolls a player makes per turn."""
    return _solve(board, jail_strategy)[1]


# Kept for callers that want a per-turn scale; exact value depends on strategy.
EXPECTED_ROLLS_PER_TURN = 1 + 1 / 6 + 1 / 36
`,ye=`"""Return on investment of complete colour groups (analytic, from the Markov model).

For a group owned outright at a given building level:
  investment          = sum of prices + houses bought on every lot
  rent per opp. turn  = sum over lots of P(an opponent's roll ends there) * rent
                        * rolls per turn
  payback (turns)     = investment / rent per opponent turn

Lower payback means the money comes back faster. With several opponents the
payback in *rounds* is payback / number of opponents.
"""

from __future__ import annotations

from typing import Any

from ..engine.board import Board
from .landing import rolls_per_turn, stationary_distribution

LEVEL_NAMES = ["monopoly, no houses", "1 house", "2 houses", "3 houses", "4 houses", "hotel"]


def group_roi(board: Board, jail_strategy: str = "leave") -> list[dict[str, Any]]:
    pi = stationary_distribution(board, jail_strategy)
    per_turn = rolls_per_turn(board, jail_strategy)
    rows: list[dict[str, Any]] = []
    for group, lots in board.groups.items():
        for level in range(6):
            invest = sum(board[i].price + level * board[i].house_cost for i in lots)
            rent = sum(
                float(pi[i]) * (board[i].rent[0] * 2 if level == 0 else board[i].rent[level]) for i in lots
            ) * per_turn
            rows.append({
                "group": group,
                "level": level,
                "level_name": LEVEL_NAMES[level],
                "investment": invest,
                "rent_per_opponent_turn": rent,
                "payback_turns": invest / rent if rent else None,
                "landing_share": float(sum(pi[i] for i in lots)),
            })
    rr = board.railroads
    if rr:
        invest = sum(board[i].price for i in rr)
        rent = sum(float(pi[i]) for i in rr) * board.railroad_rents[len(rr) - 1] * per_turn
        rows.append({"group": "railroads", "level": None, "level_name": "all owned", "investment": invest,
                     "rent_per_opponent_turn": rent, "payback_turns": invest / rent,
                     "landing_share": float(sum(pi[i] for i in rr))})
    ut = board.utilities
    if ut:
        invest = sum(board[i].price for i in ut)
        rent = sum(float(pi[i]) for i in ut) * board.utility_multipliers[len(ut) - 1] * 7 * per_turn
        rows.append({"group": "utilities", "level": None, "level_name": "all owned", "investment": invest,
                     "rent_per_opponent_turn": rent, "payback_turns": invest / rent,
                     "landing_share": float(sum(pi[i] for i in ut))})
    return rows
`,be="",he=`"""Framework-free API logic.

Used by the FastAPI server (api/main.py) and by the browser build, which runs
this package in Pyodide and calls \`handle()\` directly. Inputs are plain JSON
dicts; errors are raised as ApiError(status, detail).
"""

from __future__ import annotations

import json
from typing import Any

from ..analytics.landing import JAIL_STRATEGIES, rolls_per_turn, stationary_distribution
from ..analytics.roi import group_roi
from ..engine.board import standard_board
from ..engine.rules import RuleConfig
from ..policies.human import AnswerMismatch
from ..policies.registry import POLICIES, describe_policies
from ..simulation import session
from ..simulation.runner import ExperimentConfig, PolicySpec, run_experiment, run_single_game


class ApiError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _int(d: dict[str, Any], key: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ApiError(422, f"{key} must be a whole number")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ApiError(422, f"{key} must be between {lo} and {hi}")
    return v


def _bool(d: dict[str, Any], key: str, default: bool) -> bool:
    v = d.get(key, default)
    if not isinstance(v, bool):
        raise ApiError(422, f"{key} must be true or false")
    return v


def specs(players: Any, allow_human: bool = False) -> list[PolicySpec]:
    if not isinstance(players, list) or not 2 <= len(players) <= 8:
        raise ApiError(422, "players must list 2 to 8 seats")
    out = []
    for p in players:
        if not isinstance(p, dict) or not isinstance(p.get("policy"), str):
            raise ApiError(422, "each seat needs a policy name")
        name = p["policy"]
        if name not in POLICIES and not (allow_human and name == session.HUMAN):
            raise ApiError(422, f"Unknown policy {name!r}")
        params = p.get("params") or {}
        if not isinstance(params, dict):
            raise ApiError(422, "params must be an object")
        label = p.get("label")
        out.append(PolicySpec(name, params, label if isinstance(label, str) and label else None))
    return out


def rules(r: Any) -> RuleConfig:
    r = r or {}
    if not isinstance(r, dict):
        raise ApiError(422, "rules must be an object")
    cash = r.get("starting_cash")
    if cash is not None and (isinstance(cash, bool) or not isinstance(cash, int) or cash < 0):
        raise ApiError(422, "starting_cash must be a non-negative whole number")
    return RuleConfig(
        auctions_enabled=_bool(r, "auctions_enabled", True),
        max_rounds=_int(r, "max_rounds", 500, 10, 5000),
        starting_cash=cash,
        trading_enabled=_bool(r, "trading_enabled", True),
        max_trade_offers_per_turn=_int(r, "max_trade_offers_per_turn", 2, 0, 10),
    )


def _strategy(jail_strategy: Any) -> str:
    if jail_strategy not in JAIL_STRATEGIES:
        raise ApiError(422, f"jail_strategy must be one of {list(JAIL_STRATEGIES)}")
    return jail_strategy


# ------------------------------------------------------------------ endpoints
def board() -> dict[str, Any]:
    return standard_board().to_dict()


def policies() -> list[dict[str, Any]]:
    return describe_policies()


def landing_probabilities(jail_strategy: str = "leave") -> dict[str, Any]:
    b = standard_board()
    pi = stationary_distribution(b, _strategy(jail_strategy))
    return {
        "model": (f"Markov chain over rolls with doubles, three-doubles-to-jail, cards and jail "
                  f"('{jail_strategy}' strategy)"),
        "jail_strategy": jail_strategy,
        "rolls_per_turn": rolls_per_turn(b, jail_strategy),
        "spaces": [{"index": s.index, "name": s.name, "probability": float(pi[s.index])} for s in b.spaces],
    }


def analytics_group_roi(jail_strategy: str = "leave") -> dict[str, Any]:
    return {"jail_strategy": jail_strategy, "rows": group_roi(standard_board(), _strategy(jail_strategy))}


def play(body: dict[str, Any]) -> dict[str, Any]:
    """Advance a game with human seats: replay to the current question, apply
    the new answer if valid, and return the next question (or the result)."""
    seats = specs(body.get("players"), allow_human=True)
    cfg = rules(body.get("rules"))
    seed = _int(body, "seed", 0)
    since = _int(body, "since", 0, 0)
    answers = body.get("answers") or []
    answer = body.get("answer")
    if not isinstance(answers, list) or not all(isinstance(a, dict) for a in answers):
        raise ApiError(422, "answers must be a list of objects")
    answers = list(answers)
    try:
        current = session.play(seats, seed, cfg, answers)
        if answer is not None:
            if not isinstance(answer, dict):
                raise ApiError(422, "answer must be an object")
            if current.pending is None:
                raise ApiError(422, "The game is over; nothing to answer")
            ok, why = session.validate_answer(current.game, current.pending, answer)
            if not ok:
                raise ApiError(422, why)
            answers.append(answer)
            current = session.play(seats, seed, cfg, answers)
    except (AnswerMismatch, ValueError) as e:
        raise ApiError(422, str(e)) from e
    return {**session.describe(current, since), "answers": answers}


def experiments(body: dict[str, Any], max_games: int = 5000, workers: int = 1) -> dict[str, Any]:
    n_games = _int(body, "n_games", 500, 1)
    if n_games > max_games:
        raise ApiError(422, f"n_games is limited to {max_games} per request")
    cfg = ExperimentConfig(
        players=specs(body.get("players")),
        n_games=n_games,
        seed=_int(body, "seed", 0),
        rules=rules(body.get("rules")),
        rotate_seats=_bool(body, "rotate_seats", True),
        workers=workers if n_games >= 100 else 1,
    )
    try:
        return run_experiment(cfg).summary
    except ValueError as e:  # bad policy params
        raise ApiError(422, str(e)) from e


def replay(body: dict[str, Any]) -> dict[str, Any]:
    """Play one seeded AI game and return its full event log (for the board view)."""
    try:
        result = run_single_game(specs(body.get("players")), _int(body, "seed", 0), rules(body.get("rules")),
                                 record_events=True)
    except ValueError as e:
        raise ApiError(422, str(e)) from e
    return result.to_dict()


# ------------------------------------------------------------------ dispatch
def handle(method: str, path: str, query: dict[str, str] | None = None, body: Any = None,
           max_games: int = 5000) -> tuple[int, Any]:
    """Route a request without a web framework. Returns (status, json_payload)."""
    query = query or {}
    try:
        if method == "GET":
            if path == "/api/health":
                return 200, {"status": "ok"}
            if path == "/api/board":
                return 200, board()
            if path == "/api/policies":
                return 200, policies()
            if path == "/api/analytics/landing-probabilities":
                return 200, landing_probabilities(query.get("jail_strategy", "leave"))
            if path == "/api/analytics/group-roi":
                return 200, analytics_group_roi(query.get("jail_strategy", "leave"))
        elif method == "POST":
            if not isinstance(body, dict):
                raise ApiError(422, "request body must be a JSON object")
            if path == "/api/play":
                return 200, play(body)
            if path == "/api/experiments":
                return 200, experiments(body, max_games=max_games)
            if path == "/api/games/replay":
                return 200, replay(body)
        return 404, {"detail": f"No route for {method} {path}"}
    except ApiError as e:
        return e.status, {"detail": e.detail}


def handle_json(request_json: str) -> str:
    """Pyodide entry point: {"method","path","query","body","max_games"} -> {"status","body"}."""
    req = json.loads(request_json)
    status, payload = handle(req.get("method", "GET"), req.get("path", ""), req.get("query"),
                             req.get("body"), int(req.get("max_games", 5000)))
    return json.dumps({"status": status, "body": payload})
`,ve=`"""Command-line tournament runner.\r
\r
    python -m monopoly_lab.cli --players baseline,property_value,development --games 1000 --seed 42\r
"""\r
\r
from __future__ import annotations\r
\r
import argparse\r
import json\r
import time\r
\r
from .engine.rules import RuleConfig\r
from .simulation.runner import ExperimentConfig, PolicySpec, run_experiment\r
\r
\r
def main(argv: list[str] | None = None) -> None:\r
    ap = argparse.ArgumentParser(description="Run a Monopoly policy tournament.")\r
    ap.add_argument("--players", default="baseline,property_value,development",\r
                    help="comma-separated policy names (repeat a name for duplicates)")\r
    ap.add_argument("--games", type=int, default=1000)\r
    ap.add_argument("--seed", type=int, default=0)\r
    ap.add_argument("--max-rounds", type=int, default=500)\r
    ap.add_argument("--no-auctions", action="store_true")\r
    ap.add_argument("--no-trading", action="store_true")\r
    ap.add_argument("--no-rotate", action="store_true")\r
    ap.add_argument("--workers", type=int, default=1)\r
    ap.add_argument("--json", metavar="PATH", help="write the full summary as JSON")\r
    args = ap.parse_args(argv)\r
\r
    cfg = ExperimentConfig(\r
        players=[PolicySpec(name=n.strip()) for n in args.players.split(",") if n.strip()],\r
        n_games=args.games,\r
        seed=args.seed,\r
        rules=RuleConfig(max_rounds=args.max_rounds, auctions_enabled=not args.no_auctions,\r
                         trading_enabled=not args.no_trading),\r
        rotate_seats=not args.no_rotate,\r
        workers=args.workers,\r
    )\r
    t0 = time.perf_counter()\r
    res = run_experiment(cfg)\r
    dt = time.perf_counter() - t0\r
    s = res.summary\r
\r
    print(f"{s['n_games']} games in {dt:.1f}s  |  finished {s['finished_games']}, "\r
          f"timeouts {s['timeouts']} ({s['timeout_rate']:.1%})")\r
    gl = s["game_length_rounds"]\r
    print(f"rounds (finished games): mean {gl['mean']:.1f}, median {gl['median']:.0f}, p90 {gl['p90']:.0f}\\n")\r
    print(f"{'policy':<22}{'win rate':>10}{'95% CI':>18}{'rent in':>10}{'rent out':>10}{'houses':>8}")\r
    rows = sorted(s["policies"].values(), key=lambda r: -r["win_rate"])\r
    for r in rows:\r
        lo, hi = r["win_rate_ci95"]\r
        print(f"{r['label']:<22}{r['win_rate']:>10.1%}{f'[{lo:.1%}, {hi:.1%}]':>18}"\r
              f"{r['rent_collected_mean']:>10.0f}{r['rent_paid_mean']:>10.0f}{r['houses_built_mean']:>8.1f}")\r
    if args.json:\r
        with open(args.json, "w", encoding="utf-8") as f:\r
            json.dump(s, f, indent=2)\r
        print(f"\\nsummary written to {args.json}")\r
\r
\r
if __name__ == "__main__":\r
    main()\r
`,we="",ke=`{
  "name": "US Standard",
  "starting_cash": 1500,
  "go_salary": 200,
  "jail_fine": 50,
  "house_supply": 32,
  "hotel_supply": 12,
  "railroad_rents": [25, 50, 100, 200],
  "utility_multipliers": [4, 10],
  "groups": {
    "brown": "#8B4513",
    "lightblue": "#87CEEB",
    "pink": "#D93A96",
    "orange": "#F7941D",
    "red": "#ED1B24",
    "yellow": "#FEF200",
    "green": "#1FB25A",
    "darkblue": "#0072BB"
  },
  "spaces": [
    {"name": "Go", "type": "go"},
    {"name": "Mediterranean Avenue", "type": "street", "group": "brown", "price": 60, "rent": [2, 10, 30, 90, 160, 250], "house_cost": 50},
    {"name": "Community Chest", "type": "community_chest"},
    {"name": "Baltic Avenue", "type": "street", "group": "brown", "price": 60, "rent": [4, 20, 60, 180, 320, 450], "house_cost": 50},
    {"name": "Income Tax", "type": "tax", "amount": 200},
    {"name": "Reading Railroad", "type": "railroad", "price": 200},
    {"name": "Oriental Avenue", "type": "street", "group": "lightblue", "price": 100, "rent": [6, 30, 90, 270, 400, 550], "house_cost": 50},
    {"name": "Chance", "type": "chance"},
    {"name": "Vermont Avenue", "type": "street", "group": "lightblue", "price": 100, "rent": [6, 30, 90, 270, 400, 550], "house_cost": 50},
    {"name": "Connecticut Avenue", "type": "street", "group": "lightblue", "price": 120, "rent": [8, 40, 100, 300, 450, 600], "house_cost": 50},
    {"name": "Jail / Just Visiting", "type": "jail"},
    {"name": "St. Charles Place", "type": "street", "group": "pink", "price": 140, "rent": [10, 50, 150, 450, 625, 750], "house_cost": 100},
    {"name": "Electric Company", "type": "utility", "price": 150},
    {"name": "States Avenue", "type": "street", "group": "pink", "price": 140, "rent": [10, 50, 150, 450, 625, 750], "house_cost": 100},
    {"name": "Virginia Avenue", "type": "street", "group": "pink", "price": 160, "rent": [12, 60, 180, 500, 700, 900], "house_cost": 100},
    {"name": "Pennsylvania Railroad", "type": "railroad", "price": 200},
    {"name": "St. James Place", "type": "street", "group": "orange", "price": 180, "rent": [14, 70, 200, 550, 750, 950], "house_cost": 100},
    {"name": "Community Chest", "type": "community_chest"},
    {"name": "Tennessee Avenue", "type": "street", "group": "orange", "price": 180, "rent": [14, 70, 200, 550, 750, 950], "house_cost": 100},
    {"name": "New York Avenue", "type": "street", "group": "orange", "price": 200, "rent": [16, 80, 220, 600, 800, 1000], "house_cost": 100},
    {"name": "Free Parking", "type": "free_parking"},
    {"name": "Kentucky Avenue", "type": "street", "group": "red", "price": 220, "rent": [18, 90, 250, 700, 875, 1050], "house_cost": 150},
    {"name": "Chance", "type": "chance"},
    {"name": "Indiana Avenue", "type": "street", "group": "red", "price": 220, "rent": [18, 90, 250, 700, 875, 1050], "house_cost": 150},
    {"name": "Illinois Avenue", "type": "street", "group": "red", "price": 240, "rent": [20, 100, 300, 750, 925, 1100], "house_cost": 150},
    {"name": "B. & O. Railroad", "type": "railroad", "price": 200},
    {"name": "Atlantic Avenue", "type": "street", "group": "yellow", "price": 260, "rent": [22, 110, 330, 800, 975, 1150], "house_cost": 150},
    {"name": "Ventnor Avenue", "type": "street", "group": "yellow", "price": 260, "rent": [22, 110, 330, 800, 975, 1150], "house_cost": 150},
    {"name": "Water Works", "type": "utility", "price": 150},
    {"name": "Marvin Gardens", "type": "street", "group": "yellow", "price": 280, "rent": [24, 120, 360, 850, 1025, 1200], "house_cost": 150},
    {"name": "Go To Jail", "type": "go_to_jail"},
    {"name": "Pacific Avenue", "type": "street", "group": "green", "price": 300, "rent": [26, 130, 390, 900, 1100, 1275], "house_cost": 200},
    {"name": "North Carolina Avenue", "type": "street", "group": "green", "price": 300, "rent": [26, 130, 390, 900, 1100, 1275], "house_cost": 200},
    {"name": "Community Chest", "type": "community_chest"},
    {"name": "Pennsylvania Avenue", "type": "street", "group": "green", "price": 320, "rent": [28, 150, 450, 1000, 1200, 1400], "house_cost": 200},
    {"name": "Short Line", "type": "railroad", "price": 200},
    {"name": "Chance", "type": "chance"},
    {"name": "Park Place", "type": "street", "group": "darkblue", "price": 350, "rent": [35, 175, 500, 1100, 1300, 1500], "house_cost": 200},
    {"name": "Luxury Tax", "type": "tax", "amount": 100},
    {"name": "Boardwalk", "type": "street", "group": "darkblue", "price": 400, "rent": [50, 200, 600, 1400, 1700, 2000], "house_cost": 200}
  ],
  "chance": [
    {"id": "ch_boardwalk", "text": "Advance to Boardwalk.", "action": "advance_to", "target": 39},
    {"id": "ch_go", "text": "Advance to Go (Collect $200).", "action": "advance_to", "target": 0},
    {"id": "ch_illinois", "text": "Advance to Illinois Avenue. If you pass Go, collect $200.", "action": "advance_to", "target": 24},
    {"id": "ch_st_charles", "text": "Advance to St. Charles Place. If you pass Go, collect $200.", "action": "advance_to", "target": 11},
    {"id": "ch_railroad_1", "text": "Advance to the nearest Railroad. If owned, pay owner twice the rental.", "action": "advance_to_nearest", "kind": "railroad", "rent_multiplier": 2},
    {"id": "ch_railroad_2", "text": "Advance to the nearest Railroad. If owned, pay owner twice the rental.", "action": "advance_to_nearest", "kind": "railroad", "rent_multiplier": 2},
    {"id": "ch_utility", "text": "Advance to the nearest Utility. If owned, throw dice and pay owner ten times the amount thrown.", "action": "advance_to_nearest", "kind": "utility", "dice_multiplier": 10},
    {"id": "ch_dividend", "text": "Bank pays you dividend of $50.", "action": "collect", "amount": 50},
    {"id": "ch_goojf", "text": "Get Out of Jail Free.", "action": "get_out_of_jail_free"},
    {"id": "ch_back_3", "text": "Go Back 3 Spaces.", "action": "move_relative", "spaces": -3},
    {"id": "ch_jail", "text": "Go to Jail. Do not pass Go, do not collect $200.", "action": "go_to_jail"},
    {"id": "ch_repairs", "text": "Make general repairs on all your property: $25 per house, $100 per hotel.", "action": "repairs", "per_house": 25, "per_hotel": 100},
    {"id": "ch_speeding", "text": "Speeding fine $15.", "action": "pay", "amount": 15},
    {"id": "ch_reading", "text": "Take a trip to Reading Railroad. If you pass Go, collect $200.", "action": "advance_to", "target": 5},
    {"id": "ch_chairman", "text": "You have been elected Chairman of the Board. Pay each player $50.", "action": "pay_each_player", "amount": 50},
    {"id": "ch_loan", "text": "Your building loan matures. Collect $150.", "action": "collect", "amount": 150}
  ],
  "community_chest": [
    {"id": "cc_go", "text": "Advance to Go (Collect $200).", "action": "advance_to", "target": 0},
    {"id": "cc_bank_error", "text": "Bank error in your favor. Collect $200.", "action": "collect", "amount": 200},
    {"id": "cc_doctor", "text": "Doctor's fee. Pay $50.", "action": "pay", "amount": 50},
    {"id": "cc_stock", "text": "From sale of stock you get $50.", "action": "collect", "amount": 50},
    {"id": "cc_goojf", "text": "Get Out of Jail Free.", "action": "get_out_of_jail_free"},
    {"id": "cc_jail", "text": "Go to Jail. Do not pass Go, do not collect $200.", "action": "go_to_jail"},
    {"id": "cc_holiday", "text": "Holiday fund matures. Receive $100.", "action": "collect", "amount": 100},
    {"id": "cc_tax_refund", "text": "Income tax refund. Collect $20.", "action": "collect", "amount": 20},
    {"id": "cc_birthday", "text": "It is your birthday. Collect $10 from every player.", "action": "collect_from_each_player", "amount": 10},
    {"id": "cc_life_insurance", "text": "Life insurance matures. Collect $100.", "action": "collect", "amount": 100},
    {"id": "cc_hospital", "text": "Pay hospital fees of $100.", "action": "pay", "amount": 100},
    {"id": "cc_school", "text": "Pay school fees of $50.", "action": "pay", "amount": 50},
    {"id": "cc_consultancy", "text": "Receive $25 consultancy fee.", "action": "collect", "amount": 25},
    {"id": "cc_street_repairs", "text": "You are assessed for street repairs: $40 per house, $115 per hotel.", "action": "repairs", "per_house": 40, "per_hotel": 115},
    {"id": "cc_beauty", "text": "You have won second prize in a beauty contest. Collect $10.", "action": "collect", "amount": 10},
    {"id": "cc_inherit", "text": "You inherit $100.", "action": "collect", "amount": 100}
  ]
}
`,xe="",Te=`"""Board and property definitions.

Board data is loaded from JSON so alternative boards (other editions, house
variants, test fixtures) can be used without touching engine code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from importlib import resources
from pathlib import Path
from typing import Any


class SpaceType(str, Enum):
    GO = "go"
    STREET = "street"
    RAILROAD = "railroad"
    UTILITY = "utility"
    TAX = "tax"
    CHANCE = "chance"
    COMMUNITY_CHEST = "community_chest"
    JAIL = "jail"
    FREE_PARKING = "free_parking"
    GO_TO_JAIL = "go_to_jail"


OWNABLE_TYPES = frozenset({SpaceType.STREET, SpaceType.RAILROAD, SpaceType.UTILITY})


@dataclass(frozen=True)
class Space:
    index: int
    name: str
    type: SpaceType
    price: int = 0
    group: str | None = None
    # Street rent by building level: [base, 1h, 2h, 3h, 4h, hotel].
    rent: tuple[int, ...] = ()
    house_cost: int = 0
    tax_amount: int = 0
    mortgage_value: int = 0

    @property
    def ownable(self) -> bool:
        return self.type in OWNABLE_TYPES

    @property
    def unmortgage_cost(self) -> int:
        # Mortgage value plus 10% interest, rounded up to the dollar.
        return (self.mortgage_value * 11 + 9) // 10


@dataclass(frozen=True)
class Card:
    id: str
    text: str
    action: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Board:
    name: str
    spaces: tuple[Space, ...]
    chance_cards: tuple[Card, ...]
    community_chest_cards: tuple[Card, ...]
    starting_cash: int
    go_salary: int
    jail_fine: int
    house_supply: int
    hotel_supply: int
    railroad_rents: tuple[int, ...]
    utility_multipliers: tuple[int, ...]
    group_colors: dict[str, str]

    def __len__(self) -> int:
        return len(self.spaces)

    def __getitem__(self, index: int) -> Space:
        return self.spaces[index]

    @cached_property
    def jail_index(self) -> int:
        return self._single(SpaceType.JAIL)

    @cached_property
    def go_index(self) -> int:
        return self._single(SpaceType.GO)

    @cached_property
    def groups(self) -> dict[str, tuple[int, ...]]:
        out: dict[str, list[int]] = {}
        for s in self.spaces:
            if s.type == SpaceType.STREET and s.group:
                out.setdefault(s.group, []).append(s.index)
        return {g: tuple(ix) for g, ix in out.items()}

    @cached_property
    def railroads(self) -> tuple[int, ...]:
        return tuple(s.index for s in self.spaces if s.type == SpaceType.RAILROAD)

    @cached_property
    def utilities(self) -> tuple[int, ...]:
        return tuple(s.index for s in self.spaces if s.type == SpaceType.UTILITY)

    @cached_property
    def ownable_indices(self) -> tuple[int, ...]:
        return tuple(s.index for s in self.spaces if s.ownable)

    def _single(self, t: SpaceType) -> int:
        matches = [s.index for s in self.spaces if s.type == t]
        if len(matches) != 1:
            raise ValueError(f"Board must have exactly one {t.value} space, found {len(matches)}")
        return matches[0]

    def nearest(self, position: int, kind: SpaceType) -> int:
        """Next space of \`kind\` strictly ahead of \`position\` (wrapping)."""
        n = len(self.spaces)
        for step in range(1, n + 1):
            idx = (position + step) % n
            if self.spaces[idx].type == kind:
                return idx
        raise ValueError(f"No {kind.value} on board")

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Board":
        spaces = []
        for i, raw in enumerate(data["spaces"]):
            t = SpaceType(raw["type"])
            price = int(raw.get("price", 0))
            rent = tuple(int(r) for r in raw.get("rent", ()))
            if t == SpaceType.STREET and len(rent) != 6:
                raise ValueError(f"Street {raw['name']!r} needs 6 rent values")
            spaces.append(
                Space(
                    index=i,
                    name=raw["name"],
                    type=t,
                    price=price,
                    group=raw.get("group"),
                    rent=rent,
                    house_cost=int(raw.get("house_cost", 0)),
                    tax_amount=int(raw.get("amount", 0)),
                    mortgage_value=int(raw.get("mortgage_value", price // 2)),
                )
            )

        def cards(key: str) -> tuple[Card, ...]:
            out = []
            for raw in data.get(key, []):
                params = {k: v for k, v in raw.items() if k not in ("id", "text", "action")}
                out.append(Card(id=raw["id"], text=raw["text"], action=raw["action"], params=params))
            return tuple(out)

        board = cls(
            name=data.get("name", "Custom"),
            spaces=tuple(spaces),
            chance_cards=cards("chance"),
            community_chest_cards=cards("community_chest"),
            starting_cash=int(data.get("starting_cash", 1500)),
            go_salary=int(data.get("go_salary", 200)),
            jail_fine=int(data.get("jail_fine", 50)),
            house_supply=int(data.get("house_supply", 32)),
            hotel_supply=int(data.get("hotel_supply", 12)),
            railroad_rents=tuple(data.get("railroad_rents", (25, 50, 100, 200))),
            utility_multipliers=tuple(data.get("utility_multipliers", (4, 10))),
            group_colors=dict(data.get("groups", {})),
        )
        # Validate required singletons early.
        _ = board.jail_index, board.go_index
        return board

    @classmethod
    def from_json(cls, path: str | Path) -> "Board":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        """Serializable description (used by the API/frontend)."""
        return {
            "name": self.name,
            "starting_cash": self.starting_cash,
            "go_salary": self.go_salary,
            "jail_fine": self.jail_fine,
            "groups": self.group_colors,
            "spaces": [
                {
                    "index": s.index,
                    "name": s.name,
                    "type": s.type.value,
                    "price": s.price,
                    "group": s.group,
                    "rent": list(s.rent),
                    "house_cost": s.house_cost,
                    "tax_amount": s.tax_amount,
                    "mortgage_value": s.mortgage_value,
                }
                for s in self.spaces
            ],
        }


_STANDARD: Board | None = None


def standard_board() -> Board:
    """The standard US board (cached; Board is immutable)."""
    global _STANDARD
    if _STANDARD is None:
        text = resources.files("monopoly_lab.data").joinpath("us_standard.json").read_text("utf-8")
        _STANDARD = Board.from_dict(json.loads(text))
    return _STANDARD
`,Ae=`"""Houses, hotels and building costs.

Levels: 0-4 houses, 5 = hotel. Enforces monopoly ownership, no mortgaged lot
in the group, the even-build / even-sell rules and the bank's finite supply.
"""

from __future__ import annotations

from .board import SpaceType
from .state import GameState

HOTEL = 5


def can_build(state: GameState, pid: int, idx: int, check_cash: bool = True) -> tuple[bool, str]:
    space = state.board[idx]
    if space.type != SpaceType.STREET:
        return False, "not a street"
    if state.owner[idx] != pid:
        return False, "not owner"
    group = state.board.groups[space.group]
    if not state.owns_group(pid, space.group):
        return False, "no monopoly"
    if any(state.mortgaged[i] for i in group):
        return False, "group has a mortgaged lot"
    level = state.buildings[idx]
    if level >= HOTEL:
        return False, "already has hotel"
    if level > min(state.buildings[i] for i in group):
        return False, "even-build rule"
    if level == 4:
        if state.bank_hotels <= 0:
            return False, "no hotels in bank"
    elif state.bank_houses <= 0:
        return False, "no houses in bank"
    if check_cash and state.players[pid].cash < space.house_cost:
        return False, "insufficient cash"
    return True, ""


def build(state: GameState, pid: int, idx: int) -> None:
    ok, why = can_build(state, pid, idx)
    if not ok:
        raise ValueError(f"cannot build on {state.board[idx].name}: {why}")
    space = state.board[idx]
    state.players[pid].cash -= space.house_cost
    if state.buildings[idx] == 4:
        state.bank_hotels -= 1
        state.bank_houses += 4
    else:
        state.bank_houses -= 1
    state.buildings[idx] += 1


def can_sell_building(state: GameState, pid: int, idx: int) -> tuple[bool, str]:
    space = state.board[idx]
    if space.type != SpaceType.STREET or state.owner[idx] != pid:
        return False, "not owner"
    level = state.buildings[idx]
    if level == 0:
        return False, "no buildings"
    group = state.board.groups[space.group]
    if level < max(state.buildings[i] for i in group):
        return False, "even-sell rule"
    return True, ""


def sell_building(state: GameState, pid: int, idx: int) -> int:
    """Sell one level back to the bank at half cost. Returns cash received.

    Hotel → 4 houses when the bank has 4 houses. During a housing shortage the
    lot drops to as many houses as the bank holds, refunding each level removed.
    """
    ok, why = can_sell_building(state, pid, idx)
    if not ok:
        raise ValueError(f"cannot sell on {state.board[idx].name}: {why}")
    half = state.board[idx].house_cost // 2
    level = state.buildings[idx]
    if level == HOTEL:
        new_level = min(4, state.bank_houses)
        state.bank_hotels += 1
        state.bank_houses -= new_level
    else:
        new_level = level - 1
        state.bank_houses += 1
    refund = half * (level - new_level)
    state.buildings[idx] = new_level
    state.players[pid].cash += refund
    return refund


def building_sale_value(state: GameState, pid: int) -> int:
    """Cash a player would get by selling every building."""
    return sum(
        state.buildings[i] * (state.board[i].house_cost // 2) for i in state.properties_of(pid)
    )


def sell_all_buildings(state: GameState, pid: int) -> int:
    total = 0
    while True:
        sellable = [i for i in state.properties_of(pid) if can_sell_building(state, pid, i)[0]]
        if not sellable:
            return total
        for i in sellable:
            if can_sell_building(state, pid, i)[0]:
                total += sell_building(state, pid, i)


def buildable_lots(state: GameState, pid: int) -> list[int]:
    return [i for i in state.properties_of(pid) if can_build(state, pid, i)[0]]
`,Se=`"""Chance and Community Chest decks.

Card *effects* are applied by \`Game._apply_card\` because they need movement,
payment and landing resolution; this module owns deck mechanics.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .board import Card

GOOJF = "get_out_of_jail_free"


class Deck:
    def __init__(self, name: str, cards: tuple[Card, ...], rng: np.random.Generator | None = None):
        self.name = name
        self._by_id = {c.id: c for c in cards}
        order = list(cards)
        if rng is not None:
            perm = rng.permutation(len(order))
            order = [order[i] for i in perm]
        self.cards: deque[Card] = deque(order)

    def draw(self) -> Card:
        """Draw from the top. Ordinary cards go to the bottom; a Get Out of
        Jail Free card is held by the player until used (see \`return_card\`)."""
        card = self.cards.popleft()
        if card.action != GOOJF:
            self.cards.append(card)
        return card

    def owns(self, card_id: str) -> bool:
        return card_id in self._by_id

    def return_card(self, card_id: str) -> None:
        self.cards.append(self._by_id[card_id])

    def set_order(self, card_ids: list[str]) -> None:
        """Test helper: put the given cards on top, in order."""
        top = [self._by_id[c] for c in card_ids]
        rest = [c for c in self.cards if c.id not in set(card_ids)]
        self.cards = deque(top + rest)
`,Ee=`"""Dice. All randomness comes from a numpy Generator so games are reproducible."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np


class DiceLike(Protocol):
    def roll(self) -> tuple[int, int]: ...


class Dice:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def roll(self) -> tuple[int, int]:
        a, b = self.rng.integers(1, 7, size=2)
        return int(a), int(b)


class ScriptedDice:
    """Deterministic dice for tests: yields the given rolls in order."""

    def __init__(self, rolls: Iterable[tuple[int, int]]):
        self._rolls = list(rolls)
        self._i = 0

    def roll(self) -> tuple[int, int]:
        if self._i >= len(self._rolls):
            raise RuntimeError("ScriptedDice exhausted")
        r = self._rolls[self._i]
        self._i += 1
        return r

    @property
    def remaining(self) -> int:
        return len(self._rolls) - self._i
`,Ne=`"""Mortgages, net worth and liquidation value."""

from __future__ import annotations

from .board import SpaceType
from .buildings import building_sale_value
from .state import GameState


def can_mortgage(state: GameState, pid: int, idx: int) -> tuple[bool, str]:
    space = state.board[idx]
    if state.owner[idx] != pid:
        return False, "not owner"
    if state.mortgaged[idx]:
        return False, "already mortgaged"
    if space.type == SpaceType.STREET:
        if any(state.buildings[i] > 0 for i in state.board.groups[space.group]):
            return False, "group has buildings"
    return True, ""


def mortgage(state: GameState, pid: int, idx: int) -> int:
    ok, why = can_mortgage(state, pid, idx)
    if not ok:
        raise ValueError(f"cannot mortgage {state.board[idx].name}: {why}")
    state.mortgaged[idx] = True
    value = state.board[idx].mortgage_value
    state.players[pid].cash += value
    return value


def can_unmortgage(state: GameState, pid: int, idx: int) -> tuple[bool, str]:
    if state.owner[idx] != pid:
        return False, "not owner"
    if not state.mortgaged[idx]:
        return False, "not mortgaged"
    if state.players[pid].cash < state.board[idx].unmortgage_cost:
        return False, "insufficient cash"
    return True, ""


def unmortgage(state: GameState, pid: int, idx: int) -> int:
    ok, why = can_unmortgage(state, pid, idx)
    if not ok:
        raise ValueError(f"cannot unmortgage {state.board[idx].name}: {why}")
    cost = state.board[idx].unmortgage_cost
    state.players[pid].cash -= cost
    state.mortgaged[idx] = False
    return cost


def liquidation_value(state: GameState, pid: int) -> int:
    """Max cash obtainable: cash + selling all buildings + mortgaging all lots."""
    p = state.players[pid]
    mortgageable = sum(
        state.board[i].mortgage_value for i in state.properties_of(pid) if not state.mortgaged[i]
    )
    return p.cash + building_sale_value(state, pid) + mortgageable


def net_worth(state: GameState, pid: int) -> int:
    """Asset value at cost: cash + price of unmortgaged lots + equity in
    mortgaged lots (price - payoff) + full building cost."""
    total = state.players[pid].cash
    for i in state.properties_of(pid):
        s = state.board[i]
        total += (s.price - s.unmortgage_cost) if state.mortgaged[i] else s.price
        total += state.buildings[i] * s.house_cost
    return total
`,je=`"""Turn orchestration.\r
\r
\`Game\` is the only place where policies are consulted and where rule modules\r
are combined into a turn. All money movement goes through \`Game.pay\` so that\r
debt resolution and bankruptcy are handled uniformly.\r
"""\r
\r
from __future__ import annotations\r
\r
from typing import TYPE_CHECKING, Any\r
\r
import numpy as np\r
\r
from . import buildings as bld\r
from . import finance\r
from .board import Board, Card, SpaceType, standard_board\r
from .cards import GOOJF, Deck\r
from .dice import Dice, DiceLike\r
from .jail import JailAction, release, send_to_jail\r
from .movement import move_by, move_to\r
from .rent import compute_rent\r
from .rules import RuleConfig\r
from .state import GameState\r
from .stats import GameResult, PlayerStats\r
from .trading import Trade, execute_trade, validate_trade\r
\r
if TYPE_CHECKING:\r
    from ..policies.base import Policy\r
\r
# Upper bounds on policy round-trips per decision point; they guarantee\r
# termination even if a policy keeps proposing illegal actions.\r
_MAX_POLICY_ROUNDS = 64\r
_MAX_TURN_ACTIONS = 500\r
\r
\r
class Game:\r
    def __init__(\r
        self,\r
        policies: list["Policy"],\r
        board: Board | None = None,\r
        rules: RuleConfig | None = None,\r
        seed: int | np.random.SeedSequence | None = None,\r
        dice: DiceLike | None = None,\r
        names: list[str] | None = None,\r
        record_events: bool = False,\r
    ):\r
        if len(policies) < 2:\r
            raise ValueError("Need at least 2 players")\r
        self.board = board or standard_board()\r
        self.rules = rules or RuleConfig()\r
        self.seed = seed if isinstance(seed, int) else None\r
        self.rng = np.random.default_rng(seed)\r
        self.dice: DiceLike = dice or Dice(self.rng)\r
        self.policies = policies\r
        names = names or [f"P{i + 1} ({p.name})" for i, p in enumerate(policies)]\r
        self.state = GameState.new(self.board, names, self.rules.starting_cash)\r
        self.chance = Deck("chance", self.board.chance_cards, self.rng)\r
        self.community_chest = Deck("community_chest", self.board.community_chest_cards, self.rng)\r
        self.stats = [PlayerStats(i, names[i], p.name) for i, p in enumerate(policies)]\r
        n = len(self.board)\r
        # Arrivals on each space (incl. card squares passed through), and where\r
        # each roll's movement finally ends (after cards / Go To Jail).\r
        self.landings = [0] * n\r
        self.end_positions = [0] * n\r
        self.rent_by_space = [0] * n\r
        # Money sunk into each space: purchase/auction price + buildings - building refunds.\r
        self.invested_by_space = [0] * n\r
        self.bankruptcy_order: list[int] = []\r
        self.events: list[dict] | None = [] if record_events else None\r
        self.finished = False\r
        self.timed_out = False\r
        self.winner: int | None = None\r
        self.trade_offers_this_turn = 0\r
        self.last_dice: tuple[int, int] | None = None\r
        # Colour groups where a hotel was sold during a housing shortage.\r
        self.shortage_groups: set[str] = set()\r
        self._log("start", **self._snapshot())\r
\r
    # ================================================================ public\r
    def run(self) -> GameResult:\r
        while not self.finished:\r
            self.play_turn()\r
        return self.result()\r
\r
    def play_turn(self) -> None:\r
        if self.finished:\r
            return\r
        s = self.state\r
        pid = s.current\r
        if not s.players[pid].bankrupt:\r
            self._take_turn(pid)\r
            self._log("turn_end", pid=pid, **self._snapshot())\r
        s.turn += 1\r
        self._check_winner()\r
        if self.finished:\r
            return\r
        nxt, wrapped = pid, False\r
        while True:\r
            nxt += 1\r
            if nxt >= len(s.players):\r
                nxt, wrapped = 0, True\r
            if not s.players[nxt].bankrupt:\r
                break\r
        s.current = nxt\r
        if wrapped:\r
            s.round += 1\r
            if s.round >= self.rules.max_rounds:\r
                self.finished = True\r
                self.timed_out = True\r
                self._log("timeout", round=s.round)\r
\r
    def result(self) -> GameResult:\r
        s = self.state\r
        for st in self.stats:\r
            st.final_cash = s.players[st.player_id].cash\r
            st.final_net_worth = finance.net_worth(s, st.player_id)\r
        active = [p.id for p in s.active_players()]\r
        leader = max(active, key=lambda i: self.stats[i].final_net_worth) if active else -1\r
        return GameResult(\r
            seed=self.seed,\r
            winner=self.winner,\r
            timed_out=self.timed_out,\r
            turns=s.turn,\r
            rounds=s.round,\r
            bankruptcy_order=list(self.bankruptcy_order),\r
            net_worth_leader=leader,\r
            players=self.stats,\r
            landings=list(self.landings),\r
            end_positions=list(self.end_positions),\r
            rent_by_space=list(self.rent_by_space),\r
            invested_by_space=list(self.invested_by_space),\r
            events=self.events,\r
        )\r
\r
    # ============================================================== turn flow\r
    def _take_turn(self, pid: int) -> None:\r
        p = self.state.players[pid]\r
        self.trade_offers_this_turn = 0\r
        self._action_phase(pid, "pre_roll")\r
        if p.bankrupt or self.finished:\r
            return\r
        if p.in_jail:\r
            outcome = self._jail_turn(pid)\r
            if outcome is None:  # still in jail, or bankrupt paying the fine\r
                if not p.bankrupt and not self.finished:\r
                    self._management(pid)\r
                return\r
            if outcome != "free":\r
                # Left jail by rolling: move by that roll, no extra roll for doubles.\r
                self._advance(pid, sum(outcome))\r
                if not p.bankrupt and not self.finished:\r
                    self._management(pid)\r
                return\r
\r
        doubles = 0\r
        while True:\r
            if doubles:  # rolling again after doubles: another chance to act first\r
                self._action_phase(pid, "pre_roll")\r
                if p.bankrupt or self.finished:\r
                    return\r
            d1, d2 = self._roll()\r
            is_double = d1 == d2\r
            self._log("roll", pid=pid, dice=[d1, d2])\r
            if is_double:\r
                doubles += 1\r
                if doubles == 3:\r
                    self._jail(pid, reason="three_doubles")\r
                    self.end_positions[p.position] += 1\r
                    break\r
            self._advance(pid, d1 + d2)\r
            if p.bankrupt or p.in_jail or self.finished or not is_double:\r
                break\r
        if not p.bankrupt and not self.finished:\r
            self._management(pid)\r
\r
    def _jail_turn(self, pid: int) -> str | tuple[int, int] | None:\r
        """Returns "free" (left before rolling; play a normal turn), a dice\r
        tuple (left by rolling; move by it), or None (turn over)."""\r
        s = self.state\r
        p = s.players[pid]\r
        fine = self.board.jail_fine\r
        action = self.policies[pid].decide_jail(self, pid)\r
        if action == JailAction.USE_CARD and not p.goojf_cards:\r
            action = JailAction.ROLL\r
        if action == JailAction.PAY and finance.liquidation_value(s, pid) < fine:\r
            action = JailAction.ROLL\r
        self._decision(pid, "jail", action.value)\r
\r
        if action == JailAction.USE_CARD:\r
            card_id = p.goojf_cards.pop(0)\r
            (self.chance if self.chance.owns(card_id) else self.community_chest).return_card(card_id)\r
            release(s, pid)\r
            self._log("jail_exit", pid=pid, method="card")\r
            return "free"\r
        if action == JailAction.PAY:\r
            self.pay(pid, fine, None, "jail_fine")\r
            if p.bankrupt:\r
                return None\r
            release(s, pid)\r
            self._log("jail_exit", pid=pid, method="pay")\r
            return "free"\r
\r
        d1, d2 = self._roll()\r
        self._log("roll", pid=pid, dice=[d1, d2], in_jail=True)\r
        if d1 == d2:\r
            release(s, pid)\r
            self._log("jail_exit", pid=pid, method="doubles")\r
            return (d1, d2)\r
        p.jail_turns += 1\r
        if p.jail_turns < self.rules.max_jail_turns:\r
            self.end_positions[p.position] += 1  # this roll ends in jail\r
        if p.jail_turns >= self.rules.max_jail_turns:\r
            self.pay(pid, fine, None, "jail_fine")\r
            if p.bankrupt:\r
                return None\r
            release(s, pid)\r
            self._log("jail_exit", pid=pid, method="forced_fine")\r
            return (d1, d2)\r
        return None\r
\r
    def _roll(self) -> tuple[int, int]:\r
        self.last_dice = self.dice.roll()\r
        return self.last_dice\r
\r
    def _advance(self, pid: int, steps: int) -> None:\r
        p = self.state.players[pid]\r
        start = p.position\r
        if move_by(self.state, pid, steps):\r
            self._collect_salary(pid)\r
        self._log("move", pid=pid, frm=start, to=p.position)\r
        self._land(pid, dice_total=steps)\r
        if not p.bankrupt:\r
            self.end_positions[p.position] += 1\r
\r
    def _collect_salary(self, pid: int) -> None:\r
        self.state.players[pid].cash += self.board.go_salary\r
        self.stats[pid].salary_collected += self.board.go_salary\r
        self._log("salary", pid=pid, amount=self.board.go_salary)\r
\r
    def _jail(self, pid: int, reason: str) -> None:\r
        send_to_jail(self.state, pid)\r
        self.stats[pid].times_jailed += 1\r
        self._log("jailed", pid=pid, reason=reason)\r
\r
    # ================================================================ landing\r
    def _land(\r
        self,\r
        pid: int,\r
        dice_total: int,\r
        rent_multiplier: int = 1,\r
        utility_dice_multiplier: int | None = None,\r
    ) -> None:\r
        s = self.state\r
        idx = s.players[pid].position\r
        space = self.board[idx]\r
        self.landings[idx] += 1\r
\r
        if space.ownable:\r
            owner = s.owner[idx]\r
            if owner is None:\r
                self._offer_purchase(pid, idx)\r
            elif owner != pid and not s.mortgaged[idx]:\r
                rent = compute_rent(s, idx, dice_total, rent_multiplier, utility_dice_multiplier)\r
                paid = self.pay(pid, rent, owner, "rent")\r
                self.stats[pid].rent_paid += paid\r
                self.stats[owner].rent_collected += paid\r
                self.rent_by_space[idx] += paid\r
                self._log("rent", pid=pid, owner=owner, space=idx, amount=rent, paid=paid)\r
        elif space.type == SpaceType.TAX:\r
            paid = self.pay(pid, space.tax_amount, None, "tax")\r
            self.stats[pid].taxes_paid += paid\r
            self._log("tax", pid=pid, space=idx, amount=space.tax_amount, paid=paid)\r
        elif space.type == SpaceType.GO_TO_JAIL:\r
            self._jail(pid, reason="go_to_jail_space")\r
        elif space.type == SpaceType.CHANCE:\r
            self._apply_card(pid, self.chance.draw(), self.chance, dice_total)\r
        elif space.type == SpaceType.COMMUNITY_CHEST:\r
            self._apply_card(pid, self.community_chest.draw(), self.community_chest, dice_total)\r
        # GO, JAIL (just visiting), FREE_PARKING: nothing happens.\r
\r
    def _offer_purchase(self, pid: int, idx: int) -> None:\r
        s = self.state\r
        space = self.board[idx]\r
        p = s.players[pid]\r
        if p.cash >= space.price:\r
            want = bool(self.policies[pid].decide_buy(self, pid, idx))\r
            self._decision(pid, "buy", "yes" if want else "no")\r
        else:\r
            want = False\r
            self._decision(pid, "buy", "unaffordable")\r
        if want:\r
            p.cash -= space.price\r
            self._acquire(pid, idx, "purchase", space.price)\r
        elif self.rules.auctions_enabled:\r
            self._auction(idx)\r
\r
    def _auction(self, idx: int) -> None:\r
        s = self.state\r
        n = len(s.players)\r
        order = [(s.current + k) % n for k in range(n)]\r
        bids: list[tuple[int, int]] = []\r
        for pid in order:\r
            p = s.players[pid]\r
            if p.bankrupt:\r
                continue\r
            b = int(self.policies[pid].auction_max_bid(self, pid, idx))\r
            b = max(0, min(b, p.cash))\r
            bids.append((b, pid))\r
            self._decision(pid, "auction", "bid" if b > 0 else "pass")\r
        # Stable sort keeps seat order (from current player) as the tie-break.\r
        bids.sort(key=lambda t: -t[0])\r
        if not bids or bids[0][0] < 1:\r
            self._log("auction", space=idx, winner=None, price=0)\r
            return\r
        top, winner = bids[0]\r
        second = bids[1][0] if len(bids) > 1 else 0\r
        price = min(top, max(1, second + 1))\r
        s.players[winner].cash -= price\r
        self._acquire(winner, idx, "auction", price)\r
\r
    def _acquire(self, pid: int, idx: int, method: str, price: int) -> None:\r
        self.state.owner[idx] = pid\r
        self.invested_by_space[idx] += price\r
        self._record_acquisition(pid, idx, method)\r
        self._log("acquire", pid=pid, space=idx, method=method, price=price)\r
\r
    # ================================================================== cards\r
    def _apply_card(self, pid: int, card: Card, deck: Deck, dice_total: int) -> None:\r
        s = self.state\r
        p = s.players[pid]\r
        a = card.action\r
        prm = card.params\r
        self._log("card", pid=pid, deck=deck.name, card=card.id, text=card.text)\r
\r
        if a == "advance_to":\r
            if move_to(s, pid, int(prm["target"])):\r
                self._collect_salary(pid)\r
            self._land(pid, dice_total)\r
        elif a == "advance_to_nearest":\r
            kind = SpaceType(prm["kind"])\r
            target = self.board.nearest(p.position, kind)\r
            if move_to(s, pid, target):\r
                self._collect_salary(pid)\r
            if kind == SpaceType.UTILITY:\r
                owner = s.owner[target]\r
                if owner is not None and owner != pid and not s.mortgaged[target]:\r
                    d1, d2 = self._roll()\r
                    self._log("roll", pid=pid, dice=[d1, d2], for_card=True)\r
                    dice_total = d1 + d2\r
                self._land(pid, dice_total, utility_dice_multiplier=int(prm.get("dice_multiplier", 10)))\r
            else:\r
                self._land(pid, dice_total, rent_multiplier=int(prm.get("rent_multiplier", 1)))\r
        elif a == "collect":\r
            p.cash += int(prm["amount"])\r
        elif a == "pay":\r
            self.pay(pid, int(prm["amount"]), None, "card")\r
        elif a == GOOJF:\r
            p.goojf_cards.append(card.id)\r
        elif a == "move_relative":\r
            steps = int(prm["spaces"])\r
            if move_by(s, pid, steps):\r
                self._collect_salary(pid)\r
            self._land(pid, dice_total)\r
        elif a == "go_to_jail":\r
            self._jail(pid, reason="card")\r
        elif a == "repairs":\r
            houses, hotels = s.player_buildings(pid)\r
            cost = houses * int(prm["per_house"]) + hotels * int(prm["per_hotel"])\r
            self.pay(pid, cost, None, "card")\r
        elif a == "pay_each_player":\r
            amount = int(prm["amount"])\r
            for other in list(s.active_players()):\r
                if other.id == pid:\r
                    continue\r
                if p.bankrupt:\r
                    break\r
                self.pay(pid, amount, other.id, "card")\r
        elif a == "collect_from_each_player":\r
            amount = int(prm["amount"])\r
            for other in list(s.active_players()):\r
                if other.id != pid and not p.bankrupt:\r
                    self.pay(other.id, amount, pid, "card")\r
        else:\r
            raise ValueError(f"Unknown card action {a!r}")\r
\r
    # ================================================================= money\r
    def pay(self, pid: int, amount: int, creditor: int | None, reason: str) -> int:\r
        """Pay \`amount\` to \`creditor\` (None = bank), raising funds if needed.\r
        Returns cash actually transferred (less than \`amount\` on bankruptcy)."""\r
        if amount <= 0:\r
            return 0\r
        p = self.state.players[pid]\r
        if p.cash < amount:\r
            self._raise_funds(pid, amount)\r
        if p.cash >= amount:\r
            p.cash -= amount\r
            if creditor is not None:\r
                self.state.players[creditor].cash += amount\r
            return amount\r
        self._log("debt_unpaid", pid=pid, creditor=creditor, amount=amount, reason=reason)\r
        return self._bankrupt(pid, creditor)\r
\r
    def _raise_funds(self, pid: int, target: int) -> None:\r
        s = self.state\r
        p = s.players[pid]\r
        if finance.liquidation_value(s, pid) < target:\r
            return  # cannot be covered; bankruptcy follows\r
        from ..policies.base import default_liquidation\r
\r
        sources = [lambda: self.policies[pid].choose_liquidation(self, pid, target - p.cash),\r
                   lambda: default_liquidation(self, pid)]\r
        for source in sources:\r
            for _ in range(_MAX_POLICY_ROUNDS):\r
                if p.cash >= target:\r
                    return\r
                progress = False\r
                for kind, idx in source():\r
                    if p.cash >= target:\r
                        break\r
                    if kind == "sell" and self._do_sell(pid, idx):\r
                        self._decision(pid, "liquidate", "sell_building")\r
                        progress = True\r
                    elif kind == "mortgage" and self._do_mortgage(pid, idx):\r
                        self._decision(pid, "liquidate", "mortgage")\r
                        progress = True\r
                if not progress:\r
                    break\r
\r
    def _bankrupt(self, pid: int, creditor: int | None) -> int:\r
        s = self.state\r
        p = s.players[pid]\r
        while sellable := [i for i in s.properties_of(pid) if bld.can_sell_building(s, pid, i)[0]]:\r
            for i in sellable:\r
                self._do_sell(pid, i)\r
        cash = max(p.cash, 0)\r
        p.cash = 0\r
        props = s.properties_of(pid)\r
        cards = list(p.goojf_cards)\r
        p.goojf_cards.clear()\r
        p.bankrupt = True\r
        p.in_jail = False\r
        self.bankruptcy_order.append(pid)\r
        self.stats[pid].bankrupt_turn = s.turn\r
        self._log("bankrupt", pid=pid, creditor=creditor)\r
\r
        if creditor is not None:\r
            c = s.players[creditor]\r
            c.cash += cash\r
            c.goojf_cards.extend(cards)\r
            interest = 0\r
            for i in props:\r
                s.owner[i] = creditor\r
                self.stats[creditor].properties_acquired.append(i)\r
                self.stats[creditor].acquisition_methods["bankruptcy"] += 1\r
                if s.mortgaged[i]:\r
                    interest += (self.board[i].mortgage_value + 9) // 10\r
            # Receiving mortgaged property requires paying 10% interest now.\r
            self.pay(creditor, interest, None, "mortgage_interest")\r
        else:\r
            for card_id in cards:\r
                (self.chance if self.chance.owns(card_id) else self.community_chest).return_card(card_id)\r
            for i in props:\r
                s.owner[i] = None\r
                s.mortgaged[i] = False\r
            self._check_winner()\r
            if self.rules.auctions_enabled and not self.finished:\r
                for i in props:\r
                    self._auction(i)\r
        self._check_winner()\r
        return cash\r
\r
    def _check_winner(self) -> None:\r
        active = self.state.active_players()\r
        if len(active) <= 1 and not self.finished:\r
            self.finished = True\r
            self.winner = active[0].id if active else None\r
            self._log("game_over", winner=self.winner)\r
\r
    # ============================================================ management\r
    def _management(self, pid: int) -> None:\r
        """End-of-turn phase: the player may act, trade, unmortgage and build."""\r
        p = self.state.players[pid]\r
        policy = self.policies[pid]\r
        self._action_phase(pid, "post_roll")\r
        if p.bankrupt or self.finished:\r
            return\r
        if self.rules.trading_enabled:\r
            for trade in policy.propose_trades(self, pid):\r
                if self.trade_offers_this_turn >= self.rules.max_trade_offers_per_turn:\r
                    break\r
                if trade.proposer == pid:\r
                    self.propose_trade(trade)\r
        for idx in policy.choose_unmortgages(self, pid):\r
            if self._do_unmortgage(pid, idx):\r
                self._decision(pid, "unmortgage", "yes")\r
        for _ in range(_MAX_POLICY_ROUNDS):\r
            built = False\r
            for idx in policy.choose_builds(self, pid):\r
                if self._do_build(pid, idx):\r
                    self._decision(pid, "build", "yes")\r
                    built = True\r
            if not built:\r
                break\r
\r
    def _action_phase(self, pid: int, phase: str) -> None:\r
        """Let a player take voluntary actions until they roll / end the turn.\r
        AI policies return None immediately; a human seat answers here."""\r
        policy = self.policies[pid]\r
        for _ in range(_MAX_TURN_ACTIONS):\r
            if self.finished or self.state.players[pid].bankrupt:\r
                return\r
            act = policy.turn_action(self, pid, phase)\r
            if not act or act.get("type") in ("roll", "end_turn"):\r
                return\r
            self.apply_action(pid, act)\r
\r
    def apply_action(self, pid: int, act: dict[str, Any]) -> bool:\r
        """Apply one voluntary action if legal. Returns whether it happened."""\r
        kind = act.get("type")\r
        idx = act.get("space")\r
        if kind == "trade":\r
            if (not self.rules.trading_enabled\r
                    or self.trade_offers_this_turn >= self.rules.max_trade_offers_per_turn):\r
                self._log("action_ignored", pid=pid, action=act, reason="no trade offers left this turn")\r
                return False\r
            try:\r
                trade = Trade.from_dict({**act, "proposer": pid})\r
            except (KeyError, TypeError, ValueError):\r
                return False\r
            self.propose_trade(trade)\r
            return True\r
        if not isinstance(idx, int) or not 0 <= idx < len(self.board):\r
            return False\r
        handler = {\r
            "build": self._do_build,\r
            "sell_building": self._do_sell,\r
            "mortgage": self._do_mortgage,\r
            "unmortgage": self._do_unmortgage,\r
        }.get(str(kind))\r
        done = bool(handler and handler(pid, idx))\r
        if done:\r
            self._decision(pid, str(kind), "yes")\r
        else:\r
            self._log("action_ignored", pid=pid, action=act, reason="not allowed now")\r
        return done\r
\r
    # -------------------------------------------------- property operations\r
    def _do_build(self, pid: int, idx: int) -> bool:\r
        s = self.state\r
        if not bld.can_build(s, pid, idx)[0]:\r
            return False\r
        bld.build(s, pid, idx)\r
        self.invested_by_space[idx] += self.board[idx].house_cost\r
        if s.buildings[idx] == bld.HOTEL:\r
            self.stats[pid].hotels_built += 1\r
        else:\r
            self.stats[pid].houses_built += 1\r
        self._log("build", pid=pid, space=idx, level=s.buildings[idx])\r
        return True\r
\r
    def _do_sell(self, pid: int, idx: int) -> bool:\r
        s = self.state\r
        if not bld.can_sell_building(s, pid, idx)[0]:\r
            return False\r
        before = s.buildings[idx]\r
        self.invested_by_space[idx] -= bld.sell_building(s, pid, idx)\r
        # A hotel broken up during a housing shortage can leave the group uneven.\r
        shortage = before == bld.HOTEL and s.buildings[idx] < 4\r
        if shortage:\r
            self.shortage_groups.add(self.board[idx].group)\r
        self._log("sell_building", pid=pid, space=idx, level=s.buildings[idx], shortage=shortage)\r
        return True\r
\r
    def _do_mortgage(self, pid: int, idx: int) -> bool:\r
        if not finance.can_mortgage(self.state, pid, idx)[0]:\r
            return False\r
        finance.mortgage(self.state, pid, idx)\r
        self._log("mortgage", pid=pid, space=idx)\r
        return True\r
\r
    def _do_unmortgage(self, pid: int, idx: int) -> bool:\r
        if not finance.can_unmortgage(self.state, pid, idx)[0]:\r
            return False\r
        finance.unmortgage(self.state, pid, idx)\r
        self._log("unmortgage", pid=pid, space=idx)\r
        return True\r
\r
    # ================================================================= trades\r
    def propose_trade(self, trade: Trade) -> bool:\r
        """Offer a trade to its recipient; execute it if accepted."""\r
        self.trade_offers_this_turn += 1\r
        ok, why = validate_trade(self.state, trade)\r
        if not ok:\r
            self._log("trade_invalid", **trade.to_dict(), reason=why)\r
            return False\r
        accepted = bool(self.policies[trade.recipient].respond_to_trade(self, trade.recipient, trade))\r
        accepted = accepted and validate_trade(self.state, trade)[0]\r
        self._decision(trade.proposer, "trade_offer", "accepted" if accepted else "rejected")\r
        self._decision(trade.recipient, "trade_reply", "accept" if accepted else "reject")\r
        for pol in (self.policies[trade.proposer], self.policies[trade.recipient]):\r
            pol.on_trade_result(self, trade, accepted)\r
        if accepted:\r
            execute_trade(self.state, trade)\r
            for i in trade.give:\r
                self._record_acquisition(trade.recipient, i, "trade")\r
            for i in trade.get:\r
                self._record_acquisition(trade.proposer, i, "trade")\r
        self._log("trade", **trade.to_dict(), accepted=accepted)\r
        return accepted\r
\r
    def _record_acquisition(self, pid: int, idx: int, method: str) -> None:\r
        self.stats[pid].properties_acquired.append(idx)\r
        self.stats[pid].acquisition_methods[method] += 1\r
\r
    # ================================================================ logging\r
    def _decision(self, pid: int, kind: str, choice: str) -> None:\r
        self.stats[pid].decisions[f"{kind}:{choice}"] += 1\r
\r
    def _snapshot(self) -> dict[str, Any]:\r
        """Full public state, attached to replay events (only when logging)."""\r
        if self.events is None:\r
            return {}\r
        s = self.state\r
        return {\r
            "cash": [p.cash for p in s.players],\r
            "positions": [p.position for p in s.players],\r
            "in_jail": [p.in_jail for p in s.players],\r
            "bankrupt": [p.bankrupt for p in s.players],\r
            "owner": list(s.owner),\r
            "buildings": list(s.buildings),\r
            "mortgaged": list(s.mortgaged),\r
            "goojf": [len(p.goojf_cards) for p in s.players],\r
            "current": s.current,\r
            "turn": s.turn,\r
            "round": s.round,\r
            "dice": list(self.last_dice) if self.last_dice else None,\r
            "bank_houses": s.bank_houses,\r
            "bank_hotels": s.bank_hotels,\r
        }\r
\r
    def _log(self, type_: str, **data: Any) -> None:\r
        if self.events is not None:\r
            self.events.append({"turn": self.state.turn, "type": type_, **data})\r
`,Pe=`"""Jail mechanics (state transitions only; decisions come from policies)."""

from __future__ import annotations

from enum import Enum

from .state import GameState


class JailAction(str, Enum):
    PAY = "pay"
    USE_CARD = "use_card"
    ROLL = "roll"


def send_to_jail(state: GameState, pid: int) -> None:
    p = state.players[pid]
    p.position = state.board.jail_index
    p.in_jail = True
    p.jail_turns = 0


def release(state: GameState, pid: int) -> None:
    p = state.players[pid]
    p.in_jail = False
    p.jail_turns = 0
`,Ce=`"""Player movement and passing Go."""

from __future__ import annotations

from .state import GameState


def move_by(state: GameState, pid: int, steps: int) -> bool:
    """Move forward (or backward if negative). Returns True if Go was passed
    or landed on while moving forward. The caller pays the salary."""
    p = state.players[pid]
    n = len(state.board)
    start = p.position
    p.position = (start + steps) % n
    return steps > 0 and start + steps >= n


def move_to(state: GameState, pid: int, target: int) -> bool:
    """Advance forward to \`target\`. Returns True if Go was passed/landed on."""
    p = state.players[pid]
    passed = target <= p.position  # moving forward wraps (or lands on Go)
    p.position = target
    return passed
`,Re=`"""Rent calculation: a pure function of state."""

from __future__ import annotations

from .board import SpaceType
from .state import GameState


def compute_rent(
    state: GameState,
    idx: int,
    dice_total: int,
    rent_multiplier: int = 1,
    utility_dice_multiplier: int | None = None,
) -> int:
    """Rent owed for landing on \`idx\`.

    rent_multiplier: e.g. 2 for the "nearest railroad" Chance card.
    utility_dice_multiplier: overrides the 4x/10x rule (10 for the Chance card).
    """
    space = state.board[idx]
    owner = state.owner[idx]
    if owner is None or state.mortgaged[idx]:
        return 0

    if space.type == SpaceType.STREET:
        level = state.buildings[idx]
        rent = space.rent[level]
        if level == 0 and state.owns_group(owner, space.group):
            rent *= 2
        return rent * rent_multiplier

    if space.type == SpaceType.RAILROAD:
        count = sum(1 for r in state.board.railroads if state.owner[r] == owner)
        return state.board.railroad_rents[count - 1] * rent_multiplier

    if space.type == SpaceType.UTILITY:
        if utility_dice_multiplier is not None:
            mult = utility_dice_multiplier
        else:
            count = sum(1 for u in state.board.utilities if state.owner[u] == owner)
            mult = state.board.utility_multipliers[count - 1]
        return mult * dice_total * rent_multiplier

    return 0
`,Ge=`"""Rule toggles and limits. Defaults follow the official US rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuleConfig:
    # Declined properties are auctioned among all solvent players.
    auctions_enabled: bool = True
    # Safety cap: a game stops after this many full rounds (recorded as a timeout).
    max_rounds: int = 500
    # Override the board's starting cash (None = use board value).
    starting_cash: int | None = None
    # Maximum attempts to roll doubles in jail before the fine is forced.
    max_jail_turns: int = 3
    # Players may trade properties and cash on their turn.
    trading_enabled: bool = True
    # Trade offers a player may make per turn (stops endless haggling).
    max_trade_offers_per_turn: int = 2

    def to_dict(self) -> dict:
        return asdict(self)
`,Fe=`"""Pure game/player state containers. Rules live elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, field

from .board import Board


@dataclass
class PlayerState:
    id: int
    name: str
    cash: int
    position: int = 0
    in_jail: bool = False
    jail_turns: int = 0  # failed doubles attempts during the current stay
    goojf_cards: list[str] = field(default_factory=list)  # card ids held
    bankrupt: bool = False


@dataclass
class GameState:
    board: Board
    players: list[PlayerState]
    # Per-space ownership arrays (None / False / 0 for non-ownable spaces).
    owner: list[int | None]
    mortgaged: list[bool]
    # 0-4 houses, 5 = hotel.
    buildings: list[int]
    bank_houses: int
    bank_hotels: int
    current: int = 0
    turn: int = 0  # player-turns taken
    round: int = 0  # full rotations completed

    @classmethod
    def new(cls, board: Board, names: list[str], starting_cash: int | None = None) -> "GameState":
        cash = board.starting_cash if starting_cash is None else starting_cash
        n = len(board)
        return cls(
            board=board,
            players=[PlayerState(id=i, name=nm, cash=cash) for i, nm in enumerate(names)],
            owner=[None] * n,
            mortgaged=[False] * n,
            buildings=[0] * n,
            bank_houses=board.house_supply,
            bank_hotels=board.hotel_supply,
        )

    # ---------------------------------------------------------------- queries
    def active_players(self) -> list[PlayerState]:
        return [p for p in self.players if not p.bankrupt]

    def properties_of(self, pid: int) -> list[int]:
        return [i for i, o in enumerate(self.owner) if o == pid]

    def owns_group(self, pid: int, group: str) -> bool:
        return all(self.owner[i] == pid for i in self.board.groups[group])

    def group_of(self, idx: int) -> str | None:
        return self.board[idx].group

    def houses_on_board(self) -> int:
        return sum(b for b in self.buildings if b < 5)

    def hotels_on_board(self) -> int:
        return sum(1 for b in self.buildings if b == 5)

    def player_buildings(self, pid: int) -> tuple[int, int]:
        """(houses, hotels) owned by a player."""
        houses = hotels = 0
        for i in self.properties_of(pid):
            if self.buildings[i] == 5:
                hotels += 1
            else:
                houses += self.buildings[i]
        return houses, hotels

    def monopolies_of(self, pid: int) -> list[str]:
        return [g for g in self.board.groups if self.owns_group(pid, g)]
`,Ie=`"""Outcome tracking for a single game."""\r
\r
from __future__ import annotations\r
\r
from collections import Counter\r
from dataclasses import asdict, dataclass, field\r
\r
\r
@dataclass\r
class PlayerStats:\r
    player_id: int\r
    name: str\r
    policy: str\r
    rent_collected: int = 0\r
    rent_paid: int = 0\r
    taxes_paid: int = 0\r
    salary_collected: int = 0\r
    properties_acquired: list[int] = field(default_factory=list)\r
    acquisition_methods: Counter = field(default_factory=Counter)  # purchase/auction/bankruptcy\r
    houses_built: int = 0\r
    hotels_built: int = 0\r
    times_jailed: int = 0\r
    decisions: Counter = field(default_factory=Counter)  # "type:choice" -> count\r
    final_cash: int = 0\r
    final_net_worth: int = 0\r
    bankrupt_turn: int | None = None\r
\r
    def to_dict(self) -> dict:\r
        d = asdict(self)\r
        d["acquisition_methods"] = dict(self.acquisition_methods)\r
        d["decisions"] = dict(self.decisions)\r
        return d\r
\r
\r
@dataclass\r
class GameResult:\r
    seed: int | None\r
    winner: int | None\r
    timed_out: bool\r
    turns: int\r
    rounds: int\r
    bankruptcy_order: list[int]\r
    net_worth_leader: int\r
    players: list[PlayerStats]\r
    landings: list[int]\r
    end_positions: list[int]\r
    rent_by_space: list[int]\r
    invested_by_space: list[int]\r
    events: list[dict] | None = None\r
\r
    @property\r
    def winner_policy(self) -> str | None:\r
        return None if self.winner is None else self.players[self.winner].policy\r
\r
    def to_dict(self) -> dict:\r
        return {\r
            "seed": self.seed,\r
            "winner": self.winner,\r
            "winner_policy": self.winner_policy,\r
            "timed_out": self.timed_out,\r
            "turns": self.turns,\r
            "rounds": self.rounds,\r
            "bankruptcy_order": self.bankruptcy_order,\r
            "net_worth_leader": self.net_worth_leader,\r
            "players": [p.to_dict() for p in self.players],\r
            "landings": self.landings,\r
            "end_positions": self.end_positions,\r
            "rent_by_space": self.rent_by_space,\r
            "invested_by_space": self.invested_by_space,\r
            "events": self.events,\r
        }\r
`,Oe=`"""Player-to-player trades (rules only; whether to trade is a policy decision).

Official rules followed:
- Unimproved properties, railroads and utilities may be traded; a street
  can't change hands while any lot in its colour group has buildings.
- Mortgaged property may be traded; the receiver must immediately pay the
  bank 10% interest on its mortgage value (it stays mortgaged).
- Cash may be part of a deal. Here it must be paid from cash on hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .board import SpaceType
from .state import GameState


@dataclass(frozen=True)
class Trade:
    proposer: int
    recipient: int
    give: tuple[int, ...] = ()  # proposer -> recipient
    get: tuple[int, ...] = ()  # recipient -> proposer
    cash: int = 0  # > 0: proposer pays recipient; < 0: recipient pays proposer

    def to_dict(self) -> dict[str, Any]:
        return {"proposer": self.proposer, "recipient": self.recipient,
                "give": list(self.give), "get": list(self.get), "cash": self.cash}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trade":
        return cls(int(d["proposer"]), int(d["recipient"]),
                   tuple(int(i) for i in d.get("give", ())), tuple(int(i) for i in d.get("get", ())),
                   int(d.get("cash", 0)))


def interest_on(state: GameState, lots: tuple[int, ...]) -> int:
    """10% (rounded up) of the mortgage value of the mortgaged lots received."""
    return sum((state.board[i].mortgage_value + 9) // 10 for i in lots if state.mortgaged[i])


def tradable(state: GameState, pid: int, idx: int) -> bool:
    space = state.board[idx]
    if state.owner[idx] != pid or not space.ownable:
        return False
    if space.type == SpaceType.STREET:
        return not any(state.buildings[i] for i in state.board.groups[space.group])
    return True


def validate_trade(state: GameState, t: Trade) -> tuple[bool, str]:
    n = len(state.players)
    if not (0 <= t.proposer < n and 0 <= t.recipient < n) or t.proposer == t.recipient:
        return False, "invalid players"
    a, b = state.players[t.proposer], state.players[t.recipient]
    if a.bankrupt or b.bankrupt:
        return False, "player is bankrupt"
    if not t.give and not t.get:
        return False, "no properties change hands"
    lots = t.give + t.get
    if len(set(lots)) != len(lots):
        return False, "duplicate property"
    if not all(0 <= i < len(state.board) for i in lots):
        return False, "invalid space"
    if not all(tradable(state, t.proposer, i) for i in t.give):
        return False, "proposer can't trade one of the offered properties"
    if not all(tradable(state, t.recipient, i) for i in t.get):
        return False, "recipient can't trade one of the requested properties"
    pay_a = max(t.cash, 0) + interest_on(state, t.get)
    pay_b = max(-t.cash, 0) + interest_on(state, t.give)
    if a.cash < pay_a:
        return False, "proposer lacks the cash"
    if b.cash < pay_b:
        return False, "recipient lacks the cash"
    return True, ""


def execute_trade(state: GameState, t: Trade) -> int:
    """Apply a validated trade. Returns total mortgage interest paid to the bank."""
    ok, why = validate_trade(state, t)
    if not ok:
        raise ValueError(f"illegal trade: {why}")
    a, b = state.players[t.proposer], state.players[t.recipient]
    a.cash -= t.cash
    b.cash += t.cash
    ia, ib = interest_on(state, t.get), interest_on(state, t.give)
    a.cash -= ia
    b.cash -= ib
    for i in t.give:
        state.owner[i] = t.recipient
    for i in t.get:
        state.owner[i] = t.proposer
    return ia + ib
`,Le="",qe=`"""Policy interface and shared helpers.

A policy only *answers questions*. The engine validates every answer and
ignores illegal ones, so policies can never break the rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING, Any, ClassVar

from ..engine.board import SpaceType
from ..engine.buildings import can_sell_building
from ..engine.finance import can_mortgage
from ..engine.jail import JailAction

if TYPE_CHECKING:
    from ..engine.game import Game
    from ..engine.trading import Trade

LiquidationAction = tuple[str, int]  # ("sell", idx) | ("mortgage", idx)


@dataclass
class PolicyConfig:
    """Common knobs. Subclasses extend with their own fields."""

    # Cash kept in hand after voluntary spending (buying, building, bidding).
    cash_reserve: int = 0
    # Jail strategy: "leave" (pay/use card ASAP), "stay" (roll for doubles),
    # or "adaptive" (leave while many lots are unowned, stay later).
    jail_strategy: str = "leave"
    # For "adaptive": leave jail while at least this share of lots is unowned.
    adaptive_unowned_threshold: float = 0.3
    # Unmortgage when cash exceeds reserve by this multiple of the payoff.
    unmortgage_buffer: float = 1.0
    # --- trading (see policies/trade_ai.py) ---
    trade_enabled: bool = True
    # Extra value of a completed colour set, as a multiple of the set's price.
    monopoly_premium: float = 1.5
    # How much the opponent's gain from a deal counts against my own.
    opponent_gain_weight: float = 0.5
    # A deal must be worth at least this much (after the weighting above).
    min_trade_gain: int = 25
    # Cash kept after paying cash in a trade.
    trade_cash_reserve: int = 150
    # Turns to wait before re-offering a rejected deal.
    trade_retry_turns: int = 12


class Policy:
    """Base policy. Subclasses override the decision hooks they care about."""

    name: ClassVar[str] = "base"
    description: ClassVar[str] = ""
    config_cls: ClassVar[type[PolicyConfig]] = PolicyConfig

    def __init__(self, config: PolicyConfig | None = None, **overrides: Any):
        cfg = config or self.config_cls()
        if overrides:
            valid = {f.name for f in fields(cfg)}
            unknown = set(overrides) - valid
            if unknown:
                raise ValueError(f"Unknown {self.name} params: {sorted(unknown)}")
            cfg = type(cfg)(**{**asdict(cfg), **overrides})
        self.config = cfg
        self._rejected: dict[tuple, int] = {}  # (opponent, give, get) -> turn rejected

    # -------------------------------------------------------------- hooks
    def turn_action(self, game: "Game", pid: int, phase: str) -> dict[str, Any] | None:
        """Voluntary action before rolling / before ending the turn (human seats).
        AI policies act through choose_builds / propose_trades instead."""
        return None

    def propose_trades(self, game: "Game", pid: int) -> list["Trade"]:
        if not self.config.trade_enabled:
            return []
        from .trade_ai import propose

        turn = game.state.turn
        blocked = {k for k, t in self._rejected.items() if turn - t < self.config.trade_retry_turns}
        return propose(game, pid, self.config, blocked)

    def respond_to_trade(self, game: "Game", pid: int, trade: "Trade") -> bool:
        if not self.config.trade_enabled:
            return False
        from .trade_ai import acceptable

        return acceptable(game, pid, trade, self.config)

    def on_trade_result(self, game: "Game", trade: "Trade", accepted: bool) -> None:
        if not accepted:
            self._rejected[(trade.recipient, trade.give, trade.get)] = game.state.turn

    def decide_buy(self, game: "Game", pid: int, idx: int) -> bool:
        return game.state.players[pid].cash - game.state.board[idx].price >= self.config.cash_reserve

    def auction_max_bid(self, game: "Game", pid: int, idx: int) -> int:
        return 0

    def decide_jail(self, game: "Game", pid: int) -> JailAction:
        p = game.state.players[pid]
        strategy = self.config.jail_strategy
        if strategy == "adaptive":
            unowned = sum(1 for i in game.state.board.ownable_indices if game.state.owner[i] is None)
            share = unowned / len(game.state.board.ownable_indices)
            strategy = "leave" if share >= self.config.adaptive_unowned_threshold else "stay"
        if strategy == "stay":
            return JailAction.ROLL
        if p.goojf_cards:
            return JailAction.USE_CARD
        if p.cash - game.state.board.jail_fine >= self.config.cash_reserve:
            return JailAction.PAY
        return JailAction.ROLL

    def choose_builds(self, game: "Game", pid: int) -> list[int]:
        return []

    def choose_unmortgages(self, game: "Game", pid: int) -> list[int]:
        s = game.state
        out = []
        cash = s.players[pid].cash
        # Monopoly lots first (they block building), then most expensive.
        mortgaged = [i for i in s.properties_of(pid) if s.mortgaged[i]]
        mortgaged.sort(key=lambda i: (not _in_monopoly(game, pid, i), -s.board[i].price))
        for i in mortgaged:
            cost = s.board[i].unmortgage_cost
            if cash - cost * self.config.unmortgage_buffer >= self.config.cash_reserve:
                out.append(i)
                cash -= cost
        return out

    def choose_liquidation(self, game: "Game", pid: int, amount_needed: int) -> list[LiquidationAction]:
        return default_liquidation(game, pid)

    # ------------------------------------------------------------ utils
    def params(self) -> dict[str, Any]:
        return asdict(self.config)


def _in_monopoly(game: "Game", pid: int, idx: int) -> bool:
    g = game.state.board[idx].group
    return g is not None and game.state.owns_group(pid, g)


def default_liquidation(game: "Game", pid: int) -> list[LiquidationAction]:
    """A sensible order that preserves the most earning power:
    1. mortgage lots outside monopolies (lowest price first),
    2. sell buildings (cheapest houses first),
    3. mortgage monopoly lots."""
    s = game.state
    props = s.properties_of(pid)
    actions: list[LiquidationAction] = []
    loose = [i for i in props if not _in_monopoly(game, pid, i) and can_mortgage(s, pid, i)[0]]
    loose.sort(key=lambda i: s.board[i].price)
    actions += [("mortgage", i) for i in loose]
    sellable = [i for i in props if can_sell_building(s, pid, i)[0]]
    sellable.sort(key=lambda i: s.board[i].house_cost)
    actions += [("sell", i) for i in sellable]
    if not actions:
        rest = [i for i in props if can_mortgage(s, pid, i)[0]]
        rest.sort(key=lambda i: s.board[i].price)
        actions += [("mortgage", i) for i in rest]
    return actions


def is_street(game: "Game", idx: int) -> bool:
    return game.state.board[idx].type == SpaceType.STREET
`,De=`"""Baseline: buy whatever is affordable; build evenly when cash allows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..engine.buildings import buildable_lots
from .base import Policy, PolicyConfig

if TYPE_CHECKING:
    from ..engine.game import Game


@dataclass
class BaselineConfig(PolicyConfig):
    # Bid up to face value in auctions (limited by cash_reserve).
    bid_up_to_price: bool = True
    build_enabled: bool = True
    # Build only while cash after building stays at or above this.
    build_reserve: int = 150
    # Cautious trader: weighs the opponent's gain as much as its own.
    monopoly_premium: float = 1.0
    opponent_gain_weight: float = 1.0


def simple_builds(game: "Game", pid: int, reserve: int) -> list[int]:
    """Build one level at a time, cheapest lots first, keeping \`reserve\`."""
    s = game.state
    lots = sorted(buildable_lots(s, pid), key=lambda i: (s.board[i].house_cost, i))
    cash = s.players[pid].cash
    out = []
    for i in lots:
        cost = s.board[i].house_cost
        if cash - cost >= reserve:
            out.append(i)
            cash -= cost
    return out


class BaselinePolicy(Policy):
    name = "baseline"
    description = "Buys any property it can afford (keeping a cash reserve); builds evenly when cash allows."
    config_cls = BaselineConfig
    config: BaselineConfig

    def auction_max_bid(self, game: "Game", pid: int, idx: int) -> int:
        if not self.config.bid_up_to_price:
            return 0
        cash = game.state.players[pid].cash
        return max(0, min(game.state.board[idx].price, cash - self.config.cash_reserve))

    def choose_builds(self, game: "Game", pid: int) -> list[int]:
        if not self.config.build_enabled:
            return []
        return simple_builds(game, pid, self.config.build_reserve)
`,Je=`"""Development policy: prioritises completing monopolies and building.

- Buys anything affordable above \`cash_reserve\`.
- In auctions, pays a premium for lots that complete its own set or block an
  opponent's set.
- Builds toward \`target_level\` on the lots with the best marginal rent per
  dollar (weighted by landing probability), keeping \`build_reserve\`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..analytics.landing import stationary_distribution
from ..engine.buildings import buildable_lots
from .base import Policy, PolicyConfig
from .property_value import blocks_opponent, group_members

if TYPE_CHECKING:
    from ..engine.game import Game


@dataclass
class DevelopmentConfig(PolicyConfig):
    complete_set_premium: float = 2.0
    block_premium: float = 1.5
    base_bid_fraction: float = 1.0
    # Build up to this level (0-5; 5 = hotel).
    target_level: int = 3
    build_reserve: int = 100
    # Values completed sets more highly when trading.
    monopoly_premium: float = 2.0


class DevelopmentPolicy(Policy):
    name = "development"
    description = "Pays premiums to complete or block colour sets and builds on its best-returning lots."
    config_cls = DevelopmentConfig
    config: DevelopmentConfig

    def auction_max_bid(self, game: "Game", pid: int, idx: int) -> int:
        s = game.state
        members = group_members(game, idx)
        price = s.board[idx].price
        mult = self.config.base_bid_fraction
        mine = sum(1 for m in members if s.owner[m] == pid)
        if len(members) > 1 and mine == len(members) - 1:
            mult = max(mult, self.config.complete_set_premium)
        if blocks_opponent(game, pid, idx):
            mult = max(mult, self.config.block_premium)
        cash = s.players[pid].cash - self.config.cash_reserve
        return max(0, int(min(price * mult, cash)))

    def choose_builds(self, game: "Game", pid: int) -> list[int]:
        s = game.state
        pi = stationary_distribution(s.board)
        cash = s.players[pid].cash
        lots = [i for i in buildable_lots(s, pid) if s.buildings[i] < self.config.target_level]

        def score(i: int) -> float:
            sp = s.board[i]
            lvl = s.buildings[i]
            current = sp.rent[0] * 2 if lvl == 0 else sp.rent[lvl]
            return pi[i] * (sp.rent[lvl + 1] - current) / sp.house_cost

        out = []
        for i in sorted(lots, key=score, reverse=True):
            cost = s.board[i].house_cost
            if cash - cost >= self.config.build_reserve:
                out.append(i)
                cash -= cost
        return out
`,Be=`"""A seat played by a person.

The engine is synchronous, so a human seat answers from a queue of recorded
answers. When the queue is empty it raises \`PendingDecision\`, which stops the
game exactly at the question. Games are seeded and deterministic, so replaying
the same seed with the same answers always reaches the same point; the next
answer is appended and the game is replayed again (see simulation/session.py).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

from ..engine.jail import JailAction
from .base import LiquidationAction, Policy

if TYPE_CHECKING:
    from ..engine.game import Game
    from ..engine.trading import Trade


class PendingDecision(Exception):
    def __init__(self, pid: int, question: dict[str, Any]):
        super().__init__(f"player {pid} must answer {question['kind']}")
        self.pid = pid
        self.question = question


class AnswerMismatch(ValueError):
    pass


class AnswerQueue:
    """Answers shared by every human seat, consumed in question order."""

    def __init__(self, answers: list[dict[str, Any]]):
        self.pending = deque(answers)
        self.used = 0


class HumanPolicy(Policy):
    name = "human"
    description = "A person choosing every decision."

    def __init__(self, queue: AnswerQueue):
        super().__init__(trade_enabled=False)
        self.queue = queue

    def _ask(self, pid: int, question: dict[str, Any]) -> dict[str, Any]:
        if not self.queue.pending:
            raise PendingDecision(pid, question)
        answer = self.queue.pending.popleft()
        self.queue.used += 1
        if answer.get("kind") != question["kind"]:
            raise AnswerMismatch(
                f"answer #{self.queue.used} is for {answer.get('kind')!r} but the game asked {question['kind']!r}"
            )
        return answer

    def decide_buy(self, game: "Game", pid: int, idx: int) -> bool:
        return bool(self._ask(pid, {"kind": "buy", "space": idx}).get("value"))

    def auction_max_bid(self, game: "Game", pid: int, idx: int) -> int:
        return int(self._ask(pid, {"kind": "auction", "space": idx}).get("value") or 0)

    def decide_jail(self, game: "Game", pid: int) -> JailAction:
        value = self._ask(pid, {"kind": "jail"}).get("value")
        try:
            return JailAction(value)
        except ValueError:
            return JailAction.ROLL

    def turn_action(self, game: "Game", pid: int, phase: str) -> dict[str, Any] | None:
        return self._ask(pid, {"kind": "turn", "phase": phase}).get("action") or {"type": "roll"}

    def choose_liquidation(self, game: "Game", pid: int, amount_needed: int) -> list[LiquidationAction]:
        act = self._ask(pid, {"kind": "raise_funds", "needed": amount_needed}).get("action") or {}
        kind = {"sell_building": "sell", "mortgage": "mortgage"}.get(act.get("type", ""))
        return [(kind, int(act["space"]))] if kind and isinstance(act.get("space"), int) else []

    def respond_to_trade(self, game: "Game", pid: int, trade: "Trade") -> bool:
        return bool(self._ask(pid, {"kind": "trade_offer", "trade": trade.to_dict()}).get("value"))

    # Humans act through turn_action; never automatically.
    def propose_trades(self, game: "Game", pid: int) -> list["Trade"]:
        return []

    def choose_builds(self, game: "Game", pid: int) -> list[int]:
        return []

    def choose_unmortgages(self, game: "Game", pid: int) -> list[int]:
        return []
`,Ue=`"""Property-value policy: estimates what an acquisition is worth and buys or
bids only when the estimate justifies the price.

value = mortgage value (liquidity floor)
      + expected rent over a horizon (Markov landing odds x opponents x rent)
      + set-progress bonus (convex in the share of the group owned after buying)
      + blocking bonus (if it denies one opponent a near-complete set)

All weights are configurable; none are claimed optimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..analytics.landing import EXPECTED_ROLLS_PER_TURN, stationary_distribution
from ..engine.board import SpaceType
from .base import Policy, PolicyConfig
from .baseline import simple_builds

if TYPE_CHECKING:
    from ..engine.game import Game


@dataclass
class PropertyValueConfig(PolicyConfig):
    horizon_rounds: int = 40
    # Buy if value >= price * min_value_ratio.
    min_value_ratio: float = 1.0
    # Weight on the group's total price, scaled by (owned share)^2.
    set_bonus: float = 1.0
    # Fraction of price added when the purchase blocks an opponent's set.
    block_bonus: float = 0.5
    # Max auction bid as a fraction of estimated value.
    bid_fraction: float = 0.9
    build_enabled: bool = True
    build_reserve: int = 150


def group_members(game: "Game", idx: int) -> tuple[int, ...]:
    b = game.state.board
    t = b[idx].type
    if t == SpaceType.STREET:
        return b.groups[b[idx].group]
    if t == SpaceType.RAILROAD:
        return b.railroads
    if t == SpaceType.UTILITY:
        return b.utilities
    return (idx,)


def blocks_opponent(game: "Game", pid: int, idx: int) -> bool:
    """True if a single opponent owns every other lot in idx's group."""
    s = game.state
    members = group_members(game, idx)
    others = [s.owner[m] for m in members if m != idx and s.owner[m] not in (None, pid)]
    return bool(others) and len(set(others)) == 1 and len(others) == len(members) - 1


def estimate_value(game: "Game", pid: int, idx: int, cfg: PropertyValueConfig) -> float:
    s = game.state
    b = s.board
    space = b[idx]
    members = group_members(game, idx)
    owned_after = sum(1 for m in members if s.owner[m] == pid or m == idx)
    complete = owned_after == len(members)

    if space.type == SpaceType.STREET:
        rent = space.rent[0] * (2 if complete else 1)
    elif space.type == SpaceType.RAILROAD:
        rent = b.railroad_rents[owned_after - 1]
    else:
        rent = b.utility_multipliers[owned_after - 1] * 7  # mean of 2d6

    opponents = len(s.active_players()) - 1
    p_land = float(stationary_distribution(b)[idx]) * EXPECTED_ROLLS_PER_TURN
    income = cfg.horizon_rounds * opponents * p_land * rent

    group_value = sum(b[m].price for m in members)
    set_progress = cfg.set_bonus * group_value * (owned_after / len(members)) ** 2
    block = cfg.block_bonus * space.price if blocks_opponent(game, pid, idx) else 0.0

    return space.mortgage_value + income + set_progress + block


class PropertyValuePolicy(Policy):
    name = "property_value"
    description = (
        "Buys/bids only when an estimated value (landing odds, rent, set progress, blocking) "
        "justifies the price."
    )
    config_cls = PropertyValueConfig
    config: PropertyValueConfig

    def decide_buy(self, game: "Game", pid: int, idx: int) -> bool:
        price = game.state.board[idx].price
        if game.state.players[pid].cash - price < self.config.cash_reserve:
            return False
        return estimate_value(game, pid, idx, self.config) >= price * self.config.min_value_ratio

    def auction_max_bid(self, game: "Game", pid: int, idx: int) -> int:
        cash = game.state.players[pid].cash - self.config.cash_reserve
        value = estimate_value(game, pid, idx, self.config) * self.config.bid_fraction
        return max(0, int(min(value, cash)))

    def choose_builds(self, game: "Game", pid: int) -> list[int]:
        if not self.config.build_enabled:
            return []
        return simple_builds(game, pid, self.config.build_reserve)
`,Me=`"""Policy registry: name -> class, plus a JSON-friendly description."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .base import Policy
from .baseline import BaselinePolicy
from .development import DevelopmentPolicy
from .property_value import PropertyValuePolicy

POLICIES: dict[str, type[Policy]] = {
    cls.name: cls for cls in (BaselinePolicy, PropertyValuePolicy, DevelopmentPolicy)
}


def create_policy(name: str, params: dict[str, Any] | None = None) -> Policy:
    try:
        cls = POLICIES[name]
    except KeyError:
        raise ValueError(f"Unknown policy {name!r}; choose from {sorted(POLICIES)}") from None
    return cls(**(params or {}))


def describe_policies() -> list[dict[str, Any]]:
    return [
        {"name": cls.name, "description": cls.description, "defaults": asdict(cls.config_cls())}
        for cls in POLICIES.values()
    ]
`,$e=`"""Trade evaluation and proposal shared by the AI policies.

Valuation (all weights configurable, none claimed optimal):
  holdings value = for each group, the market value of the lots held
                   (price, or price minus payoff if mortgaged)
                 + monopoly_premium * group price when the colour set is complete
                 + a small set-progress bonus for partial sets
                 + a synergy bonus for multiple railroads / both utilities

A trade's gain for a player = change in holdings value + cash received
- mortgage interest owed. A player accepts when

  my_gain - opponent_gain_weight * their_gain >= min_trade_gain

i.e. a deal must help me, and must not help my opponent much more than me.
Proposals target completing a colour set, offering a swap that completes one
of the opponent's sets and/or the least cash that the proposer expects the
opponent to accept (modelled with the proposer's own weights).
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

from ..engine.trading import Trade, interest_on, tradable, validate_trade

if TYPE_CHECKING:
    from ..engine.game import Game
    from .base import PolicyConfig

_SET_PROGRESS = 0.1
_RAILROAD_SYNERGY = 50  # per pair of railroads held
_UTILITY_SYNERGY = 50


def _group_key(game: "Game", idx: int) -> str:
    b = game.state.board
    space = b[idx]
    if space.group:
        return space.group
    return "railroads" if idx in b.railroads else "utilities" if idx in b.utilities else f"#{idx}"


def _group_lots(game: "Game", key: str) -> tuple[int, ...]:
    b = game.state.board
    if key == "railroads":
        return b.railroads
    if key == "utilities":
        return b.utilities
    return b.groups.get(key, (int(key[1:]),) if key.startswith("#") else ())


def _group_value(game: "Game", key: str, owned: set[int], cfg: "PolicyConfig") -> float:
    s = game.state
    b = s.board
    lots = _group_lots(game, key)
    mine = [i for i in lots if i in owned]
    if not mine:
        return 0.0
    value = sum(b[i].price - b[i].unmortgage_cost if s.mortgaged[i] else b[i].price for i in mine)
    if key == "railroads":
        return value + _RAILROAD_SYNERGY * len(mine) * (len(mine) - 1) / 2
    if key == "utilities":
        return value + (_UTILITY_SYNERGY if len(mine) == len(lots) > 1 else 0)
    group_price = sum(b[i].price for i in lots)
    if len(mine) == len(lots):
        return value + cfg.monopoly_premium * group_price
    return value + _SET_PROGRESS * group_price * len(mine) / len(lots)


def holdings_value(game: "Game", owned: set[int], cfg: "PolicyConfig") -> float:
    keys = {_group_key(game, i) for i in owned}
    return sum(_group_value(game, k, owned, cfg) for k in keys)


def trade_gain(game: "Game", pid: int, t: Trade, cfg: "PolicyConfig") -> float:
    """Value change for \`pid\` (either party) if trade \`t\` happens. Only the
    groups the trade touches can change value, so only those are computed."""
    owned = set(game.state.properties_of(pid))
    if pid == t.proposer:
        out, inn, cash = set(t.give), set(t.get), -t.cash
    elif pid == t.recipient:
        out, inn, cash = set(t.get), set(t.give), t.cash
    else:
        return 0.0
    after = (owned - out) | inn
    interest = interest_on(game.state, tuple(sorted(inn)))
    delta = sum(
        _group_value(game, k, after, cfg) - _group_value(game, k, owned, cfg)
        for k in {_group_key(game, i) for i in out | inn}
    )
    return delta + cash - interest


def acceptable(game: "Game", pid: int, t: Trade, cfg: "PolicyConfig") -> bool:
    other = t.recipient if pid == t.proposer else t.proposer
    mine = trade_gain(game, pid, t, cfg)
    theirs = trade_gain(game, other, t, cfg)
    if mine - cfg.opponent_gain_weight * theirs < cfg.min_trade_gain:
        return False
    # Keep a cash cushion after paying cash in a deal.
    pays = t.cash if pid == t.proposer else -t.cash
    return pays <= 0 or game.state.players[pid].cash - pays >= cfg.trade_cash_reserve


def _completion_sets(game: "Game", me: int, other: int) -> list[tuple[int, ...]]:
    """Lots \`other\` holds that would complete one of my colour sets (all
    other lots in the set are mine, and none of it is built on)."""
    s = game.state
    out = []
    for lots in s.board.groups.values():
        theirs = tuple(i for i in lots if s.owner[i] == other)
        mine = [i for i in lots if s.owner[i] == me]
        if theirs and mine and len(theirs) + len(mine) == len(lots):
            if all(tradable(s, other, i) for i in theirs) and all(tradable(s, me, i) for i in mine):
                out.append(theirs)
    return out


def propose(game: "Game", pid: int, cfg: "PolicyConfig", blocked: set[tuple]) -> list[Trade]:
    """Candidate offers from \`pid\`, best first (by the proposer's own score)."""
    s = game.state
    me = s.players[pid]
    w = cfg.opponent_gain_weight
    scored: list[tuple[float, Trade]] = []
    for opp in s.active_players():
        if opp.id == pid:
            continue
        wants = _completion_sets(game, pid, opp.id)
        if not wants:
            continue
        swaps = [()] + _completion_sets(game, opp.id, pid)  # lots of mine completing their sets
        for get in wants:
            for give in swaps:
                base = Trade(pid, opp.id, give=give, get=get, cash=0)
                g_me = trade_gain(game, pid, base, cfg)
                g_them = trade_gain(game, opp.id, base, cfg)
                # Least cash c (rounded up to $10) that makes the deal acceptable to them:
                # (g_them + c) - w (g_me - c) >= min  =>  c >= (min - g_them + w g_me) / (1 + w)
                c = ceil((cfg.min_trade_gain - g_them + w * g_me) / (1 + w) / 10) * 10
                if c > 0 and me.cash - c - interest_on(s, get) < cfg.trade_cash_reserve:
                    continue
                if c < 0 and opp.cash + c - interest_on(s, give) < 0:
                    c = -max(0, (opp.cash - interest_on(s, give)) // 10 * 10)
                trade = Trade(pid, opp.id, give=give, get=get, cash=c)
                key = (opp.id, give, get)
                if key in blocked or not validate_trade(s, trade)[0]:
                    continue
                my_score = (g_me - c) - w * (g_them + c)
                if my_score >= cfg.min_trade_gain:
                    scored.append((my_score, trade))
    scored.sort(key=lambda st: -st[0])
    return [t for _, t in scored]
`,He="",ze=`"""Headless experiment runner.\r
\r
Reproducibility: per-game seeds are derived from the experiment seed with\r
\`numpy.random.SeedSequence\`, so game k always gets the same seed regardless\r
of \`n_games\` ordering or the number of worker processes. Any single game can\r
be replayed with \`run_single_game(..., seed=result.seed)\`.\r
"""\r
\r
from __future__ import annotations\r
\r
from collections import Counter\r
from dataclasses import asdict, dataclass, field\r
from typing import Any\r
\r
import numpy as np\r
\r
from ..engine.board import Board, standard_board\r
from ..engine.game import Game\r
from ..engine.rules import RuleConfig\r
from ..engine.stats import GameResult\r
from ..policies.registry import create_policy\r
\r
\r
@dataclass\r
class PolicySpec:\r
    name: str\r
    params: dict[str, Any] = field(default_factory=dict)\r
    label: str | None = None\r
\r
\r
@dataclass\r
class ExperimentConfig:\r
    players: list[PolicySpec]\r
    n_games: int = 1000\r
    seed: int = 0\r
    rules: RuleConfig = field(default_factory=RuleConfig)\r
    # Rotate seat order each game to cancel first-mover advantage.\r
    rotate_seats: bool = True\r
    workers: int = 1\r
    keep_games: bool = False\r
\r
\r
def resolve_labels(specs: list[PolicySpec]) -> list[str]:\r
    labels = [s.label or s.name for s in specs]\r
    counts = Counter(labels)\r
    seen: Counter = Counter()\r
    out = []\r
    for lb in labels:\r
        if counts[lb] > 1:\r
            seen[lb] += 1\r
            out.append(f"{lb}#{seen[lb]}")\r
        else:\r
            out.append(lb)\r
    return out\r
\r
\r
def game_seeds(seed: int, n_games: int) -> list[int]:\r
    return [int(x) for x in np.random.SeedSequence(seed).generate_state(n_games, dtype=np.uint64)]\r
\r
\r
def seating(n_players: int, game_index: int, rotate: bool) -> list[int]:\r
    """Spec indices in seat order for a given game."""\r
    off = game_index % n_players if rotate else 0\r
    return [(off + k) % n_players for k in range(n_players)]\r
\r
\r
def run_single_game(\r
    specs: list[PolicySpec],\r
    seed: int,\r
    rules: RuleConfig | None = None,\r
    seat_order: list[int] | None = None,\r
    board: Board | None = None,\r
    record_events: bool = False,\r
) -> GameResult:\r
    labels = resolve_labels(specs)\r
    order = seat_order or list(range(len(specs)))\r
    policies = [create_policy(specs[i].name, specs[i].params) for i in order]\r
    names = [f"{labels[i]}" for i in order]\r
    game = Game(policies, board=board, rules=rules, seed=seed, names=names, record_events=record_events)\r
    for seat, i in enumerate(order):\r
        game.stats[seat].policy = labels[i]\r
    result = game.run()\r
    result.seed = seed\r
    return result\r
\r
\r
def _run_chunk(args: tuple) -> list[GameResult]:\r
    specs, rules, jobs = args\r
    return [run_single_game(specs, seed, rules, order) for seed, order in jobs]\r
\r
\r
def run_experiment(config: ExperimentConfig) -> "ExperimentResult":\r
    if len(config.players) < 2:\r
        raise ValueError("Need at least 2 players")\r
    n = len(config.players)\r
    seeds = game_seeds(config.seed, config.n_games)\r
    jobs = [(seeds[k], seating(n, k, config.rotate_seats)) for k in range(config.n_games)]\r
\r
    if config.workers <= 1:\r
        games = [run_single_game(config.players, s, config.rules, o) for s, o in jobs]\r
    else:\r
        size = max(1, len(jobs) // (config.workers * 4))\r
        chunks = [jobs[i : i + size] for i in range(0, len(jobs), size)]\r
        # Imported here: browsers (Pyodide) have no multiprocessing, and they only run workers=1.\r
        from concurrent.futures import ProcessPoolExecutor\r
\r
        with ProcessPoolExecutor(max_workers=config.workers) as ex:\r
            games = [g for part in ex.map(_run_chunk, [(config.players, config.rules, c) for c in chunks]) for g in part]\r
\r
    summary = summarize(games, resolve_labels(config.players), standard_board())\r
    summary["config"] = {\r
        "players": [asdict(p) for p in config.players],\r
        "n_games": config.n_games,\r
        "seed": config.seed,\r
        "rules": config.rules.to_dict(),\r
        "rotate_seats": config.rotate_seats,\r
    }\r
    return ExperimentResult(summary=summary, games=games if config.keep_games else [])\r
\r
\r
@dataclass\r
class ExperimentResult:\r
    summary: dict[str, Any]\r
    games: list[GameResult]\r
\r
\r
# ---------------------------------------------------------------- statistics\r
def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:\r
    if trials == 0:\r
        return (0.0, 0.0)\r
    p = successes / trials\r
    denom = 1 + z * z / trials\r
    centre = (p + z * z / (2 * trials)) / denom\r
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom\r
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))\r
\r
\r
def _describe(values: np.ndarray) -> dict[str, float]:\r
    if values.size == 0:\r
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}\r
    return {\r
        "mean": float(values.mean()),\r
        "median": float(np.median(values)),\r
        "p10": float(np.percentile(values, 10)),\r
        "p90": float(np.percentile(values, 90)),\r
        "std": float(values.std()),\r
    }\r
\r
\r
def summarize(games: list[GameResult], labels: list[str], board: Board) -> dict[str, Any]:\r
    n_games = len(games)\r
    turns = np.array([g.turns for g in games], dtype=float)\r
    rounds = np.array([g.rounds for g in games], dtype=float)\r
    finished = np.array([not g.timed_out for g in games], dtype=bool)\r
\r
    per: dict[str, dict[str, Any]] = {}\r
    for lb in labels:\r
        rows = [(g, ps) for g in games for ps in g.players if ps.policy == lb]\r
        seats = len(rows)\r
        wins = sum(1 for g, ps in rows if g.winner == ps.player_id)\r
        leads = sum(1 for g, ps in rows if g.timed_out and g.net_worth_leader == ps.player_id)\r
        bankrupt = [g.bankruptcy_order.index(ps.player_id) + 1 for g, ps in rows if ps.player_id in g.bankruptcy_order]\r
        decisions: Counter = Counter()\r
        for _, ps in rows:\r
            decisions.update(ps.decisions)\r
\r
        def arr(attr: str) -> np.ndarray:\r
            return np.array([getattr(ps, attr) for _, ps in rows], dtype=float)\r
\r
        lo, hi = wilson_interval(wins, seats)\r
        timeouts_seen = sum(1 for g, _ in rows if g.timed_out)\r
        per[lb] = {\r
            "label": lb,\r
            "seats": seats,\r
            "wins": wins,\r
            "win_rate": wins / seats if seats else 0.0,\r
            "win_rate_ci95": [lo, hi],\r
            "timeout_leader_rate": leads / timeouts_seen if timeouts_seen else None,\r
            "bankruptcy_rate": len(bankrupt) / seats if seats else 0.0,\r
            "mean_elimination_position": float(np.mean(bankrupt)) if bankrupt else None,\r
            "final_net_worth": _describe(arr("final_net_worth")),\r
            "final_cash": _describe(arr("final_cash")),\r
            "rent_collected_mean": float(arr("rent_collected").mean()) if seats else 0.0,\r
            "rent_paid_mean": float(arr("rent_paid").mean()) if seats else 0.0,\r
            "properties_acquired_mean": float(np.mean([len(ps.properties_acquired) for _, ps in rows])) if seats else 0.0,\r
            "houses_built_mean": float(arr("houses_built").mean()) if seats else 0.0,\r
            "hotels_built_mean": float(arr("hotels_built").mean()) if seats else 0.0,\r
            "decisions": dict(sorted(decisions.items())),\r
        }\r
\r
    landings = np.array([g.landings for g in games], dtype=float) if games else np.zeros((0, len(board)))\r
    ends = np.array([g.end_positions for g in games], dtype=float) if games else np.zeros((0, len(board)))\r
    rent = np.array([g.rent_by_space for g in games], dtype=float) if games else np.zeros((0, len(board)))\r
    invested = np.array([g.invested_by_space for g in games], dtype=float) if games else np.zeros((0, len(board)))\r
    total_land = landings.sum()\r
    total_end = ends.sum()\r
    acq: Counter = Counter()\r
    for g in games:\r
        for ps in g.players:\r
            for idx in ps.properties_acquired:\r
                acq[(idx, ps.policy)] += 1\r
    spaces = []\r
    for s in board.spaces:\r
        spaces.append({\r
            "index": s.index,\r
            "name": s.name,\r
            "type": s.type.value,\r
            "group": s.group,\r
            # Arrivals, including Chance/Community Chest squares a card then moves you from.\r
            "landing_share": float(landings[:, s.index].sum() / total_land) if total_land else 0.0,\r
            # Where each roll's movement ends: comparable with the Markov model.\r
            "end_share": float(ends[:, s.index].sum() / total_end) if total_end else 0.0,\r
            "rent_per_game": float(rent[:, s.index].mean()) if n_games else 0.0,\r
            "invested_per_game": float(invested[:, s.index].mean()) if n_games else 0.0,\r
            "acquired_by": {lb: acq[(s.index, lb)] for lb in labels if acq[(s.index, lb)]},\r
        })\r
\r
    # Colour groups (plus railroads/utilities): rent earned per dollar sunk into them\r
    # (purchase/auction prices + buildings - building refunds), pooled over all games.\r
    group_members = dict(board.groups)\r
    group_members["railroads"] = board.railroads\r
    group_members["utilities"] = board.utilities\r
    groups = []\r
    for g, ix in group_members.items():\r
        r = float(rent[:, list(ix)].sum()) if n_games else 0.0\r
        inv = float(invested[:, list(ix)].sum()) if n_games else 0.0\r
        groups.append({\r
            "group": g,\r
            "rent_per_game": r / n_games if n_games else 0.0,\r
            "invested_per_game": inv / n_games if n_games else 0.0,\r
            "rent_per_dollar": r / inv if inv > 0 else None,\r
        })\r
\r
    return {\r
        "n_games": n_games,\r
        "groups": groups,\r
        "finished_games": int(finished.sum()),\r
        "timeouts": int((~finished).sum()),\r
        "timeout_rate": float((~finished).mean()) if n_games else 0.0,\r
        "game_length_turns": _describe(turns[finished]),\r
        "game_length_rounds": _describe(rounds[finished]),\r
        "policies": per,\r
        "spaces": spaces,\r
    }\r
`,Ve=`"""Interactive games with human seats, by deterministic replay.\r
\r
A session is just (seats, seed, rules, answers). \`play()\` replays the game\r
from the start, feeding the answers to the human seats, until it finishes or\r
a human must decide something. Nothing is stored server-side.\r
"""\r
\r
from __future__ import annotations\r
\r
from dataclasses import dataclass\r
from typing import Any\r
\r
from ..engine import buildings as bld\r
from ..engine import finance\r
from ..engine.board import Board\r
from ..engine.game import Game\r
from ..engine.rules import RuleConfig\r
from ..engine.trading import Trade, tradable, validate_trade\r
from ..policies.human import AnswerQueue, HumanPolicy, PendingDecision\r
from ..policies.registry import create_policy\r
from .runner import PolicySpec, resolve_labels\r
\r
HUMAN = "human"\r
\r
\r
@dataclass\r
class PlayResult:\r
    game: Game\r
    pending: dict[str, Any] | None  # includes "pid" and the question\r
\r
\r
def play(specs: list[PolicySpec], seed: int, rules: RuleConfig, answers: list[dict[str, Any]],\r
         board: Board | None = None) -> PlayResult:\r
    if not any(s.name == HUMAN for s in specs):\r
        raise ValueError("A played game needs at least one human seat")\r
    queue = AnswerQueue(answers)\r
    labels = resolve_labels([PolicySpec(s.name, s.params, s.label or ("You" if s.name == HUMAN else None))\r
                             for s in specs])\r
    policies = [HumanPolicy(queue) if s.name == HUMAN else create_policy(s.name, s.params) for s in specs]\r
    game = Game(policies, board=board, rules=rules, seed=seed, names=labels, record_events=True)\r
    pending = None\r
    try:\r
        while not game.finished:\r
            game.play_turn()\r
    except PendingDecision as p:\r
        pending = {"pid": p.pid, **p.question}\r
    if queue.pending:\r
        raise ValueError(f"{len(queue.pending)} answer(s) were never asked for")\r
    return PlayResult(game, pending)\r
\r
\r
# ------------------------------------------------------------------ options\r
def legal_options(game: Game, pending: dict[str, Any]) -> dict[str, Any]:\r
    """What the human may do at this question (drives the UI and validation)."""\r
    s = game.state\r
    pid = pending["pid"]\r
    p = s.players[pid]\r
    mine = s.properties_of(pid)\r
    kind = pending["kind"]\r
    if kind == "turn":\r
        offers_left = (game.rules.max_trade_offers_per_turn - game.trade_offers_this_turn\r
                       if game.rules.trading_enabled else 0)\r
        return {\r
            "build": [i for i in mine if bld.can_build(s, pid, i)[0]],\r
            "sell_building": [i for i in mine if bld.can_sell_building(s, pid, i)[0]],\r
            "mortgage": [i for i in mine if finance.can_mortgage(s, pid, i)[0]],\r
            "unmortgage": [i for i in mine if finance.can_unmortgage(s, pid, i)[0]],\r
            "trade_offers_left": max(0, offers_left),\r
            "tradable": {str(o.id): [i for i in s.properties_of(o.id) if tradable(s, o.id, i)]\r
                         for o in s.active_players()},\r
            "finish": "roll" if pending["phase"] == "pre_roll" else "end_turn",\r
        }\r
    if kind == "buy":\r
        return {"price": s.board[pending["space"]].price, "cash": p.cash}\r
    if kind == "auction":\r
        return {"price": s.board[pending["space"]].price, "max_bid": p.cash}\r
    if kind == "jail":\r
        opts = ["roll"]\r
        if finance.liquidation_value(s, pid) >= s.board.jail_fine:\r
            opts.insert(0, "pay")\r
        if p.goojf_cards:\r
            opts.insert(0, "use_card")\r
        return {"options": opts, "fine": s.board.jail_fine, "attempt": p.jail_turns + 1,\r
                "max_attempts": game.rules.max_jail_turns}\r
    if kind == "raise_funds":\r
        return {\r
            "needed": pending["needed"],\r
            "sell_building": [i for i in mine if bld.can_sell_building(s, pid, i)[0]],\r
            "mortgage": [i for i in mine if finance.can_mortgage(s, pid, i)[0]],\r
        }\r
    if kind == "trade_offer":\r
        return {}\r
    return {}\r
\r
\r
def _check(ok: bool, reason: str) -> tuple[bool, str]:\r
    return (True, "") if ok else (False, reason)\r
\r
\r
def validate_answer(game: Game, pending: dict[str, Any], answer: dict[str, Any]) -> tuple[bool, str]:\r
    if answer.get("kind") != pending["kind"]:\r
        return False, f"expected an answer of kind {pending['kind']!r}"\r
    opts = legal_options(game, pending)\r
    kind = pending["kind"]\r
    if kind in ("buy", "trade_offer"):\r
        return _check(isinstance(answer.get("value"), bool), "value must be true or false")\r
    if kind == "auction":\r
        v = answer.get("value")\r
        ok = isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= opts["max_bid"]\r
        return _check(ok, f"bid must be a whole number between 0 and {opts['max_bid']}")\r
    if kind == "jail":\r
        return _check(answer.get("value") in opts["options"], f"choose one of {opts['options']}")\r
    act = answer.get("action") or {}\r
    t = act.get("type")\r
    if kind == "raise_funds":\r
        if t in ("sell_building", "mortgage") and act.get("space") in opts[t]:\r
            return True, ""\r
        return False, "choose a legal building sale or mortgage"\r
    if kind == "turn":\r
        if t == opts["finish"]:\r
            return True, ""\r
        if t in ("build", "sell_building", "mortgage", "unmortgage"):\r
            return _check(act.get("space") in opts[t], f"{t} is not allowed on that space now")\r
        if t == "trade":\r
            if opts["trade_offers_left"] <= 0:\r
                return False, "no trade offers left this turn"\r
            try:\r
                trade = Trade.from_dict({**act, "proposer": pending["pid"]})\r
            except (KeyError, TypeError, ValueError):\r
                return False, "malformed trade"\r
            return validate_trade(game.state, trade)\r
        return False, f"unknown action {t!r}"\r
    return False, "unknown question"\r
\r
\r
def describe(result: PlayResult, since: int = 0) -> dict[str, Any]:\r
    """JSON view of a session for the UI."""\r
    game = result.game\r
    s = game.state\r
    pending = None\r
    if result.pending:\r
        pending = {**result.pending, "options": legal_options(game, result.pending)}\r
    events = game.events or []\r
    out: dict[str, Any] = {\r
        "status": "finished" if game.finished else "pending",\r
        "pending": pending,\r
        "state": {\r
            **game._snapshot(),\r
            "names": [p.name for p in s.players],\r
            "policies": [pol.name for pol in game.policies],\r
            "net_worth": [finance.net_worth(s, p.id) for p in s.players],\r
        },\r
        "events": events[since:],\r
        "event_count": len(events),\r
    }\r
    if game.finished:\r
        out["result"] = game.result().to_dict() | {"events": None}\r
    return out\r
`,Ye=Object.defineProperty,d=(e,n)=>Ye(e,"name",{value:n,configurable:!0}),U=(e=>typeof require<"u"?require:typeof Proxy<"u"?new Proxy(e,{get:(n,r)=>(typeof require<"u"?require:n)[r]}):e)(function(e){if(typeof require<"u")return require.apply(this,arguments);throw new Error('Dynamic require of "'+e+'" is not supported')});function M(e){return!isNaN(parseFloat(e))&&isFinite(e)}d(M,"_isNumber");function g(e){return e.charAt(0).toUpperCase()+e.substring(1)}d(g,"_capitalize");function P(e){return function(){return this[e]}}d(P,"_getter");var x=["isConstructor","isEval","isNative","isToplevel"],T=["columnNumber","lineNumber"],A=["fileName","functionName","source"],We=["args"],Ke=["evalOrigin"],E=x.concat(T,A,We,Ke);function _(e){if(e)for(var n=0;n<E.length;n++)e[E[n]]!==void 0&&this["set"+g(E[n])](e[E[n]])}d(_,"StackFrame");_.prototype={getArgs:function(){return this.args},setArgs:function(e){if(Object.prototype.toString.call(e)!=="[object Array]")throw new TypeError("Args must be an Array");this.args=e},getEvalOrigin:function(){return this.evalOrigin},setEvalOrigin:function(e){if(e instanceof _)this.evalOrigin=e;else if(e instanceof Object)this.evalOrigin=new _(e);else throw new TypeError("Eval Origin must be an Object or StackFrame")},toString:function(){var e=this.getFileName()||"",n=this.getLineNumber()||"",r=this.getColumnNumber()||"",a=this.getFunctionName()||"";return this.getIsEval()?e?"[eval] ("+e+":"+n+":"+r+")":"[eval]:"+n+":"+r:a?a+" ("+e+":"+n+":"+r+")":e+":"+n+":"+r}};_.fromString=d(function(e){var n=e.indexOf("("),r=e.lastIndexOf(")"),a=e.substring(0,n),s=e.substring(n+1,r).split(","),t=e.substring(r+1);if(t.indexOf("@")===0)var i=/@(.+?)(?::(\d+))?(?::(\d+))?$/.exec(t,""),o=i[1],l=i[2],p=i[3];return new _({functionName:a,args:s||void 0,fileName:o,lineNumber:l||void 0,columnNumber:p||void 0})},"StackFrame$$fromString");for(b=0;b<x.length;b++)_.prototype["get"+g(x[b])]=P(x[b]),_.prototype["set"+g(x[b])]=function(e){return function(n){this[e]=!!n}}(x[b]);var b;for(h=0;h<T.length;h++)_.prototype["get"+g(T[h])]=P(T[h]),_.prototype["set"+g(T[h])]=function(e){return function(n){if(!M(n))throw new TypeError(e+" must be a Number");this[e]=Number(n)}}(T[h]);var h;for(v=0;v<A.length;v++)_.prototype["get"+g(A[v])]=P(A[v]),_.prototype["set"+g(A[v])]=function(e){return function(n){this[e]=String(n)}}(A[v]);var v,C=_;function $(){var e=/^\s*at .*(\S+:\d+|\(native\))/m,n=/^(eval@)?(\[native code])?$/;return{parse:d(function(r){if(r.stack&&r.stack.match(e))return this.parseV8OrIE(r);if(r.stack)return this.parseFFOrSafari(r);throw new Error("Cannot parse given Error object")},"ErrorStackParser$$parse"),extractLocation:d(function(r){if(r.indexOf(":")===-1)return[r];var a=/(.+?)(?::(\d+))?(?::(\d+))?$/,s=a.exec(r.replace(/[()]/g,""));return[s[1],s[2]||void 0,s[3]||void 0]},"ErrorStackParser$$extractLocation"),parseV8OrIE:d(function(r){var a=r.stack.split(`
`).filter(function(s){return!!s.match(e)},this);return a.map(function(s){s.indexOf("(eval ")>-1&&(s=s.replace(/eval code/g,"eval").replace(/(\(eval at [^()]*)|(,.*$)/g,""));var t=s.replace(/^\s+/,"").replace(/\(eval code/g,"(").replace(/^.*?\s+/,""),i=t.match(/ (\(.+\)$)/);t=i?t.replace(i[0],""):t;var o=this.extractLocation(i?i[1]:t),l=i&&t||void 0,p=["eval","<anonymous>"].indexOf(o[0])>-1?void 0:o[0];return new C({functionName:l,fileName:p,lineNumber:o[1],columnNumber:o[2],source:s})},this)},"ErrorStackParser$$parseV8OrIE"),parseFFOrSafari:d(function(r){var a=r.stack.split(`
`).filter(function(s){return!s.match(n)},this);return a.map(function(s){if(s.indexOf(" > eval")>-1&&(s=s.replace(/ line (\d+)(?: > eval line \d+)* > eval:\d+:\d+/g,":$1")),s.indexOf("@")===-1&&s.indexOf(":")===-1)return new C({functionName:s});var t=/((.*".+"[^@]*)?[^@]*)(?:@)/,i=s.match(t),o=i&&i[1]?i[1]:void 0,l=this.extractLocation(s.replace(t,""));return new C({functionName:o,fileName:l[0],lineNumber:l[1],columnNumber:l[2],source:s})},this)},"ErrorStackParser$$parseFFOrSafari")}}d($,"ErrorStackParser");var Xe=new $,Qe=Xe,m=typeof process=="object"&&typeof process.versions=="object"&&typeof process.versions.node=="string"&&!process.browser,H=m&&typeof O<"u"&&typeof O.exports<"u"&&typeof U<"u"&&typeof __dirname<"u",Ze=m&&!H,en=typeof Deno<"u",z=!m&&!en,nn=z&&typeof window=="object"&&typeof document=="object"&&typeof document.createElement=="function"&&"sessionStorage"in window&&typeof importScripts!="function",rn=z&&typeof importScripts=="function"&&typeof self=="object";typeof navigator=="object"&&typeof navigator.userAgent=="string"&&navigator.userAgent.indexOf("Chrome")==-1&&navigator.userAgent.indexOf("Safari")>-1;var V,F,Y,J,L;async function q(){if(!m||(V=(await import("./__vite-browser-external-9wXp6ZBx.js")).default,J=await import("./__vite-browser-external-9wXp6ZBx.js"),L=await import("./__vite-browser-external-9wXp6ZBx.js"),Y=(await import("./__vite-browser-external-9wXp6ZBx.js")).default,F=await import("./__vite-browser-external-9wXp6ZBx.js"),D=F.sep,typeof U<"u"))return;let e=J,n=await import("./__vite-browser-external-9wXp6ZBx.js"),r=await import("./__vite-browser-external-9wXp6ZBx.js"),a=await import("./__vite-browser-external-9wXp6ZBx.js"),s={fs:e,crypto:n,ws:r,child_process:a};globalThis.require=function(t){return s[t]}}d(q,"initNodeModules");function W(e,n){return F.resolve(n||".",e)}d(W,"node_resolvePath");function K(e,n){return n===void 0&&(n=location),new URL(e,n).toString()}d(K,"browser_resolvePath");var I;m?I=W:I=K;var D;m||(D="/");function X(e,n){return e.startsWith("file://")&&(e=e.slice(7)),e.includes("://")?{response:fetch(e)}:{binary:L.readFile(e).then(r=>new Uint8Array(r.buffer,r.byteOffset,r.byteLength))}}d(X,"node_getBinaryResponse");function Q(e,n){let r=new URL(e,location);return{response:fetch(r,n?{integrity:n}:{})}}d(Q,"browser_getBinaryResponse");var j;m?j=X:j=Q;async function Z(e,n){let{response:r,binary:a}=j(e,n);if(a)return a;let s=await r;if(!s.ok)throw new Error(`Failed to load '${e}': request failed.`);return new Uint8Array(await s.arrayBuffer())}d(Z,"loadBinaryFile");var N;if(nn)N=d(async e=>await import(e),"loadScript");else if(rn)N=d(async e=>{try{globalThis.importScripts(e)}catch(n){if(n instanceof TypeError)await import(e);else throw n}},"loadScript");else if(m)N=ee;else throw new Error("Cannot determine runtime environment");async function ee(e){e.startsWith("file://")&&(e=e.slice(7)),e.includes("://")?Y.runInThisContext(await(await fetch(e)).text()):await import(V.pathToFileURL(e).href)}d(ee,"nodeLoadScript");async function ne(e){if(m){await q();let n=await L.readFile(e,{encoding:"utf8"});return JSON.parse(n)}else return await(await fetch(e)).json()}d(ne,"loadLockFile");async function re(){if(H)return __dirname;let e;try{throw new Error}catch(a){e=a}let n=Qe.parse(e)[0].fileName;if(m&&!n.startsWith("file://")&&(n=`file://${n}`),Ze){let a=await import("./__vite-browser-external-9wXp6ZBx.js");return(await import("./__vite-browser-external-9wXp6ZBx.js")).fileURLToPath(a.dirname(n))}let r=n.lastIndexOf(D);if(r===-1)throw new Error("Could not extract indexURL path from pyodide module location");return n.slice(0,r)}d(re,"calculateDirname");function te(e){let n=e.FS,r=e.FS.filesystems.MEMFS,a=e.PATH,s={DIR_MODE:16895,FILE_MODE:33279,mount:function(t){if(!t.opts.fileSystemHandle)throw new Error("opts.fileSystemHandle is required");return r.mount.apply(null,arguments)},syncfs:async(t,i,o)=>{try{let l=s.getLocalSet(t),p=await s.getRemoteSet(t),c=i?p:l,f=i?l:p;await s.reconcile(t,c,f),o(null)}catch(l){o(l)}},getLocalSet:t=>{let i=Object.create(null);function o(c){return c!=="."&&c!==".."}d(o,"isRealDir");function l(c){return f=>a.join2(c,f)}d(l,"toAbsolute");let p=n.readdir(t.mountpoint).filter(o).map(l(t.mountpoint));for(;p.length;){let c=p.pop(),f=n.stat(c);n.isDir(f.mode)&&p.push.apply(p,n.readdir(c).filter(o).map(l(c))),i[c]={timestamp:f.mtime,mode:f.mode}}return{type:"local",entries:i}},getRemoteSet:async t=>{let i=Object.create(null),o=await tn(t.opts.fileSystemHandle);for(let[l,p]of o)l!=="."&&(i[a.join2(t.mountpoint,l)]={timestamp:p.kind==="file"?(await p.getFile()).lastModifiedDate:new Date,mode:p.kind==="file"?s.FILE_MODE:s.DIR_MODE});return{type:"remote",entries:i,handles:o}},loadLocalEntry:t=>{let i=n.lookupPath(t).node,o=n.stat(t);if(n.isDir(o.mode))return{timestamp:o.mtime,mode:o.mode};if(n.isFile(o.mode))return i.contents=r.getFileDataAsTypedArray(i),{timestamp:o.mtime,mode:o.mode,contents:i.contents};throw new Error("node type not supported")},storeLocalEntry:(t,i)=>{if(n.isDir(i.mode))n.mkdirTree(t,i.mode);else if(n.isFile(i.mode))n.writeFile(t,i.contents,{canOwn:!0});else throw new Error("node type not supported");n.chmod(t,i.mode),n.utime(t,i.timestamp,i.timestamp)},removeLocalEntry:t=>{var i=n.stat(t);n.isDir(i.mode)?n.rmdir(t):n.isFile(i.mode)&&n.unlink(t)},loadRemoteEntry:async t=>{if(t.kind==="file"){let i=await t.getFile();return{contents:new Uint8Array(await i.arrayBuffer()),mode:s.FILE_MODE,timestamp:i.lastModifiedDate}}else{if(t.kind==="directory")return{mode:s.DIR_MODE,timestamp:new Date};throw new Error("unknown kind: "+t.kind)}},storeRemoteEntry:async(t,i,o)=>{let l=t.get(a.dirname(i)),p=n.isFile(o.mode)?await l.getFileHandle(a.basename(i),{create:!0}):await l.getDirectoryHandle(a.basename(i),{create:!0});if(p.kind==="file"){let c=await p.createWritable();await c.write(o.contents),await c.close()}t.set(i,p)},removeRemoteEntry:async(t,i)=>{await t.get(a.dirname(i)).removeEntry(a.basename(i)),t.delete(i)},reconcile:async(t,i,o)=>{let l=0,p=[];Object.keys(i.entries).forEach(function(u){let y=i.entries[u],k=o.entries[u];(!k||n.isFile(y.mode)&&y.timestamp.getTime()>k.timestamp.getTime())&&(p.push(u),l++)}),p.sort();let c=[];if(Object.keys(o.entries).forEach(function(u){i.entries[u]||(c.push(u),l++)}),c.sort().reverse(),!l)return;let f=i.type==="remote"?i.handles:o.handles;for(let u of p){let y=a.normalize(u.replace(t.mountpoint,"/")).substring(1);if(o.type==="local"){let k=f.get(y),ue=await s.loadRemoteEntry(k);s.storeLocalEntry(u,ue)}else{let k=s.loadLocalEntry(u);await s.storeRemoteEntry(f,y,k)}}for(let u of c)if(o.type==="local")s.removeLocalEntry(u);else{let y=a.normalize(u.replace(t.mountpoint,"/")).substring(1);await s.removeRemoteEntry(f,y)}}};e.FS.filesystems.NATIVEFS_ASYNC=s}d(te,"initializeNativeFS");var tn=d(async e=>{let n=[];async function r(s){for await(let t of s.values())n.push(t),t.kind==="directory"&&await r(t)}d(r,"collect"),await r(e);let a=new Map;a.set(".",e);for(let s of n){let t=(await e.resolve(s)).join("/");a.set(t,s)}return a},"getFsHandles");function ie(e){let n={noImageDecoding:!0,noAudioDecoding:!0,noWasmDecoding:!1,preRun:de(e),quit(r,a){throw n.exited={status:r,toThrow:a},a},print:e.stdout,printErr:e.stderr,arguments:e.args,API:{config:e},locateFile:r=>e.indexURL+r,instantiateWasm:pe(e.indexURL)};return n}d(ie,"createSettings");function ae(e){return function(n){let r="/";try{n.FS.mkdirTree(e)}catch(a){console.error(`Error occurred while making a home directory '${e}':`),console.error(a),console.error(`Using '${r}' for a home directory instead`),e=r}n.FS.chdir(e)}}d(ae,"createHomeDirectory");function se(e){return function(n){Object.assign(n.ENV,e)}}d(se,"setEnvironment");function oe(e){return n=>{for(let r of e)n.FS.mkdirTree(r),n.FS.mount(n.FS.filesystems.NODEFS,{root:r},r)}}d(oe,"mountLocalDirectories");function le(e){let n=Z(e);return r=>{let a=r._py_version_major(),s=r._py_version_minor();r.FS.mkdirTree("/lib"),r.FS.mkdirTree(`/lib/python${a}.${s}/site-packages`),r.addRunDependency("install-stdlib"),n.then(t=>{r.FS.writeFile(`/lib/python${a}${s}.zip`,t)}).catch(t=>{console.error("Error occurred while installing the standard library:"),console.error(t)}).finally(()=>{r.removeRunDependency("install-stdlib")})}}d(le,"installStdlib");function de(e){let n;return e.stdLibURL!=null?n=e.stdLibURL:n=e.indexURL+"python_stdlib.zip",[le(n),ae(e.env.HOME),se(e.env),oe(e._node_mounts),te]}d(de,"getFileSystemInitializationFuncs");function pe(e){let{binary:n,response:r}=j(e+"pyodide.asm.wasm");return function(a,s){return async function(){try{let t;r?t=await WebAssembly.instantiateStreaming(r,a):t=await WebAssembly.instantiate(await n,a);let{instance:i,module:o}=t;typeof WasmOffsetConverter<"u"&&(wasmOffsetConverter=new WasmOffsetConverter(wasmBinary,o)),s(i,o)}catch(t){console.warn("wasm instantiation failed!"),console.warn(t)}}(),{}}}d(pe,"getInstantiateWasmFunc");var B="0.26.4";async function ce(e={}){var n,r;await q();let a=e.indexURL||await re();a=I(a),a.endsWith("/")||(a+="/"),e.indexURL=a;let s={fullStdLib:!1,jsglobals:globalThis,stdin:globalThis.prompt?globalThis.prompt:void 0,lockFileURL:a+"pyodide-lock.json",args:[],_node_mounts:[],env:{},packageCacheDir:a,packages:[],enableRunUntilComplete:!1,checkAPIVersion:!0},t=Object.assign(s,e);(n=t.env).HOME??(n.HOME="/home/pyodide"),(r=t.env).PYTHONINSPECT??(r.PYTHONINSPECT="1");let i=ie(t),o=i.API;if(o.lockFilePromise=ne(t.lockFileURL),typeof _createPyodideModule!="function"){let u=`${t.indexURL}pyodide.asm.js`;await N(u)}let l;if(e._loadSnapshot){let u=await e._loadSnapshot;ArrayBuffer.isView(u)?l=u:l=new Uint8Array(u),i.noInitialRun=!0,i.INITIAL_MEMORY=l.length}let p=await _createPyodideModule(i);if(i.exited)throw i.exited.toThrow;if(e.pyproxyToStringRepr&&o.setPyProxyToStringMethod(!0),o.version!==B&&t.checkAPIVersion)throw new Error(`Pyodide version does not match: '${B}' <==> '${o.version}'. If you updated the Pyodide version, make sure you also updated the 'indexURL' parameter passed to loadPyodide.`);p.locateFile=u=>{throw new Error("Didn't expect to load any more file_packager files!")};let c;l&&(c=o.restoreSnapshot(l));let f=o.finalizeBootstrap(c);return o.sys.path.insert(0,o.config.env.HOME),f.version.includes("dev")||o.setCdnUrl(`https://cdn.jsdelivr.net/pyodide/v${f.version}/full/`),o._pyodide.set_excepthook(),await o.packageIndexReady,o.initializeStreams(t.stdin,t.stdout,t.stderr),f}d(ce,"loadPyodide");const R="numpy-1.26.4-cp312-cp312-pyodide_2024_0_wasm32.whl",an={"python_stdlib.zip":["python_stdlib.zip.b64.0.txt"],[R]:[`${R}.b64.0.txt`,`${R}.b64.1.txt`]};function sn(e){const n=atob(e),r=new Uint8Array(n.length);for(let a=0;a<n.length;a++)r[a]=n.charCodeAt(a);return r}function on(e,n){return async(r,a)=>{const t=(typeof r=="string"?r:r instanceof URL?r.href:r.url).split("?")[0].split("/").pop()??"",i=an[t];if(!i)return e(r,a);const o=await Promise.all(i.map(async p=>{const c=await e(new URL(p,n).href);if(!c.ok)throw new Error(`Could not load ${p} (${c.status})`);return c.text()})),l=sn(o.join("").trim());return new Response(l,{status:200,headers:{"Content-Type":"application/octet-stream"}})}}const ln=Object.assign({"../../../backend/monopoly_lab/__init__.py":_e,"../../../backend/monopoly_lab/analytics/__init__.py":me,"../../../backend/monopoly_lab/analytics/landing.py":ge,"../../../backend/monopoly_lab/analytics/roi.py":ye,"../../../backend/monopoly_lab/api/__init__.py":be,"../../../backend/monopoly_lab/api/core.py":he,"../../../backend/monopoly_lab/cli.py":ve,"../../../backend/monopoly_lab/data/__init__.py":we,"../../../backend/monopoly_lab/data/us_standard.json":ke,"../../../backend/monopoly_lab/engine/__init__.py":xe,"../../../backend/monopoly_lab/engine/board.py":Te,"../../../backend/monopoly_lab/engine/buildings.py":Ae,"../../../backend/monopoly_lab/engine/cards.py":Se,"../../../backend/monopoly_lab/engine/dice.py":Ee,"../../../backend/monopoly_lab/engine/finance.py":Ne,"../../../backend/monopoly_lab/engine/game.py":je,"../../../backend/monopoly_lab/engine/jail.py":Pe,"../../../backend/monopoly_lab/engine/movement.py":Ce,"../../../backend/monopoly_lab/engine/rent.py":Re,"../../../backend/monopoly_lab/engine/rules.py":Ge,"../../../backend/monopoly_lab/engine/state.py":Fe,"../../../backend/monopoly_lab/engine/stats.py":Ie,"../../../backend/monopoly_lab/engine/trading.py":Oe,"../../../backend/monopoly_lab/policies/__init__.py":Le,"../../../backend/monopoly_lab/policies/base.py":qe,"../../../backend/monopoly_lab/policies/baseline.py":De,"../../../backend/monopoly_lab/policies/development.py":Je,"../../../backend/monopoly_lab/policies/human.py":Be,"../../../backend/monopoly_lab/policies/property_value.py":Ue,"../../../backend/monopoly_lab/policies/registry.py":Me,"../../../backend/monopoly_lab/policies/trade_ai.py":$e,"../../../backend/monopoly_lab/simulation/__init__.py":He,"../../../backend/monopoly_lab/simulation/runner.py":ze,"../../../backend/monopoly_lab/simulation/session.py":Ve}),S=self,w=e=>S.postMessage(e);let G=null;async function dn(e){S.fetch=on(S.fetch.bind(S),e),w({type:"status",message:"Loading Python (WebAssembly)…"});const n=await ce({indexURL:e});w({type:"status",message:"Loading NumPy…"}),await n.loadPackage("numpy"),w({type:"status",message:"Loading the game engine…"});for(const[a,s]of Object.entries(ln)){const i=`/app/${a.slice(a.indexOf("monopoly_lab/"))}`;n.FS.mkdirTree(i.slice(0,i.lastIndexOf("/"))),n.FS.writeFile(i,s)}return n.runPython(`import sys
sys.path.insert(0, '/app')
from monopoly_lab.api.core import handle_json`),n.globals.get("handle_json")}S.onmessage=async e=>{const n=e.data;if(n.type==="init"){G??(G=dn(n.indexURL));try{await G,w({type:"ready"})}catch(r){w({type:"fatal",message:String((r==null?void 0:r.message)??r)})}return}if(n.type==="request")try{const r=await G,a=JSON.parse(r(JSON.stringify(n.req)));w({type:"response",id:n.id,status:a.status,body:a.body})}catch(r){w({type:"response",id:n.id,status:500,body:{detail:`Engine error: ${String((r==null?void 0:r.message)??r)}`}})}}});export default pn();
