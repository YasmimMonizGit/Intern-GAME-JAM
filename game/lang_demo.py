"""
lang_demo.py — test a trained text model interactively.

    python lang_demo.py --model intern_v1

A random map is shown with its tasks. You type an instruction in your own words
(or a deliberately *wrong* order) and the model plans which stations to visit,
PURELY from your text. It then executes the plan. This is how you confirm the
bot now follows language instead of the hidden task list.
"""

import argparse
import random

from gridworld import GridWorld, STATION_CONFIG, STATION_ORDER
from language import load_lang, plan_from_text, run_text_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="intern_v1")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    net, vocab = load_lang(args.model)
    rng = random.Random(args.seed)
    print(f"Loaded text model '{args.model}'. Ctrl-C to quit.\n")

    while True:
        world = GridWorld(seed=rng.randint(0, 10 ** 6), n_station_range=(2, 4))
        present = ", ".join(STATION_CONFIG[s]["name"]
                            for s in STATION_ORDER if world._find_station(s))
        print("=" * 60)
        print(f"Stations on this map: {present}")
        print("Tasks (the model will NOT see this list):")
        for t in world.tasks:
            print(f"   - {t.text}  ({STATION_CONFIG[t.station]['name']})")
        try:
            text = input("\nYour instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye"); return
        if not text:
            continue

        plan = plan_from_text(net, vocab, text)
        names = [STATION_CONFIG[STATION_ORDER[s]]["name"] for s in plan]
        print(f"Model's plan from your words: {names or '(nothing)'}")
        solved = run_text_policy(net, vocab, world, text)
        print(f"Result: {'all tasks done' if solved else 'did not finish all tasks'}\n")


if __name__ == "__main__":
    main()
