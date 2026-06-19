"""
autotester.py — automatically generate demonstrations (and optionally train).

Hand-playing maps with F is slow. This runs a *scripted expert* instead:
BFS handles movement, and a simple heuristic chooses which station to do next.
It records demonstrations through the exact same DemoRecorder / DemoStore that
play.py uses, so the data is identical in form to your hand-played runs and
lands in the same neural_room/demonstrations.pkl.

The expert is NOT a neural network -- it's the teacher. The DecisionNet you
train is the student that learns from these examples. This is the standard way
imitation-learning pipelines bootstrap data.

Examples:
  python autotester.py --n 80                      # add 80 expert demos
  python autotester.py --n 80 --strategy listed    # follow task order instead
  python autotester.py --n 100 --train --model auto_v1   # generate AND train
  python autotester.py --n 50 --fresh              # wipe dataset, then generate
"""

import argparse
import os
import random

import learning as L
from learning import (DemoRecorder, DemoStore, DecisionNet, NeuralRoom,
                      bfs_to_station, run_policy)
from gridworld import GridWorld, STATION_ORDER


def language_run(args):
    """Gemma writes an instruction from the ordered task list; the expert does
    the jobs in that order; the text-blind network learns text -> station order."""
    import language as Lang
    from language import (TextDemoStore, TextPlanNet, Vocab, train_text,
                          run_text_policy, save_lang, MAX_STEPS)
    from gemma import gemma_instruction, template_instruction, ollama_available, DEFAULT_MODEL

    rng = random.Random(args.seed if args.seed is not None else random.randint(0, 10 ** 6))
    base = args.seed if args.seed is not None else random.randint(0, 10 ** 6)

    use_gemma = not args.no_gemma and ollama_available(args.gemma_model)
    print(f"Text source: {'Gemma (' + args.gemma_model + ')' if use_gemma else 'templated fallback (Ollama not reachable)'}")

    def make_text(task_texts):
        if use_gemma:
            try:
                return gemma_instruction(task_texts, model=args.gemma_model)
            except Exception:
                return template_instruction(task_texts, rng)
        return template_instruction(task_texts, rng)

    store = TextDemoStore()
    if args.fresh and os.path.exists(store.path):
        os.remove(store.path); store.demos = []
        print("Wiped existing language dataset.")

    print(f"Generating {args.n} instruction/plan demos...")
    for i in range(args.n):
        world = GridWorld(seed=base + i)
        tasks = world.tasks[:MAX_STEPS]
        if not tasks:
            continue
        task_texts = [t.text for t in tasks]
        sequence = [STATION_ORDER.index(t.station) for t in tasks]
        text = make_text(task_texts)
        store.add({"episode_id": store.n, "text": text, "sequence": sequence})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{args.n}  e.g. \"{text[:70]}\"")

    print(f"Dataset: {store.n} instruction demos at {store.path}")

    vocab = Vocab()
    vocab.build([d["text"] for d in store.demos])
    net = TextPlanNet(len(vocab.i2w))
    print(f"\nTraining text model '{args.model}' (vocab {len(vocab.i2w)} words)...")
    hist = train_text(net, store, vocab, epochs=args.epochs)
    save_lang(net, vocab, args.model)
    curve = L._in_root("neural_room", f"{args.model}_lang_curve.png")
    L.plot_history(hist, title=f"{args.model}: text->plan learning curve",
                   save_path=curve, show=False)

    # end-to-end eval on fresh unseen maps, driven only by the instruction text
    solved = 0
    M = 40
    for i in range(M):
        w = GridWorld(seed=700000 + i)
        if not w.tasks:
            continue
        txt = make_text([t.text for t in w.tasks[:MAX_STEPS]])
        solved += run_text_policy(net, vocab, w, txt)
    va = hist["val_acc"][-1] if hist["val_acc"] else None
    if va is not None:
        print(f"val exact-sequence acc (held-out instructions): {va:.2f}")
    print(f"solved {solved}/{M} unseen maps from text alone")
    print(f"Model saved: {L._in_root('neural_room', args.model + '.lang.pt')}")
    print(f"Curve saved: {curve}")


def choose_station(world, strategy, rng):
    """Return the next station the expert will head to, or None if all done."""
    remaining = [st for st in STATION_ORDER
                 if any(t.station == st and not t.done for t in world.tasks)]
    if not remaining:
        return None
    if strategy == "listed":                          # first task in the list order
        for t in world.tasks:
            if not t.done:
                return t.station
    if strategy == "random":
        return rng.choice(remaining)
    # default "nearest": smallest BFS distance from the agent
    best = None
    for st in remaining:
        res = bfs_to_station(world, world._find_station(st))
        if res and (best is None or res[0] < best[1]):
            best = (st, res[0])
    return best[0] if best else remaining[0]


def expert_episode(world, episode_id, strategy, rng, plan_text):
    """Solve one map with the scripted expert, recording decisions exactly the
    way play.py's do_interact does."""
    rec = DemoRecorder(plan_text=plan_text)
    rec.begin(world)
    legs = 0
    while not world.all_done() and legs < 30:
        legs += 1
        st = choose_station(world, strategy, rng)
        if st is None:
            break
        res = bfs_to_station(world, world._find_station(st))
        if res is None:
            break                                     # shouldn't happen (maps are solvable)
        for dr, dc in res[1]:
            world.move(dr, dc)
        # mirror play.do_interact: commit the adjacent station we just served
        near = world.adjacent_stations()
        target = next((s for *_, s in near
                       if any(t.station == s and not t.done for t in world.tasks)), None)
        if world.interact() and target is not None:
            rec.commit(world, target)
    return rec.finalize(episode_id)


def main():
    ap = argparse.ArgumentParser(description="Auto-generate grid-world demonstrations.")
    ap.add_argument("--n", type=int, default=50, help="how many demos to generate")
    ap.add_argument("--strategy", choices=["nearest", "listed", "random"],
                    default="nearest", help="how the expert picks the next station")
    ap.add_argument("--seed", type=int, default=None, help="base seed (reproducible maps)")
    ap.add_argument("--fresh", action="store_true", help="wipe the dataset first")
    ap.add_argument("--train", action="store_true", help="train a model afterwards")
    ap.add_argument("--model", default="auto_model", help="model name when --train")
    ap.add_argument("--language", action="store_true",
                    help="Gemma writes instructions; train the text-blind model")
    ap.add_argument("--epochs", type=int, default=80, help="training epochs")
    ap.add_argument("--gemma-model", default="gemma4:e4b-it-qat",
                    help="Ollama model name for instruction text")
    ap.add_argument("--no-gemma", action="store_true",
                    help="skip Ollama, use the templated fallback")
    args = ap.parse_args()

    if args.language:
        language_run(args)
        return

    store = DemoStore()
    if args.fresh and os.path.exists(store.path):
        os.remove(store.path)
        store.demos = []
        print("Wiped existing dataset.")

    base = args.seed if args.seed is not None else random.randint(0, 10 ** 6)
    rng = random.Random(base)
    plan = {"nearest": "do the closest task next",
            "listed": "do tasks in the order listed",
            "random": "do remaining tasks in any order"}[args.strategy]

    before = store.n_episodes
    made = skipped = 0
    for i in range(args.n):
        world = GridWorld(seed=base + i)
        demo = expert_episode(world, before + made, args.strategy, rng, plan)
        if demo:
            store.add(demo)
            made += 1
        else:
            skipped += 1

    print(f"Generated {made} demos ({skipped} skipped) with '{args.strategy}' strategy.")
    print(f"Dataset now: {store.n_episodes} demos / {store.n_pairs} decisions")
    print(f"Saved to: {store.path}")

    if args.train:
        room = NeuralRoom()
        if args.model in room.list_models():
            net, _ = room.load(args.model)
            print(f"\nContinuing existing model '{args.model}'.")
        else:
            net = DecisionNet()
            print(f"\nTraining new model '{args.model}'.")
        hist = L.train(net, store, epochs=120)
        curve = L._in_root("neural_room", f"{args.model}_curve.png")
        L.plot_history(hist, title=f"{args.model}: learning curve",
                       save_path=curve, show=False)
        room.save(net, args.model,
                  meta={"demos": store.n_episodes, "decisions": store.n_pairs,
                        "strategy": args.strategy})
        # quick generalization check on fresh unseen maps
        solved = sum(run_policy(net, GridWorld(seed=900000 + i)) for i in range(40))
        va = hist["val_acc"][-1] if hist["val_acc"] else None
        if va is not None:
            print(f"val_acc (held-out maps): {va:.2f}")
        print(f"solved {solved}/40 unseen maps")
        print(f"Model saved: {L._in_root('neural_room', args.model + '.pt')}")
        print(f"Curve saved: {curve}")


if __name__ == "__main__":
    main()