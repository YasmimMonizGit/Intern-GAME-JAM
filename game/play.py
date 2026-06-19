"""
play.py — the interactive loop.

Run:  python play.py
At startup you pick or name a model (stored in neural_room/).

Per map:
  1. The task list appears (only you see it).
  2. Type your plan, press ENTER to lock it in and start recording.
  3. Play: arrows / WASD to move, E to interact with an adjacent station.
  4. Press F to finish -> your choices become training data.
Keys any time:
  N new map   T train + show learning curve   S save model   B watch the bot
  ESC quit
"""

import sys
import learning as L
from learning import DecisionNet, DemoStore, DemoRecorder, NeuralRoom, run_policy
from gridworld import GridWorld, Tile, STATION_CONFIG, STATION_ORDER, STATIONS
from gemma import gemma_instruction
from enum import Enum



class State(Enum):
    SPLASH = -1
    WAITING = 0
    PLAYING = 1
    RECORDING = 2
    WAITING_BOT = 3
    BOT_COMPUTING = 4

def pick_model(room):
    existing = room.list_models()
    print("\n=== neural room ===")
    if existing:
        print("Existing models:", ", ".join(existing))
    else:
        print("(no saved models yet)")
    name = input("Model name to load, or a NEW name to create: ").strip() or "model1"
    if name in existing:
        net, meta = room.load(name)
        print(f"Loaded '{name}'  {meta}")
    else:
        net = DecisionNet()
        print(f"Created new model '{name}'")
    return net, name


def main():
    import pygame

    bgsurf = pygame.image.load('bg.png')
    logosurf = pygame.image.load('logo.png')
    internsurf = pygame.image.load('intern.png')

    room = NeuralRoom()
    store = DemoStore()
    net, model_name = pick_model(room)
    print(f"Dataset: {store.n_episodes} demos / {store.n_pairs} decisions on disk.\n")

    CELL, PANEL, MARGIN = 52, 340, 16
    BG, FLOOR_C, WALL_C = (34, 34, 40), (60, 62, 72), (24, 24, 28)
    PLAYER_C, GRID_LINE = (86, 156, 214), (44, 46, 54)
    TEXT, MUTED, OK = (228, 228, 232), (140, 142, 150), (122, 200, 120)

    pygame.init()
    world = GridWorld()
    gp = world.size * CELL
    W, H = gp + PANEL + MARGIN * 3, gp + MARGIN * 2
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("GridWorld — demonstrate & train")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 21)
    big = pygame.font.SysFont("consolas", 25, bold=True)
    small = pygame.font.SysFont("consolas", 17)

    # episode state
    rec = None
    plan = ""
    recording = False
    episode_counter = store.n_episodes
    status = "..."
    state = State.SPLASH

    def new_map():
        nonlocal world, rec, plan, recording, status
        world = GridWorld()
        rec, plan, recording = None, "", False
        status = "New map. Type your plan, then ENTER."
        plan = gemma_instruction([t.text for t in world.tasks])
        print("plan?", plan)


        

        
        

    def begin_recording():
        nonlocal rec, recording, status
        rec = DemoRecorder(plan_text=plan)
        rec.begin(world)
        recording =  True
        status = "Recording. Arrows/WASD move, E interact, F finish."

    def do_interact():
        if not recording:
            world.interact(); return
        near = world.adjacent_stations()
        target = next((st for *_, st in near
                       if any(t.station == st and not t.done for t in world.tasks)), None)
        if world.interact() and target is not None:
            rec.commit(world, target)

    def finish_demo():
        nonlocal recording, status, episode_counter
        if not recording:
            return
        demo = rec.finalize(episode_counter)
        recording = False
        if demo:
            store.add(demo)
            episode_counter += 1
            status = (f"Demo saved (ep {demo['episode_id']}, "
                      f"{len(demo['choices'])} choices). N=new  T=train")
        else:
            status = "No choices recorded. N=new map."

    def do_train():
        nonlocal status
        if store.n_episodes == 0:
            status = "No demos yet — play one first."
            return
        print(f"\nTraining '{model_name}' on {store.n_episodes} demos "
              f"/ {store.n_pairs} decisions...")
        hist = L.train(net, store, epochs=120)
        png = L._in_root("neural_room", f"{model_name}_curve.png")
        L.plot_history(hist, title=f"{model_name}: learning curve", save_path=png)
        va = hist["val_acc"][-1] if hist["val_acc"] else None
        status = (f"Trained. val_acc={va:.2f} (held-out maps). Curve -> {png}"
                  if va is not None else
                  f"Trained on {store.n_episodes} demos (need >=4 maps for val split).")

    def watch_bot():
        nonlocal status, state
        for _ in range(12):
            if world.all_done():
                break
            import torch
            feats = torch.from_numpy(L.decision_features(world))[None]
            with torch.no_grad():
                choice = int(net(feats).argmax(1))
            pos = world._find_station(STATION_ORDER[choice])
            res = L.bfs_to_station(world, pos) if pos else None
            if not res:
                status = "Bot got stuck (untrained?)."
                return
            
            state = State.PLAYING

            for dr, dc in res[1]:
                world.move(dr, dc)
                draw(); pygame.display.flip(); pygame.time.delay(90)
                pygame.event.pump()
            world.interact()
            draw(); pygame.display.flip(); pygame.time.delay(200)
        status = "Bot finished!" if world.all_done() else "Bot stopped."

    def draw():
        nonlocal state
        screen.fill(BG)
        ox, oy = MARGIN, MARGIN
        for r in range(world.size):
            for c in range(world.size):
                rect = pygame.Rect(ox + c * CELL, oy + r * CELL, CELL, CELL)
                t = world.grid[r][c]
                if t == Tile.WALL:
                    pygame.draw.rect(screen, WALL_C, rect)
                elif t in STATIONS:
                    pygame.draw.rect(screen, STATION_CONFIG[t]["color"], rect)
                    lbl = big.render(STATION_CONFIG[t]["label"], True, (20, 20, 24))
                    screen.blit(lbl, lbl.get_rect(center=rect.center))
                else:
                    pygame.draw.rect(screen, FLOOR_C, rect)
                pygame.draw.rect(screen, GRID_LINE, rect, 1)
        ar, ac = world.agent
        pygame.draw.circle(screen, PLAYER_C,
                           (ox + ac * CELL + CELL // 2, oy + ar * CELL + CELL // 2), CELL // 3)

        px = ox + gp + MARGIN
        screen.blit(big.render("Today's Tasks", True, TEXT), (px, oy))
        y = oy + 42
        for t in world.tasks:
            box = "[x]" if t.done else "[ ]"
            col = MUTED if t.done else STATION_CONFIG[t.station]["color"]
            screen.blit(font.render(f"{box} {t.text}", True, col), (px, y)); y += 28
        if world.all_done():
            screen.blit(big.render("All done! :)", True, OK), (px, y + 6)); y += 40

        if world.all_done() and state is State.RECORDING:
            state = State.WAITING_BOT
            

        y += 14
        screen.blit(small.render(f"model: {model_name}   demos: {store.n_episodes}",
                                 True, MUTED), (px, y)); y += 22
        screen.blit(small.render(f"openness {world.openness:.2f}  intersect "
                                 f"{world.intersect:.2f}", True, MUTED), (px, y)); y += 26
        # cursor = "_" if typing else ""
        # screen.blit(font.render("Plan: " + plan + cursor, True, TEXT), (px, y)); y += 30




        if state is State.SPLASH:
            screen.blit(bgsurf, (0,0))
            screen.blit(pygame.transform.scale_by(logosurf, (0.5,0.5)), (50,50))
            screen.blit(pygame.transform.scale_by(internsurf, (0.3,0.3)), (W - 350, -20))
            screen.blit(pygame.transform.scale_by(font.render("Press SPACE to START", True, TEXT), (2,2)), (0, 0))
        if state is State.WAITING:
            screen.blit(pygame.transform.scale_by(font.render("Press SPACE to START", True, TEXT), (2,2)), (0, 0))
        if state is State.WAITING_BOT:
            screen.blit(font.render("Press SPACE to let the intern try!", True, TEXT), (0, 0))

        if state is State.BOT_COMPUTING:
            screen.blit(font.render("Please wait for the intern to make up his mind...", True, TEXT), (0, 0))


        for m in world.messages[-4:]:
            screen.blit(small.render(m, True, MUTED), (px, y)); y += 18

        screen.blit(small.render(status, True, TEXT), (ox, H - MARGIN - 22))
        screen.blit(small.render("N new  T train  S save  B watch-bot  ESC quit",
                                 True, MUTED), (ox, H - MARGIN - 4))
        pygame.display.flip()

    MOVE = {pygame.K_UP: (-1, 0), pygame.K_w: (-1, 0), pygame.K_DOWN: (1, 0),
            pygame.K_s: (1, 0), pygame.K_LEFT: (0, -1), pygame.K_a: (0, -1),
            pygame.K_RIGHT: (0, 1), pygame.K_d: (0, 1)}

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE and state in (State.WAITING, State.SPLASH):
                    new_map()
                    state = State.RECORDING
                    begin_recording()
                    recording = True
                elif e.key == pygame.K_SPACE and state is State.WAITING_BOT:
                    state = State.BOT_COMPUTING
                    draw(); pygame.display.flip(); pygame.time.delay(90)
                    pygame.event.pump()
                    new_map()
                    watch_bot()
                    state = State.WAITING
                elif e.key in MOVE and state is State.RECORDING:
                    world.move(*MOVE[e.key])
                elif e.key == pygame.K_e and state is State.RECORDING:
                    do_interact()
                elif e.key == pygame.K_f:
                    finish_demo()
                elif e.key == pygame.K_n:
                    new_map()
                elif e.key == pygame.K_t:
                    do_train()
                elif e.key == pygame.K_s:
                    room.save(net, model_name,
                                meta={"demos": store.n_episodes,
                                    "decisions": store.n_pairs})
                    status = f"Saved '{model_name}' to neural_room/."
                elif e.key == pygame.K_b:
                    watch_bot()
        draw()
        clock.tick(30)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
