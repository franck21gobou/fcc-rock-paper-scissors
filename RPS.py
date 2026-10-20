# Rock Paper Scissors player for the freeCodeCamp Machine Learning with Python certification.
#
# Strategy: an ensemble of pattern predictors (Markov / n-gram models).
#  - Some predictors look at the opponent's last k moves.
#  - Others look at the last k (my move, opponent move) pairs, which captures
#    opponents that react to what *I* played.
# Every predictor guesses the opponent's next move. Each round, every predictor
# is scored on whether following it would have won (with exponential decay so
# recent performance matters most) and we follow the best-scoring one.

import random

BEATS = {"R": "P", "P": "S", "S": "R"}  # move -> the move that beats it
MOVES = ["R", "P", "S"]
OPP_ORDERS = [1, 2, 3, 4, 5]
JOINT_ORDERS = [1, 2, 3]
DECAY = 0.9


def _new_state():
    return {
        "opp": [],    # opponent moves
        "mine": [],   # my moves
        "tables": [{} for _ in range(len(OPP_ORDERS) + len(JOINT_ORDERS))],
        "scores": [0.0] * (len(OPP_ORDERS) + len(JOINT_ORDERS)),
        "keys": None,         # context key used by each predictor last round
        "predictions": None,  # prediction of each predictor last round
    }


def _contexts(opp, mine):
    """Context key for every predictor (None if not enough history)."""
    keys = []
    for k in OPP_ORDERS:
        keys.append("".join(opp[-k:]) if len(opp) >= k else None)
    pairs = [m + o for m, o in zip(mine, opp)]
    for k in JOINT_ORDERS:
        keys.append("".join(pairs[-k:]) if len(pairs) >= k else None)
    return keys


def player(prev_play, state={}):
    # A new match starts with an empty prev_play: reset all memory.
    if prev_play == "" or not state:
        state.clear()
        state.update(_new_state())
        move = random.choice(MOVES)
        state["mine"].append(move)
        return move

    # 1) Learn from the opponent's latest move.
    if state["keys"] is not None:
        for i, (key, pred) in enumerate(zip(state["keys"], state["predictions"])):
            state["scores"][i] *= DECAY
            if pred is not None:
                mine_if_followed = BEATS[pred]
                if mine_if_followed == BEATS[prev_play]:
                    state["scores"][i] += 1
                elif BEATS[mine_if_followed] == prev_play:
                    state["scores"][i] -= 1
            if key is not None:
                counts = state["tables"][i].setdefault(key, {})
                counts[prev_play] = counts.get(prev_play, 0) + 1
    state["opp"].append(prev_play)

    # 2) Every predictor guesses the opponent's next move.
    keys = _contexts(state["opp"], state["mine"])
    predictions = []
    for i, key in enumerate(keys):
        counts = state["tables"][i].get(key) if key is not None else None
        predictions.append(max(MOVES, key=lambda m: counts.get(m, 0)) if counts else None)
    state["keys"], state["predictions"] = keys, predictions

    # 3) Follow the predictor that has been winning most recently.
    candidates = [i for i, p in enumerate(predictions) if p is not None]
    if candidates:
        best = max(candidates, key=lambda i: state["scores"][i])
        move = BEATS[predictions[best]]
    else:
        move = random.choice(MOVES)
    state["mine"].append(move)
    return move
