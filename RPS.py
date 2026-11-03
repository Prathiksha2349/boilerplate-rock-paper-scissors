BEATS = {"R": "P", "P": "S", "S": "R"}  # BEATS[x] is the move that beats x
MOVES = ["R", "P", "S"]


# ---- Models: each one predicts the OPPONENT's next move ----

def model_ngram(opp, mine):
    """Order-2 pattern predictor on the opponent's own history (beats quincy)."""
    if len(opp) < 3:
        return "R"
    key = tuple(opp[-2:])
    counts = {m: 0 for m in MOVES}
    for i in range(len(opp) - 2):
        if tuple(opp[i:i + 2]) == key:
            counts[opp[i + 2]] += 1
    if sum(counts.values()) == 0:
        return "R"
    return max(MOVES, key=lambda m: counts[m])


def model_kris(opp, mine):
    """Kris plays the move that beats my previous move."""
    last = mine[-1] if mine else "R"
    return BEATS[last]


def model_mrugesh(opp, mine):
    """Mrugesh plays the counter to my most frequent move in the last 10."""
    # The real bot's history starts with an empty string from the first call.
    last_ten = ([""] + mine)[-10:]
    most_frequent = max(["", "R", "P", "S"], key=lambda m: last_ten.count(m))
    if most_frequent == "":
        most_frequent = "S"
    return BEATS[most_frequent]


def model_abbey(opp, mine):
    """Abbey builds pair counts of my moves, predicts my next, and counters it."""
    hist = ["R"] + mine
    pairs = {a + b: 0 for a in MOVES for b in MOVES}
    for i in range(1, len(hist)):
        pairs[hist[i - 1] + hist[i]] += 1
    last = hist[-1]
    potential = [last + m for m in MOVES]
    prediction = max(potential, key=lambda k: pairs[k])[-1]
    return BEATS[prediction]


MODELS = {
    "ngram": model_ngram,
    "kris": model_kris,
    "mrugesh": model_mrugesh,
    "abbey": model_abbey,
}


def player(prev_play, opponent_history=[], my_history=[], state={}):
    # New match starts with an empty string: reset all stored state.
    if prev_play == "" or not state:
        opponent_history.clear()
        my_history.clear()
        state.clear()
        state["scores"] = {name: 0.0 for name in MODELS}
        state["preds"] = {}
    else:
        opponent_history.append(prev_play)
        # Score each model on whether it predicted the opponent's last move.
        for name, pred in state["preds"].items():
            state["scores"][name] = state["scores"][name] * 0.9 + (1.0 if pred == prev_play else 0.0)

    # Each model predicts the opponent's next move.
    preds = {name: fn(opponent_history, my_history) for name, fn in MODELS.items()}
    state["preds"] = preds

    # Trust whichever model has been most accurate recently.
    best = max(MODELS, key=lambda n: state["scores"][n])
    guess = preds[best]

    move = BEATS[guess]
    my_history.append(move)
    return move
