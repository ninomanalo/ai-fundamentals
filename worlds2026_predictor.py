"""
League of Legends Worlds 2026 predictor
=======================================
1. Load match data (Oracle's Elixir CSV)         -> one row per game
2. Build pre-game features (Elo, form, gold@15)   -> no leakage, strictly chronological
3. Train Logistic Regression + Gradient Boosting  -> P(team A wins a game)
4. Monte Carlo the tournament (Play-In -> Swiss -> Bo5 knockouts)

Usage
-----
  pip install pandas numpy scikit-learn
  python worlds2026_predictor.py --demo                       # synthetic data, just to test the pipeline
  python worlds2026_predictor.py --csv 2026_LoL_esports_match_data_from_OraclesElixir.csv

Data: download the yearly CSVs from oracleselixir.com (Downloads page) and concatenate
2023-2026 if you want more history, e.g. pd.concat([...]).to_csv("all.csv").
"""
import argparse
import math
import random
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# CONFIG - EDIT THESE
# --------------------------------------------------------------------------------------
# Names must match the `teamname` column in Oracle's Elixir. Unmatched names get a default
# rating and a warning, so check the console output.
# LCS_TEAM_x are placeholders: replace with the real LCS qualifiers.
DIRECT_TO_SWISS = [
    "Gen.G", "Hanwha Life Esports", "T1", "Dplus KIA",           # LCK
    "Anyone's Legend", "Bilibili Gaming", "TOP Esports",         # LPL
    "G2 Esports", "Movistar KOI",                                # LEC
    "Team Secret Whales", "CTBC Flying Oyster",                  # LCP
    "LOS", "FURIA",                                              # CBLOL
    "LYON", "LCS_TEAM_2",                                        # LCS
]
# Play-In: 4 teams, 1 ticket. PLACEHOLDER GUESS (LPL4 / LEC3 / LCP3 / LCS3) - verify!
PLAY_IN = ["Invictus Gaming", "Karmine Corp", "MVK Esports", "LCS_TEAM_3"]

REGION = {  # only used by --demo to generate fake data
    **{t: "LCK" for t in ["Gen.G", "Hanwha Life Esports", "T1", "Dplus KIA"]},
    **{t: "LPL" for t in ["Anyone's Legend", "Bilibili Gaming", "TOP Esports", "Invictus Gaming"]},
    **{t: "LEC" for t in ["G2 Esports", "Movistar KOI", "Karmine Corp"]},
    **{t: "LCP" for t in ["Team Secret Whales", "CTBC Flying Oyster", "MVK Esports"]},
    **{t: "CBLOL" for t in ["LOS", "FURIA"]},
    **{t: "LCS" for t in ["LYON", "LCS_TEAM_2", "LCS_TEAM_3"]},
}

# Starting Elo by league (a tunable assumption: stronger regions start higher).
BASE_ELO = 1500
LEAGUE_PRIOR = {"LCK": 1560, "LPL": 1540, "LEC": 1480, "LCS": 1450, "LTA N": 1450,
                "LTA S": 1430, "CBLOL": 1430, "LCP": 1450, "PCS": 1430, "VCS": 1420}
INTERNATIONAL = {"WLDs", "MSI", "FST", "EWC"}
K_DOMESTIC, K_INTL, FORM_N = 20, 32, 10
FEATURES = ["elo_diff", "form_diff", "gd15_diff", "exp_diff", "intl", "playoffs"]


# --------------------------------------------------------------------------------------
# 1. DATA
# --------------------------------------------------------------------------------------
def load_oracle(path, since_year=2023):
    """Oracle's Elixir CSV -> one row per game (blue vs red)."""
    df = pd.read_csv(path, low_memory=False)
    df = df[df["position"].eq("team")].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= since_year]
    if "golddiffat15" not in df:
        df["golddiffat15"] = np.nan
    df = df.drop_duplicates(["gameid", "side"])
    blue = df[df["side"] == "Blue"].set_index("gameid")
    red = df[df["side"] == "Red"].set_index("gameid")
    idx = blue.index.intersection(red.index)
    blue, red = blue.loc[idx], red.loc[idx]
    return pd.DataFrame({
        "date": blue["date"], "league": blue["league"],
        "playoffs": blue["playoffs"].fillna(0).astype(int),
        "blue": blue["teamname"], "red": red["teamname"],
        "blue_win": blue["result"].astype(int),
        "gd15": blue["golddiffat15"].fillna(0.0),
    }).reset_index(drop=True)


def make_demo_games(seed=0, n=3500):
    """Synthetic games from hidden team strengths. Only for testing the pipeline."""
    rng = np.random.default_rng(seed)
    teams = list(REGION)
    skill = {t: rng.normal(0, 1) for t in teams}
    dates = pd.date_range("2024-01-15", "2026-10-01", periods=n)
    rows = []
    for d in dates:
        intl = rng.random() < 0.12
        a, b = rng.choice(teams, 2, replace=False)
        if not intl:
            same = [t for t in teams if REGION[t] == REGION[a] and t != a]
            if not same:
                continue
            b = rng.choice(same)
        p = 1 / (1 + math.exp(-(skill[a] - skill[b]) * 0.8))
        win = int(rng.random() < p)
        rows.append((d, "WLDs" if intl else REGION[a], int(rng.random() < 0.2), a, b, win,
                     (skill[a] - skill[b]) * 300 + rng.normal(0, 800)))
    return pd.DataFrame(rows, columns=["date", "league", "playoffs", "blue", "red", "blue_win", "gd15"])


# --------------------------------------------------------------------------------------
# 2. FEATURES (chronological, pre-game only)
# --------------------------------------------------------------------------------------
def build_features(games):
    games = games.sort_values("date").reset_index(drop=True)
    elo, prior = {}, {}
    form = defaultdict(lambda: deque(maxlen=FORM_N))
    gd = defaultdict(lambda: deque(maxlen=FORM_N))
    n_games = defaultdict(int)
    last_year, rows = None, []

    def snap(t):
        f = np.mean(form[t]) if form[t] else 0.5
        g = np.mean(gd[t]) if gd[t] else 0.0
        return elo[t], f, g, math.log1p(n_games[t])

    for g in games.itertuples(index=False):
        if last_year is not None and g.date.year != last_year:   # new season: regress to prior
            for t in elo:
                elo[t] = 0.75 * elo[t] + 0.25 * prior[t]
        last_year = g.date.year
        for t in (g.blue, g.red):
            if t not in elo:
                prior[t] = LEAGUE_PRIOR.get(g.league, BASE_ELO)
                elo[t] = prior[t]
        eb, fb, gb, xb = snap(g.blue)
        er, fr, gr, xr = snap(g.red)
        intl = int(g.league in INTERNATIONAL)
        rows.append((g.date, eb - er, fb - fr, (gb - gr) / 1000, xb - xr, intl, g.playoffs, g.blue_win))

        e = 1 / (1 + 10 ** (-(eb - er) / 400))
        delta = (K_INTL if intl else K_DOMESTIC) * (g.blue_win - e)
        elo[g.blue] += delta
        elo[g.red] -= delta
        form[g.blue].append(g.blue_win); form[g.red].append(1 - g.blue_win)
        gd[g.blue].append(g.gd15);       gd[g.red].append(-g.gd15)
        n_games[g.blue] += 1;            n_games[g.red] += 1

    X = pd.DataFrame(rows, columns=["date"] + FEATURES + ["y"])
    state = {t: snap(t) for t in elo}
    return X, state


# --------------------------------------------------------------------------------------
# 3. MODEL
# --------------------------------------------------------------------------------------
def make_models():
    return [
        make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                       l2_regularization=1.0, random_state=0),
    ]


def ensemble_proba(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def train_and_evaluate(X):
    X = X.sort_values("date")
    cut = int(len(X) * 0.8)
    tr, te = X.iloc[:cut], X.iloc[cut:]
    models = make_models()
    for m in models:
        m.fit(tr[FEATURES], tr["y"])
    p = ensemble_proba(models, te[FEATURES])
    elo_p = 1 / (1 + 10 ** (-te["elo_diff"] / 400))
    print(f"Time-split test ({len(te)} games)")
    print(f"  coin flip : logloss {math.log(2):.4f}")
    print(f"  Elo only  : logloss {log_loss(te['y'], elo_p):.4f}  acc {accuracy_score(te['y'], elo_p > .5):.3f}")
    print(f"  ML model  : logloss {log_loss(te['y'], p):.4f}  acc {accuracy_score(te['y'], p > .5):.3f}")
    final = make_models()                      # refit on everything for the real predictions
    for m in final:
        m.fit(X[FEATURES], X["y"])
    return final


def pairwise_game_probs(models, teams, state):
    """P[i, j] = P(team i beats team j in one game), side-averaged, Worlds context."""
    def vec(a, b):
        ea, fa, ga, xa = state[a]
        eb, fb, gb, xb = state[b]
        return [ea - eb, fa - fb, (ga - gb) / 1000, xa - xb, 1, 1]
    n = len(teams)
    rows = [vec(teams[i], teams[j]) for i in range(n) for j in range(n)]
    p_blue = ensemble_proba(models, pd.DataFrame(rows, columns=FEATURES)).reshape(n, n)
    P = 0.5 * (p_blue + (1 - p_blue.T))        # average over who is blue / red
    np.fill_diagonal(P, 0.5)
    return P


# --------------------------------------------------------------------------------------
# 4. TOURNAMENT SIMULATION
# --------------------------------------------------------------------------------------
def series_matrix(P, best_of):
    k = best_of // 2 + 1
    out = np.zeros_like(P)
    for j in range(k):
        out += math.comb(k - 1 + j, j) * P ** k * (1 - P) ** j
    return out


def sim_play_in(four, S5, rng):
    """ASSUMED: 4-team double-elim Bo5, UB-final winner meets LB-final winner. Verify the real format."""
    def play(a, b):
        return (a, b) if rng.random() < S5[a, b] else (b, a)
    a, b, c, d = four
    w1, l1 = play(a, b)
    w2, l2 = play(c, d)
    ubw, ubl = play(w1, w2)
    lb1, _ = play(l1, l2)
    lbw, _ = play(ubl, lb1)
    return play(ubw, lbw)[0]


def pair_group(group, played, rng):
    for _ in range(30):
        g = group[:]
        rng.shuffle(g)
        pairs = [(g[i], g[i + 1]) for i in range(0, len(g), 2)]
        if all(frozenset(p) not in played for p in pairs):
            break
    return pairs


def sim_swiss(teams, S1, S3, rng):
    wins = dict.fromkeys(teams, 0)
    losses = dict.fromkeys(teams, 0)
    played = set()
    while True:
        groups = defaultdict(list)
        for t in teams:
            if wins[t] < 3 and losses[t] < 3:
                groups[(wins[t], losses[t])].append(t)
        if not groups:
            break
        for (w, l), grp in groups.items():
            S = S3 if (w == 2 or l == 2) else S1       # Bo3 when a team can qualify/be eliminated
            for a, b in pair_group(grp, played, rng):
                played.add(frozenset((a, b)))
                win, lose = (a, b) if rng.random() < S[a, b] else (b, a)
                wins[win] += 1
                losses[lose] += 1
    qualified = [t for t in teams if wins[t] == 3]
    return qualified, losses


def sim_knockouts(seeds, S5, rng):
    def play(a, b):
        return a if rng.random() < S5[a, b] else b
    qf = [play(seeds[0], seeds[7]), play(seeds[3], seeds[4]),
          play(seeds[1], seeds[6]), play(seeds[2], seeds[5])]
    sf = [play(qf[0], qf[1]), play(qf[2], qf[3])]
    return sf, play(sf[0], sf[1])


def simulate(teams_direct, teams_pi, state, P_all, names, n_sims, seed=42):
    rng = random.Random(seed)
    ix = {t: i for i, t in enumerate(names)}
    S1, S3, S5 = (series_matrix(P_all, b) for b in (1, 3, 5))
    counts = {t: defaultdict(int) for t in names}
    d = [ix[t] for t in teams_direct]
    pi = [ix[t] for t in teams_pi]
    elo = {ix[t]: state[t][0] for t in names}
    for _ in range(n_sims):
        winner_pi = sim_play_in(pi, S5, rng)
        counts[names[winner_pi]]["swiss"] += 1
        swiss_teams = d + [winner_pi]
        for t in d:
            counts[names[t]]["swiss"] += 1
        qualified, losses = sim_swiss(swiss_teams, S1, S3, rng)
        seeds = sorted(qualified, key=lambda t: (losses[t], -elo[t]))
        for t in qualified:
            counts[names[t]]["top8"] += 1
        sf, champ = sim_knockouts(seeds, S5, rng)
        for t in sf:
            counts[names[t]]["final_four"] += 1
        counts[names[champ]]["champion"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Oracle's Elixir CSV")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (pipeline test only)")
    ap.add_argument("--sims", type=int, default=20000)
    args = ap.parse_args()

    if args.demo:
        games = make_demo_games()
        print("!! DEMO MODE: synthetic data, the output below is NOT a real prediction !!\n")
    elif args.csv:
        games = load_oracle(args.csv)
    else:
        ap.error("pass --csv PATH or --demo")

    X, state = build_features(games)
    models = train_and_evaluate(X)

    names = DIRECT_TO_SWISS + PLAY_IN
    missing = [t for t in names if t not in state]
    if missing:
        print(f"\nWARNING: not found in data, using default rating: {missing}")
        for t in missing:
            state[t] = (BASE_ELO - 100, 0.5, 0.0, 0.0)

    P = pairwise_game_probs(models, names, state)
    counts = simulate(DIRECT_TO_SWISS, PLAY_IN, state, P, names, args.sims)

    out = pd.DataFrame(
        {t: {k: counts[t][k] / args.sims for k in ("swiss", "top8", "final_four", "champion")} for t in names}
    ).T.sort_values("champion", ascending=False)
    out.insert(0, "elo", [round(state[t][0]) for t in out.index])
    pd.set_option("display.width", 120)
    print(f"\nWorlds 2026 odds ({args.sims} simulations)")
    print((out.assign(**{c: (out[c] * 100).round(1) for c in ("swiss", "top8", "final_four", "champion")})).to_string())
    out.to_csv("worlds2026_predictions.csv")
    print("\nSaved worlds2026_predictions.csv")


if __name__ == "__main__":
    main()
