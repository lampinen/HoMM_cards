import numpy as np
from simple_card_games import card_game, _stringify_game 

game_types = ["high_card", "match", "pairs_and_high", "straight_flush", "sum_under"]
suits_rule = [True, False]
losers = [True, False]
black_valuable = [True, False]
max_bet = 2.

with open("baseline_data.csv", "w") as fout:
    fout.write("game, policy, expected_reward\n")
    for game_type in game_types:
        for sr in suits_rule:
            for l in losers:
                for bv in black_valuable:
                    g = card_game(game_type, bv, sr, l)
                    name = _stringify_game(g)
                    fout.write("%s, %s, %f\n" % (name,
                                                 "optimal",
                                                 g.compute_expected_return())) 

                    for sr2 in suits_rule:
                        for l2 in losers:
                            for bv2 in black_valuable:
                                if sr2 == sr and l2 == l and bv2 == bv:
                                    continue
                                g2 = card_game(game_type, bv2, sr2, l2)
                                name2 = _stringify_game(g2)
                                g2_policy = {hand: max_bet * (g2.hand_to_win_prob[hand] > 0.5) for hand in g2.hands}
                                fout.write("%s, %s, %f\n" % (
                                    name,
                                    name2,
                                    g.compute_expected_return(policy=g2_policy))) 
