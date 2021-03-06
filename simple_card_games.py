import numpy as np

from copy import deepcopy


class card_game(object):
    """Class for creating simple card games which have 4 card values and 2 
    suits (red and black), and a ranking of (two card) hands, such that your 
    probability of winning is always proportional to how high in the hand
    ranking your hand is.

    On each turn, the player (agent) bets either 0, 1, or 2, and wins or loses
    that amount if they win or lose the hand (i.e. the reward is +/- the amount
    bet).

    There are several game types:
        high_card: highest card wins
        match: the hand that differs least in value (suit counts as 0.5 pt
            difference) wins.
        pairs_and_high: same as high card, except pairs are more valuable,
            and same suit pairs are even more valuable.
        straight_flush: most valuable is adjacent numbers in same suit, i.e. 4  
            and 3 in most valuable suit wins every time (royal flush)
        sum_under: value is value_sum + 0.25 per suit if value_sum < 5 else 
            negative of this (blackjack) 

    If black_valuable is True, black cards are more valuable, if False red
    cards are.

    If suits_rule is false, suits only break ties on hands with equal values,
        else values only break ties on hands with equal suits.
    
    If losers is True, the game is reversed so that hands that would have won
    instead lose an equal amount, and vice versa."""
    def __init__(self, game_type, black_valuable=True, suits_rule=False,
                 losers=False):
        self.game_type = game_type
        self.black_valuable = black_valuable
        self.suits_rule = suits_rule
        suit_mult = 0.5 if not suits_rule else 1000
        self.losers = losers
        self.cards = [(value, suit) for suit in range(2) for value in range(4)] 
        self.hands = [(c1, c2) for c1 in self.cards for c2 in self.cards]

        def suit_val(card):
            return suit_mult * card[1] if black_valuable else suit_mult * (1-card[1])

        def high_card_value(hand):
            if hand[0][0] == hand[1][0]:
                return 1.1 * hand[0][0] + 0.01*(suit_val(hand[0]) + suit_val(hand[1]))
            else:
                index = 0 if hand[0][0] > hand[1][0] else 1 
                high_card = hand[index]
                other_card = hand[1-index]
                return high_card[0] + 0.1 * other_card[0] + 0.01 * (suit_val(high_card) + 0.5 * suit_val(other_card))
            
            
        if game_type == "high_card":
            key = high_card_value
        elif game_type == "match":
            def key(hand):
                val_d = hand[0][0] - hand[1][0] 
                suit_d = hand[0][1] - hand[1][1] 
                index = 0 if hand[0][0] > hand[1][0] else 1 
                high_card = hand[index]
                other_card = hand[1-index]
                tie_break = high_card[0] + 0.2 * other_card[0] + 0.01 *(suit_val(high_card) + 0.5 * suit_val(other_card))
                return -np.abs(val_d) - 0.5 * np.abs(suit_d) + 0.01 * tie_break 
        elif game_type == "pairs_and_high": 
            def key(hand):
                c0, c1 = hand
                if c0[0] == c1[0]:
                    return 10 + c0[0] + 0.5 * (c0[1] == c1[1])  + 0.01 * (suit_val(c0) + suit_val(c1))
                else:
                    return high_card_value(hand) 
        elif game_type == "straight_flush":
            def key(hand):
                c0, c1 = hand
                if np.abs(c0[0]-c1[0]) == 1:
                    index = 0 if hand[0][0] > hand[1][0] else 1 
                    high_card = hand[index]
                    other_card = hand[1-index]
                    return 10 + 10 * (c0[1] == c1[1]) + 2 * high_card[0] + suit_val(high_card) + 0.01 * suit_val(other_card) 
                else:
                    return high_card_value(hand) 
        elif game_type == "sum_under":
            def key(hand):
                c0, c1 = hand
                value_sum = c0[0] + c1[0]
                if value_sum < 5:
                    return value_sum + 0.5 * (suit_val(c0) + suit_val(c1))  
                else:
                    return -value_sum + 0.5 * (suit_val(c0) + suit_val(c1))  
        else:
            raise ValueError("Unrecognized game type: " + game_type)
                                                

        if losers:
            self.sort_key = lambda x: -key(x) 
        else:
            self.sort_key = key
        self.hands.sort(key=self.sort_key)
        self.num_hands = num_hands = len(self.hands)
        self.hand_to_win_prob = {}

        for i, hand in enumerate(self.hands):
            # j handles tied values
            this_val = self.sort_key(hand)
            for j in range(i+1, self.num_hands):
                if self.sort_key(self.hands[j]) > this_val:
                    self.hand_to_win_prob[hand] =  (float(j))/num_hands
                    break
            else:
                self.hand_to_win_prob[hand] = 1.0 

    def wins(self, hand):
        wp = self.hand_to_win_prob[hand]
        return np.random.rand() <= wp

    def deal(self):
        return self.hands[np.random.randint(self.num_hands)]

    def play(self, hand, bet):
        if self.wins(hand):
            return bet 
        else:
            return -bet

    def compute_expected_return(self, max_bet=2., policy=None, ties="win"):
        """If policy is None, computes the optimal expected return. If not,
        policy should be a dictionary indexed by hands that gives bets."""
        r = 0.
        num_hands = len(self.hands)
        if policy is None:
            for hand in self.hands:
                wp = self.hand_to_win_prob[hand]
                if wp > 0.5:
                    if ties == "win":
                        r += wp*max_bet - (1-wp)*max_bet
                    elif hand[0] == hand[1]: # don't win against self 
                        r += (wp - 1./num_hands)*max_bet - (1-wp)*max_bet
                    else: # don't win against self or other tie
                        r += (wp - 2./num_hands)*max_bet - (1-wp)*max_bet

        else:
            for hand in self.hands:
                bet = policy[hand]
                wp = self.hand_to_win_prob[hand]
                r += wp*bet - (1-wp)*bet

        r /= len(self.hands)
        return r

    def __str__(self):
        return "game_%s_l_%i_bv_%i_sr_%i" % (self.game_type, self.losers,
                                             self.black_valuable, self.suits_rule)


def stringify_game_def(game_def):
        return "game_%s_l_%i_bv_%i_sr_%i" % (game_def["game_type"], game_def["losers"],
                                             game_def["black_valuable"], game_def["suits_rule"])


def game_from_def(game_def):
    g = card_game(game_type=game_def["game_type"],
                  black_valuable=game_def["black_valuable"],
                  suits_rule=game_def["suits_rule"],
                  losers=game_def["losers"])
    return g


def get_meta_pairings(base_train_tasks, base_eval_tasks, meta_class_train_tasks, meta_map_train_tasks):
    """Gets which tasks map to which other tasks under the meta_tasks (i.e. the
    part of the meta datasets which is precomputable)"""
    all_meta_tasks = meta_class_train_tasks + meta_map_train_tasks
    meta_pairings = {mt: {"train": [], "eval": []} for mt in all_meta_tasks}
    for mt in all_meta_tasks:
        if mt[:6] == "toggle":
            to_toggle = mt[7:]
            for task in base_train_tasks:
                other = deepcopy(task)
                other[to_toggle] = not other[to_toggle]
                if other in base_train_tasks:
                    meta_pairings[mt]["train"].append((stringify_game_def(task),
                                                       stringify_game_def(other)))
                elif other in base_eval_tasks:
                    meta_pairings[mt]["eval"].append((stringify_game_def(task),
                                                       stringify_game_def(other)))

        elif mt[:2] == "is":
            pos_class = mt[3:]
            for task in base_train_tasks:
                truth_val = (task["game_type"] == pos_class) or (pos_class in task and task[pos_class])
                meta_pairings[mt]["train"].append((stringify_game_def(task),
                                                  1*truth_val))
            for task in base_eval_tasks:
                truth_val = (task["game_type"] == pos_class) or (pos_class in task and task[pos_class])
                meta_pairings[mt]["eval"].append((stringify_game_def(task),
                                                  1*truth_val))

    return meta_pairings

        
### simple tests
if __name__ == "__main__":
    # save win probs for javascript
    for g in ["straight_flush"]:
        for losers in [True, False]:
            this_game = card_game(game_type=g, losers=losers)
            with open("../../psych/cards/%s_l%i.json" % (g, losers), "w") as fout:
                fout.write("{\n")
                for k, v in sorted(this_game.hand_to_win_prob.items(), key= lambda x: x[1]): 
                    hand = "h_%i_%i_%i_%i" % (k[0][0], k[0][1], k[1][0], k[1][1])
                    fout.write(hand + ": " + str(v) + ",\n")
                fout.write("}")
    exit()

    game_types = ["high_card", "match", "pairs_and_high", "straight_flush", "sum_under"]
    suits_rule = [True, False]
    losers = [True, False]
    black_valuable = [True, False]

    print("testing each game")
    for g in game_types:
        print(g)
        this_game = card_game(game_type=g)
        print(this_game.hands)
        print(sorted(this_game.hand_to_win_prob.items(), key= lambda x: x[1]))
        print(this_game.compute_expected_return())
        
    print("testing methods")
    this_game = card_game(game_type="high_card")
    for _ in range(5):
        this_hand = this_game.deal()
        print(this_hand)
        print(this_game.hand_to_win_prob[this_hand])
        print(this_game.play(this_hand, 2))

    print("testing suits_rule")
    this_game = card_game(game_type="high_card", suits_rule=True)
    print(sorted(this_game.hand_to_win_prob.items(), key= lambda x: x[1]))
    print(this_game.compute_expected_return())

    print("testing losers")
    this_game = card_game(game_type="high_card", losers=True)
    print(sorted(this_game.hand_to_win_prob.items(), key= lambda x: x[1]))
    print(this_game.compute_expected_return())

    print("testing black_valuable=False")
    this_game = card_game(game_type="high_card", black_valuable=False)
    print(sorted(this_game.hand_to_win_prob.items(), key= lambda x: x[1]))
    print(this_game.compute_expected_return())

    print("all 3")
    this_game = card_game(game_type="high_card", black_valuable=False,
                          suits_rule=True, losers=True)
    print(sorted(this_game.hand_to_win_prob.items(), key= lambda x: x[1]))
    print(this_game.compute_expected_return())

    print("all 3 on sum under")
    this_game = card_game(game_type="sum_under", black_valuable=False,
                          suits_rule=True, losers=True)
    print(sorted(this_game.hand_to_win_prob.items(), key= lambda x: x[1]))
    print(this_game.compute_expected_return())
