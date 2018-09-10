import numpy as np

class card(object):
    pass

class hand(object):
    pass

class card_game(object):
    """Class for creating simple card games which have 4 card values and 2 
    suits, and a ranking of (two card) hands, such that your probability of
    winning is always proportional to how high in the hand ranking your hand is.

    On each turn, the player (agent) bets either 0, 1, or 2, and wins or loses
    that amount if the win/lose the hand (i.e. the reward is +/- the amount
    bet.

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
                 losers=False)
        self.black_valuable = black_valuable
        self.suits_rule = suits_rule
        self.losers = losers
        self.cards = [(suit, value) for suit in range(2) for value in range(4)] 
        self.hands = [(c1, c2) for c1 in cards for c2 in cards]

        if suits_rule:
            raise NotImplementedError("Come on dude")

        def suit_val(card):
            return 0.5 * card[1] if black_valuable else  0.5 * (1-card[1])
            
        if game_type == "high":
            def key(hand):
                index = np.argmax([c[0] for c in hand])
                high_card = c[index]
                return high_card[0] + suit_val(high_card) 
        elif game_type == "match":
            def key(hand):
                val_d = hand[0][0] - hand[1][0] 
                suit_d = hand[0][1] - hand[1][1] 
                return val_d + 0.5 * suit_d 
        elif game_type == "pairs_and_high":
            def key(hand):
                c0, c1 = hand
                if c0[0] == c1[0]:
                    return 4 + 4 * (c0[1] == c1[1]) + c0[0] + 0.5 * (suit_val(c0) + suit_val(c1)) 
 
                else:
                    index = np.argmax([c[0] for c in hand])
                    high_card = c[index]
                    return high_card[0] + 0.5 *suit_val(high_card) 
        elif game_type == "straight_flush":
            def key(hand):
                c0, c1 = hand
                index = np.argmax([c[0] for c in hand])
                high_card = c[index]
                if np.abs(c0[0]-c1[0]) == 1:
                    return 4 + 4 * (c0[1] == c1[1]) + high_card[0] + suit_val(high_card) 
                else:
                    return high_card[0] + 0.5 *suit_val(high_card) 
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
                                                
        self.sory_key = key
        self.hands.sort(key=self.sort_key)
        num_hands = len(self.hands)
        self.hand_to_win_prob = {hand: (float(i)+1)/num_hands for i, hand in enumerate(self.hands)}


    def wins(self, hand):
        wp = self.hand_to_win_prob[hand]
        if np.random.rand() <= wp:
            return not self.losers
        else:
            return self.losers


    def deal(self):
        return np.random.choice(self.hands)


    def play(self, hand, bet):
        if self.wins(hand):
            return bet 
        else:
            return -bet
            
        


