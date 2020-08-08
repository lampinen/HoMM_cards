from itertools import permutations

from copy import deepcopy

import numpy as np
import tensorflow as tf

from HoMM import HoMM_model
from HoMM.configs import default_run_config, default_architecture_config
import simple_card_games 

run_config = default_run_config.default_run_config
run_config.update({
    "output_dir": "presentable_results/exact_eval/",

    "game_types": ["high_card","straight_flush",  "match", "pairs_and_high", "sum_under"],
    "option_names": ["suits_rule", "losers", "black_valuable"],
    "suits_rule": [True, False],
    "losers": [True, False],
    "black_valuable": [True, False],

    "metaclass_lesion": False,  # If True, don't train meta-classifications 

    "softmax_beta": 8,

    "bets": [0, 1, 2],
    "new_tasks": [{"game_type": "straight_flush", "losers": True,
                  "black_valuable": False, "suits_rule": False},
                  {"game_type": "straight_flush", "losers": True,
                  "black_valuable": False, "suits_rule": True},
                  {"game_type": "straight_flush", "losers": True,
                  "black_valuable": True, "suits_rule": False},
                  {"game_type": "straight_flush", "losers": True,
                  "black_valuable": True, "suits_rule": True}], # will be removed
                                                                # from base tasks

    "init_learning_rate": 1e-5,
    "init_meta_learning_rate": 1e-5,

    "lr_decay": 0.85,
    "language_lr_decay": 0.8,
    "meta_lr_decay": 0.9,

    "lr_decays_every": 200,
    "min_learning_rate": 3e-8,
    "min_language_learning_rate": 3e-8,
    "min_meta_learning_rate": 3e-8,

    "num_epochs": 50000,
    "eval_every": 20,

    "num_runs": 1,
    "run_offset": 0
})

architecture_config = default_architecture_config.default_architecture_config
architecture_config.update({
   "input_shape": [(4 + 2) * 2],  # one hot card value and suit * 2
   "output_shape": [3],  # 3 possible bets

   "outcome_shape": [3 + 1],  # one-hot bet + reward
   "output_masking": True,

    "IO_num_hidden": 128,
    "optimizer": "RMSProp",

    "meta_batch_size": 768,

    "z_dim": 256,
    "F_num_hidden_layers": 0,

#    "F_weight_normalization": True,
#    "F_wn_strategy": "standard",

    "train_drop_prob": 0.,
})
if False:  # enable for persistent reps
    architecture_config.update({
        "persistent_task_reps": True,
        "combined_emb_guess_weight": "varied",
        "emb_match_loss_weight": 0.2,
    })
    run_config.update({
        "output_dir": run_config["output_dir"][:-1] + "_persistent/", 
    })

if False:  # enable for nonhomiconic, after ensuring HoMM is on correct branch 
    architecture_config.update({
        "nonhomoiconic": True,
    })
    run_config.update({
        "output_dir": run_config["output_dir"] + "nonhomoiconic/",
    })

if False:  # enable for task-conditioned
    architecture_config.update({
        "task_conditioned_not_hyper": True,
        "F_num_hidden_layers": 3,
    })
    run_config.update({
        "output_dir": run_config["output_dir"] + "tcnh_more_complex/", 
    })

if False:  # enable for language baseline
    run_config.update({
        "train_language_base": True,
        "train_base": False,
        "train_meta": False,

        "vocab": ["game", "highcard", "pairsandhigh", "match", "sumunder", "straightflush"] + ["l", "bv", "sr"] + ["0", "1"],

        "output_dir": run_config["output_dir"] + "language/",  # subfolder
    })

    architecture_config.update({
        "max_sentence_len": 8,
    })

if False:  # enable for language base + meta
    run_config.update({
        "train_language_base": True,
        "train_language_meta": True,
        "train_base": False,
        "train_meta": False,

        "vocab": ["game", "highcard", "pairsandhigh", "match", "sumunder", "straightflush"] + ["l", "bv", "sr"] + ["0", "1"] + ["toggle", "is"] + ["<PAD>"],

        "output_dir": run_config["output_dir"] + "language_meta/",  # subfolder
    })

    architecture_config.update({
        "max_sentence_len": 8,
    })

if False:  # enable for meta-class lesion
    run_config.update({
        "metaclass_lesion": True,
        "output_dir": run_config["output_dir"] + "metaclass_lesion/", 
    })


class cards_HoMM_model(HoMM_model.HoMM_model):
    def __init__(self, run_config=None):
        super(cards_HoMM_model, self).__init__(
            architecture_config=architecture_config, run_config=run_config)

    def _pre_build_calls(self):
        run_config = self.run_config

        self.bets = run_config["bets"]

        # set up the base task defs
        base_tasks = [{"game_type": g, "losers": l, "black_valuable": b,
                       "suits_rule": s} for g in run_config["game_types"] for l in run_config["losers"] for b in run_config["black_valuable"] for s in run_config["suits_rule"]]

        self.base_train_tasks = [t for t in base_tasks if t not in run_config["new_tasks"]]
        self.base_eval_tasks = run_config["new_tasks"]


        # set up the meta tasks
        if run_config["metaclass_lesion"]: 
            self.meta_class_train_tasks = []
        else:
            self.meta_class_train_tasks = ["is_" + g for g in run_config["game_types"]] + ["is_" + o for o in run_config["option_names"]] 
        self.meta_class_eval_tasks = [] 

        self.meta_map_train_tasks = ["toggle_" + o for o in run_config["option_names"]] 
        self.meta_map_eval_tasks = [] 

        # set up the meta pairings 
        self.meta_pairings = simple_card_games.get_meta_pairings(
            base_train_tasks=self.base_train_tasks,
            base_eval_tasks=self.base_eval_tasks,
            meta_class_train_tasks=self.meta_class_train_tasks,
            meta_map_train_tasks=self.meta_map_train_tasks) 

        # convert to games

        self.base_train_tasks = [simple_card_games.game_from_def(t) for t in self.base_train_tasks]
        self.base_eval_tasks = [simple_card_games.game_from_def(t) for t in self.base_eval_tasks]

        self.game_str_to_task = {str(t): t for t in self.base_train_tasks + self.base_eval_tasks}

        self.all_possible_hands = np.array([self.encode_hand(h) for h in self.base_train_tasks[0].hands], dtype=np.float32)


    def fill_buffers(self, num_data_points=1024):
        """Add new "experiences" to memory buffers."""
        self.play_games(num_turns=num_data_points,
                        epsilon=self.run_config["epsilon"])

    def get_new_memory_buffer(self):
        """Can be overriden by child"""
        return HoMM_model.memory_buffer(
            length=self.architecture_config["memory_buffer_size"],
            input_width=self.architecture_config["input_shape"][0],
            outcome_width=self.architecture_config["outcome_shape"][0])

    def encode_hand(self, hand):
        """Takes a hand tuple, returns vector appropriate for input to graph"""
        vec = np.zeros(12)
        def _card_to_vec(c):
            vec = np.zeros(6)
            vec[c[0]] = 1.
            vec[c[1] + 4] = 1.
            return vec

        vec[:6] = _card_to_vec(hand[0])
        vec[6:] = _card_to_vec(hand[1])
        return vec

    def decode_hands(self, encoded_hands):
        hands = []
        def _card_from_vec(v):
            return (np.argmax(v[:4]), np.argmax(v[4:]))
        for enc_hand in encoded_hands:
            hand = (_card_from_vec(enc_hand[:6]), _card_from_vec(enc_hand[6:]))
            hands.append(hand)
        return hands

    def encode_outcomes(self, actions, rewards):
        """Takes actions and rewards, returns matrix appropriate for input to
        graph"""
        mat = np.zeros([len(actions), 4])
        mat[range(len(actions)), actions] = 1.
        mat[:, -1] = rewards
        return mat

    def play_hands(self, encoded_hands, memory_buffer,
                   epsilon=0., return_probs=False):
        """Plays the provided hand conditioned on the game and memory buffer,
        with epsilon-greedy exploration."""
        if epsilon == 1.: # makes it easier to fill buffers before play begins
            return np.random.randint(3, size=[len(encoded_hands)])
        input_buff, outcome_buff = memory_buffer.get_memories()
        guess_mask = np.concatenate(
            [np.ones(len(input_buff), dtype=np.bool),
             np.zeros(len(encoded_hands), dtype=np.bool)], axis=0)

        inputs = np.concatenate(
            [input_buff,
             encoded_hands], axis=0)

        outcomes = np.concatenate(
            [output_buff,
             np.zeros([len(encoded_hands), outcome_buff.shape[-1]])], axis=0)

        feed_dict = {
            self.base_input_ph: inputs,
            self.guess_input_mask_ph: guess_mask,
            self.keep_prob_ph: 1.,
            self.base_outcome_ph: outcomes 
        }
        act_probs = self.sess.run(self.base_output_softmax,
                                  feed_dict=feed_dict)[-len(encoded_hands):, :]

        if return_probs: # return raw probs, e.g. for debugging
            return act_probs

        def _action_from_probs(probs, epsilon):
            if np.random.rand() > epsilon:
                return np.random.choice(range(3))
            else:
                return np.random.choice(range(3), p=probs)

        actions = [_action_from_probs(
            act_probs[i, :], epsilon) for i in range(len(act_probs))]
        return actions

    def _outcomes_to_targets(self, encoded_outcomes):
        num = len(encoded_outcomes)
        targets = np.zeros([num, 3])
        mask = encoded_outcomes[:, :3].astype(np.bool)
        targets[mask] = encoded_outcomes[:, 3]
        return targets, mask

    def _pre_loss_calls(self):
        self.base_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_unmasked_output)

        self.base_fed_emb_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_fed_emb_unmasked_output)

        self.base_cached_emb_output_softmax = tf.nn.softmax(
            self.run_config["softmax_beta"] * self.base_cached_emb_unmasked_output)

        if self.run_config["train_language_base"]:
            self.base_lang_output_softmax = tf.nn.softmax(
                self.run_config["softmax_beta"] * self.base_lang_unmasked_output)

    def fill_buffers(self, num_data_points=1024):
        """Add new "experiences" to memory buffers."""
        self.play_games(num_turns = num_data_points,
                        epsilon=1.)  # due to a bug, original results had random play

    def play_games(self, num_turns=1, epsilon=0.):
        """Plays turns games in base_tasks (and new if include_new), to add new
        experiences to memory buffers."""
        this_tasks = self.base_train_tasks + self.base_eval_tasks 
        for t in this_tasks:
            buff = self.memory_buffers[str(t)]
            encoded_hands = np.zeros([num_turns] + self.architecture_config["input_shape"])
            hands = []
            for turn in range(num_turns):
                hand = t.deal()
                hands.append(hand)
                encoded_hands[turn, :] = self.encode_hand(hand)
            acts = self.play_hands(encoded_hands, buff,
                                   epsilon=epsilon)
            bets = [self.bets[a] for a in acts]
            rs = [t.play(h, self.bets[a]) for h, a in zip(hands, acts)]
            encoded_outcomes = self.encode_outcomes(acts, rs)
            buff.insert(encoded_hands, encoded_outcomes)

    def reward_eval_helper(self, game, act_probs, encoded_hands=None, hands=None):
        if encoded_hands is not None:
            hands = self.decode_hands(encoded_hands)
        if isinstance(game, str):
            game = self.game_str_to_task[game]
        actions = [np.argmax(act_probs[i, :],
                             axis=-1) for i in range(len(act_probs))]
        bets = [self.bets[a] for a in actions]
        bet_dict = {h: b for (h, b) in zip(hands, bets)}
        #rs = [game.play(hands[i], self.bets[a]) for i, a in enumerate(actions)]
        return game.compute_expected_return(max_bet=1., policy=bet_dict, ties="nobody_wins") 

    def build_feed_dict(self, task, lr=None, fed_embedding=None,
                        call_type="base_standard_train"):
        """Build a feed dict."""
        feed_dict = super(cards_HoMM_model, self).build_feed_dict(
            task=task, lr=lr, fed_embedding=fed_embedding, call_type=call_type)

        base_or_meta, call_type, train_or_eval = call_type.split("_")

        if base_or_meta == "base":
            outcomes = feed_dict[self.base_target_ph]
            if call_type == "standard" or train_or_eval == "eval" or not self.architecture_config["persistent_task_reps"]: 
                feed_dict[self.base_outcome_ph] = outcomes 
            targets, target_mask = self._outcomes_to_targets(outcomes)
            feed_dict[self.base_target_ph] = targets 
            feed_dict[self.base_target_mask_ph] = target_mask
            if train_or_eval == "eval" and call_type != "standard":  # eval on all possible hands
                feed_dict[self.base_input_ph] = self.all_possible_hands 

        return feed_dict

    def base_eval(self, task, train_or_eval):
        feed_dict = self.build_feed_dict(task, call_type="base_cached_eval")
        fetches = [self.base_cached_emb_output_softmax]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        inputs = feed_dict[self.base_input_ph]
        expected_rewards = self.reward_eval_helper(task, res[0], inputs)
        name = str(task)
        return ([name + "_expected_rewards:" + train_or_eval],
                [expected_rewards]) 

    def base_embedding_eval(self, embedding, task):
        feed_dict = self.build_feed_dict(task, fed_embedding=embedding, call_type="base_fed_eval")
        fetches = [self.base_fed_emb_output_softmax]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        inputs = feed_dict[self.base_input_ph]
        expected_rewards = self.reward_eval_helper(task, res[0], inputs)
        name = str(task)
        return [expected_rewards] 

    def base_language_eval(self, task, train_or_eval):
        feed_dict = self.build_feed_dict(task, call_type="base_lang_eval")
        fetches = [self.base_lang_output_softmax]
        res = self.sess.run(fetches, feed_dict=feed_dict)
        inputs = feed_dict[self.base_input_ph]
        expected_rewards = self.reward_eval_helper(task, res[0], inputs)
        name = str(task)
        return ([name + "_expected_rewards:" + train_or_eval],
                [expected_rewards]) 

    def intify_task(self, task_name):  # note: only base tasks implemented at present
        words = task_name.split("_")
        if words[0] in ["is", "toggle"]:
            this_words = [words[0]]
            if words[1] == "high":
                this_words += ["highcard"]
            elif words[1] == "match":
                this_words += ["match"]
            elif words[1] == "straight":
                this_words += ["straightflush"]
            elif words[1] == "sum":
                this_words += ["sumunder"]
            elif words[1] == "pairs":
                this_words += ["pairsandhigh"]
            elif words[1] == "losers":
                this_words += ["l"]
            elif words[1] == "black":
                this_words += ["bv"]
            elif words[1] == "suits":
                this_words += ["sr"]
            else: 
                raise ValueError("Unrecognized task name: {}, {}".format(task_name, words))

            words = this_words + ["<PAD>"] * (self.architecture_config["max_sentence_len"] - len(this_words))

        else:  # base task
            if words[1] == "high":
                words = ["game", "highcard"] + words[3:]
            elif words[1] == "straight":
                words = ["game", "straightflush"] + words[3:]
            elif words[1] == "sum":
                words = ["game", "sumunder"] + words[3:]
            elif words[1] == "pairs":
                words = ["game", "pairsandhigh"] + words[4:]

        return [self.vocab_dict[x] for x in words]


## stuff
for run_i in range(run_config["run_offset"], run_config["run_offset"]+run_config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    run_config["this_run"] = run_i

    model = cards_HoMM_model(run_config=run_config)
    model.run_training()

    tf.reset_default_graph()
