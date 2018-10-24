from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from copy import deepcopy

from simple_card_games import card_game
from orthogonal_matrices import random_orthogonal

pi = np.pi
### Parameters #################################################
config = {
    "run_offset": 0,
    "num_runs": 10,
    "game_types": ["high_card","straight_flush",  "match", "pairs_and_high", "sum_under"],
    "option_names": ["suits_rule", "losers", "black_valuable"],
    "suits_rule": [True, False],
    "losers": [True, False],
    "black_valuable": [True, False],

    "bets": [0, 1, 2],

    "num_input": (4 + 2) * 2 + (5 + 2 + 2 + 2), # (4 values + 2 suits) * 2 cards
                                             # + 5 games + 3 binary options
    "num_output": 3, # bet 0, bet 1, bet 2 
    "num_outcome": (3 + 1) + (5 + 2 + 2 + 2), # 3 possible bets (actions) + reward
                                           # + 5 games + 3 binary options
    "game_hints_on_examples": False, # if true, provides game labels on input,
                                     # else replaced with zeros

    "num_hidden": 64,
    "num_hidden_hyper": 256,

    "epsilon": 0.5,
    "init_learning_rate": 1e-4,
    "init_meta_learning_rate": 1e-4,

    "new_init_learning_rate": 1e-6,
    "new_init_meta_learning_rate": 1e-6,

    "lr_decay": 0.85,
    "meta_lr_decay": 0.9,

    "lr_decays_every": 100,
    "min_learning_rate": 1e-7,
    "min_meta_learning_rate": 1e-6,

    "refresh_meta_cache_every": 1, # how many epochs between updates to meta_cache
    "refresh_mem_buffs_every": 50, # how many epochs between updates to buffers

    "max_base_epochs": 40000,
    "max_new_epochs": 1000,
    "num_task_hidden_layers": 3,
    "num_hyper_hidden_layers": 3,
    "train_drop_prob": 0.15, # dropout probability, applied on meta and hyper
                             # but NOT task or input/output at present. Note
                             # that because of multiplicative effects and depth
                             # impact can be dramatic.

    "softmax_beta": 5, # 1/temperature on action softmax, sharpens if > 1
    "task_weight_weight_mult": 1., # not a typo, the init range of the final
                                   # hyper weights that generate the task
                                   # parameters. 

    "output_dir": "/mnt/fs2/lampinen/meta_RL/results_h256_f64_dp15/",
    "save_every": 20, 
    "eval_all_hands": False, # whether to save guess probs on each hand & each game

    "memory_buffer_size": 1024, # How many memories of each task are stored
    "meta_batch_size": 768, # how many meta-learner sees
    "early_stopping_thresh": 0.05,
    "new_tasks": [{"game": "straight_flush", "losers": True,
                  "black_valuable": False, "suits_rule": False},
		  {"game": "straight_flush", "losers": True,
                  "black_valuable": True, "suits_rule": False},
		  {"game": "straight_flush", "losers": True,
                  "black_valuable": False, "suits_rule": True},
		  {"game": "straight_flush", "losers": True,
                  "black_valuable": True, "suits_rule": True}], # will be removed
                                                                # from base tasks

    "new_meta_tasks": [],

    "internal_nonlinearity": tf.nn.leaky_relu,
    "output_nonlinearity": None
}


config["base_meta_tasks"] = ["is_" + g for g in config["game_types"]] + ["is_" + o for o in config["option_names"]]
config["base_meta_mappings"] = ["toggle_" + o for o in config["option_names"]]
config["base_tasks"] = [{"game": g, "losers": l, "black_valuable": b,
                         "suits_rule": s} for g in config["game_types"] for l in config["losers"] for b in config["black_valuable"] for s in config["suits_rule"]]
config["base_tasks"] = [t for t in config["base_tasks"] if t not in config["new_tasks"]] # omit new

### END PARAMATERS (finally) ##################################

def _stringify_game(t):
    """Helper for printing, etc."""
    return "game_%s_l_%i_bv_%i_sr_%i" % (t["game"], t["losers"],
                                         t["black_valuable"], t["suits_rule"])


def _save_config(filename, config):
    with open(filename, "w") as fout:
        fout.write("key, value\n")
        for key, value in config.items():
            fout.write(key + ", " + str(value) + "\n")

var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')

#total_tasks = set(base_tasks + new_tasks)

def _get_meta_pairings(base_tasks, meta_tasks, meta_mappings):
    """Gets which tasks map to which other tasks under the meta_tasks (i.e. the
    part of the meta datasets which is precomputable)"""
    all_meta_tasks = meta_tasks + meta_mappings
    meta_pairings = {mt: {"base": [], "meta": []} for mt in all_meta_tasks}
    for mt in all_meta_tasks:
        if mt[:6] == "toggle":
            to_toggle = mt[7:]
            for task in base_tasks: 
                other = deepcopy(task) 
                other[to_toggle] = not other[to_toggle]
                if other in base_tasks:
                    meta_pairings[mt]["base"].append((_stringify_game(task),
                                                      _stringify_game(other)))

        elif mt[:2] == "is":
            pos_class = mt[3:]
            for task in base_tasks: 
                truth_val = (task["game"] == pos_class) or (pos_class in task and task[pos_class])
                meta_pairings[mt]["base"].append((_stringify_game(task),
                                                  1*truth_val))
        else: 
            raise ValueError("Unknown meta task: %s" % meta_task)

    return meta_pairings


class memory_buffer(object):
    """Essentially a wrapper around numpy arrays that handles inserting and
    removing."""
    def __init__(self, length, input_width, outcome_width):
        self.length = length 
        self.curr_index = 0 
        self.input_buffer = np.zeros([length, input_width])
        self.outcome_buffer = np.zeros([length, outcome_width])

    def insert(self, input_mat, outcome_mat):
        num_events = len(input_mat)
        if num_events > self.length:
            num_events = length
            self.input_buffer = input_mat[-length:, :]
            self.outcome_buffer = outcome_mat[-length:, :]
            self.curr_index = 0.
            return
        end_offset = num_events + self.curr_index
        if end_offset > self.length: 
            back_off = self.length - end_offset
            num_to_end = num_events + back_off
            self.input_buffer[:-back_off, :] = input_mat[num_to_end:, :] 
            self.outcome_buffer[:-back_off, :] = outcome_mat[num_to_end:, :] 
        else: 
            back_off = end_offset
            num_to_end = num_events
        self.input_buffer[self.curr_index:back_off, :] = input_mat[:num_to_end, :] 
        self.outcome_buffer[self.curr_index:back_off, :] = outcome_mat[:num_to_end, :] 
        self.curr_index = np.abs(back_off)

    def get_memories(self): 
        return self.input_buffer, self.outcome_buffer


class meta_model(object):
    """A meta-learning model for RL on simple card games."""
    def __init__(self, config):
        """args:
            config: a config dict, see above
        """
        self.config = config
        self.memory_buffer_size = config["memory_buffer_size"]
        self.meta_batch_size = config["meta_batch_size"]
        self.game_types = config["game_types"]
        self.num_input = config["num_input"]
        self.num_output = config["num_output"]
        self.num_outcome = config["num_outcome"]
        self.bets = config["bets"]
        self.game_hints_on_examples = config["game_hints_on_examples"]
        self.tkp = 1. - config["train_drop_prob"] # drop prob -> keep prob

        base_tasks = config["base_tasks"]
        base_meta_tasks = config["base_meta_tasks"]
        base_meta_mappings = config["base_meta_mappings"]

        new_tasks = config["new_tasks"]
        new_meta_tasks = config["new_meta_tasks"]

        # base datasets / memory_buffers
        self.base_tasks = base_tasks
        self.base_task_names = [_stringify_game(t) for t in base_tasks]

        # new datasets / memory_buffers
        self.new_tasks = new_tasks
        self.new_task_names = [_stringify_game(t) for t in new_tasks]

        self.all_base_tasks = self.base_tasks + self.new_tasks
        self.memory_buffers = {_stringify_game(t): memory_buffer(
            self.memory_buffer_size, self.num_input,
            self.num_outcome) for t in self.all_base_tasks}

        self.games = {_stringify_game(t): card_game(game_type=t["game"],
                          black_valuable=t["black_valuable"],
                          suits_rule=t["suits_rule"],
                          losers=t["losers"]) for t in self.all_base_tasks}

        self.base_meta_tasks = base_meta_tasks 
        self.base_meta_mappings = base_meta_mappings
        self.all_base_meta_tasks = base_meta_tasks + base_meta_mappings
        self.new_meta_tasks = new_meta_tasks 
        self.all_meta_tasks = self.all_base_meta_tasks + self.new_meta_tasks
        self.meta_dataset_cache = {t: {} for t in self.all_meta_tasks} # will cache datasets for a certain number of epochs
                                                                       # both to speed training and to keep training targets
                                                                       # consistent

        self.all_initial_tasks = self.base_tasks + self.all_base_meta_tasks 
        self.all_new_tasks = self.new_tasks + self.new_meta_tasks
        self.all_tasks = self.all_initial_tasks + self.all_new_tasks
        # think that's enough redundant variables?
        self.num_tasks = num_tasks = len(self.all_tasks)

        self.meta_pairings_base = _get_meta_pairings(
            self.base_tasks, self.base_meta_tasks, self.base_meta_mappings)

        self.meta_pairings_full = _get_meta_pairings(
            self.base_tasks + self.new_tasks,
            self.base_meta_tasks + self.new_meta_tasks,
            self.base_meta_mappings)

        # network

        # base task input
        input_size = config["num_input"] 
        outcome_size = config["num_outcome"]
        output_size = config["num_output"]
        
        self.base_input_ph = tf.placeholder(
            tf.float32, shape=[None, input_size])
        self.base_outcome_ph = tf.placeholder(
            tf.float32, shape=[None, outcome_size])
        self.base_target_ph = tf.placeholder(
            tf.float32, shape=[None, output_size])

        self.lr_ph = tf.placeholder(tf.float32)
        self.keep_prob_ph = tf.placeholder(tf.float32) # dropout keep prob

        num_hidden = config["num_hidden"] 
        num_hidden_hyper = config["num_hidden_hyper"] 
        internal_nonlinearity = config["internal_nonlinearity"]
        output_nonlinearity = config["output_nonlinearity"]
        input_processing_1 = slim.fully_connected(self.base_input_ph, num_hidden, 
                                                  activation_fn=internal_nonlinearity) 

        input_processing_2 = slim.fully_connected(input_processing_1, num_hidden, 
                                                  activation_fn=internal_nonlinearity) 

        processed_input = slim.fully_connected(input_processing_2, num_hidden_hyper, 
                                               activation_fn=internal_nonlinearity) 
        self.processed_input = processed_input

        all_target_processor_nontf = random_orthogonal(num_hidden_hyper)[:, :output_size + 1]
        self.target_processor_nontf = all_target_processor_nontf[:, :output_size]
        self.target_processor = tf.get_variable('target_processor', 
                                                shape=[num_hidden_hyper, output_size],
                                                initializer=tf.constant_initializer(self.target_processor_nontf))
        processed_targets = tf.matmul(self.base_target_ph, tf.transpose(self.target_processor)) 

        def _output_mapping(X):
            """hidden space mapped back to T/F output logits"""
            res = tf.matmul(X, self.target_processor)
            return res

        def _outcome_encoder(outcome_ph, reuse=True):
            """Outcomes mapped to hidden space"""
            with tf.variable_scope('outcome_encoder', reuse=reuse):
                oh_1 = slim.fully_connected(outcome_ph, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                oh_2 = slim.fully_connected(oh_1, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
                res = slim.fully_connected(oh_2, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity)
            return res

        processed_outcomes = _outcome_encoder(self.base_outcome_ph, reuse=False)
        self.processed_outcomes = processed_outcomes

        # meta task input
        self.meta_input_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.meta_target_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.meta_class_ph = tf.placeholder(tf.float32, shape=[None, 1]) 
        # last is for meta classification tasks

        self.class_processor_nontf = all_target_processor_nontf[:, output_size:]
        self.class_processor = tf.constant(self.class_processor_nontf, dtype=tf.float32)
        processed_class = tf.matmul(self.meta_class_ph, tf.transpose(self.class_processor)) 

#        processed_input = tf.cond(self.is_base_input,
#            lambda: processed_input,
#            lambda: self.meta_input_ph)
#        self.processed_input = processed_input
#        processed_targets = tf.cond(self.is_base_output,
#            lambda: processed_targets,
#            lambda: self.meta_target_ph)
#        self.processed_targets = processed_targets
        
        # function embedding "guessing" network / meta network
        # {(emb_in, emb_out), ...} -> emb
        self.guess_input_mask_ph = tf.placeholder(tf.bool, shape=[None]) # which datapoints get excluded from the guess

        def _meta_network(embedded_inputs, embedded_targets,
                           mask_ph=self.guess_input_mask_ph, reuse=True):
            with tf.variable_scope('meta', reuse=reuse):
                guess_input = tf.concat([embedded_inputs,
                                         embedded_targets], axis=-1)
                guess_input = tf.boolean_mask(guess_input,
                                              self.guess_input_mask_ph)
                guess_input = tf.nn.dropout(guess_input, self.keep_prob_ph)

                gh_1 = slim.fully_connected(guess_input, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity) 
                gh_1 = tf.nn.dropout(gh_1, self.keep_prob_ph)
                gh_2 = slim.fully_connected(gh_1, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity) 
                gh_2 = tf.nn.dropout(gh_2, self.keep_prob_ph)
                gh_2b = tf.reduce_max(gh_2, axis=0, keep_dims=True)
                gh_3 = slim.fully_connected(gh_2b, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity) 
                gh_3 = tf.nn.dropout(gh_3, self.keep_prob_ph)

                guess_embedding = slim.fully_connected(gh_3, num_hidden_hyper,
                                                       activation_fn=None)
                guess_embedding = tf.nn.dropout(guess_embedding, self.keep_prob_ph)
                return guess_embedding

        self.guess_base_function_emb = _meta_network(processed_input,
                                                     processed_outcomes,
                                                     reuse=False)

        self.guess_meta_t_function_emb = _meta_network(self.meta_input_ph,
                                                       processed_class)

        self.guess_meta_m_function_emb = _meta_network(self.meta_input_ph,
                                                       self.meta_target_ph)


        # hyper_network: emb -> (f: emb -> emb) 
        self.feed_embedding_ph = tf.placeholder(np.float32,
                                                [1, num_hidden_hyper])

#        self.embedding_is_fed = tf.placeholder_with_default(False, [])
#        self.function_embedding = tf.cond(self.embedding_is_fed, 
#                                          lambda: self.feed_embedding_ph,
#                                          lambda: self.guess_function_embedding)

        num_task_hidden_layers = config["num_task_hidden_layers"]

        tw_range = config["task_weight_weight_mult"]/np.sqrt(
            num_hidden * num_hidden_hyper) # yields a very very roughly O(1) map
        task_weight_gen_init = tf.random_uniform_initializer(-tw_range,
                                                             tw_range)
        
        def _hyper_network(function_embedding, reuse=True):
            with tf.variable_scope('hyper', reuse=reuse):
                hyper_hidden = function_embedding
                for _ in range(config["num_hyper_hidden_layers"]):
                    hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                        activation_fn=internal_nonlinearity)
                    hyper_hidden = tf.nn.dropout(hyper_hidden, self.keep_prob_ph)
                
                hidden_weights = []
                hidden_biases = []

                task_weights = slim.fully_connected(hyper_hidden, num_hidden*(num_hidden_hyper +(num_task_hidden_layers-1)*num_hidden + num_hidden_hyper),
                                                    activation_fn=None,
                                                    weights_initializer=task_weight_gen_init)
                task_weights = tf.nn.dropout(task_weights, self.keep_prob_ph)

                task_weights = tf.reshape(task_weights, [-1, num_hidden, (num_hidden_hyper + (num_task_hidden_layers-1)*num_hidden + num_hidden_hyper)]) 
                task_biases = slim.fully_connected(hyper_hidden, num_task_hidden_layers * num_hidden + num_hidden_hyper,
                                                   activation_fn=None)

                Wi = tf.transpose(task_weights[:, :, :num_hidden_hyper], perm=[0, 2, 1])
                bi = task_biases[:, :num_hidden]
                hidden_weights.append(Wi)
                hidden_biases.append(bi)
                for i in range(1, num_task_hidden_layers):
                    Wi = tf.transpose(task_weights[:, :, input_size+(i-1)*num_hidden:input_size+i*num_hidden], perm=[0, 2, 1])
                    bi = task_biases[:, num_hidden*i:num_hidden*(i+1)]
                    hidden_weights.append(Wi)
                    hidden_biases.append(bi)
                Wfinal = task_weights[:, :, -num_hidden_hyper:]
                bfinal = task_biases[:, -num_hidden_hyper:]

                for i in range(num_task_hidden_layers):
                    hidden_weights[i] = tf.squeeze(hidden_weights[i], axis=0)
                    hidden_biases[i] = tf.squeeze(hidden_biases[i], axis=0)

                Wfinal = tf.squeeze(Wfinal, axis=0)
                bfinal = tf.squeeze(bfinal, axis=0)
                hidden_weights.append(Wfinal)
                hidden_biases.append(bfinal)
                return hidden_weights, hidden_biases

        self.base_task_params = _hyper_network(self.guess_base_function_emb,
                                               reuse=False)
        self.meta_t_task_params = _hyper_network(self.guess_meta_t_function_emb)
        self.meta_m_task_params = _hyper_network(self.guess_meta_m_function_emb)
        self.fed_emb_task_params = _hyper_network(self.feed_embedding_ph)

        # task network
        def _task_network(task_params, processed_input):
            hweights, hbiases = task_params
            task_hidden = processed_input 
            for i in range(num_task_hidden_layers):
                task_hidden = internal_nonlinearity(
                    tf.matmul(task_hidden, hweights[i]) + hbiases[i])

            raw_output = tf.matmul(task_hidden, hweights[-1]) + hbiases[-1]

            return raw_output

        self.base_raw_output = _task_network(self.base_task_params,
                                             processed_input)
        self.base_output = _output_mapping(self.base_raw_output)
        self.base_output_softmax = tf.nn.softmax(
            config["softmax_beta"] * self.base_output)

        self.base_raw_output_fed_emb = _task_network(self.fed_emb_task_params,
                                                     processed_input)
        self.base_output_fed_emb = _output_mapping(self.base_raw_output_fed_emb)
        self.base_output_fed_emb_softmax = tf.nn.softmax(
            config["softmax_beta"] * self.base_output_fed_emb)

        self.meta_t_raw_output = _task_network(self.meta_t_task_params,
                                               self.meta_input_ph)
        self.meta_t_output = tf.nn.sigmoid(self.meta_t_raw_output)

        self.meta_m_output = _task_network(self.meta_m_task_params,
                                               self.meta_input_ph)

        # have to mask base output because can only learn about the action 
        # actually taken
        self.base_target_mask_ph = tf.placeholder(
            tf.bool, shape=[None, output_size])
        masked_base_output = tf.boolean_mask(self.base_output,
                                             self.base_target_mask_ph)
        masked_base_fed_emb_output = tf.boolean_mask(self.base_output_fed_emb,
                                                     self.base_target_mask_ph)
        masked_base_target = tf.boolean_mask(self.base_target_ph,
                                             self.base_target_mask_ph)

        self.base_loss = tf.square(masked_base_output - masked_base_target)
        self.total_base_loss = tf.reduce_mean(self.base_loss)

        self.base_fed_emb_loss = tf.square(
            masked_base_fed_emb_output - masked_base_target)
        self.total_base_fed_emb_loss = tf.reduce_mean(self.base_fed_emb_loss)

        self.meta_t_loss = tf.reduce_sum(
            tf.square(self.meta_t_output - processed_class), axis=1)
        self.total_meta_t_loss = tf.reduce_mean(self.meta_t_loss)

        self.meta_m_loss = tf.reduce_sum(
            tf.square(self.meta_m_output - self.meta_target_ph), axis=1)
        self.total_meta_m_loss = tf.reduce_mean(self.meta_m_loss)


        optimizer = tf.train.RMSPropOptimizer(self.lr_ph)

        self.base_train = optimizer.minimize(self.total_base_loss)
        self.meta_t_train = optimizer.minimize(self.total_meta_t_loss)
        self.meta_m_train = optimizer.minimize(self.total_meta_m_loss)

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.play_games(num_turns=config["memory_buffer_size"],
                        include_new=True,
                        epsilon=1)

        self.refresh_meta_dataset_cache()


    def encode_game(self, task):
        """Takes a task dict, returns vector appropriate for input to graph."""
        vec = np.zeros(11)
        if not self.game_hints_on_examples: # no hints for you!
            return vec
        game_type = t["game"]
        black_valuable = t["black_valuable"]
        suits_rule = t["suits_rule"]
        losers = t["losers"]
        vec[self.game_types.index(game_type)] = 1. 
        if black_valuable:
            vec[5] == 1.
        else:
            vec[6] == 1.
        if suits_rule:
            vec[7] == 1.
        else:
            vec[8] == 1.
        if losers:
            vec[9] == 1.
        else:
            vec[10] == 1.

        return vec


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


    def play_hands(self, encoded_hands, encoded_games, memory_buffer,
                   epsilon=0., return_probs=False):
        """Plays the provided hand conditioned on the game and memory buffer,
        with epsilon-greedy exploration."""
        if epsilon == 1.: # makes it easier to fill buffers before play begins
            return np.random.randint(3, size=[len(encoded_hands)])
        input_buff, outcome_buff = memory_buffer.get_memories()
        feed_dict = {
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: np.ones(len(input_buff), dtype=np.bool),
            self.keep_prob_ph: 1.,
            self.base_outcome_ph: outcome_buff
        }
        act_probs = self.sess.run(self.base_output_softmax,
                                  feed_dict=feed_dict)
        
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
        

    def play_games(self, num_turns=1, include_new=False, epsilon=0.):
        """Plays turns games in base_tasks (and new if include_new), to add new
        experiences to memory buffers."""
        if include_new:
            this_tasks = self.all_base_tasks
        else: 
            this_tasks = self.base_tasks
        for t in this_tasks:
            game = self.games[_stringify_game(t)]
            encoded_game = self.encode_game(t)
            buff = self.memory_buffers[_stringify_game(t)]
            encoded_games = np.tile(encoded_game, [num_turns, 1])
            encoded_hands = np.zeros([num_turns, 12])
            hands = []
            for turn in range(num_turns):
                hand = game.deal()
                hands.append(hand)
                encoded_hands[turn, :] = self.encode_hand(hand)
            acts = self.play_hands(encoded_hands, encoded_games, buff,
                                   epsilon=epsilon) 
            bets = [self.bets[a] for a in acts] 
            rs = [game.play(h, self.bets[a]) for h, a in zip(hands, acts)]
            encoded_outcomes = self.encode_outcomes(acts, rs)
            encoded_hands = np.concatenate([encoded_hands, encoded_games], axis=-1)
            encoded_outcomes = np.concatenate([encoded_outcomes, encoded_games], axis=-1)
            buff.insert(encoded_hands, encoded_outcomes)


    def _random_guess_mask(self, dataset_length):
        mask = np.zeros(dataset_length, dtype=np.bool)
        indices = np.random.permutation(dataset_length)[:config["meta_batch_size"]]
        mask[indices] = True
        return mask


    def _outcomes_to_targets(self, encoded_outcomes):
        num = len(encoded_outcomes)
        targets = np.zeros([num, 3]) 
        mask = encoded_outcomes[:, :3].astype(np.bool)
        targets[mask] = encoded_outcomes[:, 3]
        return targets, mask


    def base_train_step(self, memory_buffer, lr):
        input_buff, output_buff = memory_buffer.get_memories()
        targets, target_mask = self._outcomes_to_targets(output_buff)
        feed_dict = {
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: self._random_guess_mask(self.memory_buffer_size),
            self.base_outcome_ph: output_buff,
            self.base_target_ph: targets,
            self.base_target_mask_ph: target_mask,
            self.keep_prob_ph: self.tkp,
            self.lr_ph: lr
        }
        self.sess.run(self.base_train, feed_dict=feed_dict)


    def reward_eval_helper(self, game, act_probs, encoded_hands=None, hands=None):
        if encoded_hands is not None:
            hands = self.decode_hands(encoded_hands)
        actions = [np.argmax(act_probs[i, :],
                             axis=-1) for i in range(len(act_probs))]
        bets = [self.bets[a] for a in actions] 
        rs = [game.play(hands[i], self.bets[a]) for i, a in enumerate(actions)]
        return np.mean(rs)


    def base_eval(self, game, memory_buffer, return_rewards=True):
        input_buff, output_buff = memory_buffer.get_memories()
        targets, target_mask = self._outcomes_to_targets(output_buff)
        feed_dict = {
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: self._random_guess_mask(self.memory_buffer_size),
            self.base_outcome_ph: output_buff,
            self.base_target_ph: targets,
            self.keep_prob_ph: 1.,
            self.base_target_mask_ph: target_mask
        }
#        print(self.sess.run([self.base_task_params, self.base_output], feed_dict=feed_dict))
#        exit()
        fetches = [self.total_base_loss]
        if return_rewards:
            fetches.append(self.base_output_softmax)
        res = self.sess.run(fetches, feed_dict=feed_dict)
        if return_rewards:
            res = res[0], self.reward_eval_helper(game, res[1], input_buff[:, :12])
        return res 


    def run_base_eval(self, return_rewards=True, include_new=False):
        if include_new:
            tasks = self.all_base_tasks
        else:
            tasks = self.base_tasks

        losses = [] 
        rewards = []
        for task in tasks:
            task_str = _stringify_game(task)
            memory_buffer = self.memory_buffers[task_str]
            game = self.games[task_str]
            res = self.base_eval(game, memory_buffer, return_rewards=return_rewards)
            losses.append(res[0])
            if return_rewards:
                rewards.append(res[1])

        names = [_stringify_game(t) for t in tasks]
        if return_rewards:
            return names, losses, rewards
        else:
            return names, losses


    def base_embedding_eval(self, embedding, game, memory_buffer, return_rewards=True):
        input_buff, output_buff = memory_buffer.get_memories()
        targets, target_mask = self._outcomes_to_targets(output_buff)
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.feed_embedding_ph: embedding,
            self.base_input_ph: input_buff,
            self.base_target_ph: targets,
            self.base_target_mask_ph: target_mask
        }
        fetches = [self.total_base_fed_emb_loss]
        if return_rewards:
            fetches.append(self.base_output_fed_emb_softmax)
        res = self.sess.run(fetches, feed_dict=feed_dict)
        if return_rewards:
            loss, act_probs = res
            avg_reward = self.reward_eval_helper(game, act_probs, input_buff[:, :12])
            return loss, avg_reward 
        else:
            return res

    
    def get_base_embedding(self, memory_buffer):
        input_buff, output_buff = memory_buffer.get_memories()
        targets, target_mask = self._outcomes_to_targets(output_buff)
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: np.ones([self.memory_buffer_size]),
            self.base_outcome_ph: output_buff
        }
        res = self.sess.run(self.guess_base_function_emb, feed_dict=feed_dict)
        return res


    def get_meta_dataset(self, meta_task, include_new=False):
        """override_m2l is used to allow meta mapped class. even if not trained on it"""
        x_data = []
        y_data = []
        if include_new:
            this_base_tasks = self.meta_pairings_full[meta_task]["base"]
            this_meta_tasks = self.meta_pairings_full[meta_task]["meta"]
        else:
            this_base_tasks = self.meta_pairings_base[meta_task]["base"]
            this_meta_tasks = self.meta_pairings_base[meta_task]["meta"]
        for (task, other) in this_base_tasks:
            task_buffer = self.memory_buffers[task]
            x_data.append(self.get_base_embedding(task_buffer)[0, :])
            if other in [0, 1]:  # for classification meta tasks
                y_data.append([other])
            else:
                other_buffer = self.memory_buffers[other]
                y_data.append(self.get_base_embedding(other_buffer)[0, :])
        return {"x": np.array(x_data), "y": np.array(y_data)}


    def refresh_meta_dataset_cache(self, include_new=False):
        meta_tasks = self.all_base_meta_tasks 
        if include_new:
            meta_tasks += self.new_meta_tasks 

        for t in meta_tasks:
            self.meta_dataset_cache[t] = self.get_meta_dataset(t, include_new)


    def meta_loss_eval(self, meta_dataset):
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.meta_input_ph: meta_dataset["x"], 
            self.guess_input_mask_ph: np.ones([len(meta_dataset["x"])])
        }
        y_data = meta_dataset["y"]
        if y_data.shape[-1] == 1:
            feed_dict[self.meta_class_ph] = y_data 
            fetch = self.total_meta_t_loss
        else:
            feed_dict[self.meta_target_ph] = y_data 
            fetch = self.total_meta_m_loss

        return self.sess.run(fetch, feed_dict=feed_dict)
        

    def run_meta_loss_eval(self, include_new=False):
        meta_tasks = self.all_base_meta_tasks 
        if include_new:
            meta_tasks += self.new_meta_tasks 

        names = []
        losses = []
        for t in meta_tasks:
            meta_dataset = self.meta_dataset_cache[t]
            loss = self.meta_loss_eval(meta_dataset)
            names.append(t)
            losses.append(loss)

        return names, losses


    def get_meta_embedding(self, meta_dataset):
        feed_dict = {
            self.keep_prob_ph: 1.,
            self.meta_input_ph: meta_dataset["x"], 
            self.guess_input_mask_ph: np.ones([len(meta_dataset["x"])])
        }
        y_data = meta_dataset["y"]
        if y_data.shape[-1] == 1:
            feed_dict[self.meta_class_ph] = y_data 
            fetch = self.guess_meta_t_function_emb
        else:
            feed_dict[self.meta_target_ph] = y_data 
            fetch = self.guess_meta_m_function_emb

        return self.sess.run(fetch, feed_dict=feed_dict)

    
    def get_meta_outputs(self, meta_dataset, new_dataset=None):
        """Get new dataset mapped according to meta_dataset, or just outputs
        for original dataset if new_dataset is None"""
        meta_class = meta_dataset["y"].shape[-1] == 1

        if new_dataset is not None:
            this_x = np.concatenate([meta_dataset["x"], new_dataset["x"]], axis=0)
            if meta_class:
                this_y = np.concatenate([meta_dataset["y"], np.zeros([len(new_dataset["x"])])], axis=0)
            else:
                this_y = np.concatenate([meta_dataset["y"], np.zeros_like(new_dataset["x"])], axis=0)
            this_mask = np.zeros(len(this_x), dtype=np.bool)
            this_mask[:len(meta_dataset["x"])] = True # use only these to guess
        else:
            this_x = meta_dataset["x"]
            this_y = meta_dataset["y"]
            this_mask = np.ones(len(this_x), dtype=np.bool)

        feed_dict = {
            self.keep_prob_ph: 1.,
            self.meta_input_ph: this_x,
            self.guess_input_mask_ph: this_mask 
        }
        if meta_class:
            feed_dict[self.meta_class_ph] = this_y 
            this_fetch = self.meta_t_output 
        else:
            feed_dict[self.meta_target_ph] = this_y
            this_fetch = self.meta_m_output 

        res = self.sess.run(this_fetch, feed_dict=feed_dict)
        return res[len(meta_dataset["x"]):, :]


    def run_meta_true_eval(self, include_new=False):
        """Evaluates true meta loss, i.e. the accuracy of the model produced
           by the embedding output by the meta task"""
        meta_tasks = self.base_meta_mappings 
        meta_pairings = self.meta_pairings_base
        if include_new:
            meta_pairings = self.meta_pairings_full

        names = []
        rewards = []
        for meta_task in meta_tasks:
            meta_dataset = self.meta_dataset_cache[meta_task]
            for task, other in meta_pairings[meta_task]["base"]:
                task_buffer = self.memory_buffers[task]
                task_embedding = self.get_base_embedding(task_buffer)

                other_buffer = self.memory_buffers[other]
                other_game = self.games[other]

                mapped_embedding = self.get_meta_outputs(
                    meta_dataset, {"x": task_embedding})

                names.append(meta_task + ":" + task + "->" + other)
                _, this_rewards = self.base_embedding_eval(mapped_embedding, other_game, other_buffer)
                rewards.append(this_rewards)

        return names, rewards


    def meta_train_step(self, meta_dataset, meta_lr):
        feed_dict = {
            self.keep_prob_ph: self.tkp,
            self.meta_input_ph: meta_dataset["x"], 
            self.guess_input_mask_ph: np.ones([len(meta_dataset["x"])]),
            self.lr_ph: meta_lr
        }
        y_data = meta_dataset["y"]
        if y_data.shape[-1] == 1:
            feed_dict[self.meta_class_ph] = y_data 
            op = self.meta_t_train
        else:
            feed_dict[self.meta_target_ph] = y_data 
            op = self.meta_m_train

        self.sess.run(op, feed_dict=feed_dict)


    def run_training(self, filename_prefix, num_epochs, include_new=False):
        """Train model on base and meta tasks, if include_new include also
        the new ones."""
        config = self.config
        loss_filename = filename_prefix + "_losses.csv"
        reward_filename = filename_prefix + "_rewards.csv"
        meta_filename = filename_prefix + "_meta_true_losses.csv"
        with open(loss_filename, "w") as fout, open(reward_filename, "w") as fout_reward, open(meta_filename, "w") as fout_meta:
            base_names, base_losses, base_rewards = self.run_base_eval(
                include_new=include_new)
            meta_names, meta_losses = self.run_meta_loss_eval(
                include_new=include_new)
            meta_true_names, meta_true_losses = self.run_meta_true_eval(
                include_new=include_new)

            fout.write("epoch, " + ", ".join(base_names + meta_names) + "\n")
            fout_reward.write("epoch, " + ", ".join(base_names) + "\n")
            fout_meta.write("epoch, " + ", ".join(meta_true_names) + "\n")

            loss_format = ", ".join(["%f" for _ in base_names + meta_names]) + "\n"
            reward_format = ", ".join(["%f" for _ in base_names]) + "\n"
            meta_true_format = ", ".join(["%f" for _ in meta_true_names]) + "\n"

            s_epoch  = "0, "
            curr_losses = s_epoch + (loss_format % tuple(
                base_losses + meta_losses))
            curr_rewards = s_epoch + (reward_format % tuple(base_rewards))
            curr_meta_true = s_epoch + (meta_true_format % tuple(meta_true_losses))
            fout.write(curr_losses)
            fout_reward.write(curr_rewards)
            fout_meta.write(curr_meta_true)

            learning_rate = config["init_learning_rate"]
            meta_learning_rate = config["init_meta_learning_rate"]

            if include_new:
                tasks = self.all_tasks
            else:
                tasks = self.all_initial_tasks

            save_every = config["save_every"]
            early_stopping_thresh = config["early_stopping_thresh"]
            lr_decays_every = config["lr_decays_every"]
            lr_decay = config["lr_decay"]
            meta_lr_decay = config["meta_lr_decay"]
            min_learning_rate = config["min_learning_rate"]
            min_meta_learning_rate = config["min_meta_learning_rate"]
            for epoch in range(1, num_epochs+1):
                if epoch % config["refresh_mem_buffs_every"] == 0:
                    self.play_games(num_turns=config["memory_buffer_size"],
                                    include_new=include_new,
                                    epsilon=config["epsilon"])
                if epoch % config["refresh_meta_cache_every"] == 0:
                    self.refresh_meta_dataset_cache(include_new=include_new)

                order = np.random.permutation(len(tasks))
                for task_i in order:
                    task = tasks[task_i]
                    if task in meta_names:
                        dataset = self.meta_dataset_cache[task]
                        self.meta_train_step(dataset, meta_learning_rate)
                    else:
                        memory_buffer = self.memory_buffers[_stringify_game(task)]
                        self.base_train_step(memory_buffer, learning_rate)

                if epoch % save_every == 0:
                    s_epoch  = "%i, " % epoch
                    _, base_losses, base_rewards = self.run_base_eval(
                        include_new=include_new)
                    _, meta_losses = self.run_meta_loss_eval(
                        include_new=include_new)
                    _, meta_true_losses = self.run_meta_true_eval(
                        include_new=include_new)
                    curr_losses = s_epoch + (loss_format % tuple(
                        base_losses + meta_losses))
                    curr_rewards = s_epoch + (reward_format % tuple(base_rewards))
                    curr_meta_true = s_epoch + (meta_true_format % tuple(meta_true_losses))
                    fout.write(curr_losses)
                    fout_reward.write(curr_rewards)
                    fout_meta.write(curr_meta_true)
                    print(curr_losses)
                    if np.all(curr_losses < early_stopping_thresh):
                        print("Early stop!")
                        break

                if epoch % lr_decays_every == 0 and epoch > 0 and learning_rate > min_learning_rate:
                    learning_rate *= lr_decay

                if epoch % lr_decays_every == 0 and epoch > 0 and meta_learning_rate > min_meta_learning_rate:
                    meta_learning_rate *= meta_lr_decay


    def save_embeddings(self, filename, meta_task=None,
                        include_new=False):
        """Saves all task embeddings, if meta_task is not None first computes
           meta_task mapping on them. If include_new, will include new tasks
           (note that this has a complicated pattern of effects, since they 
           will be included in meta datasets as well)."""
        with open(filename, "w") as fout:
            if include_new:
                tasks = [_stringify_game(t) for t in self.base_tasks]
                tasks += self.all_base_meta_tasks
            else:
                tasks = [_stringify_game(t) for t in self.all_base_tasks]
                tasks += self.all_meta_tasks
            fout.write("dimension, " + ", ".join(tasks) + "\n")
            format_string = ", ".join(["%f" for _ in tasks]) + "\n"
            num_hidden_hyper = config["num_hidden_hyper"]
            task_embeddings = np.zeros([len(tasks), num_hidden_hyper])

            for task_i, task in enumerate(tasks):
                if task in self.all_meta_tasks:
                    dataset = self.get_meta_dataset(
                        task, include_new=include_new)
                    embedding = self.get_meta_embedding(dataset) 
                else:
                    memory_buffer = self.memory_buffers[task]
                    embedding = self.get_base_embedding(memory_buffer) 

                task_embeddings[task_i, :] = embedding 

            if meta_task is not None:
                meta_dataset = self.get_meta_dataset(meta_task,
                    include_new=include_new)
                task_embeddings = self.get_meta_outputs(meta_dataset,
                                                        {"x": task_embeddings})

            for i in range(num_hidden_hyper):
                fout.write(("%i, " %i) + (format_string % tuple(task_embeddings[:, i])))


    def eval_all_games_all_hands(self, directory, include_new=False):
        """Saves the probabilities for all games and all hands"""
        if include_new:
            this_tasks = self.all_base_tasks
        else: 
            this_tasks = self.base_tasks
        for t in this_tasks:
            if not os.path.exists(directory):
                os.makedirs(directory)
            t_str = _stringify_game(t)
            with open(directory + t_str + "_actions.csv", "w") as fout:
                fout.write("hand, prob_0, prob_1, prob_2\n")
                game = self.games[t_str]
                encoded_game = self.encode_game(t)
                buff = self.memory_buffers[t_str]
                num_hands = len(game.hands)
                encoded_games = np.tile(encoded_game, [num_hands, 1])
                encoded_hands = np.zeros([num_hands, 12])
                for turn, hand in enumerate(game.hands):
                    encoded_hands[turn, :] = self.encode_hand(hand)
                act_probs = self.play_hands(encoded_hands, encoded_games, buff,
                                            epsilon=0., return_probs=True) 
                for turn, hand in enumerate(game.hands):
                    fout.write(
                        '"' + hand.__repr__() + '", %f, %f, %f\n' % tuple(act_probs[turn, :]))

                

## running stuff

for run_i in range(config["run_offset"], config["run_offset"]+config["num_runs"]):
    np.random.seed(run_i)
    tf.set_random_seed(run_i)
    filename_prefix = config["output_dir"] + "run%i" % run_i
    print("Now running %s" % filename_prefix)
    _save_config(filename_prefix + "_config.csv", config)


    model = meta_model(config) 
    model.save_embeddings(filename=filename_prefix + "_init_embeddings.csv",
                          include_new=False)
    model.run_training(filename_prefix=filename_prefix,
                       num_epochs=config["max_base_epochs"],
                       include_new=False)
    model.save_embeddings(filename=filename_prefix + "_guess_embeddings.csv",
                          include_new=True)
    if config["eval_all_hands"]:
        model.eval_all_games_all_hands(directory=config["output_dir"] + "/guess_hand_actions/run%i/" % run_i,
                                       include_new=True)

    for meta_task in config["base_meta_mappings"]:
        model.save_embeddings(filename=filename_prefix + "_" + meta_task + "_guess_embeddings.csv",
                              meta_task=meta_task,
                              include_new=True)

    model.run_training(filename_prefix=filename_prefix + "_new",
                       num_epochs=config["max_new_epochs"],
                       include_new=True)

    model.save_embeddings(filename=filename_prefix + "_final_embeddings.csv",
                          include_new=True)
    for meta_task in config["base_meta_mappings"]:
        model.save_embeddings(filename=filename_prefix + "_" + meta_task + "_final_embeddings.csv",
                              meta_task=meta_task,
                              include_new=True)

    tf.reset_default_graph()

