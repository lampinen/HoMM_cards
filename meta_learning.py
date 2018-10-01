from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from copy import deepcopy

from simple_card_games import card_game
from orthogonal_matrices import random_orthogonal

pi = np.pi
### Parameters #################################################
config = {
    game_types: ["high_card", "match", "pairs_and_high", "straight_flush", "sum_under"],
    option_names: ["suits_rule", "losers", "black_valuable"],
    suits_rule: [True, False],
    losers: [True, False],
    black_valuable: [True, False],

    bets = [0, 1, 2],

    num_input: (4 + 2) * 2 + (5 + 2 + 2 + 2), # (4 values + 2 suits) * 2 cards
                                             # + 5 games + 3 binary options
    num_output: 3 # bet 0, bet 1, bet 2 
    num_outcome: (3 + 1) + (5 + 2 + 2 + 2), # 3 possible bets (actions) + reward
                                           # + 5 games + 3 binary options
    num_hidden: 64,
    num_hidden_hyper: 64,

    init_learning_rate: 1e-4,
    init_meta_learning_rate: 2e-4,

    new_init_learning_rate: 1e-6,
    new_init_meta_learning_rate: 1e-6,

    lr_decay: 0.85,
    meta_lr_decay: 0.85,

    lr_decays_every: 100,
    min_learning_rate: 1e-6,

    refresh_meta_cache_every: 1, # how many epochs between updates to meta_dataset_cache

    max_base_epochs: 4000 ,
    max_new_epochs: 200,
    num_task_hidden_layers: 3,
    num_hyper_hidden_layers: 3,

    output_dir: "results/",
    save_every: 10, 

    memory_buffer_size: 1024, # How many memories of each task are stored
    meta_batch_size: 768, # how many meta-learner sees
    early_stopping_thresh: 0.005,
    base_meta_tasks: ["is_" + g for g in game_types] + ["is_" + o for o in option_names],
    base_meta_mappings: ["toggle_" + o for o in option_names],

    new_tasks: [{"game": "straight_flush", "losers": True,
                  "black_valuable": True, "suits_rule": False}], # will be removed
                                                                 # from base tasks
    base_tasks: [{"game": g, "losers": l, "black_valuable": b,
                   "suits_rule": s} for g in game_types for l in losers for b in black_valuable for s in suits_rule],
    base_tasks: [t for t in base_tasks if t not in new_tasks], # omit new

    internal_nonlinearity: tf.nn.leaky_relu,
    output_nonlinearity: None
}
### END PARAMATERS (finally) ##################################

def _stringify_game(t):
    """Helper for printing, etc."""
    return "game_%s_l_%i_bv_%i_sr_%i" % (t["game"], t["losers"],
                                         t["black_valuable"], t["suits_rule"])


var_scale_init = tf.contrib.layers.variance_scaling_initializer(factor=1., mode='FAN_AVG')

#total_tasks = set(base_tasks + new_tasks)

def _get_meta_pairings(base_tasks, meta_tasks, meta_mappings):
    """Gets which tasks map to which other tasks under the meta_tasks (i.e. the
    part of the meta datasets which is precomputable)"""
    all_base_meta_tasks = this_base_meta_tasks + this_base_meta_mappings
    meta_pairings = {mt: {"base": [], "meta": []} for mt in all_base_meta_tasks}
    for mt in all_base_meta_tasks:
        if mt[:6] == "toggle":
            to_toggle = mt[7:]
            for task in basic_tasks: 
                other = deepcopy(task) 
                other[to_toggle] = not other[to_toggle]
                if other in base_tasks:
                    meta_pairings[mt]["base"].append((_stringify_game(task),
                                                      _stringify_game(other)))

        elif mt[:2] == "is":
            pos_class = mt[3:]
            for task in basic_tasks: 
                truth_val = (task["game_type"] == pos_class) or task[pos_class]
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
        self.input_buffer = np.zeros(length, input_width)
        self.outcome_buffer = np.zeros(length, outcome_width)

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
            num_to_end = self.num_events + back_off
            self.input_buffer[:-back_off, :] = input_mat[num_to_end:, :] 
            self.outcome_buffer[:-back_off, :] = outcome_mat[num_to_end:, :] 
        else: 
            back_off = end_offset
            num_to_end = num_events
        self.input_buffer[self.curr_index:back_off, :] = input_mat[:num_to_end, :] 
        self.outcome_buffer[self.curr_index:back_off, :] = outcome_mat[:num_to_end, :] 
        self.curr_index = np.abs(back_off)

    def get_memories(self): 
        return self.input_buffer, self.output_buffer


class meta_model(object):
    """A meta-learning model for RL on simple card games."""
    def __init__(self, base_tasks, new_tasks, base_meta_tasks,
                 base_meta_mappings, new_meta_tasks, config):
        """args:
            base_tasks: list of base binary functions to compute
            base_meta_tasks: tasks whose domain is functions but range is not
            base_meta_mappings: tasks whose domain AND range are both function
                embeddings
            new_tasks: new tasks to test on
            new_meta_tasks: new meta tasks
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

        # base datasets / memory_buffers
        self.base_tasks = base_tasks
        self.base_task_names = [_stringify_game(t) for t in base_tasks]

        # new datasets / memory_buffers
        self.new_tasks = new_tasks
        self.new_task_names = [_stringify_game(t) for t in new_tasks]

        self.all_base_tasks = self.base_tasks + self.new_tasks
        self.memory_buffers = {t: memory_buffer(
            self.memory_buffer_size, self.num_input,
            self.num_outcome) for t in self.all_base_tasks}

        self.games = {card_game(game_type=t["game"],
                                black_valuabele=t["black_valuable"],
                                suits_rule=t["suits_rule"],
                                losers=t["losers"]) for t in self.all_base_tasks}

        self.base_meta_tasks = base_meta_tasks 
        self.base_meta_mappings = base_meta_mappings
        self.all_base_meta_tasks = base_meta_tasks + base_meta_mappings
        self.new_meta_tasks = new_meta_tasks 
        self.meta_dataset_cache = {t: {} for t in self.all_base_meta_tasks + self.new_meta_tasks} # will cache datasets for a certain number of epochs
                                                                                                  # both to speed training and to keep training targets
                                                                                                  # consistent

        self.all_tasks = self.base_tasks + self.new_tasks + self.all_base_meta_tasks + self.new_meta_tasks
        self.num_tasks = num_tasks = len(self.all_tasks)
        self.task_to_index = dict(zip(self.all_tasks, range(num_tasks)))

        self.meta_pairings_base = _get_meta_pairings(
            self.base_tasks, self.base_meta_tasks, self.base_meta_mappings)

        self.meta_pairings_full = _get_meta_pairings(
            self.base_tasks + self.new_tasks,
            self.base_meta_tasks + self.new_meta_tasks,
            self.base_meta_mappings)

        # network

        # base task input
        input_size = num_input 
        self.base_input_ph = tf.placeholder(
            tf.float32, shape=[None, input_size])
        self.base_outcome_ph = tf.placeholder(
            tf.float32, shape=[None, outcome_size])
        self.base_target_ph = tf.placeholder(
            tf.float32, shape=[None, output_size])

        self.lr_ph = tf.placeholder(tf.float32)

        input_processing_1 = slim.fully_connected(self.base_input_ph, num_hidden, 
                                                  activation_fn=internal_nonlinearity) 

        input_processing_2 = slim.fully_connected(input_processing_1, num_hidden, 
                                                  activation_fn=internal_nonlinearity) 

        processed_input = slim.fully_connected(input_processing_2, num_hidden_hyper, 
                                               activation_fn=internal_nonlinearity) 

        all_target_processor_nontf = random_orthogonal(num_hidden_hyper)[:, :num_output + 1]
        self.target_processor_nontf = all_target_processor_nontf[:, :num_output]
        self.target_processor = tf.get_variable('target_processor', 
                                                shape=[num_hidden_hyper, num_output],
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

        processed_outcomes = _outcome_encoder(self.base_outcome_ph)

        # meta task input
        self.meta_input_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.meta_target_ph = tf.placeholder(tf.float32, shape=[None, num_hidden_hyper])
        self.meta_class_ph = tf.placeholder(tf.float32, shape=[None, 1]) 
        # last is for meta classification tasks

        self.class_processor_nontf = all_target_processor_nontf[:, num_output:]
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
                guess_input = tf.concat([embedded_input,
                                         embedded_targets], axis=-1)
                guess_input = tf.boolean_mask(guess_input,
                                              self.guess_input_mask_ph)

                gh_1 = slim.fully_connected(guess_input, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity) 
                gh_2 = slim.fully_connected(gh_1, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity) 
                gh_2b = tf.reduce_max(gh_2, axis=0, keep_dims=True)
                gh_3 = slim.fully_connected(gh_2b, num_hidden_hyper,
                                            activation_fn=internal_nonlinearity) 

                guess_embedding = slim.fully_connected(gh_3, num_hidden_hyper,
                                                       activation_fn=None)
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

        def _hyper_network(function_embedding, reuse=True):
            with tf.variable_scope('hyper', reuse=reuse):
                hyper_hidden = function_embedding
                for _ in range(num_hyper_hidden_layers-1):
                    hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                        activation_fn=internal_nonlinearity)
                
                hidden_weights = []
                hidden_biases = []

                hyper_hidden = slim.fully_connected(hyper_hidden, num_hidden_hyper,
                                                      activation_fn=internal_nonlinearity)
                task_weights = slim.fully_connected(hyper_hidden, num_hidden*(num_hidden_hyper +(num_task_hidden_layers-1)*num_hidden + num_hidden_hyper),
                                                    activation_fn=None)

                task_weights = tf.reshape(task_weights, [-1, num_hidden, (num_hidden_hyper + (num_task_hidden_layers-1)*num_hidden + num_hidden_hyper)]) 
                task_biases = slim.fully_connected(hyper_hidden, num_task_hidden_layers * num_hidden + num_hidden_hyper,
                                                   activation_fn=None)

                Wi = tf.transpose(task_weights[:, :, :num_hidden_hyper], perm=[0, 2, 1])
                bi = task_biases[:, :num_hidden]
                hidden_weights.append(Wi)
                hidden_biases.append(bi)
                for i in range(1, num_task_hidden_layers):
                    Wi = tf.transpose(task_weights[:, :, num_input+(i-1)*num_hidden:num_input+i*num_hidden], perm=[0, 2, 1])
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
                hidden_biases.append(Wfinal)
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
        self.base_output_softmax = tf.nn.softmax(self.base_output)

        self.base_raw_output_fed_emb = _task_network(self.fed_emb_task_params,
                                                     processed_input)
        self.base_output_fed_emb = _output_mapping(self.base_raw_output_fed_emb)
        self.base_output_fed_emb_softmax = tf.nn.softmax(
            self.base_output_fed_emb)

        self.meta_t_raw_output = _task_network(self.meta_t_task_params,
                                               self.meta_input_ph)
        self.meta_t_output = tf.nn.sigmoid(self.meta_t_raw_output)

        self.meta_m_output = _task_network(self.meta_m_task_params,
                                               self.meta_input_ph)

        # have to mask base output because can only learn about the action 
        # actually taken
        self.base_target_mask_ph = tf.placeholder(
            tf.bool, shape=[None, output_size])
        masked_base_output = tf.boolean_mask(base_output,
                                             self.base_target_mask_ph)
        masked_base_fed_emb_output = tf.boolean_mask(base_output_fed_emb,
                                                     self.base_target_mask_ph)
        masked_base_target = tf.boolean_mask(processed_targets,
                                             self.base_target_mask_ph)

        self.base_loss = tf.reduce_sum(
            tf.square(masked_base_output - masked_base_target), axis=1)
        self.total_base_loss = tf.reduce_mean(self.base_loss)

        self.base_fed_emb_loss = tf.reduce_sum(
            tf.square(masked_base_fed_emb_output - masked_base_target), axis=1)
        self.total_base_fed_emb_loss = tf.reduce_mean(self.base_fed_emb_loss)

        self.meta_t_loss = tf.reduce_sum(
            tf.square(self.meta_t_output - processed_class), axis=1)
        self.total_meta_t_loss = tf.reduce_mean(self.meta_t_loss)

        self.meta_m_loss = tf.reduce_sum(
            tf.square(self.meta_m_output - self.meta_target_ph), axis=1)
        self.total_meta_m_loss = tf.reduce_mean(self.meta_m_loss)


        optimizer = tf.train.RMSPropOptimizer(self.lr_ph)

        self.base_train = optimizer.minimize(self.total_base_loss)
        self.meta_classification_train = optimizer.minimize(self.total_meta_classification_loss)
        self.meta_mapping_train = optimizer.minimize(self.total_meta_mapping_loss)

        # initialize
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        self.refresh_meta_dataset_cache()


    def encode_game(self, task):
        """Takes a task dict, returns vector appropriate for input to graph."""
        vec = np.zeros(11)
        game_type = t["game"],
        black_valuabele = t["black_valuable"]
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

        vec[:6] = _card_to_vec[hand[0]]
        vec[6:] = _card_to_vec[hand[1]]
        return vec


    def decode_hands(self, encoded_hands):
        hands = []
        def _card_from_vec(v):
            return (np.argmax(v[:4]), np.argmax(v[4:]))
        for enc_hand in encoded_hands:
            hand = (_card_from_vec(env_hand[:6]), _card_from_vec(env_hand[6:]))
            hands.append(hand)
        return hands


    def encode_outcomes(self, actions, rewards):
        """Takes actions and rewards, returns matrix appropriate for input to
        graph"""
        mat = np.zeros(len(actions), 4)
        mat[range(len(actions)), actions] = 1.
        mat[:, -1] = rewards
        return mat


    def play_hands(self, encoded_hands, encoded_games, memory_buffer, epsilon=0.):
        """Plays the provided hand conditioned on the game and memory buffer,
        with epsilon-greedy exploration."""
        if epsilon == 1.: # makes it easier to fill buffers before play begins
            return np.random.randint(3, size=[len(encoded_hands), 3])
        input_buff, outcome_buff = memory_buffer.get_memories()
        feed_dict = {
            self.base_input_ph: input_buffer,
            self.guess_input_mask_ph: buff_mask,
            self.base_outcome_ph: outcome_buffer
        }
        act_probs = self.sess.run(self.base_output_softmax,
                                  feed_dict=feed_dict)
        actions = [np.random.choice(
            range(3), p=act_probs[i, :]) for i in range(len(act_probs))]
        return actions
        

    def play_games(self, num_turns=1, include_new=False, epsilon=epsilon)
        """Plays turns games in base_tasks (and new if include_new), to add new
        experiences to memory buffers."""
        if include_new:
            this_tasks = self.all_base_tasks
        else: 
            this_tasks = self.base_tasks
        for t in this_tasks:
            game = self.games[t]
            encoded_game = self.encode_game(t)
            buff = self.memory_buffers[t]
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
        indices = np.random.permutation(dataset_length)[:meta_batch_size]
        mask[indices] = True
        return mask


    def _outcomes_to_targets(self, encoded_outcomes):
        num = len(encoded_outcomes)
        targets = np.zeros([num, 3]) 
        mask = np.zeros_like(targets, dtype=np.bool) 
        inds = encoded_outcomes[:, :3].astype(np.bool)
        targets[encoded_outcomes[:, :3].astype(np.bool)] = encoded_outcomes[:, -1]
        mask[encoded_outcomes[:, :3].astype(np.bool)] = 1. 
        return targets, mask


    def base_train_step(self, memory_buffer, lr):
        input_buff, output_buff, _ = memory_buffer.get_memories()
        targets, target_mask = self._outcomes_to_targets(output_buff)
        feed_dict = {
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: self._random_guess_mask(self.memory_buffer_size),
            self.base_outcome_ph: output_buff,
            self.base_target_ph: targets,
            self.base_target_mask_ph: target_mask,
            self.lr_ph = lr
        }
        self.sess.run(self.base_train, feed_dict=feed_dict)


    def reward_eval_helper(self, act_probs, encoded_hands=None, hands=None):
        if encoded_hands is not None:
            hands = self.decode_hands(encoded_hands)
        actions = [np.random.choice(
            range(3), p=act_probs[i, :]) for i in range(len(act_probs))]
        bets = [self.bets[a] for a in acts] 
        rs = [game.play(hand, self.bets[a]) for a in acts]
        return np.mean(rs)


    def base_eval(self, memory_buffer, return_rewards=True):
        input_buff, output_buff, _ = memory_buffer.get_memories()
        targets, target_mask = self._outcomes_to_targets(output_buff)
        feed_dict = {
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: self._random_guess_mask(self.memory_buffer_size),
            self.base_outcome_ph: output_buff,
            self.base_target_ph: targets,
            self.base_target_mask_ph: target_mask
        }
        fetches = [self.base_total_loss]
        if return_rewards:
            fetches.append(self.base_output_softmax)
        res = self.sess.run(fetches, feed_dict=feed_dict)
        if return_rewards:
            res = res[0], self.reward_eval_helper(res[1], input_buff[:, :12])
        return res 


    def base_embedding_eval(self, embedding, memory_buffer, return_rewards=True):
        input_buff, output_buff, _ = memory_buffer.get_memories()
        targets, target_mask = self._outcomes_to_targets(output_buff)
        feed_dict = {
            self.base_feed_embedding_ph: embedding,
            self.base_input_ph: input_buffer,
            self.base_target_ph: targets,
            self.base_target_mask_ph: target_mask
        }
        fetches = [self.base_total_fed_emb_loss]
        if return_rewards:
            fetches.append(self.base_output_fed_emb_softmax)
        res = self.sess.run(fetches, feed_dict=feed_dict)
        if return_rewards:
            avg_reward = self.reward_eval_helper(act_probs, input_buff[:, :12])
        return loss, avg_reward 

    
    def get_base_embedding(self, memory_buffer):
        input_buff, output_buff, _ = memory_buffer.get_memories()
        targets, target_mask = self._outcomes_to_targets(output_buff)
        feed_dict = {
            self.base_feed_embedding_ph: embedding,
            self.base_input_ph: input_buff,
            self.guess_input_mask_ph: np.ones([self.memory_buffer_size]),
            self.base_outcome_ph: output_buff
        }
        res = self.sess.run(self.guess_base_function_emb, feed_dict=feed_dict)


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
            x_data.append(self.get_task_embedding(task_buffer)[0, :])
            if other in [0, 1]:  # for classification meta tasks
                y_data.append(other)
            else:
                other_buffer = self.memory_buffers[other]
                y_data.append(self.get_task_embedding(other_buffer)[0, :])
        return {"x": np.array(x_data), "y": np.array(y_data)}


    def refresh_meta_dataset_cache(self, include_new=False):
        meta_tasks = self.all_base_meta_tasks 
        if include_new:
            meta_tasks += self.new_meta_tasks 

        for t in meta_tasks:
            self.meta_dataset_cache[t] = self.get_meta_dataset(t, include_new)


    def meta_loss_eval(self, meta_dataset):
        feed_dict = {
            self.meta_input_ph: meta_dataset["x"] 
            self.guess_input_mask_ph: np.ones([len(meta_dataset["x"])])
        }
        y_data = meta_dataset["y"]
        if len(y_data.shape) == 1:
            feed_dict[self.meta_class_ph] = y_data 
            fetch = self.meta_t_loss
        else:
            feed_dict[self.meta_target_ph] = y_data 
            fetch = self.meta_m_loss

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

        return losses, names

    
    def get_meta_outputs(self, meta_dataset, new_dataset=None):
        """Get new dataset mapped according to meta_dataset, or just outputs
        for original dataset if new_dataset is None"""
        meta_class = len(meta_dataset["y"].shape) == 1

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
            self.meta_input_ph: this_x 
            self.guess_input_mask_ph: this_y 
        }
        if meta_class:
            feed_dict[self.meta_class_ph] = this_y 
            this_fetch = self.meta_t_output 
        else:
            feed_dict[self.meta_target_ph] = this_y
            this_fetch = self.meta_m_output 

        res = self.sess.run(this_fetch)
        return res[len(meta_dataset["x"]):, :]


    def meta_true_eval(self, include_new=False):
        """Evaluates true meta loss, i.e. the accuracy of the model produced
           by the embedding output by the meta task"""
        meta_tasks = self.all_base_meta_tasks 
        meta_pairings = self.meta_pairings_base
        if include_new:
            meta_tasks += self.new_meta_tasks 
            meta_pairings = self.meta_pairings_full

        names = []
        losses = []
        for t in meta_tasks:
            meta_dataset = self.meta_dataset_cache[t]
            for task, other in meta_pairings[meta_task]["base"]:
                task_buffer = self.memory_buffers[task]
                task_embedding = self.get_base_embedding(task_buffer)

                other_buffer = self.memory_buffers[other]

                mapped_embedding = self.get_meta_outputs(
                    meta_dataset, {"x": task_embedding})

                names.append(meta_task + ":" + task + "->" + other)
                _, this_rewards = self.base_embedding_eval(mapped_embedding, other_buffer)
                rewards.append(this_rewards)

        return rewards, names


    def meta_train_step(self, meta_dataset):
        feed_dict = {
            self.meta_input_ph: meta_dataset["x"] 
            self.guess_input_mask_ph: np.ones([len(meta_dataset["x"])])
        }
        y_data = meta_dataset["y"]
        if len(y_data.shape) == 1:
            feed_dict[self.meta_class_ph] = y_data 
            op = self.meta_t_train
        else:
            feed_dict[self.meta_target_ph] = y_data 
            op = self.meta_m_train

        self.sess.run(op, feed_dict=feed_dict)


???    def base_eval(self):
???        """Evaluates loss on the base tasks."""
???        losses = np.zeros([len(self.base_tasks) + len(self.all_base_meta_tasks)])
???        for task in self.base_tasks:
???            dataset =  self.base_datasets[task]
???            losses[self.task_to_index[task]] = self.dataset_eval(dataset)
???
???        offset = len(self.new_tasks) # new come before meta in indices
???        for task in self.all_base_meta_tasks:
???            dataset = self.meta_dataset_cache[task]
???            losses[self.task_to_index[task] - offset] = self.dataset_eval(dataset,
???                                                                          base_input=False, 
???                                                                          base_output=len(dataset["y"].shape) == 1)
???
???        return losses
???
???
???    def new_eval(self, new_task, zeros=False):
???        """Evaluates loss on a new task."""
???        new_task_full_name = [t for t in self.new_tasks if new_task in t][0] 
???        dataset = self.new_datasets[new_task_full_name]
???        return self.dataset_eval(dataset, zeros=zeros)
???
???
???    def all_eval(self):
???        """Evaluates loss on the base and new tasks."""
???        losses = np.zeros([self.num_tasks])
???        for task in self.base_tasks:
???            dataset =  self.base_datasets[task]
???            losses[self.task_to_index[task]] = self.dataset_eval(dataset)
???
???        for task in self.new_tasks:
???            dataset =  self.new_datasets[task]
???            losses[self.task_to_index[task]] = self.dataset_eval(dataset)
???
???        for task in self.all_base_meta_tasks:
???            dataset = self.meta_dataset_cache[task]
???            losses[self.task_to_index[task]] = self.dataset_eval(dataset,
???                                                                 base_input=False,
???                                                                 base_output=len(dataset["y"].shape) == 1)
???
???        return losses
???
???    def get_outputs(self, dataset, new_dataset=None, base_input=True, base_output=True, zeros=False):
???        if new_dataset is not None:
???            this_x = np.concatenate([dataset["x"], new_dataset["x"]], axis=0)
???            dummy_y = np.zeros(len(new_dataset["x"])) if base_output else np.zeros_like(new_dataset["x"])
???            this_y = np.concatenate([dataset["y"], dummy_y], axis=0)
???            this_mask = np.zeros(len(this_x), dtype=np.bool)
???            this_mask[:len(dataset["x"])] = 1. # use only these to guess
???        else:
???            this_x = dataset["x"]
???            this_y = dataset["y"]
???            this_mask = np.ones(len(dataset["x"]), dtype=np.bool)
???
???        if zeros:
???            this_mask = np.zeros_like(this_mask)
???
???        this_feed_dict = {
???            self.base_input_ph: this_x if base_input else self.dummy_base_input,
???            self.guess_input_mask_ph: this_mask,
???            self.is_base_input: base_input,
???            self.is_base_output: base_output,
???            self.base_target_ph: this_y if base_output else self.dummy_base_output,
???            self.meta_input_ph: self.dummy_meta_input if base_input else this_x,
???            self.meta_target_ph: self.dummy_meta_output if base_output else this_y
???        }
???        this_fetch = self.base_output if base_output else self.raw_output 
???        outputs = self.sess.run(this_fetch, feed_dict=this_feed_dict)
???        if base_output:
???            outputs = 2*np.argmax(outputs, axis=-1) - 1
???        if new_dataset is not None:
???            outputs = outputs[len(dataset["x"]):, :]
???        return outputs
???
???
???    def new_outputs(self, new_task, zeros=False):
???        """Returns outputs on a new task.
???           zeros: if True, will give empty dataset to guessing net, for
???           baseline"""
???        new_task_full_name = [t for t in self.new_tasks if new_task in t][0] 
???        dataset = self.new_datasets[new_task_full_name]
???        return self.get_outputs(dataset, zeros=zeros)
???
???
???    def train_base_tasks(self, filename):
???        """Train model to perform base tasks as meta task."""
???        with open(filename, "w") as fout:
???            fout.write("epoch, " + ", ".join(self.base_tasks + self.all_base_meta_tasks) + "\n")
???            format_string = ", ".join(["%f" for _ in self.base_tasks + self.all_base_meta_tasks]) + "\n"
???
???            learning_rate = init_learning_rate
???            meta_learning_rate = init_meta_learning_rate
???
???            for epoch in range(max_base_epochs):
???
???                if epoch % refresh_meta_cache_every == 0:
???                    self.refresh_meta_dataset_cache()
???
???                order = np.random.permutation(len(self.base_tasks))
???                for task_i in order:
???                    task = self.base_tasks[task_i]
???                    dataset =  self.base_datasets[task]
???                    self.dataset_train_step(dataset, learning_rate)
???
???                order = np.random.permutation(len(self.all_base_meta_tasks))
???                for task_i in order:
???                    task = self.all_base_meta_tasks[task_i]
???                    dataset =  self.meta_dataset_cache[task]
???                    self.dataset_train_step(dataset, meta_learning_rate, 
???                                            base_output=len(dataset["y"].shape) == 1,
???                                            base_input=False)
???
???                if epoch % save_every == 0:
???                    curr_losses = self.base_eval()
???                    curr_output = ("%i, " % epoch) + (format_string % tuple(curr_losses))
???                    fout.write(curr_output)
???                    print(curr_output)
???                    if np.all(curr_losses < early_stopping_thresh):
???                        print("Early stop base!")
???                        break
???
???                if epoch % lr_decays_every == 0 and epoch > 0 and learning_rate > min_learning_rate:
???                    learning_rate *= lr_decay
???
???                if epoch % lr_decays_every == 0 and epoch > 0 and meta_learning_rate > min_learning_rate:
???                    meta_learning_rate *= meta_lr_decay
???
???
???    def refresh_meta_dataset_cache(self, refresh_new=False):
???        for task in self.base_meta_tasks:
???            self.meta_dataset_cache[task] = self.get_meta_dataset(task)
???        for task in self.base_meta_mappings:
???            self.meta_dataset_cache[task] = self.get_meta_dataset(task)
???        if refresh_new:
???            for task in self.new_meta_tasks:
???                self.meta_dataset_cache[task] = self.get_meta_dataset(task, include_new=True)
???
???
???    def train_new_tasks(self, filename_prefix):
???        print("Now training new tasks...")
???
???        with open(filename_prefix + "new_losses.csv", "w") as fout:
???            with open(filename_prefix + "meta_true_losses.csv", "w") as fout_meta:
???                with open(filename_prefix + "meta_mapped_classification_true_losses.csv", "w") as fout_meta2:
???                    fout.write("epoch, " + ", ".join(self.all_tasks) + "\n")
???                    format_string = ", ".join(["%f" for _ in self.all_tasks]) + "\n"
???
???                    for new_task in self.new_tasks:
???                        dataset = self.new_datasets[new_task]
???                        with open(filename_prefix + new_task + "_outputs.csv", "w") as foutputs:
???                            foutputs.write("type, " + ', '.join(["input%i" % i for i in range(len(dataset["y"]))]) + "\n")
???                            foutputs.write("target, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(dataset["y"].flatten()) + "\n")
???
???                            curr_net_outputs = self.new_outputs(new_task, zeros=True)
???                            foutputs.write("baseline, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(curr_net_outputs) + "\n")
???                            curr_net_outputs = self.new_outputs(new_task, zeros=False)
???                            foutputs.write("guess_emb, " + ', '.join(["%f" for i in range(len(dataset["y"]))]) % tuple(curr_net_outputs) + "\n")
???
???                    self.refresh_meta_dataset_cache(refresh_new=True)
???
???                    curr_meta_true_losses, meta_true_names = self.meta_true_eval() 
???                    fout_meta.write("epoch, " + ", ".join(meta_true_names) + "\n")
???                    meta_format_string = ", ".join(["%f" for _ in meta_true_names]) + "\n"
???                    curr_meta_output = ("0, ") + (meta_format_string % tuple(curr_meta_true_losses))
???                    fout_meta.write(curr_meta_output)
???
???                    curr_meta2_true_losses, meta2_true_names = self.meta_mapped_classification_true_eval() 
???                    fout_meta2.write("epoch, " + ", ".join(meta2_true_names) + "\n")
???                    meta2_format_string = ", ".join(["%f" for _ in meta2_true_names]) + "\n"
???                    curr_meta2_output = ("0, ") + (meta2_format_string % tuple(curr_meta2_true_losses))
???                    fout_meta2.write(curr_meta2_output)
???
???                    curr_losses = self.all_eval() # guess embedding 
???                    print(len(curr_losses))
???                    curr_output = ("0, ") + (format_string % tuple(curr_losses))
???                    fout.write(curr_output)
???                    print(curr_output)
???
???                    # now tune
???                    learning_rate = new_init_learning_rate
???                    meta_learning_rate = new_init_meta_learning_rate
???                    for epoch in range(1, max_new_epochs):
???                        if epoch % refresh_meta_cache_every == 0:
???                            self.refresh_meta_dataset_cache(refresh_new=True)
???
???                        order = np.random.permutation(self.num_tasks)
???                        for task_i in order:
???                            task = self.all_tasks[task_i]
???                            base_input=True
???                            base_output=True
???                            this_lr = learning_rate
???                            if task in self.new_tasks:
???                                dataset =  self.new_datasets[task]
???                            elif task in self.all_base_meta_tasks:
???                                dataset = self.meta_dataset_cache[task]
???                                base_input=False
???                                base_output=len(dataset["y"].shape) == 1
???                                this_lr = meta_learning_rate
???                            else:
???                                dataset =  self.base_datasets[task]
???                            self.dataset_train_step(dataset, learning_rate, 
???                                                    base_input=base_input,
???                                                    base_output=base_output)
???
???                        if epoch % save_every == 0:
???                            curr_meta_true_losses, _ = self.meta_true_eval() 
???                            curr_meta_output = ("%i, " % epoch) + (meta_format_string % tuple(curr_meta_true_losses))
???                            fout_meta.write(curr_meta_output)
???
???                            curr_meta2_true_losses, _ = self.meta_mapped_classification_true_eval() 
???                            curr_meta2_output = ("%i, " % epoch) + (meta2_format_string % tuple(curr_meta2_true_losses))
???                            fout_meta2.write(curr_meta2_output)
???
???                            curr_losses = self.all_eval()
???                            curr_output = ("%i, " % epoch) + (format_string % tuple(curr_losses))
???                            fout.write(curr_output)
???                            print(curr_output)
???                            if np.all(curr_losses < early_stopping_thresh):
???                                print("Early stop new!")
???                                break
???
???                        if epoch % lr_decays_every == 0 and epoch > 0 and learning_rate > min_learning_rate:
???                            learning_rate *= lr_decay
???
???                        if epoch % lr_decays_every == 0 and epoch > 0 and meta_learning_rate > min_learning_rate:
???                            meta_learning_rate *= meta_lr_decay
???
???        for new_task in self.new_tasks:
???            with open(filename_prefix + new_task + "_outputs.csv", "a") as foutputs:
???                curr_net_outputs = self.new_outputs(new_task)
???                foutputs.write("trained_emb, " + ', '.join(["%f" for i in range(len(curr_net_outputs))]) % tuple(curr_net_outputs) + "\n")
???
???
???    def get_task_embedding(self, dataset, base_input=True, base_output=True):
???        """Gets task embedding"""
???        return self.sess.run(
???            self.function_embedding,
???            feed_dict={
???                self.base_input_ph: dataset["x"] if base_input else self.dummy_base_input,
???                self.guess_input_mask_ph: np.ones(len(dataset["x"]), dtype=np.bool),
???                self.is_base_input: base_input,
???                self.is_base_output: base_output,
???                self.base_target_ph: dataset["y"] if base_output else self.dummy_base_output,
???                self.meta_input_ph: self.dummy_meta_input if base_input else dataset["x"],
???                self.meta_target_ph: self.dummy_meta_output if base_output else dataset["y"]
???            }) 
???
???
???    def save_embeddings(self, filename, meta_task=None):
???        """Saves all task embeddings, if meta_task is not None first computes
???           meta_task mapping on them."""
???        def _simplify(t):
???            split_t = t.split(';')
???            return ';'.join([split_t[0], split_t[2]])
???        with open(filename, "w") as fout:
???            basic_tasks = self.base_tasks + self.new_tasks
???            simplified_tasks = [_simplify(t) for t in basic_tasks]
???            fout.write("dimension, " + ", ".join(simplified_tasks + self.all_base_meta_tasks) + "\n")
???            format_string = ", ".join(["%f" for _ in self.all_tasks]) + "\n"
???            task_embeddings = np.zeros([len(self.all_tasks), num_hidden_hyper])
???
???            for task in self.all_tasks:
???                base_input = True
???                base_output = True
???                if task in self.new_tasks:
???                    dataset =  self.new_datasets[task]
???                elif task in self.base_tasks:
???                    dataset =  self.base_datasets[task]
???                else:
???                    dataset = self.get_meta_dataset(task)
???                    base_input = False 
???                    base_output = len(dataset["y"].shape) == 1
???                task_i = self.task_to_index[task] 
???                task_embeddings[task_i, :] = self.get_task_embedding(dataset, 
???                                                                     base_input=base_input,
???                                                                     base_output=base_output)
???
???            if meta_task is not None:
???                meta_dataset = self.get_meta_dataset(meta_task)
???                task_embeddings = self.get_outputs(meta_dataset,
???                                                   {"x": task_embeddings},
???                                                   base_input=False,
???                                                   base_output=False)
???
???            for i in range(num_hidden_hyper):
???                fout.write(("%i, " %i) + (format_string % tuple(task_embeddings[:, i])))
???
???
???    def save_input_embeddings(self, filename):
???        """Saves embeddings of all possible inputs."""
???        raw_data, _ = datasets.X0_dataset(num_input)
???        if tf_pm:
???            raw_data = 2*raw_data - 1
???        names = [("%i" * num_input) % tuple(raw_data[i, :]) for i in range(len(raw_data))]
???        format_string = ", ".join(["%f" for _ in names]) + "\n"
???        with open(filename, "w") as fout:
???            fout.write("dimension, x0, x1, " + ", ".join(names) + "\n")
???            for perm in _get_perm_list_template(num_input):
???                this_perm = np.array(perm)
???                x_data = raw_data.copy()
???                x0 = np.where(this_perm == 0)[0] # index of X0 in permuted data
???                x1 = np.where(this_perm == 1)[0]
???
???                x_data = x_data[:, this_perm] 
???                
???                if cue_dimensions:
???                    cue_data = np.zeros_like(x_data)
???                    cue_data[:, [x0, x1]] = 1.
???                    x_data = np.concatenate([x_data, cue_data], axis=-1)
???
???                input_embeddings = self.sess.run(self.processed_input,
???                                                 feed_dict = {
???                                                     self.base_input_ph: x_data,
???                                                     self.is_base_input: True,
???                                                     self.base_target_ph: self.dummy_base_output,
???                                                     self.meta_input_ph: self.dummy_meta_input,
???                                                     self.meta_target_ph: self.dummy_meta_output
???                                                     })
???                
???                for i in range(num_hidden_hyper):
???                    fout.write(("%i, %i, %i, " %(i, x0, x1)) + (format_string % tuple(input_embeddings[:, i])))
???                
???## running stuff
???
???for run_i in range(run_offset, run_offset+num_runs):
???    for meta_two_level in [True, False]: 
???        np.random.seed(run_i)
???        perm_list_dict = {task: (np.random.permutation(_get_perm_list_template(num_input)) if task not in ["XO", "NOTX0"] else np.random.permutation(_get_single_perm_list_template(num_input))) for task in total_tasks} 
???        tf.set_random_seed(run_i)
???        filename_prefix = "m2l%r_run%i" %(meta_two_level, run_i)
???        print("Now running %s" % filename_prefix)
???
???        model = meta_model(num_input, base_tasks, base_task_repeats, new_tasks,
???                           base_meta_tasks, base_meta_mappings, new_meta_tasks,
???                           meta_two_level=meta_two_level) 
???        model.save_embeddings(filename=output_dir + filename_prefix + "_init_embeddings.csv")
???
???#        cProfile.run('model.train_base_tasks(filename=output_dir + filename_prefix + "_base_losses.csv")')
???#        exit()
???        model.train_base_tasks(filename=output_dir + filename_prefix + "_base_losses.csv")
???        model.save_embeddings(filename=output_dir + filename_prefix + "_guess_embeddings.csv")
???        if save_input_embeddings:
???            model.save_input_embeddings(filename=output_dir + filename_prefix + "_guess_input_embeddings.csv")
???        for meta_task in base_meta_mappings:
???            model.save_embeddings(filename=output_dir + filename_prefix + "_" + meta_task + "_guess_embeddings.csv",
???                                  meta_task=meta_task)
???
???        model.train_new_tasks(filename_prefix=output_dir + filename_prefix + "_new_")
???        model.save_embeddings(filename=output_dir + filename_prefix + "_final_embeddings.csv")
???        for meta_task in base_meta_mappings:
???            model.save_embeddings(filename=output_dir + filename_prefix + "_" + meta_task + "_final_embeddings.csv",
???                                  meta_task=meta_task)
???
???
???        tf.reset_default_graph()
???
