"""Extrapolates period mortality rates using a neural network. (C) Averisera Ltd 2017-2018.

NN is trained to extrapolate a sequence of rates r_0, ..., r_{N-1}, returning r_{N+K-1}.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf



# Whether the NN fits log(rate) or rate
FIT_LOGS = True

# Maximum year for extrapolation.
MAX_EXTRAPOLATION_YEAR = 2200
# Maximum year for which we need the gradients of extrapolated rates.
MAX_EXTRAPOLATION_YEAR_GRAD = 2100
# Ages for which we need the gradients of extrapolated rates.
AGES_GRAD = [0, 5, 10, 15, 20, 30, 45, 60, 65, 75, 85, 90, 100]

# Maximum year for historical data.
MAX_YEAR_HIST = 2016

# Where to save results
RESULTS_BASE = "results%d" % MAX_YEAR_HIST

# Where to save TensorFlow checkpoints
CHECKPOINTS_BASE = "checkpoints%d" % MAX_YEAR_HIST

# How many extrapolations during training
N_TRAIN = 10


def load_data(path):
	"""Load data from path."""
	df = pd.DataFrame.from_csv(path, parse_dates=False)
	ages = df.index.astype(int)
	years = df.columns.astype(int)
	values = np.log(df.values) if FIT_LOGS else df.values
	return years, ages, values
	

def create_sequence_queue(rates, sequence_length, batch_size=None, shuffle=True, num_epochs=None):
	"""Create a sequence queue."""
	num_ages, num_years = rates.shape
	if sequence_length > num_years:
		raise ValueError("Total sequence length too large")
	num_sequences_per_age = num_years - sequence_length + 1
	num_sequences = num_ages * num_sequences_per_age
	data = np.zeros([num_sequences, sequence_length])
	k = 0
	for rates_row in rates: # Iterate over rows.
		for j in range(num_sequences_per_age):
			data[k, :] = rates_row[j:(j + sequence_length)]
			k += 1
	dataset = tf.data.Dataset.from_tensor_slices(data)
	if shuffle:
		dataset = dataset.shuffle(buffer_size=1000)
	if batch_size is None:
		batch_size = num_sequences
	dataset = dataset.batch(batch_size)
	if num_epochs is not None:
		# repeat the data num_epochs times
		dataset = dataset.repeat(num_epochs)
	else:
		# repeat the data indefinitely 
		dataset = dataset.repeat()
	sequences = dataset.make_one_shot_iterator().get_next()
	return sequences
	
	
def build_cell_neural_network(inputs, num_layers, hidden_layer_size, output_size, scope_name="fcn", weights_stdev=0.0):
	"""Build fully connected cell neural network graph for given inputs.
	
	Args:
		inputs: Input tensor, shape = [batch size, copula dimension]
		num_layers: Number of layers
		hidden_layer_size: Size of the hidden (i.e. each except for the last one) layer
		output_size: Size of the last layer
	Returns:
		output: tensor with shape (batch size, output_size)
	"""
	if num_layers < 1:
		raise ValueError("Need at least 1 layer")
	x = inputs
	logging.debug("x == %s", x)
	bias_initializer = tf.constant_initializer(0.1)
	if weights_stdev:
		weight_initializer=tf.truncated_normal_initializer(stddev=weights_stdev)
	else:
		weight_initializer = tf.constant_initializer(0.0)
	for layer_idx in range(num_layers): # loop over network layers
		# x.shape[0] is the batch size, x.shape[1] is the vector length
		x_size = int(x.shape[1]) 
		
		is_hidden = layer_idx + 1 < num_layers # last layer is not "hidden"
		y_size = hidden_layer_size if is_hidden else output_size
		activation = tf.nn.relu if is_hidden else None
		
		x = tf.contrib.layers.fully_connected(x, y_size, activation_fn=activation,
			weights_initializer=weight_initializer,
			biases_initializer=bias_initializer,
			reuse=tf.AUTO_REUSE, scope="%s_%d" % (scope_name, layer_idx))
		logging.debug("y[%d] == %s", layer_idx, x)
	return x
	
	
def get_cell_neural_network_builder(*args, **kwargs):
	"""Returns a function which creates a cell NN which will be applied recursively.
	
	Builder should be called as builder(input_sequence).
	"""
	return lambda inputs: build_cell_neural_network(inputs, *args, **kwargs)
	
	
def apply_cell(cell_builder, total_sequence, input_size=None, output_size=None):
	batch_size, sequence_length = total_sequence.shape
	if input_size is None:
		input_size = sequence_length
	if output_size is None:
		output_size = sequence_length - input_size
	outputs = []
	input = total_sequence[:, :input_size]
	for i in range(output_size):
		output = cell_builder(input)
		assert output.shape[1] == 1, output
		outputs.append(output)
		input = tf.concat([input[:, 1:], output], axis=1)
	assert len(outputs) == output_size
	output = tf.concat(outputs, axis=1)
	return output

	
def split_indices_into_ABC(n, k):
	"""Split indices 0, ..., n-1 into sets A containing k-2/k of them and B and C containg 1/k of them each.
	Set A contains index 0.
	"""
	assert k > 2
	B = list(range(1, n, k))
	C = list(range(2, n, k))
	not_A = B + C
	A = [i for i in range(n) if i not in not_A]
	return A, B, C


def run(mode, run_idx, country, sex, input_size=40, num_layers=4, hidden_size=128, clip_gradient=False, restore=False, do_gradients=True, save_checkpoints=False):
	"""Args:
		mode: "train", "test" or "apply". Data are divided into sets A (60%), B (20%) and C(20%).
		run_idx: Index of the run (for CI calculations).
		country: "uk", "ew", ...
		sex: "male" or "female"
		input_size: Size of the input window.
		num_layers: Number of NN layers excluding the input layer
		clip_gradient: Whether to clip gradient values for stochastic gradient (helps to avoid blow-up)
		do_extrapolation: Whether to do extrapolation or test
		restore: Whether to load the trained model from disk.
		do_gradients: Whether to save gradients of outputs over inputs.
		save_checkpoints: Whether to save checkpoints with the model.
	"""
	basename = "%s-%s-mortality-period-qx-%d.csv" % (country, sex, MAX_YEAR_HIST)
	years, ages, mortality_rates = load_data(os.path.join("sources", basename))
	logging.debug("Years == %s", years)
	logging.debug("Ages == %s", ages)
	num_ages = len(ages)
	num_years = len(years)
	assert num_ages == mortality_rates.shape[0]
	assert num_years == mortality_rates.shape[1]
	max_input_year = max(years)
	
	results_dir = os.path.join(RESULTS_BASE, "%s_IS=%d" % (mode, input_size), str(run_idx))
	checkpoints_dir = os.path.join(CHECKPOINTS_BASE, "%s_IS=%d" % (mode, input_size), str(run_idx))
	for directory in [results_dir, checkpoints_dir]:
		ensure_dir(os.path.join(".", directory))
	
	# Reset calculation graph
	tf.reset_default_graph()
	
	weights_stdev = 1e-1
	train_target_size = N_TRAIN
	train_sequence_length = input_size + train_target_size	
	
	cell_nn_builder = get_cell_neural_network_builder(num_layers, hidden_size, 1, scope_name="qx_%s_%s" % (country, sex), weights_stdev=weights_stdev)
	
	logging.info("\n******\nCountry == %s, Sex == %s, Hidden Size == %d, Input Size == %d, Number Layers == %d\n******", country, sex, hidden_size, input_size, num_layers)
	logging.info("Saving in directories %s and %s", results_dir, checkpoints_dir)
	
	do_extrapolation = mode == "apply"
	
	if do_extrapolation:
		train_indices = list(range(num_ages))
		test_indices = []
	else:
		A, B, C = split_indices_into_ABC(num_ages, 5)
		if mode == "test":
			train_indices = A + B
			test_indices = C
		else:
			assert mode == "train", mode
			train_indices = A
			test_indices = B
	
	if test_indices:
		test_sequence = create_sequence_queue(mortality_rates[test_indices], num_years, batch_size=None, shuffle=False)
		test_output = apply_cell(cell_nn_builder, test_sequence, input_size=input_size)
		test_target = test_sequence[:, input_size:]
	
	train_sequence = create_sequence_queue(mortality_rates[train_indices], train_sequence_length, batch_size=16)
	train_output = apply_cell(cell_nn_builder, train_sequence, input_size=input_size)
	train_target = train_sequence[:, input_size:]
	
	# After we have created the graph
	trainable_variables = tf.trainable_variables()
	
	if do_extrapolation:
		# Extrapolate last input_size rates from every age group.
		# Do it age-group-by-age-group to speed up calculation of gradients.
		extrap_input = tf.constant(mortality_rates[:, -input_size:])
		num_extrapolated_years = MAX_EXTRAPOLATION_YEAR - max_input_year
		extrap_output = apply_cell(cell_nn_builder, extrap_input, output_size=num_extrapolated_years)
		assert extrap_output.shape == (num_ages, num_extrapolated_years), extrap_output
		if do_gradients:
			gradient_mask = tf.placeholder(tf.float64, shape=extrap_output.shape)
			extrap_gradient = tf.gradients(tf.reduce_sum(gradient_mask * extrap_output), extrap_input, stop_gradients=[gradient_mask])[0]
			assert extrap_gradient.shape == (num_ages, input_size), extrap_gradient
	
	# minimize L2 deviation
	train_residuals = train_output - train_target
	train_loss = tf.reduce_mean(tf.pow(train_residuals, 2))
	if not do_extrapolation:
		test_residuals = test_output - test_target
		test_loss = tf.reduce_mean(tf.pow(test_residuals, 2))
		test_bias = tf.reduce_mean(test_residuals)
	
	# learning rate for RMSProp must be small (it scales it up internally)
	# Graph operations to decrease the learning rate if required
	if mode in ("apply", "test"):
		n_steps = 0 if restore else 270001
	else:
		n_steps = 300001
	if n_steps:
		# Set up training
		initial_learning_rate = 1e-4
		log_learning_rate = tf .Variable(np.log(initial_learning_rate), name="log_learning_rate", trainable=False)
		log_learning_rate_delta = tf.constant(np.log(0.9), name="log_learning_rate_delta")
		decrease_learning_rate = tf.assign_add(log_learning_rate, log_learning_rate_delta,
			name="decrease_learning_rate")
		learning_rate = tf.exp(log_learning_rate, name="learning_rate")
		opt = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
		if clip_gradient:
			gradients, variables = zip(*opt.compute_gradients(train_loss))
			gradients_norm = tf.global_norm(gradients)
			gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
			opt_step = opt.apply_gradients(zip(gradients, variables))
		else:
			opt_step = opt.minimize(train_loss)
	
	step_delta = min(int(n_steps / 10), 10000)
	model_filename = os.path.join(checkpoints_dir, "M2_%s_%s_NL%d_HS%d_IS%d.ckpt" % (country, sex, num_layers, hidden_size, input_size))
	if save_checkpoints or restore:
		saver = tf.train.Saver()
	
	init = tf.global_variables_initializer()

	## To allow two processes at the same time.
	#config = tf.ConfigProto()
	#config.gpu_options.allow_growth = True
	#config.gpu_options.per_process_gpu_memory_fraction = 0.4
	
	#with tf.Session(config=config) as sess:
	with tf.Session() as sess:
		if restore:
			saver.restore(sess, model_filename)
			logging.info("Model restored.")
		else:
			sess.run(init)
		previous_loss = np.inf
		lowest_test_loss = np.inf
		lowest_test_loss_step = -1
		lowest_test_bias = np.inf
		for i in range(n_steps):
			_, current_loss = sess.run([opt_step, train_loss])
			if np.isnan(current_loss):
				logging.error("Loss is NaN, bailing out...")
				break
			if i % step_delta == 0:
				if current_loss >= previous_loss:
					current_learning_rate = sess.run(learning_rate)
					new_learning_rate = np.exp(sess.run(decrease_learning_rate))
					logging.info("Decreasing learning rate from %g to %g", current_learning_rate, new_learning_rate)
				previous_loss = current_loss		
				logging.info("Step %d: train loss == %g", i, current_loss)
				if mode == "train":
					test_loss_value, test_bias_value = sess.run([test_loss, test_bias])
					logging.info("Step %d: test loss == %g, test bias == %g", i, test_loss_value, test_bias_value)
					if test_loss_value < lowest_test_loss:
						lowest_test_loss = test_loss_value
						lowest_test_bias = test_bias_value
						lowest_test_loss_step = i
		if save_checkpoints:
			save_path = saver.save(sess, model_filename)
			logging.info("Model saved in path: %s" % save_path)
		if mode == "apply":
			min_year = min(years)
			extrap_years = list(range(min_year, MAX_EXTRAPOLATION_YEAR + 1))
			df = pd.DataFrame(index=ages, columns=extrap_years, dtype=float)
			for i, year in enumerate(range(min_year, max_input_year + 1)):
				df[year] = mortality_rates[:, i]
			extrap_output_data = sess.run(extrap_output)
			for i, year in enumerate(range(max_input_year + 1, MAX_EXTRAPOLATION_YEAR + 1)):
				df[year] = extrap_output_data[:, i]
			if FIT_LOGS:
				df = np.exp(df)
			df.to_csv(os.path.join(results_dir, "predicted-" + basename))
			logging.info("Saved extrapolation results.")
			if do_gradients:
				extrap_input_years = years[-input_size:]
				extrap_output_years = range(max_input_year + 1, MAX_EXTRAPOLATION_YEAR_GRAD + 1)
				for age in AGES_GRAD:
					age_idx = age - min(ages) # Assumes ages have no gaps.
					df = pd.DataFrame(index=extrap_output_years, columns=extrap_input_years)
					for j, year in enumerate(extrap_output_years):
						gradient_mask_data = np.zeros([num_ages, num_extrapolated_years], dtype=float)
						gradient_mask_data[age_idx, j] = 1.0
						extrap_gradient_data = sess.run(extrap_gradient, feed_dict={gradient_mask: gradient_mask_data})
						assert extrap_gradient_data.shape == (num_ages, input_size)
						df.loc[year] = extrap_gradient_data[age_idx, :]
					df.to_csv(os.path.join(results_dir, ("gradient-predicted-%d-" % age) + basename))
				logging.info("Saved gradients.")
		elif mode == "test":
			test_loss_value, test_bias_value = sess.run([test_loss, test_bias])
			logging.info("Test loss: %g, test bias: %g", test_loss_value, test_bias_value)
		else:
			logging.info("Lowest test loss %g after %i steps (corresponding test bias %g)", lowest_test_loss, lowest_test_loss_step, lowest_test_bias)		

			
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
	
	
if __name__ == "__main__":
	if len(sys.argv) <= 3:
		print("Run as %s <country> <sex> <mode> [num reps=1]" % sys.argv[0])
		sys.exit()
	country = sys.argv[1]
	sex = sys.argv[2]
	mode = sys.argv[3]
	
	log_filename = 'extrapolation_%s_%s_%s.log' % (country, sex, mode)
	print("Logging to %s" % log_filename)
	logging.basicConfig(filename=log_filename,
		level=logging.INFO,
		format='%(asctime)s %(levelname)-8s %(message)s',
		datefmt='%Y-%m-%d %H:%M:%S')
	
	# Optimised hyperparameters
	input_size = 40
	num_layers = 5
	# Number of hidden layers = num_layers - 1
	hidden_size = 64
	num_reps = 1
	
	if len(sys.argv) > 4:
		num_reps = int(sys.argv[4])
		
	clip_gradient = False
	restore = False
	do_gradients = True
	save_checkpoints = False
	
	for run_idx in range(num_reps):
		run(mode, run_idx, country, sex, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
			clip_gradient=clip_gradient, restore=restore, do_gradients=do_gradients, save_checkpoints=save_checkpoints)
		logging.info("--> FINISHED <--")
