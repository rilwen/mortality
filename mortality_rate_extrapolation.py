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

import itertools
import logging
logging.basicConfig(filename='extrapolation.log',level=logging.INFO)
#logging.basicConfig(level=logging.INFO)
import os

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

RESULTS = "results"
CHECKPOINTS = "checkpoints"


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
	
	
def apply_cell_with_targets(cell_builder, total_sequence, input_size):
	batch_size, sequence_length = total_sequence.shape
	output_size = sequence_length - input_size
	outputs = []
	targets = []
	input = total_sequence[:, :input_size]
	for i in range(output_size):
		output = cell_builder(input)
		assert output.shape[1] == 1, output
		j = input_size + i
		assert j < sequence_length
		assert j >= 0
		target = total_sequence[:, j:(j+1)]
		assert target.shape[1] == 1, target
		outputs.append(output)
		targets.append(target)
		input = tf.concat([input[:, 1:], output], axis=1)
	assert len(outputs) == output_size
	assert len(targets) == output_size
	output = tf.concat(outputs, axis=1)
	target = tf.concat(targets, axis=1)
	return output, target
	

def apply_cell(cell_builder, input, output_size):
	batch_size, input_size = input.shape
	outputs = []
	for i in range(output_size):
		output = cell_builder(input)
		assert output.shape[1] == 1, output
		outputs.append(output)
		input = tf.concat([input[:, 1:], output], axis=1)
	assert len(outputs) == output_size
	return tf.concat(outputs, axis=1)


def train_and_test(country, sex, num_layers=2, clip_gradient=False, do_extrapolation=False, pretrained_values=None, restore=False, do_gradients=False):
	"""Args:
		country: "uk", "ew", ...
		sex: "male" or "female"
		num_layers: Number of NN layers excluding the input layer
		clip_gradient: Whether to clip gradient values for stochastic gradient (helps to avoid blow-up)
		do_extrapolation: Whether to do extrapolation or test
		pretrained_values: Dictionary mapping variable names to their pretrained values
		restore: Whether to load the trained model from disk.
		do_gradients: Whether to save gradients of outputs over inputs (slow).
	"""
	basename = "%s-%s-mortality-period-qx.csv" % (country, sex)
	years, ages, mortality_rates = load_data(os.path.join("sources", basename))
	logging.debug("Years == %s", years)
	logging.debug("Ages == %s", ages)
	num_ages = len(ages)
	assert num_ages == mortality_rates.shape[0]
	assert len(years) == mortality_rates.shape[1]
	max_input_year = max(years)
	
	# Reset calculation graph
	tf.reset_default_graph()
	
	weights_stdev = 1e-1
	input_size = 25
	train_target_size = 10
	train_sequence_length = input_size + train_target_size
	hidden_size = 128
	
	cell_nn_builder = get_cell_neural_network_builder(num_layers, hidden_size, 1, scope_name="qx_%s_%s" % (country, sex), weights_stdev=weights_stdev)
	
	logging.info("\n******\nCountry == %s, Sex == %s, Hidden Size == %d, Input Size == %d, Number Layers == %d\n******", country, sex, hidden_size, input_size, num_layers)
	
	if do_extrapolation:
		train_indices = list(range(num_ages))
	else:
		train_test_ratio = 5 # Must be an int.
		test_indices = list(range(1, num_ages, train_test_ratio))
		train_indices = [i for i in range(num_ages) if i not in test_indices]
		test_sequence = create_sequence_queue(mortality_rates[test_indices], len(years), batch_size=None, shuffle=False)
		test_output, test_target = apply_cell_with_targets(cell_nn_builder, test_sequence, input_size)		
	
	train_sequence = create_sequence_queue(mortality_rates[train_indices], train_sequence_length, batch_size=16)
	train_output, train_target = apply_cell_with_targets(cell_nn_builder, train_sequence, input_size)
	
	# After we have created the graph
	trainable_variables = tf.trainable_variables()
	
	if do_extrapolation:
		# Extrapolate last input_size rates from every age group.
		# Do it age-group-by-age-group to speed up calculation of gradients.
		extrap_input = tf.constant(mortality_rates[:, -input_size:])
		num_extrapolated_years = MAX_EXTRAPOLATION_YEAR - max_input_year
		extrap_output = apply_cell(cell_nn_builder, extrap_input, num_extrapolated_years)
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
	
	# logging.info("Trainable variables: %s", tf.trainable_variables())
	initialisation_ops = []
	if pretrained_values:
		for variable in tf.trainable_variables():
			if variable.name in pretrained_values:
				init_value = pretrained_values[variable.name]
				if init_value.shape == variable.shape:
					initialisation_ops.append(tf.assign(variable, init_value))
	logging.info("Have %d pretrained initialisation values", len(initialisation_ops))
	
	# learning rate for RMSProp must be small (it scales it up internally)
	# Graph operations to decrease the learning rate if required
	if do_extrapolation:
		n_steps = 0 if restore else 200001
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
	model_filename = os.path.join(CHECKPOINTS, "M2_%s_%s_NL%d_HS%d_IS%d.ckpt" % (country, sex, num_layers, hidden_size, input_size))
	saver = tf.train.Saver()
	
	init = tf.global_variables_initializer()		
	
	with tf.Session() as sess:
		if restore:
			saver.restore(sess, model_filename)
			logging.info("Model restored.")
		else:
			sess.run(init)
		if initialisation_ops:
			sess.run(initialisation_ops)
		previous_loss = np.inf
		lowest_test_loss = np.inf
		lowest_test_loss_step = -1
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
				if do_extrapolation:
					logging.info("Step %d: train loss == %g", i, current_loss)
					save_path = saver.save(sess, model_filename)
					logging.info("Model saved in path: %s" % save_path)
				else:
					test_loss_value, test_bias_value = sess.run([test_loss, test_bias])
					logging.info("Step %d: train loss == %g, test loss == %g, test bias == %g", i, current_loss, test_loss_value, test_bias_value)
					if test_loss_value < lowest_test_loss:
						lowest_test_loss = test_loss_value
						lowest_test_loss_step = i
		if do_extrapolation:
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
			df.to_csv(os.path.join(RESULTS, "predicted-" + basename))
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
					df.to_csv(os.path.join(RESULTS, ("gradient-predicted-%d-" % age) + basename))
				logging.info("Saved gradients.")
		else:
			logging.info("Lowest test loss %g after %i steps", lowest_test_loss, lowest_test_loss_step)
			
		trained_values = sess.run(trainable_variables)
		return {variable.name : value for variable, value in zip(trainable_variables, trained_values)}
			
			
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
	
	
if __name__ == "__main__":
	clip_gradient = False
	restore = True
	num_layers = 4 # Excluding the input layer
	do_gradients = True
	for directory in [RESULTS, CHECKPOINTS]:
		ensure_dir(os.path.join(".", directory))
	logging.info("Saving in directories %s and %s", RESULTS, CHECKPOINTS)
	for do_extrapolation, country, sex in itertools.product([True], ["uk", "ew"], ["female", "male"]):
		train_and_test(country, sex, num_layers=num_layers, clip_gradient=clip_gradient,
				do_extrapolation=do_extrapolation, pretrained_values=None, restore=restore, do_gradients=do_gradients)