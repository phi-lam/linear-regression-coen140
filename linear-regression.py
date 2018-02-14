#PHI LAM
#COEN 140 HOMEWORK 2
#linear-regression.py

################################# IMPORTS ###################################
import numpy as np
import math
import sys
from operator import itemgetter

########################## VARIABLE DECLARATIONS ###########################

training_cols = 96        # 95 features, 1st column is ViolentCrimesPerPop
training_rows = 1595      # 1595 instances
test_cols = 96			  # 95 features, 1st col is ViolentCrimesPerPop
test_rows = 399           # 399 instances

#--- Exact copies of data files
source_training_data = [[None for x in range(training_cols)] for y in range(1+training_rows)]
source_test_data = [[None for x in range(test_cols)] for y in range(1+test_rows)]

#--- Math-ready matrices
training_data = [[None for x in range(training_cols)] for y in range(training_rows)]
training_y = [[0.0 for x in range(1)] for y in range(training_rows)]
training_x = [[0.0 for x in range(training_cols - 1)] for y in range(training_rows)]
test_y = [[0.0 for x in range(1)] for y in range(test_rows)]
test_x = [[0.0 for x in range(test_cols - 1)] for y in range(test_rows)]


#---For debugging---
# print(eu_dist, '| diff: ', diff, end='\r')  # print overwriting one line
########################## FUNCTION DEFINITIONS #############################

#-----------------------------------------------------------------
#	FUNCTION: compute_rmse(predictions)
#
#	Inputs: 1) <predictions> given from linreg_predict()
#			2) <target_y>: either training_y or test_y
#

def compute_rmse(predictions, target_y):
	return np.sqrt(((predictions - target_y) ** 2).mean())

#-----------------------------------------------------------------
#	FUNCTION: lossFunction()
#
def lossFunction(data_x, data_y, w):
	r1 = np.matmul(data_x, w)
	r2 = np.subtract(r1, data_y)

	result = np.matmul(np.transpose(r2), r2)
	return result


#-----------------------------------------------------------------
#	FUNCTION: populate()
#   Input: training-data.txt
#	Procedure:
#		1) Entire data set goes into source_training_data
#		2) First column becomes 'y', the rest is 'x'
#

def populate_training(in_file, source_array, data_y, data_x):
	global training_y, training_x, training_data
	for sampleid, line in enumerate(in_file):
		features = line.split()
		for featureid, feature in enumerate(features):
			feature = feature.strip()
			source_array[sampleid][featureid] = feature

	#---Convert to numpy arrays
	#--- Import the source data as numpy array, delete strings, convert to floats
	full_list = np.asarray(source_array)
	full_list = np.delete(full_list, 0, axis=0)
	full_list = full_list.astype(float)
	temp = full_list
	training_data = full_list
	# print(full_list)
	# training_data = np.delete(training_data, 4, 1)

	#--- Separate y and x
	data_y = full_list[:,[0]]
	data_x = np.delete(temp, 0, axis=1)

	training_y = data_y
	training_x = data_x
	#-----Debugging-------
	#print("==== data_y ====\n", np.shape(data_y))
	# print(data_y)
	#print("==== data_x ====\n", np.shape(data_x))
	# print(data_x)

#-----------------------------------------------------------------
#	FUNCTION: populate_test()
#   Input: test-data.txt
#	Procedure:
#		1) Entire data set goes into source_training_data
#		2) First column becomes 'y', the rest is 'x'
#

def populate_test(in_file, source_array, data_y, data_x):
	global test_y, test_x
	for sampleid, line in enumerate(in_file):
		features = line.split()
		for featureid, feature in enumerate(features):
			feature = feature.strip()
			source_array[sampleid][featureid] = feature

	#---Convert to numpy arrays
	#--- Import the source data as floats (also skips first row of text)
	full_list = np.asarray(source_array)
	full_list = np.delete(full_list, 0, axis=0)
	full_list = full_list.astype(float)
	temp = full_list

	# print(full_list)
	# training_data = np.delete(training_data, 4, 1)

	#--- Separate y and x
	data_y = full_list[:,[0]]
	data_x = np.delete(temp, 0, axis=1)

	test_y = data_y
	test_x = data_x
	#-----Debugging-------
	#print("==== data_y ====\n", np.shape(data_y))
	# print(data_y)
	#print("==== data_x ====\n", np.shape(data_x))
	# print(data_x)

#------------------------------------------------------------------
#	FUNCTION: linreg_calculateW()
#	Description:
#		Given <y> and <x> vectors and the number of instances,
#		return <W>, the vector of weights.
#
#		Should be identical to ridgereg_calculateW, but without lambda
#
#		1) add a column of 1's to 'x'
#

def linreg_calculateW(data_y, data_x):
	#--- Prep X
	rows = data_x.shape[0]
	data_x = np.c_[data_x, np.ones(rows)]
	x_trans = np.transpose(data_x)

	#--- Calculate w (weights), w1 and w2 simply split up the math
	w1 = np.linalg.inv(np.matmul(x_trans, data_x))
	w2 = np.matmul(x_trans, data_y)
	w = np.matmul(w1, w2)
	return w
	# print("### Added column of 1's \n",  data_x)

#-----------------------------------------------------------------
#	FUNCTION: linreg_predict(weights, test_x)
#
#	Description: Returns <y>, vector of predicted values
#				text_x represents vector of test data
#
#	Procedure:
#		1)  Prep inputs: transpose W, add 1's column to test_x
#
def linreg_predict(weights, _test_x):
	#---Prep inputs
	rows = _test_x.shape[0]
	_test_x = np.c_[_test_x, np.ones(rows)]
	w_trans = np.transpose(weights)

	y = (test_rows, 1)
	y = np.zeros(y)

	# print("### w_trans ###\n", np.shape(w_trans))
	# print("### weights ###\n", np.shape(weights))
	# print("### test_x ###\n", np.shape(_test_x))

	#----test_x is transposed for multiplication to succeed
	y = np.matmul(w_trans, np.transpose(_test_x))
	#----revert shape of y
	y = np.transpose(y)

	# print("linreg_predict()", np.shape(y))
	return y

#-----------------------------------------------------------------
#	FUNCTION: crossval_split_data()
#
#	Description:
#		Slices the training data into a fraction of n_fold partitions.
#		Returns a dict of two values: the resulting training data and test data
#		after the slicing is complete.
#
def crossval_split_data(train, iteration, n_fold):
	start_idx = int(iteration * training_rows / n_fold)
	end_idx = int((iteration + 1) * training_rows / n_fold)
	# print(start_idx)
	# print(end_idx)


	test_slice = train[start_idx : end_idx]	#slice of first column
	_test_y = test_slice[:,[0]]
	_test_x = np.delete(test_slice, 0, axis=1)

	train_slice = np.delete(train, np.s_[start_idx:end_idx], axis=0)
	_train_y = train_slice[:,[0]]
	_train_x = np.delete(train_slice, 0, axis=1)

	data_dict = {'train_x': _train_x, 'train_y': _train_y, 'test_x': _test_x, 'test_y': _test_y}
	return data_dict

#-----------------------------------------------------------------
#	FUNCTION: ridgereg_calculateW(data_y, data_x, lambda)
#
#	Description:
#
def ridgereg_calculateW(data_y, data_x, lam):
	#---- Prep X: add column of 1's, create ID matrix
	rows = data_x.shape[0]
	cols = data_x.shape[1]
	data_x = np.c_[data_x, np.ones(rows)]
	id_mtx = np.identity(cols + 1)
	x_trans = np.transpose(data_x)

	#print(np.shape(data_x))
	#print(np.shape(data_y))

	w1 = np.dot(x_trans, data_x)
	w2 = lam * id_mtx

	#print("#####", np.shape(w2))
	w3 = np.add(w1, w2)
	w4 = np.linalg.inv(w3)
	w5 = np.matmul(w4, x_trans)
	w = np.matmul(w5, data_y)

	# print("################# \n", w5)

	return w
#-----------------------------------------------------------------
#	FUNCTION: ridgereg_predict(weights, test_x)
#
#	Description: Returns <y>, vector of predicted values
#				text_x represents vector of test data
#
def ridgereg_predict(weights, _test_x):
	return linreg_predict(weights, _test_x)
	# w_trans = np.transpose(weights)
    #
	# y = (test_rows, 1)
	# y = np.zeros(y)
    #
	# print("### w_trans ###\n", np.shape(w_trans))
	# print("### weights ###\n", np.shape(weights))
	# print("### test_x ###\n", np.shape(_test_x))
    #
	# #----test_x is transposed for multiplication to succeed
	# y = np.matmul(w_trans, np.transpose(_test_x))
	# #----revert shape of y
	# y = np.transpose(y)
    #
	# # print("linreg_predict()", np.shape(y))
	# return y

#-----------------------------------------------------------------
#	FUNCTION: cross_validate()
#		TODO: modify to take in a vector of weights (probably from crossval_split)
#		TODO:
#	Description: Finds the optimal lambda in a ridge regression
#
def cross_validate(lam, trials, n_fold):
	debugging = False
	optimal_lam = lam
	current_error = 1.0
	best_error = current_error
	sum = 0

	for i in range(trials):
		if debugging: print("\n ====== Lambda:", lam, " ======")
		sum = 0

		for j in range(n_fold):
			if debugging: print("Round: ", ((n_fold*i) + j)+1)	##print round number
			# 1) Prep training and test data
			data_dict = crossval_split_data(training_data, j, n_fold)
			r_train_x = data_dict['train_x']
			r_train_y = data_dict['train_y']
			r_test_x = data_dict['test_x']
			r_test_y = data_dict['test_y']

			# 2) Retrain model
			ridge_w = ridgereg_calculateW(r_train_y, r_train_x, lam)

			# 3) Test error for given lambda
			# print(np.shape(ridge_w))
			# print(np.shape(r_test_x))
			predictions = ridgereg_predict(ridge_w, r_test_x)
			error = compute_rmse(predictions, r_test_y)
			if debugging: print(" ## Error: ", error)# " || Lambda: ", lam)

			# 4) Calculate mean error, compare to others
			sum += error

		#print("-- Sum -- ", sum)
		current_error = sum / n_fold
		if debugging: print("\n # Average Error", current_error)
		if current_error < best_error:
			best_error = current_error
			optimal_lam = lam

		lam = lam / 2
		#exit()		# JUST FOR Debugging
	print("\n== Optimal Lambda ==\n", optimal_lam)
	if debugging: print("\n== Best Error ==\n", best_error)
	return optimal_lam

#-----------------------------------------------------------------
#	FUNCTION: ridge_reg()	TODO: once cross_validate is changed, this needs
#									to be fixed as well
#
#	Description: Returns <y>, vector of predicted values
#				text_x represents vector of test data
#
def ridge_reg():
	lam = 400
	return cross_validate(lam, 10, 5)

#-----------------------------------------------------------------
#	FUNCTION: grad_desc()
#
#	Math: w^(t+1) = w^t + a(xT(y-xw^t) - lam * w^t)
#
#	tolerance = epsilon
#	alpha = step size
#
#	Math stops when Euclid distance converges

def grad_desc(data_x, data_y, lam, step, tolerance):
	#--- Initialize gradient descent (find w0), add col of 1's to x
	w_curr = np.random.uniform(0,1,(96,1))
	rows = data_x.shape[0]
	data_x = np.c_[data_x, np.ones(rows)]
	diff = 1
	x_trans = np.transpose(data_x)
	#--- calculate w1 (right-to-left)
	r1 = lam * w_curr
	r2 = np.matmul(data_x, w_curr)
	r3 = np.subtract(data_y, r2)
	r4 = np.matmul(x_trans, r3)
	r5 = np.subtract(r4, r1)
	result = step * r5
	w_next = np.add(w_curr, result)

	# print('tolerance: ', tolerance)

	while (abs(diff) > tolerance):
		w_curr = w_next
		r1 = lam * w_curr
		r2 = np.matmul(data_x, w_curr)
		r3 = np.subtract(data_y, r2)
		r4 = np.matmul(x_trans, r3)
		r5 = np.subtract(r4, r1)
		result = step * r5
		w_next = np.add(w_curr, result)
		diff = np.amin(np.subtract(w_next, w_curr))

		#---For debugging
		# eu_dist = lossFunction(data_x, data_y, w_next)
		# print(eu_dist, '| diff: ', diff, end='\r')  # print overwriting one line
		print('coeff. difference: ', abs(diff), end='\r')  # print overwriting one line

	# print()
	#print("\nConvergence value: ", lossFunction(data_x, data_y, w_next))
	return w_next

#-----------------------------------------------------------------
#	FUNCTION: cross_validate_grad_desc() TODO: rewrite cross_validate
#
#	Description: This would be integrated into the old one if i had
#				more time.
#

def cross_validate_grad_desc(lam, trials, n_fold):
	debugging = False
	optimal_lam = lam
	#print(lam)
	current_error = 1.0
	best_error = current_error
	sum = 0
	step = 0.00001			# 1e-5 four 0's and a 1
	tolerance = 0.00001 	# 1e-5 four 0's and a 1

	for i in range(trials):
		if debugging: print("\n ====== Lambda:", lam, " ======")
		sum = 0

		for j in range(n_fold):
			if debugging: print("Round: ", ((n_fold*i) + j)+1)	##print round number
			# 1) Prep training and test data
			data_dict = crossval_split_data(training_data, j, n_fold)
			r_train_x = data_dict['train_x']
			r_train_y = data_dict['train_y']
			r_test_x = data_dict['test_x']
			r_test_y = data_dict['test_y']

			# 2) Retrain model
			ridge_w = grad_desc(r_train_x, r_train_y, lam, step, tolerance)
			# 3) Test error for given lambda
			# print(np.shape(ridge_w))
			# print(np.shape(r_test_x))
			predictions = ridgereg_predict(ridge_w, r_test_x)
			error = compute_rmse(predictions, r_test_y)

			if debugging: print(" ### Error: ", error, "                             ")# " || Lambda: ", lam)

			# 4) Calculate mean error, compare to others
			sum += error

		#print("-- Sum -- ", sum)
		current_error = sum / n_fold
		if debugging: print("\n # Average Error", current_error)
		if current_error < best_error:
			best_error = current_error
			optimal_lam = lam

		lam = lam / 2
		#exit()		# JUST FOR Debugging
	print("\n== Optimal Lambda ==\n", optimal_lam)
	print("\n== Best Error ==\n", best_error)
	return optimal_lam


################################# main ###################################

# ---- Opening files/correcting user input ----
if len(sys.argv) == 3:
	try:
		f1 = open(sys.argv[1])
		f2 = open(sys.argv[2])
	except:
		print("Usage: arguments must be text files")
		exit()
elif len(sys.argv) == 4:
	try:
		f1 = open(sys.argv[1])
		f2 = open(sys.argv[2])
		sys.stdout = open(sys.argv[3], "w")
	except:
		print("Usage: arguments must be text files")
		exit()
else:
	print("Usage: linear-regression.py training.txt test.txt out_file.txt")
	print("Note: if out_file.txt missing, will print to stdout")
	exit()

# ---- Take input data ----
print("\n--- Processing training data ---")
populate_training(f1, source_training_data, training_y, training_x)
print("--- Done ---")

print("--- Processing test data ---")
populate_test(f2, source_test_data, test_y, test_x)
print("--- Done ---")

# ---- Linear Regression ----

#####################################################################

print("\n=========================================================")
print("==================== LINEAR REGRESSION ==================")
print("=========================================================")
linreg_weights = linreg_calculateW(training_y, training_x)
#print(predictions)

print("\n=====================================")
print("  Linear Regression (vs. test data)  ")
print("=====================================")
predictions_lr_test = linreg_predict(linreg_weights, test_x)
test_error_LR = compute_rmse(predictions_lr_test, test_y)
print("RMSE Error: ", test_error_LR)

print("\n=======================================")
print(" Linear Regression (vs. training data) ")
print("=======================================")
predictions_lr_train = linreg_predict(linreg_weights, training_x)
training_error_LR = compute_rmse(predictions_lr_train, training_y)
print("RMSE Error: ", training_error_LR)

# ---- Ridge Regression ----
print("\n=========================================================")
print("==================== RIDGE REGRESSION ==================")
print("========================================================")

print("\n========================")
print(" Finding optimal lambda ")
print("========================")
opt_lam = ridge_reg()

print("\n=====================================")
print("  Ridge Regression (vs. test data)  ")
print("=====================================")
#print(opt_lam)
weights_RR = ridgereg_calculateW(training_y, training_x, opt_lam)
predictions_RR = ridgereg_predict(weights_RR, test_x)
test_error_RR = compute_rmse(predictions_RR, test_y)
print("RMSE Error: ", test_error_RR)

##############################################################################
######################## GRADIENT DESCENT ####################################
##############################################################################

print("\n=========================================================")
print("================= LINEAR GRADIENT DESCENT ===============")
print("=========================================================")
step = 0.00001			# 1e-5: four 0's and a 1
tolerance = 0.00001 	# 1e-5
lam = 0					# for linear gradient descent, lambda = 0

print("\n--- Calculating weights ---")
weights_LGD = grad_desc(training_x, training_y, lam, step, tolerance)
print("--- Done ---                                           ")

print("\n========================================")
print(" Lin. Gradient Descent (vs. test data) ")
print("=======================================")
predictions_LGD = linreg_predict(weights_LGD, test_x)
test_error_LGD = compute_rmse(predictions_LGD, test_y)
print("RMSE Error: ", test_error_LGD)
# Current result: 0.14582817901908202

print("\n===========================================")
print(" Lin. Gradient Descent (vs. training data) ")
print("===========================================")
predictions_LGD_train = linreg_predict(weights_LGD, training_x)
test_error_LGD_train = compute_rmse(predictions_LGD_train, training_y)
print("RMSE Error: ", test_error_LGD)
# Current result: 0.14582817901908202


print("\n=====================================================")
print("========= RIDGE REGRESSION GRADIENT DESCENT =========")
print("=====================================================")
print(" --- Determining optimal lambda --- ")
opt_lam = cross_validate_grad_desc(400, 10, 5)	#10 trials, 5 rounds per trial
print("== Optimal Lambda ==\n ", opt_lam)

print("\n=========================================")
print(" Ridge Gradient Descent (vs. test data) ")
print("=========================================")
weights_RGD = grad_desc(training_x, training_y, opt_lam, step, tolerance)
predictions_RGD = linreg_predict(weights_RGD, test_x)
test_error_RGD = compute_rmse(predictions_RGD, test_y)
print("RMSE Error: ", test_error_RGD, "                          ")
# Current result with lambda = 25: 0.1549461254608019447693



#TODO: fix other functions so that weights don't need to be recalculated so many
#	times


print()
#end
