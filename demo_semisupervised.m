%initalize the parameters of the algorithm
parameters = struct();

%set the hyperparameters of gamma prior used for projection matrix
parameters.alpha_phi = 1;
parameters.beta_phi = 1;

%set the hyperparameters of gamma prior used for bias parameters
parameters.alpha_lambda = 1;
parameters.beta_lambda = 1;

%set the hyperparameters of gamma prior used for weight parameters
parameters.alpha_psi = 1;
parameters.beta_psi = 1;

%%% IMPORTANT %%%
%For gamma priors, you can experiment with three different (alpha, beta) values
%(1, 1) => default priors
%(1e-10, 1e+10) => good for obtaining sparsity
%(1e-10, 1e-10) => good for small sample size problems

%set the number of iterations
parameters.iteration = 200;

%set the subspace dimensionality
parameters.R = 20;

%determine whether you want to use automatic relevance determination priors for projection matrix (ard or entrywise)
parameters.prior_phi = 'entrywise';

%set the sample size used to calculate the expectation of truncated normals
parameters.sample = 200;

%set the seed for random number generator used to initalize random variables
parameters.seed = 1606;

%set the standard deviation of projected instances
parameters.sigma_z = 0.1;

%initialize the data matrix and class labels for training
Xtrain = ??; %should be an D x Ntra matrix containing input data for training samples
Ytrain = ??; %should be an L x Ntra matrix containing class labels (contains only -1s, +1s, and NaNs) where L is the number of labels

%perform training
state = bssml_semisupervised_classification_variational_train(Xtrain, Ytrain, parameters);

%initialize the data matrix for testing
Xtest = ??; %should be an D x Ntest matrix containing input data for test samples

%perform prediction
prediction = bssml_semisupervised_classification_variational_test(Xtest, state);

%display the predicted probabilities
display(prediction.P);
