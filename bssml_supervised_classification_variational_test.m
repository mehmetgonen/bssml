function prediction = bssml_supervised_classification_variational_test(X, state)
    N = size(X, 2);
    L = size(state.bW.mu, 2);
    R = size(state.bW.mu, 1) - 1;

    prediction.Z.mu = zeros(R, N);
    prediction.Z.sigma = zeros(R, N);
    for s = 1:R
        prediction.Z.mu(s, :) = state.Q.mu(:, s)' * X;
        prediction.Z.sigma(s, :) = state.parameters.sigma_z^2 + diag(X' * state.Q.sigma(:, :, s) * X);
    end

    prediction.T.mu = state.bW.mu' * [ones(1, N); prediction.Z.mu];
    prediction.T.sigma = zeros(L, N);
    for o = 1:L
        prediction.T.sigma(o, :) = 1 + diag([ones(1, N); prediction.Z.mu]' * state.bW.sigma(:, :, o) * [ones(1, N); prediction.Z.mu]);
    end

    prediction.P = 1 - normcdf(-prediction.T.mu ./ prediction.T.sigma);
end
