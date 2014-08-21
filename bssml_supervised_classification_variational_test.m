% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = bssml_supervised_classification_variational_test(X, state)
    N = size(X, 2);
    L = size(state.bW.mean, 2);
    R = size(state.bW.mean, 1) - 1;

    prediction.Z.mean = zeros(R, N);
    prediction.Z.covariance = zeros(R, N);
    for s = 1:R
        prediction.Z.mean(s, :) = state.Q.mean(:, s)' * X;
        prediction.Z.covariance(s, :) = state.parameters.sigmaz^2 + diag(X' * state.Q.covariance(:, :, s) * X);
    end

    prediction.T.mean = state.bW.mean' * [ones(1, N); prediction.Z.mean];
    prediction.T.covariance = zeros(L, N);
    for o = 1:L
        prediction.T.covariance(o, :) = 1 + diag([ones(1, N); prediction.Z.mean]' * state.bW.covariance(:, :, o) * [ones(1, N); prediction.Z.mean]);
    end

    prediction.P = 1 - normcdf(-prediction.T.mean ./ prediction.T.covariance);
end