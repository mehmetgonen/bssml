function state = bssml_semisupervised_classification_variational_train(X, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(X, 1);
    N = size(X, 2);
    L = size(Y, 1);
    R = parameters.R;
    sigma_z = parameters.sigma_z;

    switch parameters.prior_phi
        case 'ard'
            phi.alpha = (parameters.alpha_phi + 0.5 * D) * ones(R, 1);
            phi.beta = parameters.beta_phi * ones(R, 1);
        otherwise
            Phi.alpha = (parameters.alpha_phi + 0.5) * ones(D, R);
            Phi.beta = parameters.beta_phi * ones(D, R);
    end
    Q.mu = randn(D, R);
    Q.sigma = repmat(eye(D, D), [1, 1, R]);
    Z.mu = randn(R, N);
    Z.sigma = eye(R, R);
    lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(L, 1);
    lambda.beta = parameters.beta_lambda * ones(L, 1);
    Psi.alpha = (parameters.alpha_psi + 0.5) * ones(R, L);
    Psi.beta = parameters.beta_psi * ones(R, L);    
    bW.mu = randn(R + 1, L);
    bW.sigma = repmat(eye(R + 1, R + 1), [1, 1, L]);
    
    labeled = ~isnan(Y);
    unlabeled = isnan(Y);
    T.mu = zeros(L, N);
    for o = 1:L
        yo = Y(o, :);
        indices = knnsearch(X(:, labeled(o, :))', X(:, unlabeled(o, :))');
        labels = yo(labeled(o, :));
        yo(unlabeled(o, :)) = labels(indices);
        T.mu(o, :) = (abs(randn(1, N)) + 0.5 * labeled(o, :)) .* sign(yo);
    end

    XXT = X * X';
    
    gammap = 0.5 * ones(L, 1);
    gamman = 1 - gammap;
    
    lower = -1e40 * ones(L, N);
    lower(Y > 0) = 0;
    upper = +1e40 * ones(L, N);
    upper(Y < 0) = 0;
    
    phi_indices = repmat(logical(eye(D, D)), [1, 1, R]);
    psi_indices = repmat(logical(diag([0, ones(1, R)])), [1, 1, L]);

    %%%% start iterations
    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        switch parameters.prior_phi
            case 'ard'
                %%%% update phi
                for s = 1:R
                    phi.beta(s) = 1 / (1 / parameters.beta_phi + 0.5 * (Q.mu(:, s)' * Q.mu(:, s) + sum(diag(Q.sigma(:, :, s)))));
                end
                %%%% update Q
                for s = 1:R
                    Q.sigma(:, :, s) = (phi.alpha(s) * phi.beta(s) * eye(D, D) + XXT / sigma_z^2) \ eye(D, D);
                    Q.mu(:, s) = Q.sigma(:, :, s) * (X * Z.mu(s, :)' / sigma_z^2);
                end
            otherwise
                %%%% update Phi
                Phi.beta = 1 ./ (1 / parameters.beta_phi + 0.5 * (Q.mu.^2 + reshape(Q.sigma(phi_indices), D, R)));
                %%%% update Q
                for s = 1:R
                    Q.sigma(:, :, s) = (diag(Phi.alpha(:, s) .* Phi.beta(:, s)) + XXT / sigma_z^2) \ eye(D, D);
                    Q.mu(:, s) = Q.sigma(:, :, s) * (X * Z.mu(s, :)' / sigma_z^2);
                end
        end
        %%%% update Z
        Z.sigma = (eye(R, R) / sigma_z^2 + bW.mu(2:R + 1, :) * bW.mu(2:R + 1, :)' + sum(bW.sigma(2:R + 1, 2:R + 1, :), 3)) \ eye(R, R);
        Z.mu = Z.sigma * (Q.mu' * X / sigma_z^2 + bW.mu(2:R + 1, :) * T.mu - repmat(bW.mu(2:R + 1, :) * bW.mu(1, :)' + sum(bW.sigma(2:R + 1, 1, :), 3), 1, N));
        %%%% update lambda
        lambda.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * (bW.mu(1, :)'.^2 + squeeze(bW.sigma(1, 1, :))));
        %%%% update psi
        Psi.beta = 1 ./ (1 / parameters.beta_psi + 0.5 * (bW.mu(2:R + 1, :).^2 + reshape(bW.sigma(psi_indices), R, L)));
        %%%% update b and w
        for o = 1:L
            bW.sigma(:, :, o) = [lambda.alpha(o) * lambda.beta(o) + N, sum(Z.mu, 2)'; sum(Z.mu, 2), diag(Psi.alpha(:, o) .* Psi.beta(:, o)) + Z.mu * Z.mu' + N * Z.sigma] \ eye(R + 1, R + 1);
            bW.mu(:, o) = bW.sigma(:, :, o) * ([ones(1, N); Z.mu] * T.mu(o, :)');
        end
        %%%% update T
        output = bW.mu' * [ones(1, N); Z.mu];
        alpha_norm = lower(labeled) - output(labeled);
        beta_norm = upper(labeled) - output(labeled);
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        T.mu(labeled) = output(labeled) + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;        
        for o = 1:L
            normalization = gamman(o) * normcdf(-0.5 - T.mu(o, unlabeled(o, :))) + gammap(o) * normcdf(T.mu(o, unlabeled(o, :)) - 0.5);
            expectationn = output(o, unlabeled(o, :)) + (normpdf(-Inf - output(o, unlabeled(o, :))) - normpdf(-0.5 - output(o, unlabeled(o, :)))) ./ (normcdf(-0.5 - output(o, unlabeled(o, :))) - normcdf(-Inf - output(o, unlabeled(o, :))));
            expectationn(expectationn < -40) = -40;
            expectationp = output(o, unlabeled(o, :)) + (normpdf(0.5 - output(o, unlabeled(o, :))) - normpdf(Inf - output(o, unlabeled(o, :)))) ./ (normcdf(Inf - output(o, unlabeled(o, :))) - normcdf(0.5 - output(o, unlabeled(o, :))));
            expectationp(expectationp > +40) = +40;
            T.mu(o, unlabeled(o, :)) = (1 ./ normalization) .* (gamman(o) * normcdf(-0.5 - output(o, unlabeled(o, :))) .* expectationn + gammap(o) * normcdf(output(o, unlabeled(o, :)) - 0.5) .* expectationp);
        end
    end

    switch parameters.prior_phi
        case 'ard'
            state.phi = phi;
        otherwise
            state.Phi = Phi;
    end
    state.Q = Q;
    state.lambda = lambda;
    state.Psi = Psi;
    state.bW = bW;
    state.parameters = parameters;
end
