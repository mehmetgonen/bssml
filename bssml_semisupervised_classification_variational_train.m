% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bssml_semisupervised_classification_variational_train(X, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(X, 1);
    N = size(X, 2);
    L = size(Y, 1);
    R = parameters.R;
    sigmaz = parameters.sigmaz;

    switch parameters.prior_phi
        case 'ard'
            phi.shape = (parameters.alpha_phi + 0.5 * D) * ones(R, 1);
            phi.scale = parameters.beta_phi * ones(R, 1);
        otherwise
            Phi.shape = (parameters.alpha_phi + 0.5) * ones(D, R);
            Phi.scale = parameters.beta_phi * ones(D, R);
    end
    Q.mean = randn(D, R);
    Q.covariance = repmat(eye(D, D), [1, 1, R]);
    Z.mean = randn(R, N);
    Z.covariance = eye(R, R);
    lambda.shape = (parameters.alpha_lambda + 0.5) * ones(L, 1);
    lambda.scale = parameters.beta_lambda * ones(L, 1);
    Psi.shape = (parameters.alpha_psi + 0.5) * ones(R, L);
    Psi.scale = parameters.beta_psi * ones(R, L);    
    bW.mean = randn(R + 1, L);
    bW.covariance = repmat(eye(R + 1, R + 1), [1, 1, L]);
    
    labeled = ~isnan(Y);
    unlabeled = isnan(Y);
    T.mean = zeros(L, N);
    for o = 1:L
        yo = Y(o, :);
        indices = knnsearch(X(:, labeled(o, :))', X(:, unlabeled(o, :))');
        labels = yo(labeled(o, :));
        yo(unlabeled(o, :)) = labels(indices);
        T.mean(o, :) = (abs(randn(1, N)) + 0.5 * labeled(o, :)) .* sign(yo);
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
                    phi.scale(s) = 1 / (1 / parameters.beta_phi + 0.5 * (Q.mean(:, s)' * Q.mean(:, s) + sum(diag(Q.covariance(:, :, s)))));
                end
                %%%% update Q
                for s = 1:R
                    Q.covariance(:, :, s) = (phi.shape(s) * phi.scale(s) * eye(D, D) + XXT / sigmaz^2) \ eye(D, D);
                    Q.mean(:, s) = Q.covariance(:, :, s) * (X * Z.mean(s, :)' / sigmaz^2);
                end
            otherwise
                %%%% update Phi
                Phi.scale = 1 ./ (1 / parameters.beta_phi + 0.5 * (Q.mean.^2 + reshape(Q.covariance(phi_indices), D, R)));
                %%%% update Q
                for s = 1:R
                    Q.covariance(:, :, s) = (diag(Phi.shape(:, s) .* Phi.scale(:, s)) + XXT / sigmaz^2) \ eye(D, D);
                    Q.mean(:, s) = Q.covariance(:, :, s) * (X * Z.mean(s, :)' / sigmaz^2);
                end
        end
        %%%% update Z
        Z.covariance = (eye(R, R) / sigmaz^2 + bW.mean(2:R + 1, :) * bW.mean(2:R + 1, :)' + sum(bW.covariance(2:R + 1, 2:R + 1, :), 3)) \ eye(R, R);
        Z.mean = Z.covariance * (Q.mean' * X / sigmaz^2 + bW.mean(2:R + 1, :) * T.mean - repmat(bW.mean(2:R + 1, :) * bW.mean(1, :)' + sum(bW.covariance(2:R + 1, 1, :), 3), 1, N));
        %%%% update lambda
        lambda.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * (bW.mean(1, :)'.^2 + squeeze(bW.covariance(1, 1, :))));
        %%%% update psi
        Psi.scale = 1 ./ (1 / parameters.beta_psi + 0.5 * (bW.mean(2:R + 1, :).^2 + reshape(bW.covariance(psi_indices), R, L)));
        %%%% update b and w
        for o = 1:L
            bW.covariance(:, :, o) = [lambda.shape(o) * lambda.scale(o) + N, sum(Z.mean, 2)'; sum(Z.mean, 2), diag(Psi.shape(:, o) .* Psi.scale(:, o)) + Z.mean * Z.mean' + N * Z.covariance] \ eye(R + 1, R + 1);
            bW.mean(:, o) = bW.covariance(:, :, o) * ([ones(1, N); Z.mean] * T.mean(o, :)');
        end
        %%%% update T
        output = bW.mean' * [ones(1, N); Z.mean];
        alpha_norm = lower(labeled) - output(labeled);
        beta_norm = upper(labeled) - output(labeled);
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        T.mean(labeled) = output(labeled) + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;        
        for o = 1:L
            normalization = gamman(o) * normcdf(-0.5 - T.mean(o, unlabeled(o, :))) + gammap(o) * normcdf(T.mean(o, unlabeled(o, :)) - 0.5);
            expectationn = output(o, unlabeled(o, :)) + (normpdf(-Inf - output(o, unlabeled(o, :))) - normpdf(-0.5 - output(o, unlabeled(o, :)))) ./ (normcdf(-0.5 - output(o, unlabeled(o, :))) - normcdf(-Inf - output(o, unlabeled(o, :))));
            expectationn(expectationn < -40) = -40;
            expectationp = output(o, unlabeled(o, :)) + (normpdf(0.5 - output(o, unlabeled(o, :))) - normpdf(Inf - output(o, unlabeled(o, :)))) ./ (normcdf(Inf - output(o, unlabeled(o, :))) - normcdf(0.5 - output(o, unlabeled(o, :))));
            expectationp(expectationp > +40) = +40;
            T.mean(o, unlabeled(o, :)) = (1 ./ normalization) .* (gamman(o) * normcdf(-0.5 - output(o, unlabeled(o, :))) .* expectationn + gammap(o) * normcdf(output(o, unlabeled(o, :)) - 0.5) .* expectationp);
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