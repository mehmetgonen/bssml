% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bssml_supervised_classification_variational_train(X, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(X, 1);
    N = size(X, 2);
    L = size(Y, 1);
    R = parameters.R;
    sigma_z = parameters.sigma_z;

    log2pi = log(2 * pi);
    digamma = @psi;

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
    T.mu = abs(randn(L, N)) .* sign(Y);
    T.sigma = ones(L, N);

    XXT = X * X';
    
    lower = -1e40 * ones(L, N);
    lower(Y > 0) = 0;
    upper = +1e40 * ones(L, N);
    upper(Y < 0) = 0;
    
    phi_indices = repmat(logical(eye(D, D)), [1, 1, R]);
    psi_indices = repmat(logical(diag([0, ones(1, R)])), [1, 1, L]);

    if parameters.progress == 1
        bounds = zeros(parameters.iteration, 1);
    end

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
        %%%% update Psi
        Psi.beta = 1 ./ (1 / parameters.beta_psi + 0.5 * (bW.mu(2:R + 1, :).^2 + reshape(bW.sigma(psi_indices), R, L)));
        %%%% update b and W        
        for o = 1:L
            bW.sigma(:, :, o) = [lambda.alpha(o) * lambda.beta(o) + N, sum(Z.mu, 2)'; sum(Z.mu, 2), diag(Psi.alpha(:, o) .* Psi.beta(:, o)) + Z.mu * Z.mu' + N * Z.sigma] \ eye(R + 1, R + 1);
            bW.mu(:, o) = bW.sigma(:, :, o) * ([ones(1, N); Z.mu] * T.mu(o, :)');
        end
        %%%% update T
        output = bW.mu' * [ones(1, N); Z.mu];
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        T.mu = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        T.sigma = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;

        if parameters.progress == 1
            lb = 0;
            switch parameters.prior_phi
                case 'ard'
                    %%%% p(phi)
                    lb = lb + sum((parameters.alpha_phi - 1) * (digamma(phi.alpha) + log(phi.beta)) - phi.alpha .* phi.beta / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi));
                    %%%% p(Q | phi)
                    qqT.mu = zeros(D, D, R);
                    for s = 1:R
                        qqT.mu(:, :, s) = Q.mu(:, s) * Q.mu(:, s)' + Q.sigma(:, :, s);
                        lb = lb - 0.5 * phi.alpha(s) * phi.beta(s) * sum(diag(qqT.mu(:, :, s))) - 0.5 * (D * log2pi - D * (psi(phi.alpha(s)) + log(phi.beta(s))));
                    end
                otherwise
                    %%%% p(Phi)
                    lb = lb + sum(sum((parameters.alpha_phi - 1) * (digamma(Phi.alpha) + log(Phi.beta)) - Phi.alpha .* Phi.beta / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi)));
                    %%%% p(Q | Phi)
                    qqT.mu = zeros(D, D, R);
                    for s = 1:R
                        qqT.mu(:, :, s) = Q.mu(:, s) * Q.mu(:, s)' + Q.sigma(:, :, s);
                        lb = lb - 0.5 * sum(Phi.alpha(:, s) .* Phi.beta(:, s) .* diag(qqT.mu(:, :, s))) - 0.5 * (D * log2pi - sum(psi(Phi.alpha(:, s)) + log(Phi.beta(:, s))));
                    end
            end
            %%%% p(Z | Q, X)
            ZZT.mu = Z.mu * Z.mu' + N * Z.sigma;
            lb = lb - 0.5 * sum(diag(ZZT.mu)) + sum(sum((Q.mu' * X) .* Z.mu)) - 0.5 * sum(sum(sum(qqT.mu, 3) * XXT')) - 0.5 * N * R * (log2pi + 2 * log(sigma_z));
            %%%% p(lambda)
            lb = lb + sum((parameters.alpha_lambda - 1) * (digamma(lambda.alpha) + log(lambda.beta)) - lambda.alpha .* lambda.beta / parameters.beta_lambda - gammaln(parameters.alpha_lambda) - parameters.alpha_lambda * log(parameters.beta_lambda));
            %%%% p(b | lambda)
            bbT.mu = diag(bW.mu(1, :)'.^2 + squeeze(bW.sigma(1, 1, :)));
            lb = lb - 0.5 * sum(lambda.alpha .* lambda.beta .* diag(bbT.mu)) - 0.5 * (L * log2pi - sum(psi(lambda.alpha) + log(lambda.beta)));
            %%%% p(Psi)
            lb = lb + sum(sum((parameters.alpha_psi - 1) * (digamma(Psi.alpha) + log(Psi.beta)) - Psi.alpha .* Psi.beta / parameters.beta_psi - gammaln(parameters.alpha_psi) - parameters.alpha_psi * log(parameters.beta_psi)));
            %%%% p(W | Psi)
            wwT.mu = zeros(R, R, L);
            for o = 1:L
                wwT.mu(:, :, o) = bW.mu(2:R + 1, o) * bW.mu(2:R + 1, o)' + bW.sigma(2:R + 1, 2:R + 1, o);
                lb = lb - 0.5 * sum(Psi.alpha(:, o) .* Psi.beta(:, o) .* diag(wwT.mu(:, :, o))) - 0.5 * (R * log2pi - sum(psi(Psi.alpha(:, o)) + log(Psi.beta(:, o))));
            end
            %%%% p(T | b, W, Z)
            for o = 1:L            
                lb = lb - 0.5 * (T.mu(o, :) * T.mu(o, :)' + sum(T.sigma(o, :))) + (bW.mu(2:R + 1, o)' * Z.mu + bW.mu(1, o)) * T.mu(o, :)' - 0.5 * sum(sum(wwT.mu(:, :, o) .* ZZT.mu)) - sum((bW.mu(2:R + 1, o) * bW.mu(1, o) + bW.sigma(2:R + 1, 1, o))' * Z.mu) - 0.5 * N * bbT.mu(o, o) - 0.5 * N * log2pi;
            end

            switch parameters.prior_phi
                case 'ard'
                    %%%% q(phi)
                    lb = lb + sum(phi.alpha + log(phi.beta) + gammaln(phi.alpha) + (1 - phi.alpha) .* digamma(phi.alpha));
                otherwise
                    %%%% q(Phi)
                    lb = lb + sum(sum(Phi.alpha + log(Phi.beta) + gammaln(Phi.alpha) + (1 - Phi.alpha) .* digamma(Phi.alpha)));
            end
            %%%% q(Q)
            for s = 1:R
                lb = lb + 0.5 * (D * (log2pi + 1) + logdet(Q.sigma(:, :, s)));
            end
            %%%% q(Z)
            lb = lb + 0.5 * N * (R * (log2pi + 1) + logdet(Z.sigma));
            %%% q(lambda)
            lb = lb + sum(lambda.alpha + log(lambda.beta) + gammaln(lambda.alpha) + (1 - lambda.alpha) .* digamma(lambda.alpha));
            %%%% q(Psi)
            lb = lb + sum(sum(Psi.alpha + log(Psi.beta) + gammaln(Psi.alpha) + (1 - Psi.alpha) .* digamma(Psi.alpha)));
            %%%% q(b, W)
            for o = 1:L
                lb = lb + 0.5 * ((R + 1) * (log2pi + 1) + logdet(bW.sigma(:, :, o)));
            end
            %%%% q(T)
            lb = lb + 0.5 * sum(sum(log2pi + T.sigma)) + sum(sum(log(normalization)));

            bounds(iter) = lb;
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
    if parameters.progress == 1
        state.bounds = bounds;
    end
    state.parameters = parameters;
end

function ld = logdet(Sigma)
    U = chol(Sigma);
    ld = 2 * sum(log(diag(U)));
end