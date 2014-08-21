% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bssml_supervised_classification_variational_train(X, Y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    D = size(X, 1);
    N = size(X, 2);
    L = size(Y, 1);
    R = parameters.R;
    sigmaz = parameters.sigmaz;

    log2pi = log(2 * pi);
    digamma = @psi;

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
    T.mean = abs(randn(L, N)) .* sign(Y);
    T.covariance = ones(L, N);

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
        %%%% update Psi
        Psi.scale = 1 ./ (1 / parameters.beta_psi + 0.5 * (bW.mean(2:R + 1, :).^2 + reshape(bW.covariance(psi_indices), R, L)));
        %%%% update b and W        
        for o = 1:L
            bW.covariance(:, :, o) = [lambda.shape(o) * lambda.scale(o) + N, sum(Z.mean, 2)'; sum(Z.mean, 2), diag(Psi.shape(:, o) .* Psi.scale(:, o)) + Z.mean * Z.mean' + N * Z.covariance] \ eye(R + 1, R + 1);
            bW.mean(:, o) = bW.covariance(:, :, o) * ([ones(1, N); Z.mean] * T.mean(o, :)');
        end
        %%%% update T
        output = bW.mean' * [ones(1, N); Z.mean];
        alpha_norm = lower - output;
        beta_norm = upper - output;
        normalization = normcdf(beta_norm) - normcdf(alpha_norm);
        normalization(normalization == 0) = 1;
        T.mean = output + (normpdf(alpha_norm) - normpdf(beta_norm)) ./ normalization;
        T.covariance = 1 + (alpha_norm .* normpdf(alpha_norm) - beta_norm .* normpdf(beta_norm)) ./ normalization - (normpdf(alpha_norm) - normpdf(beta_norm)).^2 ./ normalization.^2;

        if parameters.progress == 1
            lb = 0;
            switch parameters.prior_phi
                case 'ard'
                    %%%% p(phi)
                    lb = lb + sum((parameters.alpha_phi - 1) * (digamma(phi.shape) + log(phi.scale)) - phi.shape .* phi.scale / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi));
                    %%%% p(Q | phi)
                    qqT.mean = zeros(D, D, R);
                    for s = 1:R
                        qqT.mean(:, :, s) = Q.mean(:, s) * Q.mean(:, s)' + Q.covariance(:, :, s);
                        lb = lb - 0.5 * phi.shape(s) * phi.scale(s) * sum(diag(qqT.mean(:, :, s))) - 0.5 * (D * log2pi - D * log(phi.shape(s) * phi.scale(s)));
                    end
                otherwise
                    %%%% p(Phi)
                    lb = lb + sum(sum((parameters.alpha_phi - 1) * (digamma(Phi.shape) + log(Phi.scale)) - Phi.shape .* Phi.scale / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi)));
                    %%%% p(Q | Phi)
                    qqT.mean = zeros(D, D, R);
                    for s = 1:R
                        qqT.mean(:, :, s) = Q.mean(:, s) * Q.mean(:, s)' + Q.covariance(:, :, s);
                        lb = lb - 0.5 * sum(Phi.shape(:, s) .* Phi.scale(:, s) .* diag(qqT.mean(:, :, s))) - 0.5 * (D * log2pi - sum(log(Phi.shape(:, s) .* Phi.scale(:, s))));
                    end
            end
            %%%% p(Z | Q, X)
            ZZT.mean = Z.mean * Z.mean' + N * Z.covariance;
            lb = lb - 0.5 * sum(diag(ZZT.mean)) + sum(sum((Q.mean' * X) .* Z.mean)) - 0.5 * sum(sum(sum(qqT.mean, 3) * XXT')) - 0.5 * N * R * (log2pi + 2 * log(sigmaz));
            %%%% p(lambda)
            lb = lb + sum((parameters.alpha_lambda - 1) * (digamma(lambda.shape) + log(lambda.scale)) - lambda.shape .* lambda.scale / parameters.beta_lambda - gammaln(parameters.alpha_lambda) - parameters.alpha_lambda * log(parameters.beta_lambda));
            %%%% p(b | lambda)
            bbT.mean = diag(bW.mean(1, :)'.^2 + squeeze(bW.covariance(1, 1, :)));
            lb = lb - 0.5 * sum(lambda.shape .* lambda.scale .* diag(bbT.mean)) - 0.5 * (L * log2pi - sum(log(lambda.shape .* lambda.scale)));
            %%%% p(Psi)
            lb = lb + sum(sum((parameters.alpha_psi - 1) * (digamma(Psi.shape) + log(Psi.scale)) - Psi.shape .* Psi.scale / parameters.beta_psi - gammaln(parameters.alpha_psi) - parameters.alpha_psi * log(parameters.beta_psi)));
            %%%% p(W | Psi)
            wwT.mean = zeros(R, R, L);
            for o = 1:L
                wwT.mean(:, :, o) = bW.mean(2:R + 1, o) * bW.mean(2:R + 1, o)' + bW.covariance(2:R + 1, 2:R + 1, o);
                lb = lb - 0.5 * sum(Psi.shape(:, o) .* Psi.scale(:, o) .* diag(wwT.mean(:, :, o))) - 0.5 * (R * log2pi - sum(log(Psi.shape(:, o) .* Psi.scale(:, o))));
            end
            %%%% p(T | b, W, Z)
            for o = 1:L            
                lb = lb - 0.5 * (T.mean(o, :) * T.mean(o, :)' + sum(T.covariance(o, :))) + (bW.mean(2:R + 1, o)' * Z.mean + bW.mean(1, o)) * T.mean(o, :)' - 0.5 * sum(sum(wwT.mean(:, :, o) .* ZZT.mean)) - sum((bW.mean(2:R + 1, o) * bW.mean(1, o) + bW.covariance(2:R + 1, 1, o))' * Z.mean) - 0.5 * N * bbT.mean(o, o) - 0.5 * N * log2pi;
            end

            switch parameters.prior_phi
                case 'ard'
                    %%%% q(phi)
                    lb = lb + sum(phi.shape + log(phi.scale) + gammaln(phi.shape) + (1 - phi.shape) .* digamma(phi.shape));
                otherwise
                    %%%% q(Phi)
                    lb = lb + sum(sum(Phi.shape + log(Phi.scale) + gammaln(Phi.shape) + (1 - Phi.shape) .* digamma(Phi.shape)));
            end
            %%%% q(Q)
            for s = 1:R
                lb = lb + 0.5 * (D * (log2pi + 1) + logdet(Q.covariance(:, :, s)));
            end
            %%%% q(Z)
            lb = lb + 0.5 * N * (R * (log2pi + 1) + logdet(Z.covariance));
            %%% q(lambda)
            lb = lb + sum(lambda.shape + log(lambda.scale) + gammaln(lambda.shape) + (1 - lambda.shape) .* digamma(lambda.shape));
            %%%% q(Psi)
            lb = lb + sum(sum(Psi.shape + log(Psi.scale) + gammaln(Psi.shape) + (1 - Psi.shape) .* digamma(Psi.shape)));
            %%%% q(b, W)
            for o = 1:L
                lb = lb + 0.5 * ((R + 1) * (log2pi + 1) + logdet(bW.covariance(:, :, o)));
            end
            %%%% q(T)
            lb = lb + 0.5 * sum(sum(log2pi + T.covariance)) + sum(sum(log(normalization)));

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