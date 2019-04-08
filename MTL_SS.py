@mfunction("initial_model, initial_model_iterations, posterior_w_all, Xi_0_w, Xi_l_w_m_task, loglikelihood_labeled, logLike_prior")
def MTL_SS(phi=None, L_idx=None, y_idx=None, posterior_xi=None, alpha=None, gamma=None, Eta_square_w=None, initial_model=None, Maxiter=None, tol=None):

    task_count = length(phi)#  The number of tasks
    [Nw, Ndata1] = size(phi(1))
 
 if nargin < 5:
        alpha = 0.3    
    # precision matrix for the first part of the prior distribution for w: the
    # classifier in Eqn(2)

    if nargin < 6:
        gamma = 10 * eye(Nw)    
    #variance matrix for the second part of the prior distribution in Eqn(2)

    if nargin < 7:
        Eta_square_w = 1 * eye(Nw)    

        if nargin < 8:
        for task in mslice[1:task_count]:
            initial_model(task).lvalue = randn(Nw, 1)

            #initial_model{task} = zeros(Nw,1);   
    if nargin < 9:
        Maxiter = 200
    
    if nargin < 10:
        tol = 1e-6
    
    #EM Algorithm
    iter = 0
    Converged = 0
    while (not logical_and(Converged, iter < Maxiter)):
        iter = iter + 1
        # E Step,
        for m in mslice[1:task_count]:
            Xi_0_part2("Xi_l_w_m")

            if m == 1:                                        
                Xi_0_w(m).lvalue = 1
                Xi_l_w_m_task(m).lvalue = mcat([])
                logLike_prior(m, iter).lvalue = log(alpha *elmul* gausspdf(initial_model(m).cT, zeros(1, Nw), inv(gamma)))
            else:
 
                Xi_0_part1 = alpha *elmul* gausspdf(initial_model(m).cT, zeros(1, Nw), inv(gamma))
                for ml in mslice[1:m - 1]:
                    Xi_0_part2(ml).lvalue = gausspdf((initial_model(m) - initial_model(ml)).cT, zeros(Nw, 1), Eta_square_w)
                
                Xi_0_w(m).lvalue = Xi_0_part1 /eldiv/ (Xi_0_part1 + sum(Xi_0_part2))

                Xi_l_w_m = Xi_0_part2 / (Xi_0_part1 + sum(Xi_0_part2))
                Xi_l_w_m_task(m).lvalue = Xi_l_w_m

                logLike_prior(m, iter).lvalue = Xi_0_w(m) *elmul* (log(Xi_0_part1) - log(Xi_0_w(m)))
                logLike_prior(m, iter).lvalue = logLike_prior(m, iter) + sum(Xi_l_w_m *elmul* (log(Xi_0_part2) - log(Xi_l_w_m)))

            [Nw, Ndata] = size(phi(m))
            for k in mslice[1:length(L_idx(m))]:
                y_k = y_idx(m);
                i = mcat([mslice[1:Ndata]])
                mslice[:]
                Prob_withRandomWalks(i, k).lvalue = Prob_yk_Given_xiAndw(i, k) *elmul* posterior_xi(m); print Prob_withRandomWalks
                like_sum = sum(Prob_withRandomWalks(mslice[:], k), 1)
                delta_ik(mslice[:], k).lvalue = Prob_withRandomWalks(mslice[:], k) / like_sum
                loglik(k).lvalue = log(like_sum)
            
                Zeta_ik_alltask(m).lvalue = delta_ik                                                # equation{11}
                Prob_yk_Given_xiAndw_alltask(m).lvalue = Prob_yk_Given_xiAndw
                loglikelihood_labeled(m, iter).lvalue = sum(loglik)                                                # log-likelihood for the mth task

        for m in mslice[1:task_count]:
            posterior_w_m(m, iter).lvalue = logLike_prior(m, iter) + loglikelihood_labeled(m, iter)
            
        sum_pos = sum(posterior_w_m, 1)
        posterior_w_all(iter).lvalue = sum_pos(iter)
        if m == 1:   
            #Algorithm part 3: Block-cooridate gradient-asing for w1...wM
            for m in mslice[1:task_count]:
                #Compute the Q functions, its Derivatives and its Hessian Matrix.
                Gradient_Q1 = zeros(Nw, 1)                                                        #The Gradient of the prior part
                Hessian_Q1 = zeros(Nw, Nw)                                                        #The Hessian Matrix of the likelihood part
                Gradient_Q1 = Xi_0_w(m) *elmul* (-initial_model(m).cT * gamma)
                Hessian_Q1 = -Xi_0_w(m) *elmul* gamma
                    if m == 1:
                        for lm2 in mslice[m + 1:task_count]:
                            Gradient_Q1 = Gradient_Q1 + Xi_l_w_m_task(lm2)(m) *elmul* (-(initial_model(m) - initial_model(lm2)).cT * inv(Eta_square_w));
                            Hessian_Q1 = Hessian_Q1 - Xi_l_w_m_task(lm2)(m) *elmul* inv(Eta_square_w);
                        
                    else
                        for lm in mslice[1:m-1]:
                            Gradient_Q1 = Gradient_Q1 + Xi_l_w_m_task(m)(lm) *elmul* (-(initial_model(m) - initial_model(lm2)).cT * inv(Eta_square_w));
                            Hessian_Q1 = Hessian_Q1 - Xi_l_w_m_task(m)(lm) *elmul* inv(Eta_square_w);
                        
                        for lm2 in mslice[m_1:task_count]:                            for lm in mslice[1:m-1]:
                            Gradient_Q1 = Gradient_Q1 + Xi_l_w_m_task(lm2)(m) *elmul* (-(initial_model(m) - initial_model(lm2)).cT * inv(Eta_square_w));
                            Hessian_Q1 = Hessian_Q1 - Xi_l_w_m_task(lm2)(m) *elmul* inv(Eta_square_w);
                    Gradient_Q2 = zeros(Nw, 1)#The Gradient of the likelihood part
                    Hessian_Q2 = zeros(Nw, Nw)#The Hessian Matrix of the likelihood part
                    for k in mslice[1:length(L_idx(m))]:
                        y_k = y_idx(m); print y_k
                        (k)

                        Delta_k = Zeta_ik_alltask(m); print Delta_k
                        (mslice[:], k)

                        Pm_vector = Prob_yk_Given_xiAndw_alltask(m); print Pm_vector
                        (mslice[:], k)
                        # column vector
                        Gradient_Q2 = Gradient_Q2 + phi(m) * ((1 - Pm_vector) *elmul* Delta_k) *elmul* y_k
                        phi_star = phi(m) *elmul* repmat((1 - Pm_vector).cT *elmul* Pm_vector.cT *elmul* Delta_k.cT, mcat([Nw, 1]))
                        Hessian_Q2 = Hessian_Q2 - phi_star * phi(m).cT# negative definite
                        end

                        Gradient_Q = Gradient_Q1.cT + Gradient_Q2#column vector
                        Hessian_Q = Hessian_Q1 + Hessian_Q2# + diag(ones(1,Nw)*10^(-2));
                        while rank(Hessian_Q) < size(Hessian_Q, 1):
                            Hessian_Q = Hessian_Q + eye(size(Hessian_Q, 1)) * mean(mean(abs(Hessian_Q))) * 10 ** (-10)
                        Direct = inv(Hessian_Q) * Gradient_Q
                        if norm(Direct) != 0:
                            Direct = Direct / norm(Direct)        #normalize
                        #The initial objective function and its derivative
                        wo = initial_model(m)        #The old w.
                        [obj0, dw0] = Q_logiClassifier_MTL(wo, m, alpha, initial_model, gamma, Eta_square_w, Xi_0_w, Xi_l_w_m_task, L_idx, y_idx, Zeta_ik_alltask, phi)
                        ogparams = mcellarray([m, alpha, initial_model, gamma, Eta_square_w, Xi_0_w, Xi_l_w_m_task, L_idx, y_idx, Zeta_ik_alltask, phi])
                        [gamma, obj, dx] = cgLineSearch_Q(wo, obj0, dw0, -Direct, @, ogparams)
                        #gamma = 0.005;
                        initial_model(m).lvalue = wo - gamma *elmul* Direct        #the new w
                    if iter >= 2:            # Stopping criterion is no more change in loglikelihood
                        diffW = abs(posterior_w_all(iter) - posterior_w_all(iter - 1))
                        Converged = diffW /eldiv/ abs(posterior_w_all(iter - 1)) <= tol            # stopping criterion
                    initial_model_iterations(iter).lvalue = initial_model
                                                   
