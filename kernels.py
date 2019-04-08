@mfunction("phi")
def kernels(x=None, x_sv=None, kernel_type=None, kernel_parameter=None):
    # phi = kernels(x,x_sv,kernel_type,kernel_parameter);
    # Inputs:
    #       x          -> input data
    #       x-sv       -> support data
    #       kernel_type     ->  kernel type which can be 'rbf', 'poly','direct' or 'linear'
    #       kernel_parameter -> a vector, each corresponds to a support data; paramters are for 'rbf' and 'poly'
    # Ouput:
    #       phi        -> the design matrix

    [dim, N] = size(x)

    
    if  kernel_type is 'rbf':
        for i in mslice[1:size(x_sv, 2)]:
            xdif = x_sv(mslice[:], i) * ones(1, N) - x
            d = sum(xdif *elmul* xdif, 1)
            phi(i, mslice[:]).lvalue = exp(-0.5 * d / kernel_parameter(i) ** 2)
    if kernel_type is 'poly':
        phi          = (x_sv'*x+1).^kernel_parameter;
    if kernel_type is 'linear':
        phi          = x_sv'*x;
    if kernel_type is 'direct':
        phi = x;
    phi     = [ones(1,N);phi];
