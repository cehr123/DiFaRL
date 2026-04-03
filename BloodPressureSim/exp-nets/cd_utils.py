import numpy as np
from causalai.models.common.CI_tests.kci import KCI
import numpy as np
from causalai.models.common.CI_tests.kernels import GaussianKernel

def get_mask(model, X, A_fac, X_next, dA):
    if model == 'baseline':
        mask = np.ones((dA, dA))
    elif model == 'factored':
        mask = np.identity(dA)
    elif model == 'oracle':
        mask = np.identity(dA)
        mask[2][0] = mask[0][2] = 1
    elif model == 'DiFaRL':
        mask = get_mask_entangled(X, A_fac, X_next, alpha = 0.005)
    return mask

def get_mask_entangled(X, A, X_next, alpha = 0.005):
    dA = A.shape[1]
    mask = np.identity(dA)
    X = np.argmax(X, axis = 1).reshape(-1, 1)
    X_next = np.argmax(X_next, axis = 1).reshape(-1, 1)

    kci_test = KCI(
        Xkernel=GaussianKernel(width='median'),
        Ykernel=GaussianKernel(width='median'),
        Zkernel=GaussianKernel(width='median'),
        null_space_size=5000,
        approx=True,
        chunk_size=5000
    )
    kci_test.epsilon_x = 1e-3
    kci_test.epsilon_y = 1e-3

    for i in range(dA):
        for j in range(i + 1, dA):
            x = A[:,i].reshape(-1, 1)
            y = A[:,j].reshape(-1, 1)
            _, p_value = kci_test.run_test(data_x=x, data_y=y, data_z=np.hstack((X, X_next)))
            if p_value < alpha:
                mask[i][j] = mask[j][i] = 1
    
    return mask

def get_groups(model, X, A_fac, X_next, dA):
    if model == 'baseline':
        groups = [0, 0, 0, 0]
    elif model == 'factored':
        groups =  [0, 1, 2, 3]
    elif model == 'oracle':
        groups = [0, 1, 0, 2]
    elif model == 'DiFaRL':
        groups = get_groups_entangled(X, A_fac, X_next, alpha = 0.005)
    return groups

def get_groups_entangled(X, A, X_next, alpha=0.005):
    dA = A.shape[1]
    interactions = np.zeros((dA, dA), dtype=int)

    X = np.argmax(X, axis=1).reshape(-1, 1)
    X_next = np.argmax(X_next, axis=1).reshape(-1, 1)

    kci_test = KCI(
        Xkernel=GaussianKernel(width='median'),
        Ykernel=GaussianKernel(width='median'),
        Zkernel=GaussianKernel(width='median'),
        null_space_size=5000,
        approx=True,
        chunk_size=2500
    )
    kci_test.epsilon_x = 1e-3
    kci_test.epsilon_y = 1e-3

    for i in range(dA):
        for j in range(i + 1, dA):
            x = A[:, i].reshape(-1, 1)
            y = A[:, j].reshape(-1, 1)
            _, p_value = kci_test.run_test(
                data_x=x, data_y=y, data_z=np.hstack((X, X_next))
            )
            if p_value < alpha:
                interactions[i, j] = interactions[j, i] = 1

    visited = [False] * dA
    groups = [-1] * dA
    group_id = 0

    for i in range(dA):
        if not visited[i]:
            stack = [i]
            visited[i] = True
            groups[i] = group_id

            while stack:
                u = stack.pop()
                for v in range(dA):
                    if interactions[u, v] == 1 and not visited[v]:
                        visited[v] = True
                        groups[v] = group_id
                        stack.append(v)

            group_id += 1

    return groups
