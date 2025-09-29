import numpy as np

def generate_adj(N=6, p=0.4, seed=42):
    rng = np.random.default_rng(seed)
    A = (rng.random((N, N)) < p).astype(int)
    A = np.triu(A, 1)
    A = A + A.T
    return A

def degrees(A):
    return A.sum(axis=1)

def paths_length_2(A):
    return A @ A

def unique_friends_of_friends(A):
    A2 = paths_length_2(A)
    mask = (A2 > 0).astype(int)
    np.fill_diagonal(mask, 0)
    fof = mask - A
    fof[fof < 0] = 0
    return fof.sum(axis=1)

def eigen_spectral_radius(A):
    vals = np.linalg.eigvals(A)
    return vals, float(np.max(np.abs(vals)))

def demo():
    A = generate_adj()
    deg = degrees(A)
    A2 = paths_length_2(A)
    fof = unique_friends_of_friends(A)
    eig, rho = eigen_spectral_radius(A)
    print("Adjacency matrix:\n", A)
    print("Degrees:", deg)
    print("Argmax degree:", int(np.argmax(deg)))
    print("A^2:\n", A2)
    print("Friends-of-friends counts:", fof)
    print("Eigenvalues:", eig)
    print("Spectral radius:", rho)

if __name__ == "__main__":
    demo()
