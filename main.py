import numpy as np


def data_to_adj_mat(time_series_data):
    pass


def page_rank():
    pass


def graph_part(adj_mat):
    D = np.diag(np.sum(adj_mat, axis=0))
    L = D - adj_mat
    eig_vals, eig_vecs = np.linalg.eigh(L)
    p = np.argsort(eig_vals)

    fiedler = np.array(eig_vecs[:, p[1]]).ravel()
    perm = np.argsort(fiedler)

    return fiedler, perm


if __name__ == '__main__':
    pass
