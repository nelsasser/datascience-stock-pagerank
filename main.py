import numpy as np


def data_to_adj_mat(time_series_data):
    pass


def page_rank():
    pass


def graph_part(adj_mat, k, kmeans_iters=50):
    def fit_kmeans(k, data, iters=50):
        p = np.random.permutation(data.shape[0])
        centriods = data[p[:k]]

        dists = np.zeros((k, data.shape[0]))

        for iter_num in range(iters + 1):
            for m in range(k):
                dists[m, :] = np.linalg.norm(data - centriods[m], axis=1)

            classes = np.argmin(dists, axis=0)

            old_centriods = centriods.copy()

            if iter_num < iters:
                for m in range(k):
                    if np.sum(classes == m) != 0:
                        centriods[m] = np.dot((classes == m), data)
                        centriods[m] /= np.sum(classes == m)

                if np.allclose(old_centriods, centriods):
                    break

        return centriods, classes

    D = np.diag(np.sum(adj_mat, axis=0))
    L = D - adj_mat
    eig_vals, eig_vecs = np.linalg.eigh(L)
    p = np.argsort(eig_vals)

    fiedler = np.array(eig_vecs[:, p[1]]).ravel()

    classes = fit_kmeans(k, fiedler[:, np.newaxis], iters=kmeans_iters)[1]

    return classes


if __name__ == '__main__':
    pass
