import numpy as np
import datetime

def sharpe(x, r):
    """Get sharpe ratio for x

    Args:
        x (numpy.array): array of returns data
        r (float): risk free rate

    Returns:
        float: sharpe ratio
    """

    expected_return = np.mean(x)
    return_deviation = np.std(x)
    return (expected_return - r) / return_deviation

def beta(x, y):
    """Get beta coefficient for x compared against y

    Args:
        x (np.array): asset returns
        y (np.array): index returns

    Returns:
        float: beta coefficient
    """

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    cov = np.sum((x - x_mean) * (y - y_mean)) / (x.shape[0] - 1)
    var = np.var(x)

    return cov/var

def data_to_adj_mat(time_series_data):
    pass


def page_rank(adj_mat):

    n = len(adj_mat)
    r_old = np.repeat(1 / n, n)
    for t in range(50):
        r_new = np.repeat(1 / n, n)
        for page in range(n):
            for dest in adj_mat[page]:
                r_new[dest] += r_old[page] / (len(adj_mat[page]))
        r_new = r_new / np.sum(r_new)
        r_old = r_new
    return r_new


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


def backtest(start_date, end_date, window_size, timestep, time_aggregate, heuristic):
    current_end = start_date
    current_start = current_end - datetime.timedelta(days=window_size)

    balance = 100_000 # starting amount of money

    current_positions = []
    balance_history = []

    while(current_end <= end_date):
        # 1) aggregate data using current_start, current_end, time_aggregate
        # 2) if there are current positions, sell them, push old balance to balance_history, and update balance
        # 3) calculate heuristics and construct M
        # 4) run pagerank power iteration to get r
        # 5) calculate new current positions from r and balance
        # 6) update window


        current_end = current_end + datetime.timedelta(days=timestep)
        current_start = current_start + datetime.timedelta(days=timestep)

    # from balance_history calculate the sharpe, beta, etc whatever metrics we want to use against our index
    # return backtest metrics

if __name__ == '__main__':
    pass
