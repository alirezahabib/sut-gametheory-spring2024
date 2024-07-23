from itertools import chain, combinations
import numpy as np


# Finds vector of probabilities that makes player indifferent between rows
def solve_indifference(A, rows, columns, tolerance=1e-6):
    # Make differences between pairs of pure strategies the same
    M = (A[np.array(rows)] - np.roll(A[np.array(rows)], 1, axis=0))[:-1]
    # Columns that must not be played (prob = 0)
    zero_columns = set(range(A.shape[1])) - set(columns)

    if zero_columns:
        M = np.append(M, [[int(i == j) for i, col in enumerate(M.T)] for j in zero_columns], axis=0)

    M = np.append(M, np.ones((1, M.shape[1])), axis=0)
    b = np.append(np.zeros(len(M) - 1), [1])

    try:
        prob = np.linalg.solve(M, b)

        # Zero probabilities can be slightly negative due to numerical errors
        prob[(-tolerance < prob) & (prob < 0)] = 0

        if all(prob >= 0) and all(prob <= 1):
            return prob
        return False
    except np.linalg.LinAlgError:
        return False


def non_empty_powerset(n):
    # Based on https://docs.python.org/2/library/itertools.html#recipes
    return list(chain.from_iterable(combinations(range(n), r) for r in range(n + 1)))[1:]


def all_support_pairs(A):
    pairs = []
    for s1 in non_empty_powerset(A.shape[0]):
        for s2 in non_empty_powerset(A.shape[1]):
            pairs.append((s1, s2))
    return pairs


# All potential strategies that are indifferent on each potential support.
def indifference_strategies(A, B):
    strategies = []
    for pair in all_support_pairs(A):
        s1 = solve_indifference(B.T, *(pair[::-1]))
        if s1 is False:
            continue

        s2 = solve_indifference(A, *pair)
        if s2 is not False:
            strategies.append((s1, s2, pair[0], pair[1]))
    return strategies


# Check if strategy pair is a pair of best responses
def both_best_response(strategy_pair, support_pair, payoff_matrices):
    A, B = payoff_matrices
    # Payoff against opponents strategies:
    u = strategy_pair[1].reshape(strategy_pair[1].size, 1)
    row_payoffs = np.dot(A, u)

    v = strategy_pair[0].reshape(strategy_pair[0].size, 1)
    column_payoffs = np.dot(B.T, v)

    # Pure payoffs on current support:
    row_support_payoffs = row_payoffs[np.array(support_pair[0])]
    column_support_payoffs = column_payoffs[np.array(support_pair[1])]

    return (row_payoffs.max() == row_support_payoffs.max()
            and column_payoffs.max() == column_support_payoffs.max())


def support_enumeration(A, B):
    for s1, s2, sup1, sup2 in indifference_strategies(A, B):
        if both_best_response((s1, s2), (sup1, sup2), (A, B)):
            return s1, s2


if __name__ == '__main__':
    n, m = map(int, input().split())

    profits_abolf = [list(map(int, input().split())) for _ in range(n)]
    profits_behzad = [list(map(int, input().split())) for _ in range(n)]

    s1, s2 = support_enumeration(np.array(profits_abolf), np.array(profits_behzad))

    abolf_strategy = list(map(lambda x: f'{x:.6f}', s1))
    behzad_strategy = list(map(lambda x: f'{x:.6f}', s2))

    print(*abolf_strategy)
    print(*behzad_strategy)
