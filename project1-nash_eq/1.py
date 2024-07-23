from itertools import chain, combinations
from typing import Generator, Any, Tuple, Iterator, Union

import numpy as np
import numpy.typing as npt


def solve_indifference(A, rows=None, columns=None) -> Union[bool, Any]:
    """
    Solve the indifference for a payoff matrix assuming support for the
    strategies given by columns

    Finds vector of probabilities that makes player indifferent between
    rows.  (So finds probability vector for corresponding column player)

    Parameters
    ----------
    A : array
        The row player utility matrix.
    rows : array
        Array of integers corresponding to rows to consider.
    columns : array
        Array of integers corresponding to columns to consider.

    Returns
    -------
    Union
        The solution to the indifference equations.
    """
    # Ensure differences between pairs of pure strategies are the same
    M = (A[np.array(rows)] - np.roll(A[np.array(rows)], 1, axis=0))[:-1]
    # Columns that must be played with prob 0
    zero_columns = set(range(A.shape[1])) - set(columns)

    if zero_columns != set():
        M = np.append(
            M,
            [[int(i == j) for i, col in enumerate(M.T)] for j in zero_columns],
            axis=0,
        )

    # Ensure have probability vector
    M = np.append(M, np.ones((1, M.shape[1])), axis=0)
    b = np.append(np.zeros(len(M) - 1), [1])

    try:
        prob = np.linalg.solve(M, b)
        prob[(prob < 0) & (-1e-8 < prob)] = 0
        if all(prob >= 0):
            return prob
        return False
    except np.linalg.LinAlgError:
        return False


def powerset(n: int) -> Iterator[Tuple[Any, ...]]:
    """
    A power set of range(n)

    Based on recipe from python itertools documentation:

    https://docs.python.org/2/library/itertools.html#recipes

    Parameters
    ----------
    n : int
        The defining parameter of the powerset.

    Returns
    -------
    Iterator
        The powerset
    """
    return chain.from_iterable(combinations(range(n), r) for r in range(n + 1))


def potential_support_pairs(
        A: npt.NDArray, B: npt.NDArray, non_degenerate: bool = False
) -> Generator[tuple, Any, None]:
    """
    A generator for the potential support pairs


    Parameters
    ----------
    A : array
        The row player utility matrix.
    B : array
        The column player utility matrix
    non_degenerate : bool
        Whether or not to consider supports of equal size. By default
        (False) only considers supports of equal size.

    Yields
    -------
    Generator
        A pair of possible supports.
    """
    p1_num_strategies, p2_num_strategies = A.shape
    for support1 in (s for s in powerset(p1_num_strategies) if len(s) > 0):
        for support2 in (
                s
                for s in powerset(p2_num_strategies)
                if (len(s) > 0 and not non_degenerate) or len(s) == len(support1)
        ):
            yield support1, support2


def obey_support(strategy, support: npt.NDArray, tol: float = 10 ** -16) -> bool:
    """
    Test if a strategy obeys its support

    Parameters
    ----------
    strategy: array
        A given strategy vector
    support: array
        A strategy support
    tol : float
        A tolerance parameter for equality.

    Returns
    -------
    bool
        whether or not that strategy does indeed have the given support
    """
    if strategy is False:
        return False
    if not all(
            (i in support and value > tol) or (i not in support and value <= tol)
            for i, value in enumerate(strategy)
    ):
        return False
    return True


def indifference_strategies(
        A: npt.NDArray, B: npt.NDArray, non_degenerate: bool = False, tol: float = 10 ** -16
) -> Generator[Tuple[bool, bool, Any, Any], Any, None]:
    """
    A generator for the strategies corresponding to the potential supports

    Parameters
    ----------
    A : array
        The row player utility matrix.
    B : array
        The column player utility matrix
    non_degenerate : bool
        Whether or not to consider supports of equal size. By default
        (False) only considers supports of equal size.
    tol : float
        A tolerance parameter for equality.

    Yields
    ------
    Generator
        A generator of all potential strategies that are indifferent on each
        potential support. Return False if they are not valid (not a
        probability vector OR not fully on the given support).
    """
    if non_degenerate:
        tol = min(tol, 0)

    for pair in potential_support_pairs(A, B, non_degenerate=non_degenerate):
        s1 = solve_indifference(B.T, *(pair[::-1]))
        s2 = solve_indifference(A, *pair)

        if obey_support(s1, pair[0], tol=tol) and obey_support(s2, pair[1], tol=tol):
            yield s1, s2, pair[0], pair[1]


def is_ne(
        strategy_pair: tuple,
        support_pair: Tuple[npt.NDArray, npt.NDArray],
        payoff_matrices: Tuple[npt.NDArray, npt.NDArray],
) -> bool:
    """
    Test if a given strategy pair is a pair of best responses

    Parameters
    ----------
    strategy_pair: tuple
        a 2-tuple of numpy arrays.
    support_pair: tuple
        a 2-tuple of numpy arrays of integers.
    payoff_matrices: tuple
        a 2-tuple of numpy array of payoff matrices.

    Returns
    -------
    bool
        True if a given strategy pair is a pair of best responses.
    """
    A, B = payoff_matrices
    # Payoff against opponents strategies:
    u = strategy_pair[1].reshape(strategy_pair[1].size, 1)
    row_payoffs = np.dot(A, u)

    v = strategy_pair[0].reshape(strategy_pair[0].size, 1)
    column_payoffs = np.dot(B.T, v)

    # Pure payoffs on current support:
    row_support_payoffs = row_payoffs[np.array(support_pair[0])]
    column_support_payoffs = column_payoffs[np.array(support_pair[1])]

    return (
            row_payoffs.max() == row_support_payoffs.max()
            and column_payoffs.max() == column_support_payoffs.max()
    )


def support_enumeration(A, B, non_degenerate = False, tol = 10 ** -16):
    for s1, s2, sup1, sup2 in indifference_strategies(A, B, non_degenerate=non_degenerate, tol=tol):
        if is_ne((s1, s2), (sup1, sup2), (A, B)):
            return s1, s2


# Step 1: Read input matrices
n, m = map(int, input().split())

profits_abolf = [list(map(int, input().split())) for _ in range(n)]
profits_behzad = [list(map(int, input().split())) for _ in range(n)]

# Step 3: Compute mixed Nash equilibrium
eqs = list(support_enumeration(np.array(profits_abolf), np.array(profits_behzad), non_degenerate=False, tol=1e-12))

# Step 4: Print strategies
abolf_strategy = list(map(lambda x: f"{x:.6f}", eqs[0]))
behzad_strategy = list(map(lambda x: f"{x:.6f}", eqs[1]))

print(*abolf_strategy)
print(*behzad_strategy)
