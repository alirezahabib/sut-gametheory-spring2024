import numpy as np


import numpy.typing as npt
from typing import Set, List, Iterable, Optional, Tuple
from itertools import cycle


class Tableau(object):
    """
    An implementation of a standard Tableau
    Tableaus are well known in linear optimizations
    problems and e.g. part of the simplex algorithm.
    """

    def __init__(
            self, tableau: npt.NDArray, original_basic_labels: Optional[Iterable] = None
    ):
        """
        Constructs a Tableau for solving lemke-howson algorithm.

        Parameters
        ----------
        tableau : array
            The tableau off a payoff matrix
        original_basic_labels : Optional[Iterable]
            By default this corresponds to the non-basic variables.
            There should be no need to override this unless tableau
            matrix was manipulated prior to calling constructor
        """
        self._tableau = tableau
        if original_basic_labels is not None:
            self._original_basic_labels = set(original_basic_labels)
        else:
            self._original_basic_labels = self.non_basic_variables

    @property
    def labels(self) -> Set:
        """
        The full set of labels

        Returns
        -------
        Set
            All lables
        """
        h, w = self._tableau.shape
        real_w = w - h - 1
        return set(range(h + real_w))

    @property
    def non_basic_variables(self) -> Set:
        """
        Identifies the non basic variables of the tableau,
        these correspond to the basic labels.

        Returns
        -------
        Set
            The indices of the non basic variables.
        """

        columns = self._tableau[:, :-1].transpose()
        return set(np.where([np.count_nonzero(col) != 1 for col in columns])[0])

    @property
    def basic_variables(self) -> Set:
        """
        Identifies the basic variables of the tableau
        these correspond to the non-basic labels

        Returns
        -------
        Set
            The indices of the basic variables.
        """
        return self.labels - self.non_basic_variables

    def _find_pivot_row(self, column_index: int) -> int:
        """
        Uses minratio test to find the row to pivot against.
        To avoid divide-by-zeros this is implemented using max ratio

        Parameters
        ----------
        column_index : int
            The column/label to pivot.

        Returns
        -------
        int
            The row to pivot against
        """
        row_ratios = self._tableau[:, column_index] / self._tableau[:, -1]
        return int(np.argmax(row_ratios))

    def _apply_pivot(self, pivot_col: int, pivot_row: int, applying_row: int):
        """
        Dropping pivot value on the applying row

        Parameters
        ----------
        pivot_col : int
            The column/label to pivot.
        pivot_row : int
            The row to pivot
        applying_row : int
            The row to drop pivot column from
        """
        pivot_element = self._tableau[pivot_row, pivot_col]
        row_pivot_val = self._tableau[applying_row, pivot_col]
        row = self._tableau[applying_row, :] * pivot_element
        row -= self._tableau[pivot_row, :] * row_pivot_val
        self._tableau[applying_row, :] = row

    def _pivot(self, column_index: int, pivot_row_index: int):
        """
        Perform row operations to drop column from all but the pivot row

        Parameters
        ----------
        column_index : int
            The column/label to pivot.
        pivot_row_index : int
            The row to pivot
        """
        for i in range(self._tableau.shape[0]):
            if i != pivot_row_index:
                self._apply_pivot(column_index, pivot_row_index, i)

    def _pivot_on_column(self, column_index: int):
        """
        Perform a column pivot, returning the row/dropped label

        Parameters
        ----------
        column_index : int
            The column/label to pivot.

        Returns
        -------
        int
            The row chosen to pivot against
        """
        pivot_row_index = self._find_pivot_row(column_index)
        self._pivot(column_index, pivot_row_index)
        return pivot_row_index

    def _find_dropped(self, pivot_row_index: int, prev_basic_variables: Set) -> int:
        """
        Identifies the dropped label

        Parameters
        ----------
        pivot_row_index : int
            The row to find dropped label from
        prev_basic_variables : Set
            The candidates for labels that might have been dropped

        Returns
        -------
        int
            The dropped label

        Raises
        ------
        ValueError
            if no dropped label is identified.
        """
        for i in prev_basic_variables:
            if self._tableau[pivot_row_index, i] != 0:
                return i
        raise ValueError("could not find dropped label")

    def pivot_and_drop_label(self, column_index: int) -> int:
        """
        Pivots the tableau and returns the dropped label

        Parameters
        ----------
        column_index : int
            The index of a tableau on which to pivot.

        Returns
        -------
        int
            The dropped label.
        """
        prev_basic_vars = self.basic_variables
        row = self._pivot_on_column(column_index)
        dropped = self._find_dropped(row, prev_basic_vars)
        return dropped

    def _extract_label_values(self, column_index: int) -> List:
        """
        Calculates equlibria for a basic label in strategy

        Parameters
        ----------
        column_index : int
            The basic label to compute strategy for

        Returns
        -------
        List
            The computed unnormalized strategy
        """
        vertex = []
        for row, value in zip(self._tableau[:, column_index], self._tableau[:, -1]):
            if row != 0:
                vertex.append(value / row)
        return vertex

    def to_strategy(self, basic_labels: Set) -> npt.NDArray:
        """
        Return a strategy vector from a tableau

        Parameters
        ----------
        basic_labels : Set
            the set of basic labels in the other tableau.

        Returns
        -------
        array
            A strategy.
        """

        vertex = []
        for column in self._original_basic_labels:
            if column in basic_labels:
                vertex += self._extract_label_values(column)
            else:
                vertex.append(0)
        strategy = np.array(vertex)
        return strategy / sum(strategy)


class TableauLex(Tableau):
    """
    A tableau with lexiographic sorting to break ties when pivoting.
    This avoids endless looping that might occur with a standard
    tableau when applied on degenerate games.
    """

    def __init__(self, *kargs, **kwargs):
        """
        Constructs a lex Tableau for solving degenerate games.
        Parameters are inherited from Tableau

        Parameters
        ----------
        *kargs : Any
            Positional arguments passed to Tableau constructor
        **kwargs : Any
            Key value arguments passed to Tableau constructor
        """
        self._non_basic_variables = None
        super().__init__(*kargs, **kwargs)


def _build_tableau_matrix(payoffs: npt.NDArray, shifted: bool) -> npt.NDArray:
    """
    Build the tableau matrix from payoff. Can be shifted to preserve label indices.
    As required in lemke howson, payoffs are ensured to be positive.
    Adding a constant would anyways never affect the equilibrium

    Parameters
    ----------
    payoffs : array
        The payoff matrix
    shifted : bool
        When True, first indices will be slack vars

    Returns
    -------
    array
        the tableau matrix
    """
    if np.min(payoffs) <= 0:
        payoffs = payoffs + abs(np.min(payoffs)) + 1
    slack_vars = np.eye(payoffs.shape[0])
    targets = np.ones((payoffs.shape[0], 1))
    if shifted:
        return np.concatenate([slack_vars, payoffs, targets], axis=1)
    return np.concatenate([payoffs, slack_vars, targets], axis=1)


def create_row_tableau(payoffs: npt.NDArray, lexicographic=True):
    tableau = _build_tableau_matrix(payoffs.transpose(), False)
    if lexicographic:
        return TableauLex(tableau)
    return Tableau(tableau)


def create_col_tableau(payoffs: npt.NDArray, lexicographic=False):
    tableau = _build_tableau_matrix(payoffs, True)
    if lexicographic:
        return TableauLex(tableau)
    return Tableau(tableau)


def lemke_howson(
        A: npt.NDArray,
        B: npt.NDArray,
        initial_dropped_label: int = 0,
        lexicographic: bool = True,
) -> Tuple[npt.NDArray, npt.NDArray]:

    col_tableau = create_col_tableau(A, lexicographic)
    row_tableau = create_row_tableau(B, lexicographic)

    if initial_dropped_label in row_tableau.non_basic_variables:
        tableux = cycle((row_tableau, col_tableau))
    else:
        tableux = cycle((col_tableau, row_tableau))

    full_labels = col_tableau.labels
    fully_labeled = False
    entering_label = initial_dropped_label
    while not fully_labeled:
        tableau = next(tableux)
        entering_label = tableau.pivot_and_drop_label(entering_label)
        current_labels = col_tableau.non_basic_variables.union(
            row_tableau.non_basic_variables
        )
        fully_labeled = current_labels == full_labels

    row_strat = row_tableau.to_strategy(col_tableau.non_basic_variables)
    col_strat = col_tableau.to_strategy(row_tableau.non_basic_variables)

    return row_strat, col_strat


class Game:
    def __init__(self, *args) -> None:
        if len(args) == 2:
            if (not len(args[0]) == len(args[1])) or (
                    not len(args[0][0]) == len(args[1][0])
            ):
                raise ValueError("Unequal dimensions for matrices A and B")
            self.payoff_matrices = tuple([np.asarray(m) for m in args])
        if len(args) == 1:
            self.payoff_matrices = np.asarray(args[0]), -np.asarray(args[0])
        self.zero_sum = np.array_equal(
            self.payoff_matrices[0], -self.payoff_matrices[1]
        )

    def lemke_howson(self, initial_dropped_label):
        """
        Obtain the Nash equilibria using the Lemke Howson algorithm implemented
        using integer pivoting.

        Algorithm implemented here is Algorithm 3.6 of [Nisan2007]_.

        1. Start at the artificial equilibrium (which is fully labeled)
        2. Choose an initial label to drop and move in the polytope for which
           the vertex has that label to the edge
           that does not share that label. (This is implemented using integer
           pivoting)
        3. A label will now be duplicated in the other polytope, drop it in a
           similar way.
        4. Repeat steps 2 and 3 until have Nash Equilibrium.

        Parameters
        ----------
        initial_dropped_label: int
            The initial dropped label.

        Returns
        -------
        Tuple
            An equilibria
        """
        return lemke_howson(
            *self.payoff_matrices, initial_dropped_label=initial_dropped_label
        )


# Step 1: Read input matrices
n, m = map(int, input().split())

profits_abolf = [list(map(int, input().split())) for _ in range(n)]
profits_behzad = [list(map(int, input().split())) for _ in range(n)]

# Step 2: Create the game
game = Game(np.array(profits_abolf), np.array(profits_behzad))

# Step 3: Compute Nash equilibrium using Lemke-Howson algorithm
eqs = game.lemke_howson(initial_dropped_label=1)

# Step 4: Print strategies
abolf_strategy = list(map(lambda x: f"{x:.6f}", eqs[0]))
behzad_strategy = list(map(lambda x: f"{x:.6f}", eqs[1]))

print(*abolf_strategy)
print(*behzad_strategy)
