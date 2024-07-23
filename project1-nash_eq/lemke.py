from itertools import cycle
from typing import Set, List, Iterable, Optional, Tuple

import numpy as np
import numpy.typing as npt

class Tableau:
    def __init__(
            self, tableau: npt.NDArray, original_basic_labels: Optional[Iterable] = None
    ):

        self._tableau = tableau
        if original_basic_labels is not None:
            self._original_basic_labels = set(original_basic_labels)
        else:
            self._original_basic_labels = self.non_basic_variables

    @property
    def labels(self) -> Set:
        h, w = self._tableau.shape
        real_w = w - h - 1
        return set(range(h + real_w))

    @property
    def non_basic_variables(self) -> Set:
        columns = self._tableau[:, :-1].transpose()
        return set(np.where([np.count_nonzero(col) != 1 for col in columns])[0])

    @property
    def basic_variables(self) -> Set:
        return self.labels - self.non_basic_variables

    def find_pivot_row(self, column_index: int) -> int:
        row_ratios = self._tableau[:, column_index] / self._tableau[:, -1]
        return int(np.argmax(row_ratios))

    def apply_pivot(self, pivot_col: int, pivot_row: int, applying_row: int):
        pivot_element = self._tableau[pivot_row, pivot_col]
        row_pivot_val = self._tableau[applying_row, pivot_col]
        row = self._tableau[applying_row, :] * pivot_element
        row -= self._tableau[pivot_row, :] * row_pivot_val
        self._tableau[applying_row, :] = row

    def pivot(self, column_index: int, pivot_row_index: int):
        for i in range(self._tableau.shape[0]):
            if i != pivot_row_index:
                self.apply_pivot(column_index, pivot_row_index, i)

    def pivot_on_column(self, column_index: int):
        pivot_row_index = self.find_pivot_row(column_index)
        self.pivot(column_index, pivot_row_index)
        return pivot_row_index

    def find_dropped(self, pivot_row_index: int, prev_basic_variables: Set) -> int:
        for i in prev_basic_variables:
            if self._tableau[pivot_row_index, i] != 0:
                return i
        raise ValueError("Could not find dropped label")

    def pivot_and_drop_label(self, column_index: int) -> int:
        prev_basic_vars = self.basic_variables
        row = self.pivot_on_column(column_index)
        dropped = self.find_dropped(row, prev_basic_vars)
        return dropped

    def extract_label_values(self, column_index: int) -> List:
        vertex = []
        for row, value in zip(self._tableau[:, column_index], self._tableau[:, -1]):
            if row != 0:
                vertex.append(value / row)
        return vertex

    def to_strategy(self, basic_labels: Set) -> npt.NDArray:
        vertex = []
        for column in self._original_basic_labels:
            if column in basic_labels:
                vertex += self.extract_label_values(column)
            else:
                vertex.append(0)
        strategy = np.array(vertex)
        return strategy / sum(strategy)


class TableauLex(Tableau):
    def __init__(self, *kargs, **kwargs):
        self._non_basic_variables = None
        super().__init__(*kargs, **kwargs)


def build_tableau_matrix(payoffs: npt.NDArray, shifted: bool) -> npt.NDArray:
    if np.min(payoffs) <= 0:
        payoffs = payoffs + abs(np.min(payoffs)) + 1
    slack_vars = np.eye(payoffs.shape[0])
    targets = np.ones((payoffs.shape[0], 1))
    if shifted:
        return np.concatenate([slack_vars, payoffs, targets], axis=1)
    return np.concatenate([payoffs, slack_vars, targets], axis=1)


def create_row_tableau(payoffs: npt.NDArray, lexicographic=True):
    tableau = build_tableau_matrix(payoffs.transpose(), False)
    if lexicographic:
        return TableauLex(tableau)
    return Tableau(tableau)


def create_col_tableau(payoffs: npt.NDArray, lexicographic=False):
    tableau = build_tableau_matrix(payoffs, True)
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
        current_labels = col_tableau.non_basic_variables.union(row_tableau.non_basic_variables)
        fully_labeled = current_labels == full_labels

    row_strat = row_tableau.to_strategy(col_tableau.non_basic_variables)
    col_strat = col_tableau.to_strategy(row_tableau.non_basic_variables)

    return row_strat, col_strat


# Step 1: Read input matrices
n, m = map(int, input().split())

profits_abolf = [list(map(int, input().split())) for _ in range(n)]
profits_behzad = [list(map(int, input().split())) for _ in range(n)]

# Step 3: Compute Nash equilibrium using Lemke-Howson algorithm
eqs = lemke_howson(np.array(profits_abolf), np.array(profits_behzad), initial_dropped_label=1)

# Step 4: Print strategies
abolf_strategy = list(map(lambda x: f"{x:.6f}", eqs[0]))
behzad_strategy = list(map(lambda x: f"{x:.6f}", eqs[1]))

print(*abolf_strategy)
print(*behzad_strategy)
