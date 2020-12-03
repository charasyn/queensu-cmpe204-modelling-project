
from nnf import Var
from lib204 import Encoding

from nnf import NNF
from nnf.operators import iff

def implication(l, r):
    return l.negate() | r

def neg(f):
    return f.negate()

NNF.__invert__ = neg
NNF.__rshift__ = implication

# We are going to define some non-boolean variables here to represent
# the puzzle, for ease of setting our constraints later.
PUZZLE_NUM_ROWS = 5
PUZZLE_NUM_COLS = 5
PUZZLE_GRID_STATE = [
    "    t",
    "xtx x",
    "t    ",
    "xt   ",
    "   tx",
]
PUZZLE_HINT_ROWS = [0,3,0,1,1]
PUZZLE_HINT_COLS = [2,0,1,0,2]

# The maximum number of tents per row/column is equal to the
# number of columns/rows in the puzzle, divided by 2 and rounded up.
# Example:
#                                                _ _ _ _ _
#   # of col = 5 -> ceil(5/2) = ceil(2.5) = 3 : |T| |T| |T|
# 
#                                             _ _ _ _ _ _           _ _ _ _ _ _
#   # of col = 6 -> ceil(6/2) = ceil(3) = 3: |T| |T| |T| |  --or-- | |T| |T| |T|
# 
# Doing ((n+1) // 2) is equivalent to doing ceil(n/2).
MAX_TENTS_PER_ROW = (PUZZLE_NUM_COLS + 1) // 2
MAX_TENTS_PER_COL = (PUZZLE_NUM_ROWS + 1) // 2

# This should be all the data required about our puzzle. Let's guarantee it's correct.
def verify_puzzle_data_format():
    # Verify grid dimensions and cell data
    assert len(PUZZLE_GRID_STATE) == PUZZLE_NUM_ROWS, "Puzzle grid does not match given number of rows"
    for i,row in enumerate(PUZZLE_GRID_STATE):
        assert len(row) == PUZZLE_NUM_COLS, f"Puzzle grid (row {i}) does not match given number of columns"
        for j,cell in enumerate(row):
            assert cell in 'tx ', f"Cell at ({i},{j}) has invalid value, must be 't' for tree, 'x' for tent, or ' ' for blank"
    
    # Verify hint dimensions and data
    assert len(PUZZLE_HINT_ROWS) == PUZZLE_NUM_ROWS, "Puzzle row hints do not match given number of rows"
    for i,row_hint in enumerate(PUZZLE_HINT_ROWS):
        assert 0 <= row_hint and row_hint <= MAX_TENTS_PER_ROW, f"Row {i} has invalid hint {row_hint}, must be from 0 to {MAX_TENTS_PER_ROW}"

    assert len(PUZZLE_HINT_COLS) == PUZZLE_NUM_COLS, "Puzzle column hints do not match given number of columns"
    for j,col_hint in enumerate(PUZZLE_HINT_COLS):
        assert 0 <= col_hint and col_hint <= MAX_TENTS_PER_COL, f"Column {j} has invalid hint {col_hint}, must be from 0 to {MAX_TENTS_PER_COL}"

verify_puzzle_data_format()

# Note: All 2D arrays in this project will be using [row][column] indexing, starting at 0.
# Define a helper function to build a 2D array
def build_2D_var_array(num_rows, num_cols, format_string='({i},{j})'):
    """A helper function to build a 2D array of variables
    
    Inputs:

    num_rows: the number of rows in this array

    num_cols: the number of columns in this array

    format_string: a string defining the name of each variable, where
    \{i\} and \{j\} will be replaced with the row and column number respectively
    
    Outputs:
    
    array: an array with the specified number of rows and columns"""

    array = []
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            row.append(Var(format_string.format(i=i,j=j)))
        array.append(row)
    return array

###############################################################################
## PRIMARY PROPOSITIONS
# t_i,j : true iff there is a tree at location (i,j) in the puzzle
t = build_2D_var_array(PUZZLE_NUM_ROWS, PUZZLE_NUM_COLS, format_string='t_{i},{j}')

# x_i,j : true iff there is a tent at location (i,j) in the puzzle
x = build_2D_var_array(PUZZLE_NUM_ROWS, PUZZLE_NUM_COLS, format_string='x_{i},{j}')

# r_i;n : true iff row i should have n tents in the row
r = build_2D_var_array(PUZZLE_NUM_ROWS, MAX_TENTS_PER_ROW+1, format_string='r_{i};{j}')

# c_j;n : true iff column j should have n tents in the column
c = build_2D_var_array(PUZZLE_NUM_COLS, MAX_TENTS_PER_COL+1, format_string='c_{i};{j}')

###############################################################################
## SECONDARY PROPOSITIONS
# TODO: add the rest of the propositions

#
# Build an example full theory for your setting and return it.
#
#  There should be at least 10 variables, and a sufficiently large formula to describe it (>50 operators).
#  This restriction is fairly minimal, and if there is any concern, reach out to the teaching staff to clarify
#  what the expectations are.
def example_theory():
    E = Encoding()

    # Add constraints for grid state based on PUZZLE_* variables
    # Grid state constraints for t_i,j and x_i,j
    for i,row_data in enumerate(PUZZLE_GRID_STATE):
        for j,cell_data in enumerate(row_data):
            if cell_data == 't':
                # Cell is a tree
                E.add_constraint(t[i][j] & ~x[i][j])
            elif cell_data == 'x':
                # Cell is a tent
                E.add_constraint(~t[i][j] & x[i][j])
            else:
                # Cell is empty
                E.add_constraint(~t[i][j] & ~x[i][j])

    # Hint constraints for r_i;n and c_j;n
    for i,row_hint in enumerate(PUZZLE_HINT_ROWS):
        # Get positive constraint
        constraint_accumulator = r[i][row_hint]

        # Add in negative constraints using loop
        for n in range(MAX_TENTS_PER_ROW+1):
            if n == row_hint:
                continue
            constraint_accumulator &= ~r[i][n]

        # Add final combined constraint
        E.add_constraint(constraint_accumulator)

    for j,col_hint in enumerate(PUZZLE_HINT_COLS):
        # Get positive constraint
        constraint_accumulator = c[j][col_hint]

        # Add in negative constraints using loop
        for n in range(MAX_TENTS_PER_COL+1):
            if n == col_hint:
                continue
            constraint_accumulator &= ~c[j][n]

        # Add final combined constraint
        E.add_constraint(constraint_accumulator)

    # Checking that each row and column have the correct number of tents
    # Will need to be changed from exhaustive listing to some adder later
    for i in range(PUZZLE_NUM_ROWS):
        # no tents
        E.add_constraint(iff(r[i][0],  ~x[i][0] & ~x[i][1] & ~x[i][2] & ~x[i][3] & ~x[i][4]))
        # 1 tent
        E.add_constraint(iff(r[i][1], ( x[i][0] & ~x[i][1] & ~x[i][2] & ~x[i][3] & ~x[i][4]) | \
                                      (~x[i][0] &  x[i][1] & ~x[i][2] & ~x[i][3] & ~x[i][4]) | \
                                      (~x[i][0] & ~x[i][1] &  x[i][2] & ~x[i][3] & ~x[i][4]) | \
                                      (~x[i][0] & ~x[i][1] & ~x[i][2] &  x[i][3] & ~x[i][4]) | \
                                      (~x[i][0] & ~x[i][1] & ~x[i][2] & ~x[i][3] &  x[i][4])))
        # 2 tents
        E.add_constraint(iff(r[i][2], ( x[i][0] & ~x[i][1] &  x[i][2] & ~x[i][3] & ~x[i][4]) | \
                                      ( x[i][0] & ~x[i][1] & ~x[i][2] &  x[i][3] & ~x[i][4]) | \
                                      ( x[i][0] & ~x[i][1] & ~x[i][2] & ~x[i][3] &  x[i][4]) | \
                                      (~x[i][0] &  x[i][1] & ~x[i][2] &  x[i][3] & ~x[i][4]) | \
                                      (~x[i][0] &  x[i][1] & ~x[i][2] & ~x[i][3] &  x[i][4]) | \
                                      (~x[i][0] & ~x[i][1] &  x[i][2] & ~x[i][3] &  x[i][4])))
        # 3 tents
        E.add_constraint(iff(r[i][3],   x[i][0] & ~x[i][1] &  x[i][2] & ~x[i][3] &  x[i][4]))

    for j in range(PUZZLE_NUM_COLS):
        # no tents
        E.add_constraint(iff(c[j][0],  ~x[0][j] & ~x[1][j] & ~x[2][j] & ~x[3][j] & ~x[4][j]))
        # 1 tent
        E.add_constraint(iff(c[j][1], ( x[0][j] & ~x[1][j] & ~x[2][j] & ~x[3][j] & ~x[4][j]) | \
                                      (~x[0][j] &  x[1][j] & ~x[2][j] & ~x[3][j] & ~x[4][j]) | \
                                      (~x[0][j] & ~x[1][j] &  x[2][j] & ~x[3][j] & ~x[4][j]) | \
                                      (~x[0][j] & ~x[1][j] & ~x[2][j] &  x[3][j] & ~x[4][j]) | \
                                      (~x[0][j] & ~x[1][j] & ~x[2][j] & ~x[3][j] &  x[4][j])))
        # 2 tents
        E.add_constraint(iff(c[j][2], ( x[0][j] & ~x[1][j] &  x[2][j] & ~x[3][j] & ~x[4][j]) | \
                                      ( x[0][j] & ~x[1][j] & ~x[2][j] &  x[3][j] & ~x[4][j]) | \
                                      ( x[0][j] & ~x[1][j] & ~x[2][j] & ~x[3][j] &  x[4][j]) | \
                                      (~x[0][j] &  x[1][j] & ~x[2][j] &  x[3][j] & ~x[4][j]) | \
                                      (~x[0][j] &  x[1][j] & ~x[2][j] & ~x[3][j] &  x[4][j]) | \
                                      (~x[0][j] & ~x[1][j] &  x[2][j] & ~x[3][j] &  x[4][j])))
        # 3 tents
        E.add_constraint(iff(c[j][3],   x[0][j] & ~x[1][j] &  x[2][j] & ~x[3][j] &  x[4][j]))

    return E

def get_row_hint(tprops, row):
    search_str = 'r_{};'.format(row)
    matching = [prop for prop in tprops if search_str in prop]
    assert len(matching) == 1, "Solver returned multiple true row hints {}".format(matching)
    return int(matching[0].replace(search_str,''))
    
def get_col_hint(tprops, col):
    search_str = 'c_{};'.format(col)
    matching = [prop for prop in tprops if search_str in prop]
    assert len(matching) == 1, "Solver returned multiple true col hints {}".format(matching)
    return int(matching[0].replace(search_str,''))

def visualize_solution(soln):
    true_propositions = [prop for (prop, assignment) in soln.items() if assignment]
    row_hints = [get_row_hint(true_propositions, row) for row in range(PUZZLE_NUM_ROWS)]
    col_hints = [get_col_hint(true_propositions, col) for col in range(PUZZLE_NUM_COLS)]
    print('Row hints:', row_hints)
    print('Col hints:', col_hints)
    print('Puzzle grid:')
    print('+','-' * PUZZLE_NUM_ROWS,'+',sep='')
    for i in range(PUZZLE_NUM_ROWS):
        line_join = []
        for j in range(PUZZLE_NUM_COLS):
            idx_str = '_{},{}'.format(i,j)
            t = soln['t'+idx_str]
            x = soln['x'+idx_str]
            assert not (t and x), "Tree and tent at same location ({},{}) in solution".format(i,j)
            char = ' '
            if t:
                char = 't'
            if x:
                char = 'x'
            line_join.append(char)
        print('|',''.join(line_join),'|',sep='')
    print('+','-' * PUZZLE_NUM_ROWS,'+',sep='')

def build_negated_solution_nnf(soln):
    nnf = None
    # Suppose our solution is {a=True, b=False, c=True}.
    # We could construct ~(a&~b&c), but the NNF library
    # would then have to perform the negation for us.
    # We can instead construct something that's already
    # in NNF, which is equivalent to what the NNF library
    # would give us: ~a|b|~c
    for prop,assignment in soln.items():
        var = Var(prop, true=not assignment)
        if nnf is None:
            nnf = var
        else:
            nnf |= var
    return nnf

if __name__ == "__main__":

    T = example_theory()

    print("\nSatisfiable: %s" % T.is_satisfiable())
    soln_count = T.count_solutions()
    print("# Solutions: %d" % soln_count)
    for i in range(min(10,soln_count)):
        print("=" * 72)
        print('Solution', i+1)
        soln = T.solve()
        # print("   Solution: %s" % soln)
        visualize_solution(soln)
        T.add_constraint(build_negated_solution_nnf(soln))
    if soln_count > 10:
        print('Only first 10 solutions were printed...')

    # We are (initially) trying to detect if a given puzzle is properly solved.
    # Because of this, we are really only looking to see if the set of constraints
    # is satisfiable, and there will be only 1 or 0 solutions.
    # In the future, if we want to check and see if the puzzle -can- be solved, or
    # use this to solve a puzzle, then I think the variable likelihoods would
    # make more sense.
    # For now, this is commented out because I deleted the "abcxyz" variables.
    # print("\nVariable likelihoods:")
    # for v,vn in zip([a,b,c,x,y,z], 'abcxyz'):
    #     print(" %s: %.2f" % (vn, T.likelihood(v)))
    print()
