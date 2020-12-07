
from nnf import Var
from lib204 import Encoding
from functools import reduce
from random import random, seed

from dictattraccess import DictAttrAccess

from nnf import NNF
from nnf.operators import iff

def implication(l, r):
    return l.negate() | r

def neg(f):
    return f.negate()

NNF.__invert__ = neg
NNF.__rshift__ = implication


class TentsAndTreesTheory:
    DIRECTIONS = [(-1,0),(1,0),(0,-1),(0,1)]
    # Note: All 2D arrays in this project will be using [row][column] indexing, starting at 0.
    # Define a helper function to build a 2D array
    @staticmethod
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
    
    @staticmethod
    def generate_random_trees(size=(5,5), density=0.3):
        trees = []
        for i in range(size[0]):
            line_join = []
            for j in range(size[1]):
                line_join.append('t' if random() < density else ' ')
            trees.append(''.join(line_join))
        return trees

    def __init__(self, size=(5,5)):
        self.num_rows = size[0]
        self.num_cols = size[1]
        assert self.num_cols == 5 and self.num_rows == 5, "Code doesn't support sizes other than 5x5"
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
        self.max_tents_per_row = (self.num_cols + 1) // 2
        self.max_tents_per_col = (self.num_rows + 1) // 2

        self.theory = Encoding()
        self.build_propositions()
        self.build_general_constraints()

    def verify_puzzle_grid_state_format(self, puzzle_grid_state):
        # Verify grid dimensions and cell data
        assert len(puzzle_grid_state) == self.num_rows, "Puzzle grid does not match given number of rows"
        for i,row in enumerate(puzzle_grid_state):
            assert len(row) == self.num_cols, f"Puzzle grid (row {i}) does not match given number of columns"
            for j,cell in enumerate(row):
                assert cell in 'tx ', f"Cell at ({i},{j}) has invalid value, must be 't' for tree, 'x' for tent, or ' ' for blank"

    def build_propositions(self):
        self.props = DictAttrAccess()
        ###############################################################################
        ## PRIMARY PROPOSITIONS
        # t_i,j : true iff there is a tree at location (i,j) in the puzzle
        self.props.t = TentsAndTreesTheory.build_2D_var_array(self.num_rows, self.num_cols, format_string='t_{i},{j}')

        # x_i,j : true iff there is a tent at location (i,j) in the puzzle
        self.props.x = TentsAndTreesTheory.build_2D_var_array(self.num_rows, self.num_cols, format_string='x_{i},{j}')

        # r_i;n : true iff row i should have n tents in the row
        self.props.r = TentsAndTreesTheory.build_2D_var_array(self.num_rows, self.max_tents_per_row+1, format_string='r_{i};{j}')

        # c_j;n : true iff column j should have n tents in the column
        self.props.c = TentsAndTreesTheory.build_2D_var_array(self.num_cols, self.max_tents_per_col+1, format_string='c_{i};{j}')

        ###############################################################################
        ## SECONDARY PROPOSITIONS
        # Adjacency array: A_d_i,j : true iff the tree at location (i,j) is paired with the tent in dir'n d
        #   d ranges from 0-3, and represents [u,d,l,r] - see TentsAndTreesTheory.DIRECTIONS
        #   Directions that don't exist are skipped (ex. you can't go up from 0,0) and are set to None in the table.

        A = []
        for d,(offset_i,offset_j) in enumerate(TentsAndTreesTheory.DIRECTIONS):
            array = []
            for i in range(self.num_rows):
                row = []
                for j in range(self.num_cols):
                    adj_i, adj_j = offset_i + i, offset_j + j
                    if 0 <= adj_i and adj_i < self.num_rows and 0 <= adj_j and adj_j < self.num_cols:
                        row.append(Var('A_{}_{},{}'.format(d,i,j)))
                    else:
                        row.append(None)
                array.append(row)
            A.append(array)
        self.props.A = A

    def safe_A(self, d, i, j):
        return self.props.A[d][i][j] if 0 <= i and i < self.num_rows and 0 <= j and j < self.num_cols else None

    ### Build constraints for the theory
    def verify_puzzle_hint_format(self, puzzle_hint_rows, puzzle_hint_cols):
        # Verify hint dimensions and data
        assert len(puzzle_hint_rows) == self.num_rows, "Puzzle row hints do not match given number of rows"
        for i,row_hint in enumerate(puzzle_hint_rows):
            assert 0 <= row_hint and row_hint <= self.max_tents_per_row, f"Row {i} has invalid hint {row_hint}, must be from 0 to {self.max_tents_per_row}"
        for j,col_hint in enumerate(puzzle_hint_cols):
            assert 0 <= col_hint and col_hint <= self.max_tents_per_col, f"Column {j} has invalid hint {col_hint}, must be from 0 to {self.max_tents_per_col}"

    def build_board_hint_constraints(self, puzzle_hint_rows, puzzle_hint_cols):
        E = self.theory
        t = self.props.t
        x = self.props.x
        r = self.props.r
        c = self.props.c
        A = self.props.A

        # Ensure the data we were given is valid
        self.verify_puzzle_hint_format(puzzle_hint_rows, puzzle_hint_cols)

        # Hint constraints for r_i;n and c_j;n
        for i,row_hint in enumerate(puzzle_hint_rows):
            # Get positive constraint
            constraint_accumulator = r[i][row_hint]

            # Add in negative constraints using loop
            for n in range(self.max_tents_per_row+1):
                if n == row_hint:
                    continue
                constraint_accumulator &= ~r[i][n]

            # Add final combined constraint
            E.add_constraint(constraint_accumulator)

        for j,col_hint in enumerate(puzzle_hint_cols):
            # Get positive constraint
            constraint_accumulator = c[j][col_hint]

            # Add in negative constraints using loop
            for n in range(self.max_tents_per_col+1):
                if n == col_hint:
                    continue
                constraint_accumulator &= ~c[j][n]

            # Add final combined constraint
            E.add_constraint(constraint_accumulator)

    def build_board_tree_constraints(self, puzzle_grid_state):
        E = self.theory
        t = self.props.t
        x = self.props.x
        r = self.props.r
        c = self.props.c
        A = self.props.A

        # Ensure the data we were given is valid
        self.verify_puzzle_grid_state_format(puzzle_grid_state)

        # Grid state constraints for t_i,j and x_i,j
        for i,row_data in enumerate(puzzle_grid_state):
            for j,cell_data in enumerate(row_data):
                if cell_data == 't':
                    # Cell is a tree
                    E.add_constraint(t[i][j])
                else:
                    # Cell is not a tree
                    E.add_constraint(~t[i][j])

    def build_board_tent_constraints(self, puzzle_grid_state):
        E = self.theory
        t = self.props.t
        x = self.props.x
        r = self.props.r
        c = self.props.c
        A = self.props.A

        # Ensure the data we were given is valid
        self.verify_puzzle_grid_state_format(puzzle_grid_state)

        # Grid state constraints for t_i,j and x_i,j
        for i,row_data in enumerate(puzzle_grid_state):
            for j,cell_data in enumerate(row_data):
                if cell_data == 'x':
                    # Cell is a tent
                    E.add_constraint(x[i][j])
                else:
                    # Cell is not a tent
                    E.add_constraint(~x[i][j])

    def build_general_constraints(self):
        E = self.theory
        t = self.props.t
        x = self.props.x
        r = self.props.r
        c = self.props.c
        A = self.props.A

        # Checking that each row and column have the correct number of tents
        # Will need to be changed from exhaustive listing to some adder later
        # This also verifies that the row and column hints are set properly
        for i in range(self.num_rows):
            # Ensure only one of r[i][...] is set
            E.add_constraint(( r[i][0] & ~r[i][1] & ~r[i][2] & ~r[i][3]) | \
                             (~r[i][0] &  r[i][1] & ~r[i][2] & ~r[i][3]) | \
                             (~r[i][0] & ~r[i][1] &  r[i][2] & ~r[i][3]) | \
                             (~r[i][0] & ~r[i][1] & ~r[i][2] &  r[i][3]))
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

        for j in range(self.num_cols):
            # Ensure only one of c[j][...] is set
            E.add_constraint(( c[j][0] & ~c[j][1] & ~c[j][2] & ~c[j][3]) | \
                             (~c[j][0] &  c[j][1] & ~c[j][2] & ~c[j][3]) | \
                             (~c[j][0] & ~c[j][1] &  c[j][2] & ~c[j][3]) | \
                             (~c[j][0] & ~c[j][1] & ~c[j][2] &  c[j][3]))
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

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                ### Constrain that tents and trees can't be in the same square
                E.add_constraint((~x[i][j] & ~t[i][j]) | (~x[i][j] &  t[i][j]) | ( x[i][j] & ~t[i][j]))

                ### Constrain that tents can't be adjacent
                # For each cell in the grid, if there is a tent there, then there can't be one:
                # to the right, below, below and to the right, or below and to the left.
                # I.e.
                #  |x|1
                # -+-+-
                # 4|2|3
                # If x is a tent, then 1,2,3,4 must not be tents.
                # We need to be careful to not exceed the bounds of our arrays however.
                adjacent_cells = []
                # Check if 1 is in bounds
                if j < (self.num_cols - 1):
                    adjacent_cells.append(x[i][j+1])
                # Check if 2 is in bounds
                if i < (self.num_rows - 1):
                    adjacent_cells.append(x[i+1][j])
                # 3 is in bounds iff both 1 and 2 are in bounds.
                if len(adjacent_cells) == 2:
                    adjacent_cells.append(x[i+1][j+1])
                # Check if 4 is in bounds
                if i < (self.num_rows - 1) and j > 0:
                    adjacent_cells.append(x[i+1][j-1])
                # Reduce adjacent cell list into formula saying that those cells are false
                # (if we have any adjacent cells)
                if len(adjacent_cells) > 0:
                    formula = reduce(lambda acc,x: acc & x, (~x for x in adjacent_cells))
                    E.add_constraint(x[i][j] >> formula)

                ### Constrain that tent must be adjacent to tree
                # For each cell in the grid, if there is a tent there, then there must be
                # a tree adjacent.
                # I.e.
                #  |1|
                # -+-+-
                # 3|x|4
                # -+-+-
                #  |2|
                # If x is a tent, then one (or more) of 1,2,3,4 must be a tree.
                # We need to be careful to not exceed the bounds of our arrays however.
                adjacent_cells = []
                # Check if 1 is in bounds
                if i > 0:
                    adjacent_cells.append(t[i-1][j])
                # Check if 2 is in bounds
                if i < (self.num_rows - 1):
                    adjacent_cells.append(t[i+1][j])
                # Check if 3 is in bounds
                if j > 0:
                    adjacent_cells.append(t[i][j-1])
                # Check if 4 is in bounds
                if j < (self.num_cols - 1):
                    adjacent_cells.append(t[i][j+1])
                # Reduce adjacent cell list into formula saying that one of those cells is true
                # (if we have any adjacent cells)
                if len(adjacent_cells) > 0:
                    formula = reduce(lambda acc,x: acc | x, adjacent_cells)
                    E.add_constraint(x[i][j] >> formula)

                ### Constrain that if t(i,j) is true, then exactly one of A(d,i,j) must be true for all d
                # Get all A(d,i,j) that are valid (not out of bounds)
                adjacency_entries = [A[d][i][j] for d in range(len(TentsAndTreesTheory.DIRECTIONS)) if A[d][i][j] is not None]
                # Build up (A & ~B & ~C) | (~A & B & ~C) | (~A & ~B & C) expression
                parts = []
                for positive_adj in adjacency_entries:
                    # Build individual (A & ~B & ~C) and put it into parts
                    part_formula = positive_adj
                    for other_adj in adjacency_entries:
                        if other_adj == positive_adj:
                            continue
                        part_formula &= ~other_adj
                    parts.append(part_formula)
                # OR all individual parts together
                formula = reduce(lambda acc,x: acc | x, parts)
                E.add_constraint(t[i][j] >> formula)

                ### Constrain that if x(i,j) is true, then exactly one of the A entries pointing at it must be true
                # Get all A(d,i,j) that are valid (not out of bounds)
                adjacency_entries = [self.safe_A(d,i-oi,j-oj) for d,(oi,oj) in enumerate(TentsAndTreesTheory.DIRECTIONS) if self.safe_A(d,i-oi,j-oj) is not None]
                # Build up (A & ~B & ~C) | (~A & B & ~C) | (~A & ~B & C) expression
                parts = []
                for positive_adj in adjacency_entries:
                    # Build individual (A & ~B & ~C) and put it into parts
                    part_formula = positive_adj
                    for other_adj in adjacency_entries:
                        if other_adj == positive_adj:
                            continue
                        part_formula &= ~other_adj
                    parts.append(part_formula)
                # OR all individual parts together
                formula = reduce(lambda acc,x: acc | x, parts)
                E.add_constraint(x[i][j] >> formula)

                ### More adjacency constraints, direction-specific
                for d,(offset_i,offset_j) in enumerate(TentsAndTreesTheory.DIRECTIONS):
                    # Check if grid in this direction exists by checking the A matrix
                    if A[d][i][j] is None:
                        continue
                    ### Constrain that if A(d,i,j) is true, then t(i,j) is true
                    E.add_constraint(A[d][i][j] >> t[i][j])
                    ### Constrain that if A(d,i,j) is true, then x((i,j) + dirn) is true
                    E.add_constraint(A[d][i][j] >> x[i+offset_i][j+offset_j])
        self.theory = E

class TentsAndTreesTheorySolution:
    @staticmethod
    def get_row_hint(tprops, row):
        search_str = 'r_{};'.format(row)
        matching = [prop for prop in tprops if search_str in prop]
        assert len(matching) == 1, "Solver returned multiple true row hints {}".format(matching)
        return int(matching[0].replace(search_str,''))

    @staticmethod
    def get_col_hint(tprops, col):
        search_str = 'c_{};'.format(col)
        matching = [prop for prop in tprops if search_str in prop]
        assert len(matching) == 1, "Solver returned multiple true col hints {}".format(matching)
        return int(matching[0].replace(search_str,''))

    def __init__(self, theory_obj: TentsAndTreesTheory, soln):
        self.theory_obj = theory_obj
        self.soln = soln
        # Build list of true propositions
        self.tprops = [prop for (prop, assignment) in soln.items() if assignment]

        self.build_hints()
        self.build_board_state()
        self.build_adjacency_state()
    
    def build_hints(self):
        self.row_hints = [TentsAndTreesTheorySolution.get_row_hint(self.tprops, row) for row in range(self.theory_obj.num_rows)]
        self.col_hints = [TentsAndTreesTheorySolution.get_col_hint(self.tprops, col) for col in range(self.theory_obj.num_cols)]
    
    def build_board_state(self):
        st = []
        for i in range(self.theory_obj.num_rows):
            line_join = []
            for j in range(self.theory_obj.num_cols):
                idx_str = '_{},{}'.format(i,j)
                t = self.soln['t'+idx_str]
                x = self.soln['x'+idx_str]
                assert not (t and x), "Tree and tent at same location ({},{}) in solution".format(i,j)
                char = ' '
                if t:
                    char = 't'
                if x:
                    char = 'x'
                line_join.append(char)
            st.append(''.join(line_join))
        self.board_state = st

    def build_adjacency_state(self):
        A = []
        for d,_ in enumerate(TentsAndTreesTheory.DIRECTIONS):
            Ad = []
            for i in range(self.theory_obj.num_rows):
                line = []
                for j in range(self.theory_obj.num_cols):
                    a = self.soln.get(f'A_{d}_{i},{j}', False)
                    line.append(a)
                Ad.append(line)
            A.append(Ad)
        self.adjacency_state = A
    
    def visualize_solution(self):
        print('Row hints:', self.row_hints)
        print('Col hints:', self.col_hints)
        print('Puzzle grid:')
        print('+','-' * self.theory_obj.num_cols,'+',sep='')
        for line in self.board_state:
            print('|',line,'|',sep='')
        print('+','-' * self.theory_obj.num_cols,'+',sep='')

    def visualize_solution_svg(self,html_file):
        svg_scale = 50
        # print(f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
        print(f'''<svg xmlns="http://www.w3.org/2000/svg"
        width="{svg_scale*(2+self.theory_obj.num_cols)}"
        height="{svg_scale*(2+self.theory_obj.num_rows)}"
        viewBox="-1 -1 {2+self.theory_obj.num_cols} {2+self.theory_obj.num_rows}">''', file=html_file)
        print('''
        <style>
            .hint_text { font: 0.5px sans-serif; }
            .rect_grid { fill: none; stroke: #000; stroke-width: 0.1; }
            .rect_grid_empty { fill: #eee; stroke: #000; stroke-width: 0.1; }
            .tent { fill: #e11; }
            .tree { fill: #1e1; }
            .assoc { fill: none; stroke: #444; stroke-width: 0.03; }
        </style>
        ''', file=html_file)
        for i,h in enumerate(self.row_hints):
            print(f'<text class="hint_text" x="-0.5" y="{i+0.5}">{h}</text>', file=html_file)
        for j,h in enumerate(self.col_hints):
            print(f'<text class="hint_text" y="-0.5" x="{j+0.5}">{h}</text>', file=html_file)
        for i,line in enumerate(self.board_state):
            for j,char in enumerate(line):
                t = char == 't'
                x = char == 'x'
                assert not (t and x), "Tree and tent at same location ({},{}) in solution".format(i,j)
                print(f'<rect x="{j}" y="{i}" width="1" height="1" class="{"rect_grid" if t or x else "rect_grid_empty"}" />', file=html_file)
                if t or x:
                    print(f'<rect x="{j+0.2}" y="{i+0.2}" width="0.6" height="0.6" class="{"tent" if x else "tree"}" />', file=html_file)
                doff = 0.125
                for d,(x,y,w,h) in enumerate([
                    (doff,-1+doff,1-2*doff,2-2*doff),
                    (doff,   doff,1-2*doff,2-2*doff),
                    (-1+doff,doff,2-2*doff,1-2*doff),
                    (   doff,doff,2-2*doff,1-2*doff)
                    ]):
                    a = self.adjacency_state[d][i][j]
                    if not a:
                        continue
                    assert t, "T must be true for A to be true"
                    print(f'<rect x="{j+x}" y="{i+y}" width="{w}" height="{h}" class="assoc" />', file=html_file)
        print('</svg>', file=html_file)

    def build_negated_solution_nnf(self):
        nnf = None
        # Suppose our solution is {a=True, b=False, c=True}.
        # We could construct ~(a&~b&c), but the NNF library
        # would then have to perform the outer negation for us.
        # We can instead construct something that's already
        # in NNF, which is equivalent to what the NNF library
        # would give us: ~a|b|~c
        for prop,assignment in self.soln.items():
            var = Var(prop, true=not assignment)
            if nnf is None:
                nnf = var
            else:
                nnf |= var
        return nnf
    
    def remove_solution(self):
        for i, line in enumerate(self.board_state):
            self.board_state[i] = line.replace('x',' ')
            for j, _ in enumerate(line):
                # If there's an association here, remove it
                for d in range(4):
                    self.adjacency_state[d][i][j] = False
    
    def to_json(self, json_filename):
        import json
        json_obj = {
            'rows': self.theory_obj.num_rows,
            'cols': self.theory_obj.num_cols,
            'grid_state': self.board_state,
            'row_hints': self.row_hints,
            'col_hints': self.col_hints,
        }
        with open(json_filename, 'w') as f:
            json.dump(json_obj, f, sort_keys=True, indent=4)

def main(arguments=None):
    import sys
    import argparse

    # arguments = ['--output-html','test.html']

    global config

    parser = argparse.ArgumentParser(description='Work with the group 24 Tents and Trees model.')
    parser.add_argument('-q', '--quiet', action='store_true', help='print less output')
    parser.add_argument('-t', '--print-theory', action='store_true', help='print all constraints in the theory')
    parser.add_argument('-s', '--print-raw-solution', action='store_true', help='print all variables in each solution (raw output from dsharp)')
    parser.add_argument('--mode', default='solve', help='select one mode of: solve, generate')
    parser.add_argument('--input-json', metavar='<puzzle.json>', help='specify a .json file to load the puzzle from, instead of using the default built-in puzzle')
    parser.add_argument('--output-html', metavar='<puzzle.html>', help='specify an .html output file to print a nicer version to')
    parser.add_argument('--size', default='5x5', help='specify size of generated puzzle')
    parser.add_argument('--seed', help='specify starting seed of generated puzzle (default: random)')
    parser.add_argument('--output-json', metavar='<puzzle.json>', help='specify a .json file to save the generated puzzle to')
    
    if arguments:
        config = parser.parse_args(arguments)
    else:
        config = parser.parse_args()
    do_html = config.output_html
    do_json = config.input_json
    do_json_output = config.output_json
    quiet = config.quiet
    print_theory = config.print_theory
    print_raw_solution = config.print_raw_solution
    solve = config.mode == 'solve'
    generate = config.mode == 'generate'
    generate_size = tuple(int(n) for n in config.size.split('x'))
    generate_seed = config.seed

    if solve:
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
        # This is an example which IS NOT a valid puzzle. We need to be able to reject this.
        # PUZZLE_GRID_STATE = [
        #     "  tt ",
        #     " t   ",
        #     "     ",
        #     "   t ",
        #     " t   ",
        # ]
        # PUZZLE_HINT_ROWS = [1,1,2,0,1]
        # PUZZLE_HINT_COLS = [1,2,0,1,1]
        if do_json:
            import json
            jsondata = None
            with open(do_json, 'r') as f:
                jsondata = json.load(f)
            PUZZLE_NUM_ROWS = jsondata['rows']
            PUZZLE_NUM_COLS = jsondata['cols']
            PUZZLE_HINT_ROWS = jsondata['row_hints']
            PUZZLE_HINT_COLS = jsondata['col_hints']
            PUZZLE_GRID_STATE = jsondata['grid_state']

        Tclass = TentsAndTreesTheory(size=(PUZZLE_NUM_ROWS,PUZZLE_NUM_COLS))
        Tclass.build_board_hint_constraints(PUZZLE_HINT_ROWS,PUZZLE_HINT_COLS)
        Tclass.build_board_tree_constraints(PUZZLE_GRID_STATE)

        if do_html:
            html_file = open(do_html, 'w')
            print('''<!DOCTYPE html>\n<html><head><style>
                body {
                    font-family: sans-serif;
                }
            </style></head><body>''', file=html_file)

        if print_theory:
            if not quiet:
                print('Constraints:')
                for constr in Tclass.theory.constraints:
                    print(constr)
            if do_html:
                print(f'<p>Constraints:</p><ul>', file=html_file)
                for constr in Tclass.theory.constraints:
                    print(f'<li>{constr}</li>', file=html_file)
                print(f'</ul>', file=html_file)

        soln_count = Tclass.theory.count_solutions()
        if not quiet:
            print(f'# Solutions: {soln_count}')
        if do_html:
            print(f'<p># Solutions: {soln_count}</p>', file=html_file)
        
        for i in range(min(10,soln_count)):
            soln = Tclass.theory.solve()
            if soln is None:
                if not quiet:
                    print('Unable to find another solution!')
                if do_html:
                    print(f'<p>Unable to find another solution!</p>', file=html_file)
                break
            solnclass = TentsAndTreesTheorySolution(Tclass, soln)
            if not quiet:
                print("=" * 72)
                print('Solution', i+1)
                if print_raw_solution:
                    print("Solution: %s" % soln)
                solnclass.visualize_solution()
            if do_html:
                print(f'<hr><p>Solution #{i+1}</p>', file=html_file)
                if print_raw_solution:
                    print("<p>Solution:</p><ul>", file=html_file)
                    for prop,assign in sorted(soln.items()):
                        print(f'<li>{prop}: {assign}</li>', file=html_file)
                    print('</ul>', file=html_file)
                solnclass.visualize_solution_svg(html_file)
            Tclass.theory.add_constraint(solnclass.build_negated_solution_nnf())
        if soln_count > 10:
            if not quiet:
                print('Only first 10 solutions were printed...')
            if do_html:
                print(f'<p>Only first 10 solutions were printed...</p>', file=html_file)

        if do_html:
            print('</body></html>', file=html_file)
            html_file.close()
        
        return 1 if soln_count == 0 else 0
    
    elif generate:
        if generate_seed is not None:
            seed(generate_seed)
        puzzle_solved = None
        puzzle_unsolved = None
        # While we haven't managed to generate a solvable puzzle, keep on trying
        while not puzzle_solved:
            puzzle = TentsAndTreesTheory.generate_random_trees(generate_size)
            Tclass = TentsAndTreesTheory(generate_size)
            # We will provide only the tree locations to the SAT solver,
            # and let it tell us how many tents are in each row
            Tclass.build_board_tree_constraints(puzzle)
            if not Tclass.theory.is_satisfiable():
                if not quiet:
                    print("Puzzle not satisfiable, retrying...")
                continue
            # We know it's satisfiable. Get a solution from solver.
            if not quiet:
                print("Found satisfiable puzzle.")
            soln = Tclass.theory.solve()
            puzzle_solved = TentsAndTreesTheorySolution(Tclass, soln)
            # This solution represents our board! :)
            # Now we're going to build the solution object, and clear the solved state from it
            puzzle_unsolved = TentsAndTreesTheorySolution(Tclass, soln)
            puzzle_unsolved.remove_solution()
        if not quiet:
            print("=" * 72)
            print('Unsolved:')
            puzzle_unsolved.visualize_solution()
            print("=" * 72)
            print('Solved:')
            puzzle_solved.visualize_solution()
        if do_json_output:
            puzzle_unsolved.to_json(do_json_output)
        if do_html:
            html_file = open(do_html, 'w')
            print('''<!DOCTYPE html>\n<html><head><style>
                body {
                    font-family: sans-serif;
                }
            </style></head><body>''', file=html_file)
            print('<p>Unsolved:</p>', file=html_file)
            puzzle_unsolved.visualize_solution_svg(html_file)
            print('<hr><p>Solved:</p>', file=html_file)
            puzzle_solved.visualize_solution_svg(html_file)
            print('</body></html>', file=html_file)
            html_file.close()

    else:
        mode = config.mode
        print(f'Unknown mode "{mode}"!', file=sys.stderr)
        return 2

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


if __name__ == "__main__":
    sys.exit(main())
