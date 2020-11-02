# Project Summary
TODO: ???

# Propositions
TODO: All

# Constraints
TODO: All

# Model Exploration
TODO: Alex

# First-Order Extension
Our Tents and Trees problem extends well to first-order logic.
To extend it to first-order logic, we could define the following functions over the set of positions in the grid:
- `T(p)`: Tree function. It is T if the grid at position `p` contains a tree, and F otherwise.
- `X(p)`: Tent function. It is T if the grid at position `p` contains a tent, and F otherwise.
- `P(p1,p2)`: Partial adjacency function. This function is commutative. It is T if positions `p1` and `p2` are adjacent horizontally, or vertically, but not diagonally. It is F otherwise.
- `F(p1,p2)`: Full adjacency function. This function is commutative. It is T if positions `p1` and `p2` are adjacent horizontally, vertically, or diagonally. It is F otherwise.
- `A(p1,p2)`: Association function. This function is commutative. It is T if the tent/tree at `p1` is associated (paired) with the tree/tent at `p2`, and F otherwise.
- `E(p1,p2)`: Equality function. It is T if `p1` is the same position as `p2`, and F otherwise.

This would allow us to represent our constraints more effectively. Instead of having one constraint per cell in the grid, we could do as follows: (note that two-letter variables are used as shorthand for subscript, meaning pt -> p_{t})
- A tent cannot occupy the same spot as a tree
  - `∀p. ¬(T(p) ∧ X(p))`
- A tent associated with a tree must be partially adjacent to it
  - `∀pt. (T(pt) → ∀px. (X(px) ∧ A(pt,px) → P(pt,px)))`
- A tree must be associated (paired) with a tent
  - `∀pt. (T(pt) → ∃px1. (X(px1) ∧ A(pt,px1)))`
- A position cannot be associated with multiple other positions
  - `∀p1. (∀p2. (A(p1,p2) → (∀p3. (A(p1,p3) → E(p2,p3)))))`
  - If any p1 is associated with any p2, then any p3 which is also associated with p1 must be equal to p2
