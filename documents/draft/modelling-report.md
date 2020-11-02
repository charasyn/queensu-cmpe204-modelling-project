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
To extend it to first-order logic, we could define the following functions:
- `T(p)`: Tree function. It is T if the grid at position `p` contains a tree, and F otherwise.
- `X(p)`: Tent function. It is T if the grid at position `p` contains a tent, and F otherwise.
- `P(p1,p2)`: Partial adjacency function. It is T if positions `p1` and `p2` are adjacent horizontally, or vertically, but not diagonally. It is F otherwise.
- `F(p1,p2)`: Full adjacency function. It is T if positions `p1` and `p2` are adjacent horizontally, vertically, or diagonally. It is F otherwise.
- `A(pt,px)`: Association function. It is T if the tent at `px` is associated with the tree at `pt`, and F otherwise.
- `E(p1,p2)`: Equality function. It is T if `p1` is the same position as `p2`, and F otherwise.

This would allow us to represent our constraints more effectively. We could do as follows: (note that two-letter variables are used as shorthand for subscript, meaning pt -> p_{t})
- A tent cannot occupy the same spot as a tree
  - `∀p. ¬(T(p) ∧ X(p))`
- A tent associated with a tree must be partially adjacent to it
  - `∀pt. (T(pt) → ∀px. (X(px) ∧ A(pt,px) → P(pt,px)))`
- A tree must be paired with exactly one tent
  - `∀pt. (T(pt) → ∃px1. (X(px1) ∧ A(pt,px1) ∧ (∀px2. (A(pt,px2) → E(px1,px2)))))`
  - Summarized: for all trees, there must exist a tent that is associated with it, and for all other positions, an association between the tree and said position implies that the two positions are equal
