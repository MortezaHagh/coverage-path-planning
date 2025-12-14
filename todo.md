# CovPP

## Paper

1. pipeline

<!-- ----------------------------------------------------------------------- -->

## ToDo

### Sequence

- add selection of start and end points to the MILP and other sequence creation
- enable to oprimize selection of the start-finish from end points
- GA for sequence opt
- GA for selecting the critical points
- another seq optimization: multiple sequences, each till there is no adj (so only one pass from each node)

### General Implementaiton

- more use of numpy arrays

### contour method

- ...

### boustraphedon

- non-geometric method (cv)

<!-- ----------------------------------------------------------------------- -->

## ToDo Now

- segmentation method
- merging condition (small, thin not based on robot size (...))
- cell_planning as plugin (methods for merging and processing cells)
- better or more stale main direction detection
- in energy min method: add find last point in the obstacle and adjust l based on that
- new boustraphedon
- add to inputs: positions of docking stations (later check in which cell they fell in to)


## CMD

devcontainer exec --workspace-folder ~/Documents/00-CovPP/covpp -- /bin/zsh