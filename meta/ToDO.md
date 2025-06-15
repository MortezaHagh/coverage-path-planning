- statistics and path analyzer class

- add selectino of start and end points to the MILP
- optim seq based on start exit points not centers that may not even be in the cell area
- GA for sequence opt
- optimizing entry and exit points
- Create Sample maps with few rooms (no need to detect main sweeping direction)

- more use of numpy arrays

- another seq optimization: multiple sequences, each till there is no adj (so only one pass from each node)

- contour method: voronoi decomposition params as optimization params

- two classes, merged cells (for planning) and normal cells (with less functionalities and data)
- better boustrophedon



# loggings

import logging

- Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)

- Log with different levels
logging.debug("Debug message")
logging.info("Info message")
logging.warning("Warning message")
logging.error("Error message")
logging.critical("Critical message")