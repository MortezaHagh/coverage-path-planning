- statistics and path analyzer class

- add selectino of start and end points to the MILP
- optim seq based on start exit points not centers that may not even be in the cell area
- GA for sequence opt
- optimizing entry and exit points
- Create Sample maps with few rooms (no need to detect main sweeping direction)

- more use of numpy arrays

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