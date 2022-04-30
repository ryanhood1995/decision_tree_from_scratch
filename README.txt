The required libraries to run everything include:
    - pandas
    - math
    - copy
    - numpy
    - sklearn

There are two python files of interest: decision_tree_from_scratch.py and random_forest.py.  decision_tree_from_scratch.py will
output one accuracy value every time the program is run.  The algorithm and dataset is chosen at the beginning of program
execution from the command line.  Random_forest.py will output every accuracy score each time the program is run, so there
are no choices to be made by the user.


-- Before running decision_tree_from_scratch.py, you need to input the location of the datasets.  This is done through the code
(not the command line), and the location in the code where it is set is line 615.


Note: The performance of decision_tree_from_scratch.py for Reduced Error Pruning for d5000 datasets is poor.  Although I believe the
implementation to be correct, I have adjusted the code so that only a few prunes occur before stopping.  The technically correct
implementation using a while loop has been commented out and replaced with a for loop.  This only needs to be changed if a technically
correct implementation is desired.  The relevant code is lines 516 - 556.


-- Before running random_forest.py, you also need to set the location of the data similarly to the above situation.  The location
in this file is line 14.
