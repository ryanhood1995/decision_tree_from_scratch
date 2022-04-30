# ================================================================================================================================
# Author: Ryan C Hood
#
# Description: This is the main python file for homework #1.  The goal is to design a Decision Tree from scratch capable of
# using two different impurity measures and performing two different types of post pruning techniques.  The resulting code
# should work for different size data with binary feature values and binary class variable.
#
# ================================================================================================================================

# I pull in 3 basic libraries for the task, pandas to read the csv files, math for the log function (needed for entropy impurity measure), and copy
# to perform a deep copy.
import pandas as pd
import math
import copy


# ================================================================================================================================
# The first section of code is concerned with defining classes that will be useful in the following code.
# ================================================================================================================================
class Decision_Node:
    """A Decision Node is an interior node of the tree.  It holds a reference to a question, the rows of the
    training data that reach that point, a reference to a true node, a reference to a false node, a node depth,
    and a node number
    """

    def __init__(self, question, rows, true_branch, false_branch, depth, number):
        self.question = question
        self.rows = rows
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.number = number


class Leaf:
    """A Leaf node classifies new data rows.  To define a leaf, you need to input the rows that successfully
    make it to that leaf.  The predictions attribute holds a dictionary of the class results at the leaf,
    along with the number of time the leaf sees that particular class result.  Also a depth of the leaf node is stored.
    """

    def __init__(self, rows, depth):
        self.predictions = class_counts(rows)
        self.depth = depth


class Question:
    """A question is tied to a Decision Node and is used to partition the data set.  It is defined by a column and a
    value for that column.  There is a method, match(), which compares a given row's value to the value for the corresponding
    stored in the Question.
    """

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        """Compare the feature value in an example to the
        feature value in this question."""
        val = example[self.column]

        # If the particular example's value for the column equals the value of this column, then return true.
        return val == self.value

    # A method for printing a question used later.
    def __repr__(self):
        condition = "=="
        return "Is %s == %s?" % (
            header[self.column], str(self.value))



# ================================================================================================================================
# The second section of code contains some general methods used by all of the different algorithms.
# ================================================================================================================================



def get_header_list(training_data):
    """
    This method takes the training data, picks off the first row.  Then the headers list increases for every attribute
    present in the data.  This is used for printing the tree.
    """

    row_length = len(training_data[0]) - 1 # -1 because one of the entries is the class.
    headers = []
    for entry in range(0, row_length):
        headers.append("X" + str(entry))
    return headers

def print_tree(node, spacing=""):
    """
    This method prints the tree as neatly as possible.
    """
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print (spacing + "Predict", node.predictions)
        return

    # Print the question at this node
    print (spacing + str(node.question))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")

"""
def print_tree(node):
    # Recursive function to print a tree
    if isinstance(node, Leaf):
        print("Leaf at depth: ", node.depth)
        return

    # If reach here, we must be at a Decision Node.
    print("Decision Node #", node.number, " at depth ", node.depth)
    print_tree(node.true_branch)
    print_tree(node.false_branch)
    return
"""

def unique_vals(rows, col):
    """
    This method finds the unique values in a column given some rows of the dataset.
    For the most part, in this application, the number of unique_vals will be =2, since
    our features are binary.
    """
    return set([row[col] for row in rows])


def class_counts(rows):
    """
    Counts the number of each type of example in a dataset.  This is used by leaf nodes
    to determine what to classify examples as.
    """
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] = counts[label] + 1
    return counts


def partition(rows, question):
    """Partitions a dataset.
    For each row in the dataset, check if it matches the question.
    """
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def entropy(rows):
    """
    This method takes rows (a list of lists) and computes the entropy of those rows.
    The entropy is a measure of the impurity of the class labels of a set of rows.
    """
    counts = class_counts(rows)
    entropy = 0
    for value in counts:
        prob_of_value = counts[value] / float(len(rows))
        entropy = entropy + (-1)*prob_of_value*(math.log(prob_of_value)/math.log(2))
    return entropy


def info_gain_entropy_impurity(left_rows, right_rows, current_uncertainty):
    """
    Information Gain using entropy.
    The impurity of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left_rows)) / (len(left_rows) + len(right_rows))
    return current_uncertainty - p * entropy(left_rows) - (1 - p) * entropy(right_rows)

def info_gain_variance_impurity(left_rows, right_rows, current_uncertainty):
    """
    Information Gain using variance.  Similar to above.
    """
    p = float(len(left_rows)) / (len(left_rows) + len(right_rows))
    return current_uncertainty - p * variance(left_rows) - (1 - p) * variance(right_rows)


def variance(rows):
    """
    This method calculates the variance of a set of rows.  The variance is an impurity measure.
    """
    # We need to get the number of training examples with class 0.
    num_class_zero = 0
    for row in rows:
        if row[-1] == 0:
            num_class_zero = num_class_zero + 1
    # Now we need the number of training examples with class 1.
    num_class_one = 0
    for row in rows:
        if row[-1] == 1:
            num_class_one = num_class_one + 1
    return (num_class_one)*(num_class_zero)/(len(rows))**2



def find_best_split(rows, impurity_heuristic):
    """
    Finds the best question to ask at the current step.
    """
    best_gain = 0
    best_question = None

    if impurity_heuristic == 0:
        current_impurity = entropy(rows)
    else:
        current_impurity = variance(rows)


    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature

        for val in [0,1]:  # for each value
            # a question is generated here.
            question = Question(col, val)

            # The dataset is split over the question.
            true_rows, false_rows = partition(rows, question)

            # Skip this split if it doesn't do anything.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            if impurity_heuristic == 0:
                gain = info_gain_entropy_impurity(true_rows, false_rows, current_impurity)
            else:
                gain = info_gain_variance_impurity(true_rows, false_rows, current_impurity)

            # Replace the best_gain if there is one better.
            if gain >= best_gain:
                best_gain, best_question = gain, question

    # By the end of the double for loop, we will have the best question and the best gain.  Return both.
    return best_gain, best_question

def classify(row, node):
    """
    This method takes a row and a node and recursively uses the tree to classify the row.
    """
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        # predictions is a dict containing the class_counts at a particular leaf.
        return node.predictions

    # Decide whether to follow the true-branch or the false-branch.
    # Compare the feature / value stored in the node,
    # to the example we're considering.
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def pick_best_choice(classification_dict):
    """
    This method takes a classification_dict (returns by the classify method) and returns
    a list of the form [winner (0 or 1), accuracy_of_winner(between 0 and 1)]
    """
    if len(classification_dict) == 2:
        zero_prediction = classification_dict[0]
        one_prediction = classification_dict[1]
        total_counts = zero_prediction + one_prediction
        if zero_prediction >= one_prediction:
            return [0, zero_prediction / total_counts]
        else:
            return [1, one_prediction / total_counts]
    elif 0 in classification_dict.keys():
        return [0,1]
    else:
        return [1,1]

def accuracy(predictions, values):
    """
    This method takes a list of predictions (which are 0 or 1) and a list of values (also 0 or 1), and compares
    them.  If they match for a given index, then the prediction was correct.  If they don't, then it wasn't correct.
    """
    num_correct = 0
    for index in range(0, len(predictions)-1):
        if predictions[index] == values[index]:
            num_correct = num_correct + 1
    return num_correct / len(predictions)

def get_values(test_data_set):
    """
    This method takes a test data set, and returns a list of the class labels.  This is to be used in the accuracy method above.
    """
    values = []
    for row in test_data_set:
        values.append(row[-1])
    return values

def get_predictions(tree, test_data_set):
    """
    This method takes a tree and the test data set, and returns a list containing the predictions
    by the tree on the test data set.
    """
    predictions = []
    for row in test_data_set:
        predictions.append(pick_best_choice(classify(row, tree))[0])
    return predictions




def find_max_number_in_tree(node):
    """
    Theoretically, a tree's Decision nodes should be numbered 0 to n-1 where n is the number
    of Decision nodes in the tree.  This method is designed to look at a tree's Decision Nodes
    and return the largest value among them.
    """
    # Root node is not a Leaf.
    root_node_number = node.number
    if isinstance(node.true_branch, Leaf) and isinstance(node.false_branch, Leaf):
        return root_node_number
    if isinstance(node.true_branch, Leaf) and isinstance(node.false_branch, Decision_Node):
        return max(root_node_number, find_max_number_in_tree(node.false_branch))
    if isinstance(node.false_branch, Leaf) and isinstance(node.true_branch, Decision_Node):
        return max(root_node_number, find_max_number_in_tree(node.true_branch))

    # If we reach here then the subtree_root_node has two decision nodes as children.
    return max(find_max_number_in_tree(node.true_branch), find_max_number_in_tree(node.false_branch))


def assign_numbers_to_nodes(node, number):
    """
    This method takes an already built tree and assigns each Decision Node within that tree a number.  The numbers
    start with 0 and increase sequentially up to n-1 where n is the number of Decision Nodes.
    """

    if isinstance(node, Leaf):
        return

    # The current node must be a decision node, so assign it the given number.
    node.number = number

    if isinstance(node.true_branch, Leaf) and isinstance(node.false_branch, Leaf):
        return
    if isinstance(node.true_branch, Leaf) and isinstance(node.false_branch, Decision_Node):
        assign_numbers_to_nodes(node.false_branch, number + 1)
        return
    if isinstance(node.true_branch, Decision_Node) and isinstance(node.false_branch, Leaf):
        assign_numbers_to_nodes(node.true_branch, number + 1)
        return

    # If we reach this point, then both children are decision nodes.
    # Start by assigning numbers to the left subtree.
    assign_numbers_to_nodes(node.true_branch, number + 1)

    # Now find the largest number assigned in left subtree.
    largest_number = find_max_number_in_tree(node.true_branch)

    # Now assign numbers to the right subtree starting from the largest number in the left subtree.
    assign_numbers_to_nodes(node.false_branch, max(largest_number, number) + 1)

    return

# ================================================================================================================================
# The below section of code is focused on the methods used when doing depth-based pruning.
# ================================================================================================================================

def prune_tree_by_depth(node, prune_depth):
    """
    This will be a recursive function which traverses a tree and when it reaches a decision node
    at the prune_depth, then it will make that decision node a leaf node.
    """

    # Base case.
    if node.depth == prune_depth-1 and isinstance(node, Decision_Node):
        if isinstance(node.true_branch, Decision_Node):
            new_leaf = Leaf(node.true_branch.rows, node.true_branch.depth)
            node.true_branch = new_leaf
        if isinstance(node.false_branch, Decision_Node):
            new_leaf = Leaf(node.false_branch.rows, node.false_branch.depth)
            node.false_branch = new_leaf
    if isinstance(node, Leaf):
        return

    # If we reach this point, then the node is not a leaf and it is not a decision node at the distance threshold.
    # So
    prune_tree_by_depth(node.true_branch, prune_depth)
    prune_tree_by_depth(node.false_branch, prune_depth)
    return


def print_depths(node):
    """
    This method prints the depths of the nodes in the tree.  Mainly used to check if correct.
    """
    print(node.depth)
    if isinstance(node, Leaf):
        return
    if isinstance(node.true_branch, Decision_Node) and isinstance(node.false_branch, Leaf):
        print_depths(node.true_branch)
        return
    if isinstance(node.false_branch, Decision_Node) and isinstance(node.true_branch, Leaf):
        print_depths(node.false_branch)
        return

    print_depths(node.true_branch)
    print_depths(node.false_branch)



def find_best_depth_pruned_tree(rows, validation_rows, impurity_heuristic, current_depth, current_number):
    """
    This method uses a list of depth_parameters, to prune the original tree to have max depth equal to each of
    the depth_parameters.  The accuracy on the validation data is calculated, and the tree with the best accuracy
    is returned.
    """

    depth_parameters = [5,10,15,20,50,100]

    # Build initial tree.
    initial_tree = build_tree(rows, validation_rows, 0, impurity_heuristic, current_depth, current_number)

    # initialize best accuracy with the un-pruned tree.
    best_accuracy = accuracy(get_predictions(initial_tree, validation_rows), get_values(validation_rows))

    # If final best_depth == -1, then we will know pruning should not be done (and we return initial tree)
    depth_for_best_accuracy = -1

    # initialize best tree
    best_tree = initial_tree

    # Now for each possible depth parameter, trim the tree to that length, and then test the accuracy.
    for depth in depth_parameters:
        # Call function which prunes and returns the resulting tree.

        pruned_tree = copy.deepcopy(initial_tree)
        prune_tree_by_depth(pruned_tree, depth)

        pruned_tree_accuracy = accuracy(get_predictions(pruned_tree, validation_rows), get_values(validation_rows))
        if pruned_tree_accuracy > best_accuracy:
            best_accuracy = pruned_tree_accuracy
            depth_for_best_accuracy = depth
            best_tree = pruned_tree
    print("Depth for best accuracy is: ", depth_for_best_accuracy)
    return best_tree


# ================================================================================================================================
# The below section of code is focused on the methods used when doing Reduced Error Pruning
# ================================================================================================================================


def prune_particular_node(node, node_number):
    """
    This method takes a tree and an integer corresponding to the number attribute of a Decision Node in the given tree.
    The method prunes the Decision Node out, replacing it with a Leaf Node.
    """
    # The purpose of this method is to take a tree, and prune a particular node number from that tree.
    # The method returns the resulting tree.

    # We go through the tree recursively.  If the true child of the node in question is the node_number, then the true branch for the
    # current node is set to be a leaf node.  Similarly for the false child.
    if isinstance(node, Leaf):
        return
    if isinstance(node.true_branch, Leaf) and isinstance(node.false_branch, Leaf):
        return
    if isinstance(node.true_branch, Decision_Node) and node.true_branch.number == node_number:
        # Make node.true_branch a Leaf
        new_leaf = Leaf(node.true_branch.rows, node.depth + 1)
        node.true_branch = new_leaf
        return
    if isinstance(node.false_branch, Decision_Node) and node.false_branch.number == node_number:
        # Make node.false_branch a Leaf
        new_leaf = Leaf(node.false_branch.rows, node.depth + 1)
        node.false_branch = new_leaf
        return
    if isinstance(node.false_branch, Leaf) and isinstance(node.true_branch, Decision_Node):
        prune_particular_node(node.true_branch, node_number)
    if isinstance(node.true_branch, Leaf) and isinstance(node.false_branch, Decision_Node):
        prune_particular_node(node.false_branch, node_number)

    # If we reach here, then both children are decision nodes, but neither contain the node_number, so we just go down left path, then right path.
    prune_particular_node(node.true_branch, node_number)
    prune_particular_node(node.false_branch, node_number)
    return


def find_number_of_internal_nodes_in_tree(node):
    """
    This method takes a tree and returns the number of Decision Nodes (internal nodes) in the tree.
    """
    if isinstance(node, Leaf):
        return 0
    if isinstance(node, Decision_Node):
        return 1 + find_number_of_internal_nodes_in_tree(node.true_branch) + find_number_of_internal_nodes_in_tree(node.false_branch)



def reduced_error_pruned_tree(initial_tree, validation_rows):
    """
    This method takes a tree, labels all decision nodes, then for each decision node, a pruned tree is created and the accuracy of this pruned tree on the validation set
    is calculated.  If any of the pruned trees accuracy is higher than the original tree, then the maximum tree is kept.  This process continues until none of the pruned trees
    is better than the original.  The original tree is then returned.
    """
    # Note: The correct algorithm uses a while loop (commented out).  For peformance reasons, I stopped after the third iteration.
    i = 0
    for i in range(0, 3):
        i= i + 1

    #while True:
        # First we get the accuracy of the original tree and use it to initialize the best_accuracy_initial.  best_accuracy will change throughout.
        best_accuracy_initial = accuracy(get_predictions(initial_tree, validation_rows), get_values(validation_rows))
        best_accuracy = best_accuracy_initial

        best_tree = initial_tree

        # Now we get the total number of internal nodes.
        # number_of_internal_nodes_old = find_max_number_in_tree(initial_tree) # internal node counting starts at 0.
        number_of_internal_nodes = find_number_of_internal_nodes_in_tree(initial_tree)
        print("Number of internal nodes: ", number_of_internal_nodes)

        for number in range(0, number_of_internal_nodes):
            # Create the tree with the Decision Node removed.
            temp_tree = copy.deepcopy(initial_tree)
            assign_numbers_to_nodes(temp_tree, 0)
            prune_particular_node(temp_tree, number)
            print("Node number being checked: ", number)

            # Test the tree against the validation data.
            temp_tree_accuracy = accuracy(get_predictions(temp_tree, validation_rows), get_values(validation_rows))
            print("This temp tree accuracy: ", temp_tree_accuracy)
            print("The best tree accuracy: ", best_accuracy)
            # In Below if time, implement ability to choose smaller tree if accuracy is the same.
            if temp_tree_accuracy >= best_accuracy:
                best_accuracy = temp_tree_accuracy
                best_tree = temp_tree
                print("Best tree being updated.")

        print("Another iteration of while loop has been completed!")
        # Now best_tree should point to the tree that has the highest accuracy (best_accuracy)
        # If the best_tree at this point is the initial_tree, then pruning is bad and we can simply return the initial tree.
        # If the best_tree is not the initial_tree, the pruning is still helpful and we go back through the while loop.
        if best_accuracy == best_accuracy_initial:
            return best_tree
        else:
            initial_tree = best_tree

    # If using a while loop, the below line of code is not needed.
    return best_tree

# ================================================================================================================================
# The below code contains a single method, build_tree(), which is the key driving method of the whole program.
# ================================================================================================================================

def build_tree(rows, validation_rows, pruning_choice, impurity_heuristic, current_depth, current_number):
    """
    This method is half-recursive: Depending on a set of parameters the user chooses in the main function, a few options can be taken.
    If no pruning is desired, a tree is built with the given impurity heuristic.  This is the part that is recursive, as the tree keeps
    growing until the best question is unable to produce any gain.  If pruning is desired, different actions are taken.
    """
    if pruning_choice == 0: # Then we do not want to do any pruning.
        if impurity_heuristic == 0: # Then we want to use entropy heuristic.
            gain, question = find_best_split(rows, impurity_heuristic)

            if gain == 0:
                return Leaf(rows, current_depth)

            true_rows, false_rows = partition(rows, question)

            true_branch = build_tree(true_rows, validation_rows, 0, 0, current_depth + 1, current_number + 1)
            false_branch = build_tree(false_rows, validation_rows, 0, 0, current_depth + 1, current_number + 1)

            return Decision_Node(question, rows, true_branch, false_branch, current_depth, current_number)
        else: # Then we want to do variance heuristic.
            gain, question = find_best_split(rows, impurity_heuristic)

            if gain == 0:
                return Leaf(rows, current_depth)

            true_rows, false_rows = partition(rows, question)

            true_branch = build_tree(true_rows, validation_rows, 0, 1, current_depth + 1, current_number)
            false_branch = build_tree(false_rows, validation_rows, 0, 1, current_depth + 1, current_number)

            return Decision_Node(question, rows, true_branch, false_branch, current_depth, current_number + 1)

    elif pruning_choice == 1: # Then we want to do some reduced error pruning.
        # First we generate the tree with no pruning
        initial_tree = build_tree(rows, validation_rows, 0, impurity_heuristic, 0, 0)

        # Now we do the pruning.
        pruned_tree = reduced_error_pruned_tree(initial_tree, validation_rows)
        return pruned_tree
    # Insert code here.
    else: # Then we want to do some depth-based pruning.
        # First, we need to use the validation set to find the best depth to prune.
        best_tree = find_best_depth_pruned_tree(rows, validation_rows, impurity_heuristic, current_depth, current_number)
        # Return that tree.
        return best_tree


# ================================================================================================================================
# The below code is the "main" function.  The correct data is loaded, user desired parameters are set, and the accuracy of the resulting set-up is calculated.
# ================================================================================================================================

# First, we provide the directory where the data is stored.
# ******* NEW USERS MAY NEED TO CHANGE THIS VALUE ********
data_directory = "C:\\Users\\User\\Dropbox\\CS 6375 - Machine Learning\\homework\\CS6375 HW1\\hw1_data\\all_data\\"

if __name__ == '__main__':
    data_set_choice = 0
    while data_set_choice not in [*range(1,16,1)]:
        print("First select a number representing the correct data-set you wish to process from the following options.")
        print("1: c300 d100")
        print("2: c300 d1000")
        print("3: c300 d5000")
        print("4: c500 d100")
        print("5: c500 d1000")
        print("6: c500 d5000")
        print("7: c1000 d100")
        print("8: c1000 d1000")
        print("9: c1000 d5000")
        print("10: c1500 d100")
        print("11: c1500 d1000")
        print("12: c1500 d5000")
        print("13: c1800 d100")
        print("14: c1800 d1000")
        print("15: c1800 d5000")
        data_set_choice = int(input("Choice: "))
        if data_set_choice == 1:
            train = pd.read_csv(data_directory + "train_c300_d100.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c300_d100.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c300_d100.csv", header=None).values.tolist()
        elif data_set_choice == 2:
            train = pd.read_csv(data_directory + "train_c300_d1000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c300_d1000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c300_d1000.csv", header=None).values.tolist()
        elif data_set_choice == 3:
            train = pd.read_csv(data_directory + "train_c300_d5000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c300_d5000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c300_d5000.csv", header=None).values.tolist()
        elif data_set_choice == 4:
            train = pd.read_csv(data_directory + "train_c500_d100.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c500_d100.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c500_d100.csv", header=None).values.tolist()
        elif data_set_choice == 5:
            train = pd.read_csv(data_directory + "train_c500_d1000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c500_d1000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c500_d1000.csv", header=None).values.tolist()
        elif data_set_choice == 6:
            train = pd.read_csv(data_directory + "train_c500_d5000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c500_d5000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c500_d5000.csv", header=None).values.tolist()
        elif data_set_choice == 7:
            train = pd.read_csv(data_directory + "train_c1000_d100.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c1000_d100.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c1000_d100.csv", header=None).values.tolist()
        elif data_set_choice == 8:
            train = pd.read_csv(data_directory + "train_c1000_d1000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c1000_d1000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c1000_d1000.csv", header=None).values.tolist()
        elif data_set_choice == 9:
            train = pd.read_csv(data_directory + "train_c1000_d5000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c1000_d5000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c1000_d5000.csv", header=None).values.tolist()
        elif data_set_choice == 10:
            train = pd.read_csv(data_directory + "train_c1500_d100.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c1500_d100.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c1500_d100.csv", header=None).values.tolist()
        elif data_set_choice == 11:
            train = pd.read_csv(data_directory + "train_c1500_d1000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c1500_d1000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c1500_d1000.csv", header=None).values.tolist()
        elif data_set_choice == 12:
            train = pd.read_csv(data_directory + "train_c1500_d5000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c1500_d5000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c1500_d5000.csv", header=None).values.tolist()
        elif data_set_choice == 13:
            train = pd.read_csv(data_directory + "train_c1800_d100.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c1800_d100.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c1800_d100.csv", header=None).values.tolist()
        elif data_set_choice == 14:
            train = pd.read_csv(data_directory + "train_c1800_d1000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c1800_d1000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c1800_d1000.csv", header=None).values.tolist()
        elif data_set_choice == 15:
            train = pd.read_csv(data_directory + "train_c1800_d5000.csv", header=None).values.tolist()
            test = pd.read_csv(data_directory + "test_c1800_d5000.csv", header=None).values.tolist()
            validate = pd.read_csv(data_directory + "valid_c1800_d5000.csv", header=None).values.tolist()
        else:
            print("You did not select a valid choice.  You must input an integer between 1 and 15.  Try again.")


    algorithm_choice = 0
    while algorithm_choice not in [*range(1,7,1)]:
        print("Now select a number representing the algorithm you wish to run.")
        print("1: Naive Learner w/ Entropy")
        print("2: Naive Learner w/ Variance")
        print("3: Reduced Error Pruning w/ Entropy")
        print("4: Reduced Error Pruning w/ Variance")
        print("5: Depth-Based Pruning w/ Entropy")
        print("6: Depth-Based Pruning w/ Variance")
        algorithm_choice = int(input("Choice: "))
        if (algorithm_choice == 1):
            pruning_choice = 0
            impurity_heuristic = 0
        elif (algorithm_choice == 2):
            pruning_choice = 0
            impurity_heuristic = 1
        elif (algorithm_choice == 3):
            pruning_choice = 1
            impurity_heuristic = 0
        elif (algorithm_choice == 4):
            pruning_choice = 1
            impurity_heuristic = 1
        elif (algorithm_choice == 5):
            pruning_choice = 2
            impurity_heuristic = 0
        elif (algorithm_choice == 6):
            pruning_choice = 2
            impurity_heuristic = 1
        else:
            print("You did not select a valid choice.  You must input an integer between 1 and 6.  Try again.")

    # We start building the tree.  The training and validation data is passed along, along with the pruning and impurity choices.
    # Lastly, the parameters depth and numbers are initialized to 0, since we start building the tree at the root node.
    tree = build_tree(train, validate, pruning_choice, impurity_heuristic, 0, 0)


    predictions = get_predictions(tree, test)
    values = get_values(test)

    accuracy = accuracy(predictions, values)
    print("Accuracy of the chosen algorithm with the chosen data set: ", accuracy)
