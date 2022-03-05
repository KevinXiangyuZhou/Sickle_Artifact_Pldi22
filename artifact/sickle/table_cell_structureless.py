# two special symbols used in the language
HOLE = "_?_"
UNKNOWN = "_UNK_"
"""
this class represents a cell stored in the data frame with its trace
"""
class SimpleCell:
    def __init__(self, value, argument, operator):
        self.value = value
        self.argument = argument  # a list of (table_id, coordinate_x, coordinate_y)
        self.operator = operator

    def matches(self, other):
        # looser check if the target cell contain some uninstantiated parts
        if other.operator == HOLE and other.argument == HOLE:
            return True
        elif other.operator == HOLE:
            return self.is_sublist(self.argument, other.argument)
        elif other.argument == HOLE:
            return self.is_sublist(self.operator, other.operator)
        # firstly, check this argument is a subset of other's argument
        # we assume that if argument is not None then operator should not be None
        if self.argument is not None and self.operator is not None:
            return self.argument.issubset(other.argument) and self.operator.issubset(other.operator)
            # return self.is_subset(other) and self.is_subset_op(other)
        return False

    def is_subset(self, other):
        return [e for e in self.argument if e not in other.argument] == []

    def is_subset_op(self, other):
        return [e for e in self.operator if e not in other.operator] == []

    def to_dict(self):
        return {
            "argument": self.argument,
            "operator": self.operator,
        }

    def get_value(self):
        return self.value

    def get_argument(self):
        if self.argument == HOLE:
            return HOLE
        return self.argument.copy()

    def get_operator(self):
        if self.operator == HOLE:
            return HOLE
        return self.operator.copy()

    def to_stmt(self):
        return f"<{self.operator}, {self.argument}>"
