"""
this class represents a cell stored in the data frame with its trace
"""
import copy
import random
from configuration import config

# two special symbols used in the language
HOLE = "_?_"
UNKNOWN = "_UNK_"

class TableCell(object):
    def __init__(self, value, exp):
        self.value = value
        self.exp = exp

    def to_dict(self):
        return {
            "value": self.value,
            "exp": self.exp.to_dict()
        }

    def matches(self, other, cmp_val=False, check_value=False):
        # firstly, check this argument is a subset of other's argument
        # we assume that if argument is not None then operator should not be None
        if cmp_val:
            if other.value == HOLE or self.value == HOLE:
                return True
            if other.value == self.value:
                return True
            return False
        else:
            if self.exp is not None and other.exp is not None and not semantically_equiv(other.exp, self.exp):
                return False
            if check_value and self.value is not None and self.value != other.value:
                return False
            return True

    def get_value(self):
        return self.value

    def get_exp(self):
        return copy.copy(self.exp)

    def to_stmt(self):
        return f"<{self.value},{self.exp}>"

    def get_flat_args(self):
        res = set()
        if isinstance(self.exp, list):
            for e in self.exp:
                if isinstance(e, ExpNode) or isinstance(e, ArgOr):
                    res.update(e.to_flat_list())
                else:
                    res.add(e)
        else:
            res.update(self.exp.to_flat_list())
        return res

    def get_flat_ops(self):
        res = set()
        if isinstance(self.exp, list):
            for e in self.exp:
                if isinstance(e, ExpNode):
                    res |= e.get_flat_ops()
        elif isinstance(self.exp, ExpNode):
            res |= self.exp.get_flat_ops()
        return res

    def randomize(self):
        if isinstance(self.exp, list):
            rand_exp = []
            for e in self.exp:
                if isinstance(e, ExpNode):
                    rand_exp += [e.randomize()]
                else:
                    rand_exp += [e]
        else:
            rand_exp = self.exp.randomize()
        return TableCell(self.value, rand_exp)

    def compress_sum(self):
        if isinstance(self.exp, list):
            compressed_exp = []
            for e in self.exp:
                if isinstance(e, ExpNode):
                    compressed_exp += [e.compress_sum()]
                else:
                    compressed_exp += [e]
        else:
            compressed_exp = self.exp.compress_sum()
        return TableCell(self.value, compressed_exp)

    def count_ptrs(self, count_or):
        count = 0
        if isinstance(self.exp, list):
            for e in self.exp:
                if isinstance(e, ExpNode):
                    count += e.count_ptrs(count_or)
                elif count_or and isinstance(e, ArgOr):
                    count += e.count_ptrs(count_or)
                else:
                    count += 1
        else:
            count = self.exp.count_ptrs(count_or)
        return count


def semantically_equiv(exp1, exp2, containment=True):
    # exp2 is our target expression and exp1 is the output expression
    # looser check if the target cell contain some unknown parts
    if exp1 == HOLE:
        return True
    if isinstance(exp1, ExpNode) and isinstance(exp2, ExpNode):
        if exp1.op != HOLE and exp2.op != HOLE and exp1.op != exp2.op:
            sum_ops = ["cumsum", "sum", "lambda x, y: x + y"]
            if not (exp1.op in sum_ops and exp2.op in sum_ops):  # add ambiguity to operators
                # print(f"failed {exp1.op} vs {exp2.op}!")
                return False
        if HOLE == exp1.children:
            return True
        return semantically_equiv(exp1.get_children(), exp2.get_children(), containment)
    elif isinstance(exp1, list) and isinstance(exp2, list):
        used = []
        if containment or UNKNOWN in exp2:  # check for trace containment
            for c2 in exp2:
                if c2 == UNKNOWN:  # skip the indicator
                     continue
                exist = False
                for i in range(len(exp1)):
                    if semantically_equiv(exp1[i], c2, containment) and i not in used:
                        exist = True
                        used.append(i)
                        break
                if not exist:
                    return False
            return True
        else:  # the trace in user example is exact
            if len(exp1) != len(exp2):
                return False
            for c2 in exp2:
                exist = False
                for i in range(len(exp1)):
                    if semantically_equiv(exp1[i], c2, containment) and i not in used:
                        exist = True
                        used.append(i)
                if not exist:
                    return False
            return len(used) == len(exp1)  # everything in exp1 is been used
    elif isinstance(exp1, list) and isinstance(exp2, ExpNode):
        return semantically_equiv(exp1, [exp2], containment)
    elif isinstance(exp1, ExpNode) and isinstance(exp2, list):
        return semantically_equiv([exp1], exp2, containment)
    else:
        if exp1 == HOLE or exp2 == HOLE:
            return True
        return exp1 == exp2


class ExpNode(object):
    def __init__(self, op, children):
        self.op = op
        self.children = children  # children can be a set of ExpNode or CellCoordinates

    def __hash__(self):
        return hash(str(self.to_dict()))

    def __repr__(self):
        return str((self.op, self.children))

    def __eq__(self, other):
        def exact_equiv(exp1, exp2):
            if isinstance(exp1, ExpNode) and isinstance(exp2, ExpNode):
                return (exp1.op == exp2.op and len(exp1.children) == len(exp2.children)
                        and all([exact_equiv(exp1.children[i], exp2.children[i])
                                 for i in range(len(exp1.children))]))
            else:
                # then it's a leaf note, both exp are coordinates
                return exp1 == exp2
        if isinstance(other, ArgOr):
            for e in other.arguments:
                if self.__eq__(e):
                    return True
        if not isinstance(other, ExpNode):
            return False
        return exact_equiv(self, other)

    def to_dict(self):
        return {
            "op": self.op,
            "children": [c.to_dict() if isinstance(c, ExpNode) else c for c in self.children]
        }

    def get_op(self):
        return self.op

    def get_children(self):
        return copy.copy(self.children)

    def to_flat_list(self):
        # decompose to get a list of coordinates
        def add_leaves(children, out):
            for e in children:
                if isinstance(e, ExpNode):
                    add_leaves(e.children, out)
                elif isinstance(e, ArgOr):
                    out.update(e.to_flat_list())
                else:
                    out.add(e)
        res = self.children
        output = set()
        add_leaves(res, output)
        return output

    def get_flat_ops(self):
        res = set()
        if isinstance(self.op, ArgOr):
            res.update(self.op.to_flat_list())
        elif isinstance(self.op, list):
            res.update(self.op)
        else:
            res.add(self.op)
        for e in self.children:
            if isinstance(e, ExpNode):
                res |= e.get_flat_ops()
        # print(res)
        return res

    def randomize(self):
        new_op = self.op
        new_exp = []
        children_count = 0
        for e in self.children:
            if isinstance(e, ExpNode):
                new_exp += [e.randomize()]
            elif children_count >= 4:
                if UNKNOWN not in new_exp:
                    new_exp += [UNKNOWN]
                continue
            else:
                new_exp += [e]
                children_count += 1

        return ExpNode(new_op, new_exp)

    def compress_sum(self, compress=False, manual=False):
        compressed_children = []
        for e in self.children:
            if isinstance(e, ExpNode):
                sum_ops = ArgOr(["cumsum", "sum", "lambda x, y: x + y"])
                # only compress when the operators are determined
                # if not (isinstance(self.op, str) and isinstance(e.get_op(), str)):
                #     print(self)
                #     print(e)
                #     print()
                if self.op == sum_ops and e.get_op() == sum_ops:
                    if isinstance(self.op, str) and isinstance(e.get_op(), str):
                        compressed_children += e.compress_sum(compress=True)
                    else:
                        compressed_children += e.compress_sum(compress=True)
                        compressed_exp = e.compress_sum()
                        compressed_children.append(compressed_exp)
                else:
                    compressed_exp = e.compress_sum()
                    compressed_children.append(compressed_exp)
            else:
                compressed_children += [e]
        if compress:
            return compressed_children
        else:
            return ExpNode(self.op, compressed_children)

    def count_ptrs(self, count_or):
        count = 0
        for e in self.children:
            if isinstance(e, ExpNode):
                count += e.count_ptrs(count_or)
            elif count_or and isinstance(e, ArgOr):
                count += e.count_ptrs(count_or)
            elif e != UNKNOWN:
                count += 1

        return count


def dict_to_exp(source):
    new_children = []
    for k in source["children"]:
        if isinstance(k, dict):
            new_children.append(dict_to_exp(k))
        else:
            new_children.append(k)
    return ExpNode(source["op"], new_children)


class ArgOr:
    """this class represent some arguments that are alternatives to each other
        used for comparing traces"""

    def __init__(self, arguments):
        self.arguments = arguments  # a list of (note, coordinate_x, coordinate_y)

    def __hash__(self):
        return hash(str(self.arguments))

    def __eq__(self, other):
        if not isinstance(other, ArgOr):
            return other in self.arguments
        return [1 for i in self.arguments for j in other.arguments if i == j] != []

    def __repr__(self):
        return "ArgOr" + str(self.arguments)

    def is_subset(self, other):
        if not isinstance(other, ArgOr):
            return other in self.arguments
        return all([i in other.arguments for i in self.arguments])

    def contains(self, exp):
        for val in self.arguments:
            if semantically_equiv(val, exp):
                return True
        return False

    def to_stmt(self):
        return "ArgOr" + str(self.arguments)

    def to_flat_list(self):
        res = set()
        for e in self.arguments:
            if isinstance(e, ExpNode) or isinstance(e, ArgOr):
                res.update(e.to_flat_list())
            else:
                res.add(e)
        return res

    def count_ptrs(self, count_or):
        count = 0
        for e in self.arguments:
            if isinstance(e, ExpNode):
                count += e.count_ptrs(count_or)
            elif count_or and isinstance(e, ArgOr):
                count += e.count_ptrs(count_or)
            else:
                count += 1

        return count

