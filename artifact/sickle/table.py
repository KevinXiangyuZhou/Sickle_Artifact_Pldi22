# 2020/7/16

import json
import pandas as pd
from table_cell import *
from table_cell_structureless import *
import time
import numpy as np

# two special symbols used in the language
HOLE = "_?_"
UNKNOWN = "_UNK_"

"""
a table is represented by set of cells
"""


class AnnotatedTable:
    """
    construct the table with the given dataset.
    """

    def __init__(self, source, from_source=False):
        """load from a dictionary represented annotated table"""
        self.df = []  # stored as a two-level array with columns to be the inner level
        if not from_source:
            self.load_from_dict(source)
        else:
            self.load_from_source(source)

    def equals(self, other):
        if self.get_col_num() != other.get_col_num():
            return False
        if self.get_row_num() != other.get_row_num():
            return False
        for cid in range(self.get_col_num()):
            for rid in range(self.get_row_num()):
                if self.get_cell(cid, rid).get_value() != other.get_cell(cid, rid).get_value():
                    return False
        return True

    # load from two-level array of cells
    def load_from_source(self, source):
        self.df = source.copy()

    # dict source should be a two-level array [[{val, exp}], [{val, exp}]]
    def load_from_dict(self, source):
        for col_id in range(len(source)):
            self.df.append([])
            for cell_id in range(len(source[col_id])):
                cell = source[col_id][cell_id]
                self.df[col_id].append(TableCell(cell["value"], cell["exp"]))

    def add_column(self, new_column):
        self.df.append(new_column.copy())

    def add_row(self, new_row):
        if self.is_empty():
            # initialize then append when the current space is empty
            for i in range(len(new_row)):
                self.df.append([new_row[i]])
        else:
            if len(new_row) != self.get_col_num():
                print("[error] new row with inconsistent column number added")
            for i in range(len(self.df)):
                self.df[i].append(new_row[i])

    def get_cell(self, x, y):
        return copy.copy(self.df[x][y])

    def get_column(self, col_index):
        return self.df[col_index].copy()

    def get_row(self, row_index):
        # get a list of cells in the same row index
        rlt = []
        for i in range(len(self.df)):
            rlt.append(self.df[i][row_index])
        return rlt

    def get_col_num(self):
        if self.is_empty():
            return 0
        return len(self.df)

    def get_row_num(self):
        if self.is_empty():
            return 0
        return len(self.df[0])

    def is_empty(self):
        return self.df == [] or [e for e in self.df if e != []] == []

    def round(self):
        for cid in range(len(self.df)):
            for rid in range(len(self.df[cid])):
                cell = self.df[cid][rid]
                if isinstance(cell.get_value(), float):
                    self.df[cid][rid] = TableCell(np.round(cell.get_value(), 2), cell.get_exp())

    def extract_values(self):
        """ convert annotated table to a dataframe 
            (drop trace information and keep only values and store it as a dataframe)
        """
        data = {}
        for i in range(self.get_col_num()):
            attribute = "COL_" + str(i)  # COL_0, COL_1, ...
            if attribute not in data.keys():
                data[attribute] = []
            for j in range(self.get_row_num()):
                cell = self.df[i][j]
                data[attribute].append(cell.get_value())
        return pd.DataFrame.from_dict(data)

    def extract_traces(self):
        """ version that only keeps trace info"""
        data = {}
        for i in range(self.get_col_num()):
            attribute = "COL_" + str(i)  # COL_0, COL_1, ...
            if attribute not in data.keys():
                data[attribute] = []
            for j in range(self.get_row_num()):
                cell = self.df[i][j]
                data[attribute].append(cell.get_exp())
        return pd.DataFrame.from_dict(data)

    def to_dataframe(self):
        """ convert annotated table to a dataframe
            cells in the dataframe are represented as <{self.value}, {self.operator}, {self.argument}>
        """
        data = {}
        for i in range(self.get_col_num()):
            attribute = "COL_" + str(i)  # COL_0, COL_1, ...
            if attribute not in data.keys():
                data[attribute] = []
            for j in range(self.get_row_num()):
                cell = self.df[i][j]
                data[attribute].append(cell.to_stmt())
        return pd.DataFrame.from_dict(data)

    def to_dict(self):
        """convert to a dictionary for easy import export"""
        dicts = []
        for i in range(len(self.df)):
            for j in range(len(self.df[i])):
                cell = self.df[i][j]
                dicts.append(cell.to_dict())
        return dicts

    def to_plain_dict(self):
        """for print use"""
        dicts = []
        for j in range(len(self.df[0])):
            d = {}
            for i in range(len(self.df)):
                cell = self.df[i][j]
                temp = cell.to_dict()
                for k in temp:
                    d[k] = temp[k]["value"]
            dicts.append(d)
        return dicts

    def select_region(self, x_range, y_range):
        if x_range[0] < 0 or x_range[1] > self.get_col_num()\
           or y_range[0] < 0 or y_range[1] > self.get_row_num():
            return None
        x1, x2 = x_range[0], x_range[1]
        y1, y2 = y_range[0], y_range[1]
        if x1 == x2:
            x2 += 1
        if y1 == y2:
            y2 += 1
        selected = []
        for x in range(x1, x2):
            selected.append([])
            for y in range(y1, y2):
                selected[-1].append(self.get_cell(x, y))
        return AnnotatedTable(selected, from_source=True)

    def select_rows(self, rids):
        new_source = []
        for i in range(self.get_col_num()):
            new_col = []
            for j in rids:
                cell = self.df[i][j]
                new_col.append(cell)
            new_source.append(new_col)
        return AnnotatedTable(new_source, from_source=True)

    def randomize(self):
        rand_table = []
        for x in range(self.get_col_num()):
            rand_table.append([])
            for y in range(self.get_row_num()):
                rand_table[-1].append(self.get_cell(x,y).randomize())
        return AnnotatedTable(rand_table, from_source=True)

    def compress_sum(self):
        new_source = []
        for i in range(self.get_col_num()):
            new_col = []
            for j in range(self.get_row_num()):
                cell = self.df[i][j]
                new_col.append(cell.compress_sum())
            new_source.append(new_col)
        return AnnotatedTable(new_source, from_source=True)

    def count_ptrs(self, count_or=False):
        count = 0
        for i in range(self.get_col_num()):
            for j in range(self.get_row_num()):
                cell = self.df[i][j]
                count += cell.count_ptrs(count_or)
        return count



"""
from format of eg.
[{"cust_country": "UK", "grade": 2, "outstanding_amt": 3600},
{"cust_country": "USA", "grade": 2, "outstanding_amt": 5400}]
"""

""" ----- annotated table util functions ----- """
def select_columns(att, cols):
    cell_list = []
    for col in cols:
        temp = []
        for i in range(att.get_row_num()):
            cell = att.get_cell(col, i)
            temp.append({"value": cell.get_value(), "exp": cell.get_exp()})
        cell_list.append(temp)
    return AnnotatedTable(cell_list)


def get_flat_table(t):
    table = copy.deepcopy(t)
    rownum = table.get_row_num()
    colnum = table.get_col_num()
    new_source = []
    for cid in range(colnum):
        new_source.append([])
        for rid in range(rownum):
            curr_cell = table.get_cell(cid, rid)
            new_cell = SimpleCell(curr_cell.get_value(), curr_cell.get_flat_args(), curr_cell.get_flat_ops())
            new_source[cid].append(new_cell)
    return AnnotatedTable(new_source, from_source=True)


# return the list of exact trace on the last level
def extract_last_level(table):
    traces = []
    rownum = table.get_row_num()
    colnum = table.get_col_num()
    for cid in range(colnum):
        for rid in range(rownum):
            curr_cell = table.get_cell(cid, rid)
            if isinstance(curr_cell.get_exp(), ExpNode):
                # only add complete expression node for check
                traces += [c for c in curr_cell.get_exp().get_children() if isinstance(c, ExpNode)]
    return traces


def check_concrete_list(exp_list, traces):
    for exp2 in traces:
        exist = False
        for exp1 in exp_list:
            # print(f"{exp1} vs {exp2}")
            if semantically_equiv(exp1, exp2, containment=False):
                exist = True
                break
        if not exist:
            return False
    return True


def check_concrete(table, traces):
    rownum = table.get_row_num()
    colnum = table.get_col_num()
    for exp2 in traces:
        exist = False
        stop = False
        for cid in range(colnum):
            for rid in range(rownum):
                exp1 = table.get_cell(cid, rid).get_exp()
                # print(f"{exp1} vs {exp2}")
                if semantically_equiv(exp1, exp2, containment=False):
                    exist = True
                    stop = True
                    break
            if stop:
                break
        if not exist:
            return False
    return True


"""checker function for pruning annotated outputs
actual: the table generated by synthesizer
target: the annotated table generated based on user inputs
"""
def checker_function(actual, target, check_relations=True, print_result=False, print_time=False, cmp_val=False, check_value=False):
    if actual == "Pass":
        return "PASS"
    if actual is None or target is None:
        return None

    # find mappings from cells in target to actual for each cell
    # store for each cell with format: {(x, y): [(0,0), (1,2)]}
    start_t = time.time()
    mapping = find_mapping(target, actual, print_result, cmp_val, check_value)
    mapping_t = time.time()
    if print_time:
        print(f"search for mapping cost: {mapping_t - start_t}")
    # print(mapping)
    if mapping is None:
        # print("##########PRUNED BY MAPPING################")
        return None

    s1_mapping = mapping
    # use column and row to remove infeasible mappings
    if check_relations:
        target_df = target.extract_values()
        prune_by_row_column(mapping, target_df, print_result)

        # search for possible mappings
        # stop whenever we find on feasible mapping, and return the mapping
        relation_t = time.time()
        if print_time:
            print(f"prune by relation cost: {relation_t - mapping_t}")
        if not check_mappings(mapping):
            return None

    # extract mappings
    # use dfs to search for the valid mapping
    keys = [*mapping.keys()]
    final_mapping = extract_mappings(mapping, keys)
    if final_mapping is None and print_result:
        print("########PRUNED BY RELATIONS####################################################")
        print("failed mapping: ")
        print(mapping)
    findresult_t = time.time()
    if print_time:
        print(f"find result cost: {findresult_t - relation_t}")
    return final_mapping


"""search for valid mapping for each cell in target table"""
def find_mapping(target, actual, print_result, cmp_val, check_value):
    mapping = {}
    for cid in range(target.get_col_num()):
        for rid in range(target.get_row_num()):
            mapping[(cid, rid)] = search_values(actual, target.get_cell(cid, rid), cmp_val, check_value)
            # let it fail here
            if not check_mappings(mapping):
                if print_result:
                    print("##########PRUNED BY MAPPING################")
                    print("failed to find mapping for:")
                    print([k for k in mapping.keys() if mapping[k] == []])
                return None
    return mapping


# the given table is the actual table we generated
# cell is a cell from target table
def search_values(table, cell, cmp_val, check_value):
    rlt = []
    for cid in range(table.get_col_num()):
        for rid in range(table.get_row_num()):
            if cell.matches(table.get_cell(cid, rid), cmp_val=cmp_val, check_value=check_value):
                rlt.append((cid, rid))
    return rlt


"""prune mapping by relative column and row positions"""
def prune_by_row_column(mapping, target_df, print_result=False):
    if print_result:
        # print(target_df)
        print("step1 mapping")
        print(mapping)
    # pruning each column
    # if two cells are in the same row in the output,
    # then their source (matching cells in actual table) must be in the same row in actual
    for x in range(len(target_df.columns)):
        l = [mapping[(x, y)] for y in range(len(target_df))]
        smallest = find_smallest_array(l)
        # get list of x value in the smallest mapping
        x_list = [a for (a, b) in smallest]
        for y in range(len(target_df)):
            mapping[(x, y)] = [t for t in mapping[(x, y)] if t[0] in x_list]
    if print_result:
        print("prune by col")
        print(mapping)
    # pruning each row
    # same pruning law as column
    for y in target_df.index.tolist():
        l = [mapping[(x, y)] for x in range(len(target_df.columns))]
        smallest = find_smallest_array(l)
        y_list = [b for (a, b) in smallest]
        for x in range(len(target_df.columns)):
            mapping[(x, y)] = [t for t in mapping[(x, y)] if t[1] in y_list]
    if print_result:
        print("prune by row")
        print(mapping)


"""use dfs to search for the valid mapping"""
def extract_mappings(mappings, keys):
    rlt = {}
    closed = []
    open = []
    search_mappings(rlt, 0, mappings, keys, closed, open)
    if check_valid_mapping(rlt, keys):
        return rlt
    else:
        return None


# helper function for search mappings
def search_mappings(rlt, index, mappings, keys, closed, open):
    if index == len(keys):
        # we are done searching for mappings
        return True
    else:
        # iterate over maps
        coord = keys[index]
        # fail if there is no choice for current key
        open += mappings[coord]
        if not [m for m in mappings[coord] if m not in closed]:
            # we cannot find any possible mappings for keys[index]
            if not [v for v in closed if v not in open]:
                # there is no possibility for this coord to find a valid mapping
                return True
            return False

        # iterate over mappings[coord]
        for i in range(len(mappings[coord])):
            if coord not in rlt:
                rlt[coord] = []
            if mappings[coord][i] not in closed:
                closed.append(mappings[coord][i])
                rlt[coord].append(mappings[coord][i])
                if check_valid_mapping(rlt, keys):
                    # if we found a valid mapping
                    return True
                found = search_mappings(rlt, index + 1, mappings, keys, closed, open)
                if found:
                    return True
                rlt[coord].pop()
                closed.pop()
        return False


def check_valid_mapping(mapping, keys):
    if list(mapping.keys()) != keys:
        return False
    return check_mappings(mapping)


# check if there is no empty list in mappings
def check_mappings(mapping):
    if len(mapping) == 0:
        return False
    for k in mapping:
        if len(mapping[k]) == 0:
            return False
    return True


def find_smallest_array(list):
    # choose the firstly found array if tie
    if len(list) == 0:
        return []
    rlt = list[0]
    for array in list:
        if len(array) < len(rlt):
            rlt = array
    return rlt
