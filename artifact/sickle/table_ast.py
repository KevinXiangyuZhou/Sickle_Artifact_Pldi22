import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import copy
import itertools
from table import *
from tabulate import tabulate
from table_cell import *
from table_cell_structureless import *


# two special symbols used in the language
HOLE = "_?_"
UNKNOWN = "_UNK_"
# = 4

# global used dataN storage
n_program_search = []
n_program_search_analysis = []
run_time = []
run_time_analysis = []

class Node(ABC):
	def __init__(self):
		super(AbstractExpression, self).__init__()

	@abstractmethod
	def eval(self, inputs):
		"""the inputs are dataframes,
			it returns a pandas dataframe representation"""
		pass

	@abstractmethod
	def to_dict(self):
		pass

	@staticmethod
	def load_from_dict(ast):
		"""given a dictionary represented AST, load it in to a program form"""
		constructors = {
			"select": Select,
			"filter": Filter,
			"group_sum": GroupSummary,
			"group_mutate": GroupMutate,
			"mutate_arithmetic": Mutate_Arithmetic,
			"join": Join
		}
		if ast["op"] == "table_ref":
			return Table(ast["children"][0]["value"])
		else:
			if ast["op"] == "join":
				node = constructors[ast["op"]](
					Node.load_from_dict(ast["children"][0]),
					Node.load_from_dict(ast["children"][1]),
					*[arg["value"] for arg in ast["children"][2:]])
			else:
				node = constructors[ast["op"]](
					Node.load_from_dict(ast["children"][0]),
					*[arg["value"] for arg in ast["children"][1:]])
			return node

	def to_stmt_dict(self):
		"""translate the expression into a  """
		def _recursive_translate(ast, used_vars):
			if ast["op"] == "table_ref":
				# create a variable to capture the return variable
				stmt_dict = copy.copy(ast)
				var = get_temp_var(used_vars)
				stmt_dict["return_as"] = var
				return [stmt_dict], used_vars + [var]
			else:
				stmt_dict = copy.copy(ast)
				# iterate over all possible subtrees
				sub_tree_stmts = []
				for i, arg in enumerate(ast["children"]):
					# check if the argument is an ast
					if isinstance(arg, (dict,)) and arg["type"] == "node":
						stmts, used_vars = _recursive_translate(ast["children"][i], used_vars)
						sub_tree_stmts += stmts
						# the subtree is replaced by a reference to the variable
						retvar = stmts[-1]["return_as"]
						stmt_dict["children"][i] = {"value": retvar, "type": "variable"}

				# use a temp variable to wrap the current statement, and add it to the coolection
				var = get_temp_var(used_vars)
				stmt_dict["return_as"] = var
				return sub_tree_stmts + [stmt_dict], used_vars + [var]
		stmts, _ = _recursive_translate(self.to_dict(), [])
		return stmts

	def infer_colnum(self, inputs):
		""" infer the number of columns in the over-approximated table"""
		pass

	def infer_rownum(self, inputs):
		""" infer the number of rows in the over-approximated table"""
		pass

	def is_abstract(self):
		"""Check if the subtree is abstract (contains any holes)"""
		def contains_hole(node):
			for i, arg in enumerate(node["children"]):
				if arg["type"] == "node":
					if contains_hole(arg):
						return True
				elif arg["value"] == HOLE:
					# we find a variable to infer
					return True
			return False
		return contains_hole(self.to_dict())

	def is_fully_abstract(self):
		"""Check if the subtree is fully abstract (contains only holes)"""
		def contains_val(node):
			for i, arg in enumerate(node["children"]):
				if arg["type"] == "node":
					if contains_val(arg):
						return True
				elif arg["value"] != HOLE:
					# we find a variable to infer
					return True
			return False

		return not contains_val(self.to_dict())

	def program_list(self):
		def add_op(node, ops):
			for i, arg in enumerate(node["children"]):
				if isinstance(arg, (dict,)) and arg["type"] == "node":
					ops.append(arg["children"][i])
					add_op(arg, ops)
		ops = []
		add_op(self.to_dict(), ops)
		return ops

	def stmt_string(self):
		"""generate a string from stmts, for the purpose of pretty printing"""
		def val_to_str(x):
			if x["value"] == HOLE:
				return "?"
			if x["type"] == "bv_filter":
				return "".join(["|" if k else "." for k in x["value"][:min(5, len(x["value"]))]])
			else:
				return str(x["value"])

		stmts = self.to_stmt_dict()
		result = []
		for s in stmts:
			lhs = s['return_as']
			f = s['op']
			arg_str = ', '.join([val_to_str(x) for x in s["children"]])
			result.append(f"{lhs} <- {f}({arg_str})")
		return "; ".join(result)


class Table(Node):
	def __init__(self, data_id):
		self.data_id = data_id

	def infer_domain(self, arg_id, inputs, config):
		assert False, "Table has no args to infer domain."

	def infer_output_info(self, inputs):
		"""infer output schema """
		inp = inputs[self.data_id]
		if isinstance(inp, (list,)):
			df = pd.DataFrame.from_dict(inp)
		else:
			df = inp
		schema = extract_table_schema(df)
		return schema

	def eval(self, inputs):
		inp = inputs[self.data_id]
		source = [[] for i in range(len(inp[0]))]
		rid = 0
		for row in inp:
			for col in row.keys():
				cid = list(row.keys()).index(col)
				source[cid].append(TableCell(row[col], [f"{self.data_id}_{get_alphabet(cid)}{rid}"]))
			rid += 1
		return AnnotatedTable(source, from_source=True)

	def to_dict(self):
		return {
			"type": "node",
			"op": "table_ref",
			"children": [
				value_to_dict(self.data_id, "table_id")
			]
		}

	def get_id(self):
		return self.data_id

	def infer_trace(self, inputs, config):
		# return self.eval(inputs)
		inp = inputs[self.data_id]
		source = [[] for i in range(len(inp[0]))]
		rid = 0
		for row in inp:
			for col in row.keys():
				cid = list(row.keys()).index(col)
				source[cid].append(TableCell(row[col], [f"{self.data_id}_{get_alphabet(cid)}{rid}"]))
			rid += 1
		return AnnotatedTable(source, from_source=True)

	def infer_value(self, inputs, config):
		return self.eval(inputs)

	def infer_colnum(self, inputs):
		return len(inputs[self.data_id][0])

	def infer_rownum(self, inputs):
		return len(inputs[self.data_id])

class Join(Node):
	def __init__(self, q1, q2, predicate, is_left_outer):
		self.q1 = q1
		self.q2 = q2
		self.predicate = predicate
		self.is_left_outer = is_left_outer

	def infer_domain(self, arg_id, inputs, config):
		if arg_id == 2:
			return config["join_predicates"]
		elif arg_id == 3:
			return [False, True]
		else:
			assert False, "[Join] No args to infer domain for id > 3."

	def infer_output_info(self, inputs):
		schema_1 = self.q1.infer_output_info(inputs)
		schema_2 = self.q2.infer_output_info(inputs)
		return schema_1 + schema_2

	def get_id(self):
		ids = []
		if isinstance(self.q1.get_id(), list):
			ids += self.q1.get_id()
		else:
			ids += [self.q1.get_id()]
		if isinstance(self.q2.get_id(), list):
			ids += self.q2.get_id()
		else:
			ids += [self.q2.get_id()]

		return ids

	def infer_left_colnum(self, inputs):
		return self.q1.infer_colnum(inputs)

	def eval(self, inputs):
		# make a copy of table for argument reference
		table1 = self.q1.eval(inputs)
		table2 = self.q2.eval(inputs)

		# evaluate the dataframes from inputs
		eval_predicate = eval(self.predicate)
		join_keys = list(eval_predicate[1])
		if isinstance(self.q1.get_id(), list):  # the last level is a join operator
			if eval_predicate[0][0] not in self.q1.get_id() or self.q2.get_id() != eval_predicate[0][1]:
				return AnnotatedTable([], from_source=True)
			join_keys[0] = join_keys[0] + self.q1.infer_left_colnum(inputs)
			if join_keys[0] >= table1.get_col_num() or join_keys[1] >= table2.get_col_num():
				return AnnotatedTable([], from_source=True)
		else:
			if self.q1.get_id() != eval_predicate[0][0] or self.q2.get_id() != eval_predicate[0][1]:
				# return empty table to the next level
				return AnnotatedTable([], from_source=True)
			if join_keys[0] >= table1.get_col_num() or join_keys[1] >= table2.get_col_num():
				return AnnotatedTable([], from_source=True)

		source = []

		# build annotated table of joined tables
		for rid1 in range(table1.get_row_num()):
			# left outer join will include the row in left table when there is no match in the right table
			exist_matches = False
			for rid2 in range(table2.get_row_num()):
				# add the a combination of table1[rid] & table2[rid] if the values of keys are the same
				# else exclude the row from final output
				if table1.get_cell(join_keys[0], rid1).get_value() != table2.get_cell(join_keys[1], rid2).get_value():
					continue
				exist_matches = True
				for cid in range(table1.get_col_num() + table2.get_col_num()):
					if cid >= len(source):
						source.append([])
					if cid < table1.get_col_num():
						arg = table1.get_cell(cid, rid1).get_exp()
						val = table1.get_cell(cid, rid1).get_value()
						if not isinstance(arg, list):
							arg = [arg]
					else:
						arg = table2.get_cell(cid - table1.get_col_num(), rid2).get_exp()
						val = table2.get_cell(cid - table1.get_col_num(), rid2).get_value()
						if not isinstance(arg, list):
							arg = [arg]
					source[cid].append(TableCell(val, arg))   # not storing predicates
			if self.is_left_outer and not exist_matches:
				for cid in range(table1.get_col_num() + table2.get_col_num()):
					if cid >= len(source):
						source.append([])
					if cid < table1.get_col_num():
						arg = table1.get_cell(cid, rid1).get_exp()
						val = table1.get_cell(cid, rid1).get_value()
						if not isinstance(arg, list):
							arg = [arg]
					else:
						arg = []
						val = 0.0
					source[cid].append(TableCell(val, arg))
		# it is okay to pass empty joined result to the next level
		return AnnotatedTable(source, from_source=True)

	def to_dict(self):
		return {
			"type": "node",
			"op": "join",
			"children": [
				self.q1.to_dict(),
				self.q2.to_dict(),
				value_to_dict(self.predicate, "func"),
				value_to_dict(self.is_left_outer, "bool")
			]
		}

	def infer_value(self, inputs, config):
		# a cross product of computed intermediate of the two joined programs
		if self.predicate != HOLE and self.is_left_outer != HOLE:
			return self.eval(inputs)
		table1 = self.q1.infer_trace(inputs, config)
		table2 = self.q2.infer_trace(inputs, config)

		# two empty table we will build and merge together
		empty_table1 = AnnotatedTable([])
		empty_table2 = AnnotatedTable([])
		# build the table by first replicating table2 by row(table1) times
		# (add the result as new rows; also added one by one due to implementation decision)
		for i in range(table1.get_row_num()):
			# repeat all rows
			for j in range(table2.get_row_num()):
				empty_table2.add_row(table2.get_row(j))

		# then replicate each row of table1 by row(table2) times
		for i in range(table1.get_row_num()):
			# each row is repeated by row(table2) times
			for j in range(table2.get_row_num()):
				empty_table1.add_row(table1.get_row(i))
		for i in range(empty_table2.get_col_num()):
			empty_table1.add_column(empty_table2.get_column(i))
		return empty_table1

	def infer_trace(self, inputs, config):
		# a cross product of computed intermediate of the two joined programs
		if self.predicate != HOLE and self.is_left_outer != HOLE:
			return self.eval(inputs)
		table1 = self.q1.infer_trace(inputs, config)
		table2 = self.q2.infer_trace(inputs, config)

		# two empty tables we will build and merge together
		empty_table1 = AnnotatedTable([])
		empty_table2 = AnnotatedTable([])
		# build the table by first replicating table2 by row(table1) times
		# (add the result as new rows; also added one by one due to implementation decision)
		for i in range(table1.get_row_num()):
			# repeat all rows
			for j in range(table2.get_row_num()):
				empty_table2.add_row(table2.get_row(j))

		# then replicate each row of table1 by row(table2) times
		for i in range(table1.get_row_num()):
			# each row is repeated by row(table2) times
			for j in range(table2.get_row_num()):
				empty_table1.add_row(table1.get_row(i))
		for i in range(empty_table2.get_col_num()):
			empty_table1.add_column(empty_table2.get_column(i))
		return empty_table1


	def infer_colnum(self, inputs):
		return self.q1.infer_colnum(inputs) + self.q2.infer_colnum(inputs)

	def infer_rownum(self, inputs):
		if self.predicate != HOLE:
			val = self.eval(inputs)

			if val.is_empty():
				print(0)
				return 0
			return val.get_row_num()
		return self.q1.infer_rownum(inputs) * self.q2.infer_rownum(inputs)

class Select(Node):
	def __init__(self, q, cols):
		self.q = q
		self.cols = cols

	def infer_domain(self, arg_id, inputs, config):
		if arg_id == 1:
			input_schema = self.q.infer_output_info(inputs)
			col_num = len(input_schema)
			col_list_candidates = []
			for size in range(1, col_num + 1):
				col_list_candidates += list(itertools.permutations(list(range(col_num)), size))  # want permutation
			return col_list_candidates
		else:
			assert False, "[Select] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		schema = self.q.infer_output_info(inputs)
		return [s for i, s in enumerate(schema) if i in self.cols]

	def eval(self, inputs):
		res = self.q.eval(inputs)
		df = res.extract_values()  # of type pandas dataframe
		# check if df has input cols
		for col in self.cols:
			if col >= len(df.columns):
				print("error in select")
				assert False

		df_prev = df.copy()
		select_cols = [df.columns[i] for i in self.cols]
		df = df[select_cols]
		arguments = {}
		for rid in range(len(df)):
			for colname in df.columns:
				if rid not in arguments:
					arguments[rid] = {}
				arguments[rid][colname] = [(get_col_index_by_name(df_prev, colname), rid)]
		return df_to_annotated_table_index_colname(df, None, arguments, res)

	def to_dict(self):
		return {
			"type": "node",
			"op": "select",
			"children": [self.q.to_dict(), value_to_dict(self.cols, "col_index_list")]
		}

class Filter(Node):
	def __init__(self, q, col_index, op, const):
		self.q = q
		self.col_index = col_index
		self.op = op
		self.const = const

	def infer_domain(self, arg_id, inputs, config):
		if arg_id == 1:
			col_num = len(self.q.infer_output_info(inputs))
			return list(range(col_num))
		elif arg_id == 2:
			return config["filer_op"]
		elif arg_id == 3:
			return config["constants"]
		else:
			assert False, "[Filter] No args to infer domain for id > 3."

	def infer_output_info(self, inputs):
		return self.q.infer_output_info(inputs)

	def eval(self, inputs):
		df = self.q.eval(inputs)
		if df is None:
			return None
		if not isinstance(df, pd.DataFrame):
			df = df.extract_values()
		col = df.columns[self.col_index]
		if self.op == "==":
			ret = df[df[col] == self.const]
		elif self.op == "!=":
			ret = df[df[col] != self.const]
		elif self.op == ">":
			ret = df[df[col] > self.const]
		elif self.op == "<":
			ret = df[df[col] < self.const]
		else:
			sys.exit(-1)
		arguments = generate_direct_arguments(df)
		return df_to_annotated_table_index_colname(ret, "filter", arguments)

	def to_dict(self):
		return {
			"type": "node",
			"op": "filter",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.col_index, "col_index"),
				value_to_dict(self.op, "filter_op"),
				value_to_dict(self.const, "constants")
			]}

class GroupSummary(Node):
	def __init__(self, q, group_cols, aggr_func, aggr_col):
		self.q = q
		self.group_cols = group_cols
		self.aggr_col = aggr_col
		self.aggr_func = aggr_func

	def infer_domain(self, arg_id, inputs, config):
		schema = self.q.infer_output_info(inputs)
		if arg_id == 1:
			# approximation: only get fields with more than one values
			# for the purpose of avoiding empty fields
			try:
				df = self.q.eval(inputs)
				df = df.extract_values()
				if df.empty:
					return []
			except Exception as e:
				print(f"[eval error in infer_domain] {e}")
				return []
			# use this list to store primitive table keys,
			# use them to eliminate column combinations that contain no duplicates
			table_keys = []
			col_num = len(schema)
			col_list_candidates = []
			for size in range(1, col_num):
				for gb_keys in itertools.combinations(list(range(col_num)), size):
					if any([set(banned).issubset(set(gb_keys)) for banned in table_keys]):
						# current key group is subsumed by a table key, so all fields will be distinct
						continue
					# print(df)
					gb_cols = df[[df.columns[k] for k in gb_keys if k < len(df.columns)]]
					if not gb_cols.duplicated().any():
						# a key group is valid for aggregation
						#   if there exists at least a key appear more than once
						table_keys.append(gb_keys)
						continue
					col_list_candidates += [gb_keys]
			return col_list_candidates
		elif arg_id == 3:
			number_fields = [i for i, s in enumerate(schema) if s == "number"]
			if self.group_cols != HOLE:
				cols = [i for i in number_fields if i not in self.group_cols]
			else:
				cols = number_fields
			if self.aggr_func != HOLE and not isinstance(self.aggr_func, str):
				func_list = self.aggr_func
				cols = itertools.permutations(cols, len(func_list))
				cols = [list(e) for e in cols]
			return cols
		elif arg_id == 2:
			valid_cols = [i for i, s in enumerate(schema) if s == "number" and i not in self.group_cols]
			func_candidates = copy.copy(config["aggr_func"])
			valid_len = len(valid_cols)
			if "count" in config["aggr_func"] and len(valid_cols) < len(schema):
				valid_len += 1
			for size in range(2, valid_len + 1):
				func_candidates += itertools.combinations(config["aggr_func"], size)
			func_candidates = [list(e) if not isinstance(e, str) else e for e in func_candidates]
			return func_candidates
		else:
			assert False, "[Gather] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		input_schema = self.q.infer_output_info(inputs)
		output_schema = [s for i, s in enumerate(input_schema) if i in self.group_cols]
		if isinstance(self.aggr_col, list):
			output_schema += ["number" for col in self.aggr_col if input_schema[col] == "number"]
		else:
			output_schema += ["number"]
		return output_schema

	def eval(self, inputs):
		# make a copy of table for argument reference
		table = self.q.eval(inputs)
		if table.is_empty():
			return table
		df = table.extract_values()
		res = df.copy()
		group_keys = [res.columns[idx] for idx in self.group_cols]
		# case that aggregate on multiple columns
		if isinstance(self.aggr_func, str):
			target = res.columns[self.aggr_col]
		else:
			target = self.aggr_col  # not column name
		res = res.groupby(group_keys)

		# map argument for keys and groups
		arguments = {}  # {rid: {colname: argument}}
		gid = 0
		for (key, group) in res:
			arguments[gid] = {}
			for colname in group.columns:
				val_arg = []
				for row_index in group.to_dict()[colname]:
					# (value, cid, rid)
					if (isinstance(self.aggr_func, str) and colname == target) or \
							(isinstance(target, list) and get_col_index_by_name(group, colname) in target):  # handle agg_columns
						val_arg.append((get_col_index_by_name(group, colname),
										row_index))
					elif colname in [group.columns[i] for i in self.group_cols]:  # handle group columns
						# check if it is an expnode obj
						temp_exp = table.get_cell(get_col_index_by_name(group, colname),
												  row_index).get_exp()
						if isinstance(temp_exp, ExpNode):
							temp_exp = [temp_exp]
						elif len(temp_exp) == 1 and isinstance(temp_exp[0], ArgOr):
							temp_exp = temp_exp[0].to_flat_list()
						elif len(temp_exp) == 1 and isinstance(temp_exp[0], list):
							temp_exp = temp_exp[0]
						val_arg += temp_exp
				# map the group argument with the target col
				if colname in [group.columns[i] for i in self.group_cols]:
					arguments[gid][colname] = [ArgOr(val_arg)]
				else:
					arguments[gid][colname] = val_arg
			gid += 1

		if isinstance(self.aggr_func, str):
			res = res.agg({target: self.aggr_func}).fillna(0)
		else:
			res = res.agg({df.columns[col_index]: func for func, col_index in zip(self.aggr_func, self.aggr_col)}).fillna(0)
		res = res.reset_index()
		res = round_df(res)
		# change name of the target col name in arguments
		target_names = [df.columns[i] for i in self.aggr_col] if isinstance(self.aggr_col, list)\
			else df.columns[self.aggr_col]
		return df_to_annotated_table_index_colname(res, self.aggr_func, arguments, table, target_cols=target_names)

	def to_dict(self):
		return {
			"type": "node",
			"op": "group_sum",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.group_cols, "col_index_list"),
				value_to_dict(self.aggr_func, "aggr_func"),
				value_to_dict(self.aggr_col, "col_index")
			]}

	def infer_colnum(self, inputs):
		n = self.q.infer_colnum(inputs)
		if self.group_cols == HOLE:
			return n + 1
		elif self.aggr_func != HOLE:
			return len(self.group_cols) + len(self.aggr_func) if isinstance(self.aggr_func, list) else 1
		else:
			return len(self.group_cols) + 1

	def infer_rownum(self, inputs):
		if self.aggr_col != HOLE:
			val = self.eval(inputs)
			if val.is_empty():
				return 0
			return val.get_row_num()
		return self.q.infer_rownum(inputs)

	def infer_value(self, inputs, config):
		if self.group_cols != HOLE and self.aggr_func != HOLE and self.aggr_col != HOLE:
			# the program has all parameters
			return self.eval(inputs)
		table = self.q.infer_value(inputs, config)
		rownum = table.get_row_num()
		colnum = table.get_col_num()
		new_source = []
		df = table.extract_values()
		agg_allowed = len(config["aggr_func"])
		if self.group_cols == HOLE:  # we know nothing about parameters
			for cid in range(colnum + agg_allowed):
				col_cells = []
				if cid < colnum:  # group columns
					group_key = df.columns[cid]
					group_df = df.groupby(group_key)
					for (key, group) in group_df:
						index_list = group.index.tolist()
						for rid in index_list:
							new_cell = TableCell(table.get_cell(cid, rid).get_value(), HOLE)
							# add the new cells to the table
							col_cells.append(new_cell)
				else:
					for rid in range(rownum):
						new_cell = TableCell(HOLE, HOLE)
						col_cells.append(new_cell)
				new_source.append(col_cells)
			new_table = AnnotatedTable(new_source, from_source=True)
			new_table.round()
			return new_table

		group_keys = [df.columns[idx] for idx in self.group_cols]
		df = df.groupby(group_keys)
		if self.aggr_func == HOLE:
			agg_allowed = len(config["aggr_func"])
		elif isinstance(self.aggr_func, list):
			agg_allowed = len(self.aggr_func)
		else:
			agg_allowed = 1
		agg_cols = [colnum + i for i in range(agg_allowed)]
		new_cols = list(self.group_cols) + agg_cols
		start_row = 0
		for (key, group) in df:
			index_list = group.index.tolist()
			for cid in range(colnum + agg_allowed):  # group_cols + new col
				if cid not in self.group_cols and cid < colnum:
					continue
				if new_cols.index(cid) >= len(new_source):  # create column at the first group
					new_source.append([])
				if cid >= colnum:
					new_cell = TableCell(HOLE, HOLE)
				else:
					new_cell = TableCell(table.get_cell(cid, index_list[0]).get_value(), HOLE)
				new_source[new_cols.index(cid)].append(new_cell)
			start_row += len(group)
		new_table = AnnotatedTable(new_source, from_source=True)
		new_table.round()
		return new_table

	def infer_trace(self, inputs, config):
		if self.group_cols != HOLE and self.aggr_func != HOLE and self.aggr_col != HOLE:
			# the program has all parameters
			return self.eval(inputs)
		start_time = time.time()
		table = self.q.infer_trace(inputs, config)
		if table == "Pass":
			return "Pass"
		rownum = table.get_row_num()
		colnum = table.get_col_num()
		new_source = []
		# number of possible aggregated columns
		df = table.extract_values()
		agg_allowed = len(config["aggr_func"])
		if self.group_cols == HOLE:  # we know nothing about parameters
			for cid in range(colnum + agg_allowed):
				col_cells = []
				if cid < colnum:  # group columns
					group_key = df.columns[cid]
					group_df = df.groupby(group_key)
					for (key, group) in group_df:
						index_list = group.index.tolist()
						trace = [(cid, y) for y in index_list]
						args = []
						for c in trace:
							if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
								args += table.get_cell(c[0], c[1]).get_exp()
							else:
								args += [table.get_cell(c[0], c[1]).get_exp()]
						args = remove_duplicates(args)
						for rid in index_list:
							new_cell = TableCell(table.get_cell(cid, rid).get_value(), args)
							# add the new cells to the table
							col_cells.append(new_cell)
				else:
					trace = [(x, y) for x in range(colnum) for y in range(rownum)]
					args = []
					for c in trace:
						if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
							args += table.get_cell(c[0], c[1]).get_exp()
						else:
							args += [table.get_cell(c[0], c[1]).get_exp()]
					args = remove_duplicates(args)
					if cid >= colnum:
						func = ArgOr(config["aggr_func"])
						new_cell = TableCell(HOLE, ExpNode(func, args))
					else:
						new_cell = TableCell(HOLE, args)
					for rid in range(rownum):
						# add the new cells to the table
						col_cells.append(new_cell)
				new_source.append(col_cells)
			return AnnotatedTable(new_source, from_source=True)

		group_keys = [df.columns[idx] for idx in self.group_cols]
		df = df.groupby(group_keys)
		if self.aggr_func == HOLE:
			agg_allowed = len(config["aggr_func"])
		elif isinstance(self.aggr_func, list):
			agg_allowed = len(self.aggr_func)
		else:
			agg_allowed = 1
		agg_cols = [colnum + i for i in range(agg_allowed)]
		new_cols = list(self.group_cols) + agg_cols
		start_row = 0
		for (key, group) in df:
			index_list = group.index.tolist()
			for cid in range(colnum + agg_allowed):  # group_cols + new col
				if cid not in self.group_cols and cid < colnum:
					continue
				if new_cols.index(cid) >= len(new_source):  # create column at the first group
					new_source.append([])
				if cid >= colnum:
					# the new cell in new column can come from any cell
					# but it should not be placed in group cols
					trace = [(x, y) for x in range(colnum) for y in index_list if x not in self.group_cols]
				else:
					# this column is group column
					# its trace should be ArgOr of all cells in the column
					trace = [(cid, y) for y in index_list]
				args = []
				for c in trace:
					if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
						args += table.get_cell(c[0], c[1]).get_exp()
					else:
						args += [table.get_cell(c[0], c[1]).get_exp()]
				if cid >= colnum:
					if self.aggr_func == HOLE:
						func = ArgOr(config["aggr_func"])
					elif agg_allowed > 1:
						func = self.aggr_func[cid - colnum]
					else:
						func = self.aggr_func
					# print(set(args))
					new_cell = TableCell(HOLE, ExpNode(func, args))
				else:
					new_cell = TableCell(table.get_cell(cid, index_list[0]).get_value(), args)
				new_source[new_cols.index(cid)].append(new_cell)
			start_row += len(group)
		return AnnotatedTable(new_source, from_source=True)


class GroupMutate(Node):
	def __init__(self, q, group_cols, aggr_func, target_col, sort_by=None, asc=True):
		self.q = q
		self.group_cols = group_cols
		self.target_col = target_col
		self.aggr_func = aggr_func
		self.sort_by = sort_by
		self.asc = asc

	def infer_domain(self, arg_id, inputs, config):
		schema = self.q.infer_output_info(inputs)
		if arg_id == 1:
			# approximation: only get fields with more than one values
			# for the purpose of avoiding empty fields
			try:
				df = self.q.eval(inputs)
				df = df.extract_values()
				if df.empty:
					return []
			except Exception as e:
				print(f"[eval error in infer_domain] {e}")
				return []
			# use this list to store primitive table keys,
			# use them to elimiate column combinations that contain no duplicates
			table_keys = []
			col_num = len(schema)
			col_list_candidates = []
			for size in range(1, col_num):
				for gb_keys in itertools.combinations(list(range(col_num)), size):
					if any([set(banned).issubset(set(gb_keys)) for banned in table_keys]):
						# current key group is subsumbed by a table key, so all fields will be distinct
						continue
					gb_cols = df[[df.columns[k] for k in gb_keys if k < len(df.columns)]]
					if not gb_cols.duplicated().any():
						# a key group is valid for aggregation
						#   if there exists at least a key appear more than once
						table_keys.append(gb_keys)
						continue
					col_list_candidates += [gb_keys]
			valid_candidates = []
			groups_list = []
			for group_keys in col_list_candidates:
				keys = [df.columns[idx] for idx in group_keys]
				# Format: {index: {colname: argument}}
				# iterate through df, map each cell in resulting table with its argument
				temp = df.groupby(keys, sort=False)
				groups = [list(temp.groups[k]) for k in temp.groups]
				seen = False
				for found in groups_list:
					if all(map(lambda x, y: x == y, found, groups)):
						seen = True
						break
				if not seen:
					groups_list.append(groups)
					valid_candidates.append(group_keys)
			valid_candidates.append([])
			return valid_candidates
		elif arg_id == 3:
			number_fields = [i for i, s in enumerate(schema) if s == "number"]
			if self.group_cols != HOLE:
				cols = [i for i in number_fields if i not in self.group_cols]
			else:
				cols = number_fields
			return cols
		elif arg_id == 2:
			if self.target_col != HOLE:
				return [f for f in config["mutate_func"] if f != "count"]
			else:
				return config["mutate_func"]
		else:
			assert False, "[Gather] No args to infer domain for id > 1."

	def infer_output_info(self, inputs):
		input_schema = self.q.infer_output_info(inputs)
		# aggr_type = input_schema[self.aggr_col] if self.aggr_func != "count" else "number"
		# aggr_type = input_schema[self.target_col]
		output_schema = [s for i, s in enumerate(input_schema)]
		# if aggr_type == "number":
		output_schema += ["number"]
		return output_schema

	def eval(self, inputs):
		# make a copy of table for argument reference
		table = self.q.eval(inputs)
		if table.is_empty():
			return table
		df = table.extract_values()

		res = df.copy()
		target = df.columns[self.target_col]
		if self.sort_by is not None:
			res = res.sort_values(self.sort_by, ascending=self.asc)
		new_col = get_fresh_col(list(res.columns))[0]
		arguments = generate_direct_arguments(res)
		if self.group_cols:
			group_keys = [df.columns[idx] for idx in self.group_cols]
			# Format: {index: {colname: argument}}
			# iterate through df, map each cell in resulting table with its argument
			grouped_res = res.groupby(group_keys, sort=False)

			# map argument for keys and groups
			for (key, group) in grouped_res:
				# key can be a tuple if there are multiple group cols
				# get the group argument for the target column
				temp_arg = []
				# print(target)
				index_list = group.to_dict()[target]
				# group.to_dict() in {col_name:{rid:}} format
				# special trace handler for cumsum
				if self.aggr_func == "cumsum":
					# map argument for col cumsum
					for index in index_list:
						temp_arg.append((self.target_col, index))
						arguments[index][new_col] = temp_arg.copy()
				elif self.aggr_func == "rank":  # special trace handler for rank
					for row_index in index_list:
						arguments[row_index][new_col] = [(self.target_col, row_index)]
				else:
					arg = [(self.target_col, rid) for rid in index_list]
					for row_index in index_list:
						arguments[row_index][new_col] = arg
			# do aggregation work
			if self.aggr_func == "rank":
				res[new_col] = grouped_res[target].rank(method='first')
			elif self.aggr_func == "rownum":
				res[new_col] = grouped_res.cumcount() + 1
			else:
				res[new_col] = grouped_res.transform(self.aggr_func)[target]
				res = res.fillna(0)
		else:
			# if we have not group_key, we simply do a mutate
			if self.aggr_func == "cumsum":
				res[new_col] = res[target].cumsum()
				# map argument for col cumsum
				temp_arg = []
				for index in res.to_dict()[target]:
					temp_arg.append((self.target_col, index))
					arguments[index][new_col] = temp_arg.copy()
			elif self.aggr_func == "rank":
				res[new_col] = res[target].rank(method='first')
				for index in arguments:
					arguments[index][new_col] = arguments[index][target].copy()
			else:
				if self.aggr_func == "rownum":
					res[new_col] = np.arange(len(res))
				else:
					res[new_col] = res.apply(self.aggr_func)[target]
					res = res.fillna(0)
				arg = [(self.target_col, rid) for rid in arguments]
				for index in arguments:
					arguments[index][new_col] = arg
		res = round_df(res)
		return df_to_annotated_table_index_colname(res, self.aggr_func, arguments,
												   table, target_cols=[new_col])

	def to_dict(self):
		return {
			"type": "node",
			"op": "group_mutate",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.group_cols, "col_index_list"),
				value_to_dict(self.aggr_func, "func"),
				value_to_dict(self.target_col, "col_index")
			]}

	def infer_colnum(self, inputs):
		return self.q.infer_colnum(inputs) + 1

	def infer_rownum(self, inputs):
		return self.q.infer_rownum(inputs)

	def infer_value(self, inputs, config):
		if self.group_cols != HOLE and self.aggr_func != HOLE and self.target_col != HOLE:
			# the program has all parameters
			return self.eval(inputs)
		table = self.q.infer_value(inputs, config)
		rownum = table.get_row_num()
		colnum = table.get_col_num()
		new_source = []
		new_cell = TableCell(HOLE, HOLE)
		for rid in range(rownum):
			new_source.append(new_cell)
		table.add_column(new_source)  # add a new column
		table.round()
		return table

	def infer_trace(self, inputs, config):
		if self.group_cols != HOLE and self.aggr_func != HOLE and self.target_col != HOLE:
			# the program has all parameters
			return self.eval(inputs)

		table = self.q.infer_trace(inputs, config)
		if table == "Pass":
			return "Pass"
		rownum = table.get_row_num()
		colnum = table.get_col_num()
		new_source = []
		if self.group_cols == HOLE:  # we know nothing about parameters
			trace = [(x, y) for x in range(colnum) for y in range(rownum)]
			args = []
			for c in trace:
				if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
					args += table.get_cell(c[0], c[1]).get_exp()
				else:
					args += [table.get_cell(c[0], c[1]).get_exp()]
			func = ArgOr(config["mutate_func"])
			# print(set(args))
			new_cell = TableCell(HOLE, ExpNode(func, args))
			for rid in range(rownum):
				new_source.append(new_cell)
			table.add_column(new_source)  # add a new column
			return table
		df = table.extract_values()
		if not self.group_cols:
			df = [('', df)]
		else:
			group_keys = [df.columns[idx] for idx in self.group_cols]
			df = df.groupby(group_keys)
		new_cols = [i for i in range(colnum)] + [colnum]
		group_arg = {}  # gid: args
		member_gid = {}  # member1 : gid
		gid = 0
		for (key, group) in df:
			group_trace = [(x, y) for x in range(colnum)
								for y in group.index.values.tolist() if x not in self.group_cols]
			args = []
			for c in group_trace:
				if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
					args += table.get_cell(c[0], c[1]).get_exp()
				else:
					args += [table.get_cell(c[0], c[1]).get_exp()]
			args = remove_duplicates(args)
			group_arg[gid] = args
			for m in group.index.values.tolist():
				member_gid[m] = gid
			gid += 1
		for rid in range(rownum):
			# the new cell in new column can come from any cell
			# but it should not be placed in group cols
			args = group_arg[member_gid[rid]]
			func = self.aggr_func
			if self.aggr_func == HOLE:
				func = ArgOr(config["mutate_func"])
			new_cell = TableCell(HOLE, ExpNode(func, args))
			new_source.append(new_cell)
		table.add_column(new_source)  # add a new column
		return table


class Mutate_Arithmetic(Node):
	def __init__(self, q, func, cols):
		self.q = q
		self.cols = cols
		self.func = func

	def infer_domain(self, arg_id, inputs, config):
		schema = self.q.infer_output_info(inputs)
		if arg_id == 1:
			return config["mutate_function"]
		elif arg_id == 2:
			if self.func != HOLE:
				expression = eval(self.func)
				argc = expression.__code__.co_argcount
				columns = [i for i, s in enumerate(schema) if s == "number"]
				combinations_objects = list(itertools.permutations(columns, argc))
				return combinations_objects
			else:
				return []
		else:
			assert False, "[Mutate] No args to infer domain for id > 3."

	def infer_output_info(self, inputs):
		return self.q.infer_output_info(inputs) + ["number"]

	def eval(self, inputs):
		# make a copy of table for argument reference
		table = self.q.eval(inputs)
		if table.is_empty():
			return table
		df = table.extract_values()

		res = df.copy()
		arguments = generate_direct_arguments(res)
		col_name = str(self.cols)

		expression = eval(self.func)  # lambda x, y: x - y - 0.1 * x
		argc = expression.__code__.co_argcount
		if argc == 0 or len(self.cols) != argc:
			print("invalid number of arguments")
			assert False
		x_func = "lambda f: lambda x: f(x[0]"
		for i in range(1, argc):
			x_func += ", x[" + str(i) + "]"
		x_func += ")"
		# get the result of the function and make a new column with the value
		new_col = eval(x_func)(expression)([res[res.columns[i]] for i in self.cols])
		new_colname = get_fresh_col(res.columns)[0]
		function = {new_colname: new_col}
		res = res.assign(**function)

		# add arguments for the new column
		for index in arguments:
			arguments[index][new_colname] = []
			for i in self.cols:
				arguments[index][new_colname] += arguments[index][res.columns[i]]

		res = round_df(res)
		res = res.fillna(0)

		return df_to_annotated_table_index_colname(res, self.func, arguments,
												   table, target_cols=[new_colname])

	def to_dict(self):
		return {
			"type": "node",
			"op": "mutate_arithmetic",
			"children": [
				self.q.to_dict(),
				value_to_dict(self.func, "func"),
				value_to_dict(self.cols, "cols")
			]}

	def infer_value(self, inputs, config):
		if self.func != HOLE and self.cols != HOLE:  # the program has all parameters
			return self.eval(inputs)
		table = self.q.infer_value(inputs, config)
		rownum = table.get_row_num()
		new_source = []
		new_cell = TableCell(HOLE, HOLE)
		for rid in range(rownum):
			new_source.append(new_cell)
		table.add_column(new_source)  # add a new column
		table.round()
		return table

	def infer_trace(self, inputs, config):
		if self.func != HOLE and self.cols != HOLE:
			# the program has all parameters
			return self.eval(inputs)

		table = self.q.infer_trace(inputs, config)
		if table == "Pass":
			return "Pass"
		rownum = table.get_row_num()
		colnum = table.get_col_num()
		new_source = []
		cid = colnum
		for rid in range(rownum):
			# the new cell in new column can come from any cell
			trace = [(x, rid) for x in range(colnum)]
			args = []
			for c in trace:
				if isinstance(table.get_cell(c[0], c[1]).get_exp(), list):
					args += table.get_cell(c[0], c[1]).get_exp()
				else:
					args += [table.get_cell(c[0], c[1]).get_exp()]
			# args = remove_duplicates(args)

			func = self.func
			if self.func == HOLE:
				func = ArgOr(config["mutate_function"])
			new_cell = TableCell(HOLE, ExpNode(func, args))
			new_source.append(new_cell)
		table.add_column(new_source)  # add a new column
		return table

	def infer_colnum(self, inputs):
		return self.q.infer_colnum(inputs) + 1

	def infer_rownum(self, inputs):
		return self.q.infer_rownum(inputs)

""" ----- utility functions -----"""
def get_fresh_col(used_columns, n=1):
	"""get a fresh column name used in pandas evaluation"""
	names = []
	for i in range(0, 1000):
		if "COL_{}".format(i) not in used_columns:
			names.append("COL_{}".format(i))
		if len(names) >= n:
			break
	return names


def get_temp_var(used_vars):
	"""get a temp variable name """
	for i in range(0, 1000):
		var_name = "t{}".format(i)
		if var_name not in used_vars:
			return var_name


def value_to_dict(val, val_type):
	"""given the value and its type, dump it to a dict
		the helper function to dump values into dict ast
	"""
	return {"type": val_type, "value": val}


def extract_table_schema(df):
	"""Given a dataframe, extract it's schema """
	def dtype_mapping(dtype):
		"""map pandas datatype to c """
		dtype = str(dtype)
		if dtype == "object" or dtype == "string":
			return "string"
		elif "int" in dtype or "float" in dtype:
			return "number"
		elif "bool" in dtype:
			return "bool"
		else:
			print(f"[unknown type] {dtype}")
			sys.exit(-1)

	schema = [dtype_mapping(s) for s in df.infer_objects().dtypes]
	return schema


"""round data in the given df to degree two"""
def round_df(res):
	for col_name in res.columns:
		if res.dtypes[col_name] == 'double':
			res[col_name] = res[col_name].round(2)
	return res


def generate_direct_arguments(df, data_id=None):
	"""generate direct argument mapping for the given df
	returned argument format eg. {0: {'COL1':[(0,1,0), (0,1,1)]}}
	3/25 update format eg.{0: {'COL1':['0_a0', 'COL2_0']}}"""
	arguments = {}
	for index in df.index.tolist():
		arguments[index] = {}
		for col_name in df.to_dict().keys():
			arguments[index][col_name] = []
			if data_id is not None:
				arguments[index][col_name]\
					.append(f"{data_id}_{get_alphabet(get_col_index_by_name(df, col_name))}{index}")
			else:
				arguments[index][col_name].append((get_col_index_by_name(df, col_name), index))  # only coordinate
	return arguments


def df_to_annotated_table_index_colname(df, op, arguments, table, target_cols=None):
	"""convert df to annotated table with given op and arguments (trace info)"""
	cell_list = []
	for colName in df.columns.tolist():
		cid = get_col_index_by_name(df, colName)
		cell_list.append([])
		for index in df.index.tolist():
			# get full arguments for this level
			this_arguments = arguments[index][colName]
			cell_arg = []
			if table is None:
				exp = this_arguments.copy()
			else:
				for arg in this_arguments:
					if isinstance(arg, ArgOr):
						cell_arg += [arg]
					else:
						temp_exp = table.get_cell(arg[0], arg[1]).get_exp()
						if not isinstance(temp_exp, list):
							temp_exp = [temp_exp]
						if isinstance(temp_exp, list):
							temp_exp = [e[0] if isinstance(e, list) else e for e in temp_exp]
						cell_arg += temp_exp
				# there might be duplicate coord representing the same source
				cell_arg = remove_duplicates(cell_arg)  # TODO: change this to make it fast
				if isinstance(op, list) and colName in target_cols:
					target_index = target_cols.index(colName)
					exp = ExpNode(op[target_index], cell_arg)
				elif colName in target_cols:
					exp = ExpNode(op, cell_arg)
				else:
					exp = cell_arg
			cell_list[cid].append({"value": df.to_dict()[colName][index], "exp": exp})
	return AnnotatedTable(cell_list)


""" special handler for join"""
def df_to_annotated_table_join(df, op, arguments, table1, table2):
	"""special handler for join operation which include two tables"""
	cell_list = []
	for colName in df.columns.tolist():
		cid = get_col_index_by_name(df, colName)
		cell_list.append([])
		for index in df.index.tolist():
			# get full arguments for this level
			this_arguments = arguments[index][colName]
			args = []
			for arg in this_arguments:
				temp_exp = []
				if arg[0] == "table_1":
					temp_exp = table1.get_cell(arg[1], arg[2]).get_exp()
				elif arg[0] == "table_2":
					temp_exp = table2.get_cell(arg[1], arg[2]).get_exp()
				if not isinstance(temp_exp, list):
					temp_exp = [temp_exp]
				args += temp_exp
			args = remove_duplicates(args)
			cell_list[cid].append({"value": df.to_dict()[colName][index], "exp": args})
	return AnnotatedTable(cell_list)


def get_col_index_by_name(df, colName):
	return df.columns.get_loc(colName)

# not used after revisions
def get_index_by_rownum(df, rid):
	return df.index.tolist()[rid]


def get_value_by_row_col(df, rid, cid):
	return df.iloc[rid][df.columns[cid]]

def get_alphabet(i):

	alphabet_list = ['a', 'b', 'c', 'd', 'e', 'f',
					 'g', 'h', 'i', 'j', 'k', 'l',
					 'm', 'n', 'o', 'p', 'q', 'r',
					 's', 't', 'u', 'v', 'w', 'x',
					 'y', 'z']
	return alphabet_list[i]

def remove_duplicates(x):
	return list(set(x))


def dict_to_program(l):
	def to_program(q, dict):
		operators = ["select", "filter", "group_sum", "group_mutate", "mutate_arithmetic", "join"]
		if dict["op"] not in operators:
			return None
		op = dict["op"]
		if op == "select":
			return Select(q, dict["0"])
		if op == "group_sum":
			return GroupSummary(q, dict["0"], dict["1"], dict["2"])
		if op == "group_mutate":
			return GroupMutate(q, dict["0"], dict["1"], dict["2"])
		if op == "mutate_arithmetic":
			return Mutate_Arithmetic(q, dict["0"], dict["1"])
		if op == "join":
			# we assume user should specify join at the first step with some column provided
			return Join(q, Table(dict["0"]), dict["1"], dict["2"])
	q = Table(0)
	for i in range(1, len(l)):
		q = to_program(q, l[i])
	return q
