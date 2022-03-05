import sys
from table_ast import *
from synthesizer import *
from tabulate import tabulate
from table_cell import *

test_config = {
        "operators": ["group_sum", "mutate_arithmetic", "group_mutate", "join"],
        "parameter_config": {
            "filer_op": ["=="],
            "constants": [3000],
            "aggr_func": ["mean", "sum", "count", "max", "min"],
            "mutate_func": ["mean", "sum", "max", "min", "count", "cumsum", "rank"],
            "join_predicates": ["[(0, 1), (0, 0)]"],
            "join_outer": [False, True],
            "mutate_function": ["lambda x, y: x - y",
                                "lambda x, y: x + y",
                                "lambda x, y: x * y",
                                "lambda x, y: x / y",
                                "lambda x: x - (x * 0.1)",
                                "lambda x, y: y / (x - y)",
                                "lambda x: 1",
                                "lambda x: x * 1000",
                                "lambda x, y: x / y * 100"
                                ]
        },
        "permutation_test": False,
        "random_test": False,
        "partial_table": False,
        "partial_trace": False,
        "level_limit": 6,
        "time_limit": 600,
        "solution_limit": 5
    }

if __name__ == '__main__':
    # sys.argv
    with open(sys.argv[1], 'r') as filehandler:
        data = json.load(filehandler)
        try:
            inputs = data["input_data"]
            demonstration = data["demonstration"]
            print(data["parameter_config"])
            curr_config = test_config
            curr_config["parameter_config"] = data["parameter_config"]
            curr_config["time_limit"] = int(sys.argv[2])
            curr_config["solution_limit"] = int(sys.argv[3])
        except:
            print("[load error] incorrect format of components. Please refer to example.json for the correct format.")
        # construct a new annotated table with the given demonstration
        parsed = []
        for row in demonstration:
            for cid in range(len(row)):
                field = row[cid]
                if len(parsed) <= cid:
                    parsed.append([])
                if isinstance(field, dict):
                    parsed[cid].append(TableCell("_?_", dict_to_exp(field)))
                else:
                    parsed[cid].append(TableCell("_?_", field))
        manual = AnnotatedTable(parsed, from_source=True)

        print("=======Computation Demonstration==========")
        print(manual.to_dataframe())
        candidates = []
        print("=======Running Synthesizer==========")
        for i in range(0, curr_config["level_limit"]):
            print(f"start with level {i}")
            candidates += Synthesizer(curr_config) \
                .enumerative_synthesis(inputs, manual, None, i,
                                       time_limit_sec=curr_config["time_limit"],
                                       solution_limit=curr_config["solution_limit"])
            if len(candidates) > 0:
                break

        for i in range(len(candidates)):
            p = candidates[i]
            # print(alignment_result)
            print(f"candidate program {i}:")
            print(p.stmt_string())
            print("output table:")
            print(tabulate(p.eval(inputs).compress_sum().extract_values(), headers='keys', tablefmt='psql'))
            print("corresponding provenance information:")
            print(tabulate(p.eval(inputs).compress_sum().extract_traces(), headers='keys', tablefmt='psql'))
            print()
        print(f"number of programs: {len(candidates)}")
        print("\n\n\n\n\n\n")
        print("------------------------------------------------------------------------------------------")
