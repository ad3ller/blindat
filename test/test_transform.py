import numpy as np
import pandas as pd
import blindat as bd

COLUMNS = ["A", "B"]
NUM_ROWS = int(1e2)
SPECS = {
    "A": {"offset": 0.0, "scale": 2.0},
    "B": {"offset": 0.5, "scale": 1.0},
}

# generate data
data = np.ones((NUM_ROWS, len(COLUMNS)))
df = pd.DataFrame(data, columns=COLUMNS)


def test_transform():
    # generate rules
    rules = bd.generate_rules(SPECS)

    # blind
    left = bd.blind(df, rules)

    # test
    right = pd.DataFrame(data, columns=COLUMNS)
    right["A"] = right["A"] * 2
    right["B"] = right["B"] + 0.5
    pd.testing.assert_frame_equal(left, right)
