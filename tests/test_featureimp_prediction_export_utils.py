import warnings

import numpy as np
import pandas as pd

from autobehaver.postinf.featureimp.prediction_export_utils import insert_group_separator_rows


def test_insert_group_separator_rows_preserves_order_without_futurewarning():
    df = pd.DataFrame(
        {
            "individual": ["a1", "a2", "b1", "b2"],
            "group": ["A", "A", "B", "B"],
            "Prob_A": [0.1, 0.2, 0.3, 0.4],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        out = insert_group_separator_rows(df)

    assert not [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(out) == 5
    assert out.loc[0, "individual"] == "a1"
    assert out.loc[1, "individual"] == "a2"
    assert pd.isna(out.loc[2, "individual"])
    assert pd.isna(out.loc[2, "group"])
    assert np.isnan(out.loc[2, "Prob_A"])
    assert out.loc[3, "individual"] == "b1"
    assert out.loc[4, "individual"] == "b2"
