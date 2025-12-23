"""_summary_
"""

from pandas import DataFrame


def print_mem_mb(df: DataFrame):
    """_summary_

    Args:
        df (DataFrame): _description_
    """
    print(f"{df.memory_usage().sum()*1e-06:0.2f} MB")
