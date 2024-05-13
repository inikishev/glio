import polars as pl


def list_to_str(df:pl.DataFrame, col):
    return df.with_columns(
                center=pl.col(col).map_elements(lambda x: str(x.to_list())[1:-1]),
                src=pl.lit(col))

def str_to_list(df:pl.DataFrame, col):
    return df.with_columns(
                center=pl.col(col).map_elements(lambda x: [float(i) for i in str(x[1:-1]).split(", ")]),
                src=pl.lit(col),)