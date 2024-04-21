SYSTEM_PROMPT = """
You are an expect python coder and debugger.
Allowed imports: pandas as pd, matplotlib.pyplot as plt, numpy as np, datetime
"""

DATAFRAME_PROMPT_PREFIX = """
You are working with {n_dataframes} pandas dataframe in Python.
Their names, the file name from which they were imported and a few rows from them are provided below in order.
"""

DATAFRAME_PROMPT = """
The dataframe number {i}: It is imported from file: {filename}.
The name of the dataframe in the global environment is df{num} .
The first few rows of the dataframe are as follows:
{df_head}
"""

USER_PROMPT_SUFFIX = """
{previous_conversation_history}
Based on the above information about the dataframes, answer the following user query.
Query: {query}
"""

USER_PROMPT_SUFFIX_DEBUGGING = """
For the query: {query},
the following python code was generated
{python_code}
This yielded the following error.
{error}

Your task is to correct the code.
"""