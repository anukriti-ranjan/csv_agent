from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from enum import Enum

class OutputType(Enum):
    DICT = "dict"
    DATAFRAME = "pandas dataframe"
    FIGURE = "figure"
    LIST = "list"
    SINGLE_ITEM = "single item"
    PANDAS_SERIES = "pandas series"

class PythonCodeForDataFrame(BaseModel):

    python_code: Optional[str] = Field(None, description="The Python code required to fulfill the query.")
    expected_output_type: OutputType = Field(description = "The type of output expected from the query")
    python_libraries: List[str] = Field(default_factory=list, description="List of Python libraries required to execute the generated code.")
    dataframes_required: List[int] = Field(..., description="The list of dataframes denoted by their numbers required to be used in code for fulfilling the query")
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                
                "python_code": "df['dates'] = pd.to_datetime(df['dates'])\ndf[month']=df['dates'].dt.month\ntotal_sales = df.groupby('month').sum()\nplt.bar(total_sales.index, total_sales['sales'])\nplt.show()",
                "expected_output_type": "figure",
                "python_libraries": ["pandas", "matplotlib"],
                "dataframes_required":[1],
                }
            }
    )


class CorrectedPythonCode(BaseModel):

    corrected_python_code: Optional[str] = Field(None, description="The corrected python code re-generated to fulfill the query.")