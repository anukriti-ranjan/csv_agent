
import os
import glob
import pandas as pd
import numpy as np
import openai
import instructor
import ast
import inspect
import base64
import io
import sys
import matplotlib.pyplot as plt


from prompts.openai_prompts import SYSTEM_PROMPT, DATAFRAME_PROMPT, DATAFRAME_PROMPT_PREFIX, USER_PROMPT_SUFFIX, USER_PROMPT_SUFFIX_DEBUGGING
from response_models.openai_response_models import PythonCodeForDataFrame, CorrectedPythonCode


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

class PandasAgentOpenAI:
    def __init__(self, api_key=None, temperature=0.1,data_dir="data"):
        if api_key is None:
            raise ValueError("API Key is required.")
        self.client = instructor.patch(openai.OpenAI(api_key=api_key))
        self.data_dir = data_dir
        self.list_of_filenames = self.get_list_of_filenames()
        self.system_prompt = SYSTEM_PROMPT
        self.dataframe_prompt = DATAFRAME_PROMPT
        self.dataframe_prompt_prefix = DATAFRAME_PROMPT_PREFIX
        self.user_prompt_suffix = USER_PROMPT_SUFFIX
        self.user_prompt_suffix_debugging = USER_PROMPT_SUFFIX_DEBUGGING
        self.query_count = 0
        self.previous_conversation_history = ""
        self.max_retries_first_response = 2
        self.max_retries_code_correction = 3
        self.temperarure = temperature

    def get_list_of_filenames(self):
        return glob.glob(os.path.join(self.data_dir, '*.csv'))

    def make_dataframe_prompt(self):
        dataframe_prompt = ""
        file_count = 0
        for idx, filename in enumerate(self.list_of_filenames, start=1):
            try:
                globals()[f"df{idx}"] = pd.read_csv(filename)
                dataframe_prompt += DATAFRAME_PROMPT.format(
                    i=idx, num=idx, filename=os.path.basename(filename),
                    df_head=str(globals()[f"df{idx}"].head(5).to_markdown())
                )
                file_count += 1
            except Exception as e:
                print(f"Error reading {filename}: {e}")
        
        dataframe_aggregated = DATAFRAME_PROMPT_PREFIX.format(n_dataframes=file_count) + "\n\n" + dataframe_prompt
        return dataframe_aggregated

    def add_context(self, context):
        self.previous_conversation_history += "\n" + context

    def fetch_conversation_history(self):
        return self.previous_conversation_history

    def make_user_prompt(self):
        user_prompt_prefix = self.make_dataframe_prompt()
        return user_prompt_prefix


    def run_python_code(self,code):
        try:
            # Parse and modify the AST
            tree = ast.parse(code)
            last_node = tree.body[-1] if tree.body else None

        # Modify AST to capture the result
            temp_name = "_result"
            if isinstance(last_node, ast.Expr):
                new_assign = ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())],
                                        value=last_node.value)
                new_assign.lineno = last_node.lineno
                new_assign.col_offset = last_node.col_offset
                tree.body[-1] = new_assign
            elif isinstance(last_node, ast.Assign):
                last_node.targets[0].id = temp_name

            # Set proper lineno and col_offset
            for node in ast.walk(tree):
                if 'lineno' in node._attributes:
                    node.lineno = 1
                if 'col_offset' in node._attributes:
                    node.col_offset = 0

            # Setup stdout capture
            buffer_stdout = io.StringIO()
            sys.stdout = buffer_stdout
            
            # Prepare to capture plots
            plots_captured = []
            original_show = plt.show
            
            def custom_show(*args, **kwargs):
                nonlocal plots_captured
                capture_active_figures(plots_captured)
                original_show(*args, **kwargs)
            
            def capture_active_figures(plot_list):
                figures = [plt.figure(i) for i in plt.get_fignums()]
                for fig in figures:
                    buffer_plot = io.BytesIO()
                    fig.savefig(buffer_plot, format='png', bbox_inches='tight')
                    buffer_plot.seek(0)
                    plot_data = base64.b64encode(buffer_plot.read()).decode('utf-8')
                    plot_list.append(plot_data)
                    plt.close(fig)  # Optionally close the fig after capture

            plt.show = custom_show
            
            # Execute the code
            local_state = {}
            exec(compile(tree, filename='<ast>', mode='exec'), globals(), local_state)

            # Capture any remaining figures not handled by plt.show
            capture_active_figures(plots_captured)

            # Restore plt.show to its original state
            plt.show = original_show

            # Restore stdout to capture printed outputs
            sys.stdout = sys.__stdout__
            output = buffer_stdout.getvalue().strip()
            
            # Prepare the result dictionary
            result_dict = {'printed_output': output}
            if plots_captured:
                result_dict['plot_data'] = plots_captured

            # Check for the last expression result
            last_result = local_state.get(temp_name, None)
            if last_result is not None:
                result_dict['last_expression'] = last_result

            return result_dict
        except Exception as e:
            return {'error': f"Error: {type(e).__name__}: {str(e)}"}
        
    def format_result(self, result_dict):
        formatted_dict = result_dict.copy()
        if 'last_expression' in result_dict:
            last_exp = result_dict.get('last_expression')
            if isinstance(last_exp, pd.DataFrame):
                formatted_dict["dataframe"] = last_exp
            elif isinstance(last_exp, list):
                formatted_dict["list"] = last_exp
            elif isinstance(last_exp, pd.Series):
                formatted_dict["pandas_series"] = last_exp.to_frame()
            elif isinstance(last_exp, dict):
                formatted_dict["dict"] = last_exp
            elif isinstance(last_exp, (int, float , str)):
                formatted_dict["result_value"] = last_exp
        return formatted_dict

    def get_answer_to_query(self, query):
        self.current_code = ""
        self.query_count += 1
        previous_conversation_history = self.fetch_conversation_history()
        user_prompt_prefix = self.make_user_prompt()
        user_prompt_suffix = self.user_prompt_suffix.format(
            previous_conversation_history=previous_conversation_history, query=query
        )
        user_prompt = user_prompt_prefix + "\n\n" + user_prompt_suffix
        #print(user_prompt)

        model_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_retries=self.max_retries_first_response,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature = self.temperarure,
            response_model=PythonCodeForDataFrame,
        )
        self.current_code = model_response.python_code
        print(model_response.python_code)
        result_dict = self.run_python_code(model_response.python_code)
        final_result = self.format_result(result_dict)

        context = f"Query no. {self.query_count}: {query}"
        context += "\n" f"Code generated for the query: {model_response.python_code}"


        if "error" in final_result:
            context += "\n" + f"Error: {final_result.get('error', None)}"
        else:

            if "dict" in final_result:
                answer = str(final_result.get("dict"))
                context += "\n" + f"Answer to the query: {answer}"
            if "list" in final_result:
                answer = str(final_result.get("list"))
                context += "\n" + f"Answer to the query: {answer}"
            if "result_value" in final_result:
                answer = str(final_result.get("result_value"))
                context += "\n" + f"Answer to the query: {answer}"
            if "dataframe" in final_result:
                final_result_df_head = str(final_result.get("dataframe").head(5).to_markdown())
                context += "\n" + f"Answer to the query: The first few rows of the generated answer: \n{final_result_df_head}"
            if "pandas_series" in final_result:
                final_result_df_head = str(final_result.get("pandas_series").head(5).to_markdown())
                context += "\n" + f"Answer to the query: The first few rows of the generated answer: \n{final_result_df_head}"
            if "plot_data" in final_result:
                context += "\n" + f"Answer to the query: A figure was plotted."

        self.add_context(context)
        #print(context)
        return final_result

    def get_code_correction(self, query, error):
        self.current_code = ""
        user_prompt_prefix = self.make_user_prompt()
        user_prompt_suffix = self.user_prompt_suffix_debugging.format(query=query, python_code = self.current_code, error=error)
        user_prompt = user_prompt_prefix + "\n\n" + user_prompt_suffix

        model_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature = self.temperarure,
            response_model=CorrectedPythonCode,
        )
        self.current_code = model_response.corrected_python_code
        print(model_response.corrected_python_code)
        result_dict = self.run_python_code(model_response.corrected_python_code)
        final_result = self.format_result(result_dict)

        context = f"Query no. {self.query_count}: {query}"
        context += "\n" f"Code re-generated for the query: {model_response.corrected_python_code}"

        if "error" in final_result:
            context += "\n" + f"Error: {final_result.get('error', None)}"
        else:

            if "dict" in final_result:
                answer = str(final_result.get("dict"))
                context += "\n" + f"Answer to the query: {answer}"
            if "list" in final_result:
                answer = str(final_result.get("list"))
                context += "\n" + f"Answer to the query: {answer}"
            if "result_value" in final_result:
                answer = str(final_result.get("result_value"))
                context += "\n" + f"Answer to the query: {answer}"
            if "dataframe" in final_result :
                final_result_df_head = str(final_result.get("dataframe").head(5).to_markdown())
                context += "\n" + f"Answer to the query: The first few rows of the generated answer: \n{final_result_df_head}"
            if "pandas_series" in final_result:
                final_result_df_head = str(final_result.get("pandas_series").head(5).to_markdown())
                context += "\n" + f"Answer to the query: The first few rows of the generated answer: \n{final_result_df_head}"
            if "plot_data" in final_result:
                context += "\n" + f"Answer to the query: A figure was plotted."

        self.add_context(context)
        return final_result


    def run_agent(self, query):
        self.current_answer = self.get_answer_to_query(query)
        if "error" in self.current_answer or "":
            for retry_num in range(self.max_retries_code_correction):
                error = self.current_answer.get("error")
                self.current_answer = self.get_code_correction(query, error)
                if "error" not in self.current_answer:
                    break
        return self.current_answer