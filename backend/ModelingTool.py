from dotenv import load_dotenv
from typing import Any, Dict, Union
from smolagents import CodeAgent, Tool, OpenAIServerModel
load_dotenv()# Load environment variables from .env file
import pandas as pd
import numpy as np
import os
from ucimlrepo import fetch_ucirepo
import h2o
import sys
import time
import re
import json
import traceback

class ModelingTool(Tool):
    name = "modeling_tool"
    description = "Runs H2O AutoML for classification (target 'Y'), saves the best model, and reports metrics."
    inputs = {
        "dataset": {
            "type": "object",
            "description": "Pandas DataFrame for modeling (must contain target 'Y')."
        }
    }
    output_type = "string"

    def __init__(self, model_save_path="best_model", **kwargs):
        super().__init__(**kwargs)
        self.model_save_path = os.path.abspath(model_save_path).replace("\\", "/")
        os.makedirs(self.model_save_path, exist_ok=True)
        print(f"DEBUG [ModelingTool]: Models will be saved to: {self.model_save_path}")
        try:
            self.model = OpenAIServerModel(
                model_id="models/gemini-2.0-flash",
                api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.getenv("GEMINI_API_KEY")
            )
            self.modeling_agent = CodeAgent(
                tools=[],
                model=self.model,
                additional_authorized_imports=[
                    "numpy",
                    "pandas",
                    "h2o",
                    "os",
                    "h2o.automl",
                    "h2o.estimators.random_forest",
                    "h2o.estimators.gbm",
                    "h2o.estimators.glm",
                    "json",
                    "traceback"
                ],
            )
            print("DEBUG [ModelingTool]: Internal CodeAgent initialized.")
        except Exception as e:
            print(f"ERROR [ModelingTool]: Failed to initialize internal CodeAgent: {e}", file=sys.stderr)
            self.modeling_agent = None

    def forward(self, dataset: pd.DataFrame) -> str:
        if self.modeling_agent is None:
            return "Error: Modeling agent not initialized."
        if not isinstance(dataset, pd.DataFrame):
            return "Error: Invalid input. 'dataset' must be a pandas DataFrame."

        target_col = 'Y'
        if target_col not in dataset.columns:
            target_col_found = None
            possible_targets = ['default', 'default.payment.next.month', 'default_payment_next_month']
            for col in dataset.columns:
                if col.lower() in possible_targets:
                    target_col_found = col
                    break
            if target_col_found:
                print(f"WARN [ModelingTool]: Target 'Y' not found, using '{target_col_found}' instead and renaming.")
                dataset = dataset.rename(columns={target_col_found: 'Y'})
            else:
                return f"Error: Target column '{target_col}' (or common variants) not found in the dataset columns: {dataset.columns.tolist()}"

        model_filename = f"h2o_automl_model_{int(time.time())}"
        safe_model_save_dir = self.model_save_path

        analysis_prompt = f"""
        As an ML Engineer, perform the following tasks using the h2o library on the provided dataset:

        1. Import necessary libraries and initialize h2o:
        ```python
        import h2o
        from h2o.estimators.random_forest import H2ORandomForestEstimator
        from h2o.estimators.gbm import H2OGradientBoostingEstimator
        from h2o.estimators.glm import H2OGeneralizedLinearEstimator
        from h2o.automl import H2OAutoML
        import pandas as pd
        import os
        import json
        import traceback

        results = {{}}

        try:
            h2o.init(max_mem_size="2G", log_level="WARN")
            print("H2O initialized.")

            target_column_name = "Y"
            col_types = {{col: "numeric" for col in dataframe.columns}}
            if target_column_name in col_types:
                col_types[target_column_name] = "enum"
            else:
                raise ValueError(f"Target column '{{target_column_name}}' not found in dataframe passed to agent.")

            print(f"Using column types: {{col_types}}")

            print("Converting DataFrame to H2OFrame...")
            df_h2o = h2o.H2OFrame(dataframe, column_types=col_types)
            print("H2OFrame created.")

            y = target_column_name
            x = df_h2o.columns
            if y in x:
                x.remove(y)
            id_col_to_remove = None
            for col_name in x:
                if col_name.upper() == "ID":
                    id_col_to_remove = col_name
                    break
            if id_col_to_remove:
                x.remove(id_col_to_remove)
                print(f"Removed predictor column: '{{id_col_to_remove}}'")

            print(f"Data split complete. Train rows: {{train.nrows}}, Test rows: {{test.nrows}}")
            print(f"Using Predictors: {{x}}")
            print(f"Using Target: {{y}}")

            print("Starting AutoML...")
            aml = H2OAutoML(max_models=10, seed=1, max_runtime_secs=180)
            aml.train(x=x, y=y, training_frame=train)
            print("AutoML training finished.")

            lb = aml.leaderboard
            print("--- AutoML Leaderboard (Top 5) ---")
            print(lb.head(rows=5).as_data_frame().to_string())
            print("--- End Leaderboard ---")

            if lb.nrows > 0:
                best_model = aml.leader
                results['model_type'] = best_model.model_id
                print(f"Best Model ID: {{results['model_type']}}")
            else:
                results['error'] = "AutoML did not produce any models."
                print(results['error'])
                raise Exception(results['error'])

            print("Evaluating best model on test data...")
            performance = best_model.model_performance(test)

            metrics = {{}}
            try:
                metrics['auc'] = performance.auc()
                metrics['accuracy'] = performance.accuracy()[0][1]
                metrics['precision'] = performance.precision()[0][1]
                metrics['recall'] = performance.recall()[0][1]
                metrics['f1'] = performance.F1()[0][1]
                metrics['logloss'] = performance.logloss()
                results['metrics'] = metrics
                print(f"Calculated Metrics: AUC={{metrics['auc']:.4f}}, Accuracy={{metrics['accuracy']:.4f}}")
            except Exception as metric_err:
                print(f"Warning: Could not calculate all metrics: {{metric_err}}")
                results['metrics_error'] = str(metric_err)

            save_dir = "{safe_model_save_dir}"
            file_name = best_model.model_id.replace(":", "_").replace("/", "_")
            print(f"Saving model to: {{save_dir}} with name based on ID: {{file_name}}")
            try:
                model_path = h2o.save_model(model=best_model, path=save_dir, filename=file_name, force=True)
                results['model_path'] = model_path
                print(f"MODEL_SAVE_PATH: {{model_path}}")
            except Exception as save_err:
                print(f"Error saving model: {{save_err}}")
                results['save_error'] = str(save_err)

            try:
                cm_df = performance.confusion_matrix().as_data_frame()
                cm_dict = cm_df.to_dict()
                tn = cm_dict.get('0', {{}}).get('0', 0)
                fp = cm_dict.get('1', {{}}).get('0', 0)
                fn = cm_dict.get('0', {{}}).get('1', 0)
                tp = cm_dict.get('1', {{}}).get('1', 0)
                results['confusion_matrix'] = {{"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
                print(f"Confusion Matrix values: TN={{tn}}, FP={{fp}}, FN={{fn}}, TP={{tp}}")
            except Exception as cm_err:
                print(f"Warning: Could not extract confusion matrix: {{cm_err}}")
                results['cm_error'] = str(cm_err)

        except Exception as e:
            error_msg = f"ERROR_DURING_EXECUTION: {{str(e)}}"
            results["error"] = error_msg
            print(error_msg)

        finally:
            h2o.shutdown(prompt=False)
            print("H2O Shutdown.")

        print("---RESULTS_JSON_START---")
        output_dict_serializable = json.loads(json.dumps(results, default=str))
        print(json.dumps(output_dict_serializable))
        print("---RESULTS_JSON_END---")
        """

        print(f"DEBUG [ModelingTool]: Running agent with prompt expecting target 'Y'.")
        try:
            agent_result = self.modeling_agent.run(
                analysis_prompt,
                additional_args={"dataframe": dataset.copy()}
            )

            if isinstance(agent_result, dict):
                print(f"DEBUG [ModelingTool]: Agent returned a dictionary directly.")
                formatted_output = self._format_output(agent_result)
            elif isinstance(agent_result, str):
                print(f"DEBUG [ModelingTool]: Agent returned a string. Raw length: {len(agent_result)}")
                formatted_output = self._format_output(agent_result)
            else:
                print(f"WARN [ModelingTool]: Agent returned unexpected type: {type(agent_result)}")
                formatted_output = f"Error: Agent returned unexpected result type '{type(agent_result)}'."

            return formatted_output

        except Exception as e:
            print(f"ERROR [ModelingTool]: Error running modeling agent: {e}", file=sys.stderr)
            print(traceback.format_exc())
            return f"Error during agent execution: {str(e)}"

    def _format_output(self, result_input: Union[str, dict]) -> str:
        """Parses the JSON output (passed as dict or str) and formats it."""
        print("DEBUG [_format_output]: Formatting agent result.")
        results_dict = None
        try:
            if isinstance(result_input, dict):
                print("DEBUG [_format_output]: Received pre-parsed dictionary.")
                results_dict = result_input
            elif isinstance(result_input, str):
                print("DEBUG [_format_output]: Received string, attempting to parse JSON.")
                json_match = re.search(r"---RESULTS_JSON_START---(.*?)---RESULTS_JSON_END---", result_input, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1).strip()
                    results_dict = json.loads(json_str)
                    print("DEBUG [_format_output]: Successfully parsed JSON from string.")
                else:
                    print("ERROR [_format_output]: JSON markers not found in string input.")
                    error_log_match = re.search(r"ERROR_DURING_EXECUTION: (.*)", result_input, re.DOTALL)
                    if error_log_match:
                        error_detail = error_log_match.group(1).strip().split('\n')[0]
                        return f"## Modeling Error\n\nAn error occurred during H2O processing:\n```\n{error_detail}\n```"
                    h2o_error_match = re.search(r"(h2o\..*?Error: .*?)\n", result_input, re.IGNORECASE)
                    if h2o_error_match:
                        return f"## Modeling Error\n\nAn H2O error occurred:\n```\n{h2o_error_match.group(1).strip()}\n```"
                    if "error" in result_input.lower():
                        return f"## Agent Error\n\n{result_input}"
                    return f"Error: Could not parse results from modeling agent string. Preview: {result_input[:250]}..."
            else:
                return f"Error: _format_output received unexpected type '{type(result_input)}'."

            if not results_dict:
                return "Error: Failed to obtain results dictionary for formatting."

            if "error" in results_dict and results_dict["error"]:
                print(f"ERROR [_format_output]: Error reported in results JSON: {results_dict['error']}")
                if "did not produce any models" in results_dict["error"]:
                    return "## H2O AutoML Results\n\nProcess completed, but AutoML did not build any models within the specified constraints."
                error_detail = str(results_dict['error']).replace('\\n', '\n')
                return f"## Modeling Error\n\n{error_detail}"

            output = "## H2O AutoML Results\n\n"
            output += f"**Best Model ID:** `{results_dict.get('model_type', 'N/A')}`\n"
            output += f"**Model Saved:** {'✅ Yes' if results_dict.get('model_path') else '❌ No'}\n\n"

            metrics = results_dict.get('metrics', {})
            if metrics and metrics.get('auc') is not None:
                output += "### Performance Metrics (Test Set)\n\n"
                output += f"- **AUC:** {metrics.get('auc', 'N/A'):.4f}\n"
                output += f"- **Accuracy:** {metrics.get('accuracy', 'N/A'):.4f}\n"
                output += f"- **Precision:** {metrics.get('precision', 'N/A'):.4f}\n"
                output += f"- **Recall:** {metrics.get('recall', 'N/A'):.4f}\n"
                output += f"- **F1 Score:** {metrics.get('f1', 'N/A'):.4f}\n"
                if 'logloss' in metrics:
                    output += f"- **LogLoss:** {metrics.get('logloss', 'N/A'):.4f}\n"
                output += "\n"

                cm = results_dict.get('confusion_matrix')
                cm_error = results_dict.get('cm_error')

                if cm and isinstance(cm, dict):
                    output += "### Confusion Matrix\n\n"
                    tn = cm.get('tn', 'N/A')
                    fp = cm.get('fp', 'N/A')
                    fn = cm.get('fn', 'N/A')
                    tp = cm.get('tp', 'N/A')
                    output += "| Actual / Predicted | Predicted 0 | Predicted 1 |\n"
                    output += "|--------------------|-------------|-------------|\n"
                    output += f"| **Actual 0**       | {tn} (TN)   | {fp} (FP)   |\n"
                    output += f"| **Actual 1**       | {fn} (FN)   | {tp} (TP)   |\n\n"
                elif cm_error:
                    output += f"### Confusion Matrix\n\nCould not extract confusion matrix.\nWarning: `{cm_error}`\n\n"

                output += "### Interpretation\n"
                output += "- **AUC:** Model's ability to separate classes.\n"
                output += "- **Accuracy:** Overall correct predictions.\n"
                output += "- **Precision:** Correct positive predictions rate.\n"
                output += "- **Recall:** True positive rate.\n"
                output += "- **F1 Score:** Balance of Precision and Recall.\n"

            else:
                output += "Performance metrics were not successfully extracted or calculated.\n"
                if results_dict.get('metrics_error'):
                    output += f"Metric Calculation Warning: {results_dict['metrics_error']}\n"

            return output

        except json.JSONDecodeError as e:
            print(f"ERROR [_format_output]: Failed to decode JSON content: {e}")
            return f"Error: Failed to decode results JSON. Error: {e}"
        except Exception as e:
            print(f"ERROR [_format_output]: Unexpected error formatting output: {e}")
            print(traceback.format_exc())
            return f"Error: Unexpected error formatting results. Error: {e}"