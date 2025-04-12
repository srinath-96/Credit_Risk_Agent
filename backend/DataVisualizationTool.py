from dotenv import load_dotenv
from typing import Any, Dict
from smolagents import CodeAgent, Tool,OpenAIServerModel
load_dotenv()# Load environment variables from .env file
import pandas as pd
import numpy as np
import os
from PyPDF2 import PdfReader
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from ucimlrepo import fetch_ucirepo
import h2o
import sys
import time

class DataVisualizationTool(Tool):
    name = "data_visualizer"
    description = "Uses a CodeAgent to create insightful visualizations (saved as .png files) from dataframes. Provide the dataframe."
    inputs = {
        "dataset": {
            "type": "object",
            "description": "Pandas DataFrame to analyze and visualize"
        },
        "visualization_type": {
             "type": "string",
             "description": "Optional: Specific type of analysis/plot requested.",
             "nullable": True
        }
    }
    output_type = "string" # The CodeAgent's text output

    def __init__(self, plot_save_dir="generated_plots", **kwargs):
        super().__init__(**kwargs)
        self.plot_save_dir = os.path.abspath(plot_save_dir) # Store absolute path
        os.makedirs(self.plot_save_dir, exist_ok=True)
        print(f"DEBUG [DataVisualizationTool]: Plots will be saved to: {self.plot_save_dir}")
        # Initialize the internal agent ONCE here if the model/config is static
        # Or initialize it within forward() if model/params change per call
        # Make sure GEMINI_API_KEY is accessible (e.g., set globally in api.py)
        try:
            # Use the same model as the main agent, or specify if different
            # Ensure model_id is valid for LiteLLM and your key
            self.model = OpenAIServerModel(
    model_id="models/gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key= os.getenv("GEMINI_API_KEY")  # Ensure your API key is set in .env
            )
            self.visualization_agent = CodeAgent(
                tools=[], # This agent likely doesn't need tools itself
                model=self.model,
                additional_authorized_imports=[
                    "numpy",
                    "pandas",
                    "plotly.express",
                    "plotly.graph_objects",
                    "plotly.io", # Explicitly for write_image
                    "os",
                    "matplotlib",
                    "matplotlib.pyplot"# Authorize 'os' for path joining
                ],
                # add_base_tools=False # Usually not needed for specific tasks
            )
            print("DEBUG [DataVisualizationTool]: Internal CodeAgent initialized.")
        except Exception as e:
            print(f"ERROR [DataVisualizationTool]: Failed to initialize internal CodeAgent: {e}", file=sys.stderr)
            self.visualization_agent = None # Mark as unavailable

    def forward(self, dataset: pd.DataFrame, visualization_type=None) -> str:
        """
        Uses an internal CodeAgent to execute plotting commands based on a prompt,
        instructing it to save plots to the configured directory.
        """
        if self.visualization_agent is None:
             return "Error: Visualization agent not initialized."
        if not isinstance(dataset, pd.DataFrame):
             return "Error: Invalid input. 'dataset' must be a pandas DataFrame."
        if 'default' not in dataset.columns:
             return "Error: Target column 'default' not found in the dataset."

        # Create timestamp for unique filenames
        timestamp = int(time.time())
        
        # Ensure the directory exists and is properly formatted for the agent
        os.makedirs(self.plot_save_dir, exist_ok=True)
        safe_plot_save_dir = self.plot_save_dir.replace("\\", "/")
        print(f"DEBUG [DataVisualizationTool]: Plots will be saved to: {safe_plot_save_dir}")

        # Simplified, highly specific analysis prompt
        analysis_prompt = f"""Generate 5 essential credit risk visualizations using ONLY Matplotlib and save them as PNG files. Follow these exact steps:

```python
import matplotlib
matplotlib.use('Agg')  # Force PNG backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

# Setup
OUTPUT_DIR = './generated_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Will save plots to: {{OUTPUT_DIR}}")

# Target variable
target = 'default.payment.next.month'

# Basic dataset info
print(f"Dataset shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")

# 1. TARGET DISTRIBUTION - Create and save
plt.figure(figsize=(10, 6))
ax = sns.countplot(x=target, data=df)
plt.title(f'Distribution of {{target}}')
plt.xlabel('Default (1=Yes, 0=No)')
plt.ylabel('Count')
target_plot = os.path.join(OUTPUT_DIR, 'target_distribution.png')
plt.tight_layout()
plt.savefig(target_plot, format='png')
plt.close()
print(f"SAVED PNG: {{target_plot}}")

# 2. CORRELATION HEATMAP - Create and save
numeric_cols = df.select_dtypes(include=['number']).columns
plt.figure(figsize=(12, 10))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
corr_plot = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
plt.tight_layout()
plt.savefig(corr_plot, format='png')
plt.close()
print(f"SAVED PNG: {{corr_plot}}")

# 3. FEATURE DISTRIBUTIONS - Create and save
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols[:6]):  # First 6 numeric columns
    if col != target:
        plt.subplot(2, 3, i+1)
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {{col}}')
        plt.tight_layout()
feature_plot = os.path.join(OUTPUT_DIR, 'feature_distributions.png')
plt.savefig(feature_plot, format='png')
plt.close()
print(f"SAVED PNG: {{feature_plot}}")

# 4. BOXPLOTS BY TARGET - Create and save
plt.figure(figsize=(15, 10))
important_features = numeric_cols[:6]  # First 6 numeric features
for i, col in enumerate(important_features):
    if col != target:
        plt.subplot(2, 3, i+1)
        sns.boxplot(x=target, y=col, data=df)
        plt.title(f'{{col}} by Default Status')
        plt.tight_layout()
box_plot = os.path.join(OUTPUT_DIR, 'boxplots_by_target.png')
plt.savefig(box_plot, format='png')
plt.close()
print(f"SAVED PNG: {{box_plot}}")

# 5. PAIRPLOT OF KEY FEATURES - Create and save
key_features = [col for col in numeric_cols[:4] if col != target] + [target]
pairplot = sns.pairplot(df[key_features], hue=target, corner=True)
pair_plot = os.path.join(OUTPUT_DIR, 'feature_pairplot.png')
pairplot.savefig(pair_plot, format='png')
plt.close('all')
print(f"SAVED PNG: {{pair_plot}}")

# Verify all plots were saved
png_files = [
    'target_distribution.png',
    'correlation_heatmap.png', 
    'feature_distributions.png',
    'boxplots_by_target.png',
    'feature_pairplot.png'
]

print("\\nSummary of PNG files created:")
successful_plots = []
for png in png_files:
    full_path = os.path.join(OUTPUT_DIR, png)
    if os.path.exists(full_path):
        print(f"✓ Successfully saved {{png}}")
        successful_plots.append(png)
    else:
        print(f"✗ Failed to save {{png}}")

print(f"\\nTotal PNG plots created: {{len(successful_plots)}}/5")
print("Visualization completed successfully!")
```

IMPORTANT NOTES:
1. Use ONLY Matplotlib/Seaborn (no Plotly)
2. Save ALL plots as PNG files using plt.savefig() with format='png'
3. Close each figure after saving to free memory
4. Print "SAVED PNG: <filepath>" for each successful save
5. Each plot should be saved with a specific filename as shown above
"""

        print(f"DEBUG [DataVisualizationTool]: Running internal agent with save dir: {self.plot_save_dir}")
        try:
            # Run the internal agent, passing the dataset
            result = self.visualization_agent.run(
                analysis_prompt,
                additional_args={"df": dataset.copy()} # Pass a copy to the agent
            )
            print(f"DEBUG [DataVisualizationTool]: Internal agent result: {result}")
            
            # Check for success indicators in the result
            successful_plots = []
            for line in result.split('\n'):
                if "SAVED PNG:" in line:
                    # Extract the filename from the line
                    filepath = line.split("SAVED PNG:")[-1].strip()
                    filename = os.path.basename(filepath)
                    successful_plots.append(filename)
                elif "✓ Successfully saved" in line:
                    # Extract the filename from the line
                    filename = line.split("✓ Successfully saved")[-1].strip()
                    successful_plots.append(filename)
            
            # If no plots were found from the output, directly check the directory
            if not successful_plots:
                try:
                    # Check for recently created PNG files (within the last 2 minutes)
                    current_time = time.time()
                    recent_pngs = []
                    for f in os.listdir(self.plot_save_dir):
                        if f.endswith('.png'):
                            file_path = os.path.join(self.plot_save_dir, f)
                            # Check if file was created/modified in the last 2 minutes
                            if current_time - os.path.getmtime(file_path) < 120:
                                recent_pngs.append(f)
                
                    if recent_pngs:
                        successful_plots = recent_pngs
                except Exception as e:
                    print(f"ERROR [DataVisualizationTool]: Error checking plot directory: {e}")
            
            # Add information about created plots to the result
            if successful_plots:
                result += f"\n\nVisualization tool successfully created {len(successful_plots)} plots: {', '.join(successful_plots)}"
            else:
                result += "\n\nVisualization tool attempted to create plots, but no PNG files were found."
            
            return result
        except Exception as e:
            print(f"ERROR [DataVisualizationTool]: Error running internal visualization agent: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return f"Error during visualization agent execution: {e}"