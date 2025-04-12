import os
import sys
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, DuckDuckGoSearchTool,ToolCallingAgent # Base smolagents imports
#from ucimlrepo import fetch_ucirepo
import h2o
import logging # Using logging module
import time

# Import your custom tools
from DataVisualizationTool import DataVisualizationTool
from ModelingTool import ModelingTool
from RetrieverTool import RetrieverTool

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration for paths (use environment variables or defaults)
# Ensure defaults point to directories within BASE_DIR
PDF_DIRECTORY = os.getenv("PDF_DIRECTORY", os.path.join(BASE_DIR, "research_papers"))
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", os.path.join(BASE_DIR, "best_model"))
PLOT_SAVE_DIRECTORY = os.getenv("PLOT_SAVE_DIRECTORY", os.path.join(BASE_DIR, "generated_plots"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
FLASK_PORT = int(os.getenv("PORT", 5001))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173") # Default Vite port

# --- Initialization ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": FRONTEND_URL}}) # Use env var for origin

# Ensure essential directories exist
os.makedirs(PLOT_SAVE_DIRECTORY, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
if not os.path.isdir(PDF_DIRECTORY):
    logging.warning(f"PDF directory '{PDF_DIRECTORY}' not found. Creating it, but it might be empty.")
    os.makedirs(PDF_DIRECTORY, exist_ok=True)


# --- Pre-load Data and Initialize Services ---
logging.info("--- Initializing Application ---")
main_dataframe = None
@app.route('/api/upload-dataset', methods=['POST'])
def upload_dataset():
    global main_dataframe 
    if 'dataset' not in request.files:
        return jsonify({"error": "No dataset file provided"}), 400

    dataset_file = request.files['dataset']
    if dataset_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if the file is a CSV
    if not dataset_file.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files are accepted"}), 400

    # Save the file to a designated directory
    save_path = os.path.join(BASE_DIR, 'datasets', dataset_file.filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    dataset_file.save(save_path)

    # Load the dataset into a DataFrame for further processing
    try:
        main_dataframe = pd.read_csv(save_path)
        logging.info(f"Dataset uploaded and loaded successfully. Shape: {main_dataframe.shape}")
    except Exception as e:
        logging.error(f"Failed to load dataset into DataFrame: {e}", exc_info=True)
        return jsonify({"error": "Failed to load dataset into DataFrame."}), 500

    return jsonify({"message": "Dataset uploaded successfully", "file_path": save_path}), 200

# Initialize H2O
logging.info("Initializing H2O...")
try:
    # Reduce log spam from H2O itself
    h2o.init(log_level="WARN")
    logging.info("H2O initialized successfully.")
except Exception as e:
    logging.warning(f"H2O initialization failed: {e}. Modeling tool may not function.", exc_info=True)

# Check for API Key
if not GEMINI_API_KEY:
    logging.error("GEMINI_API_KEY environment variable not set. AI Agent will not work.")
    sys.exit("Exiting due to missing API key.")
else:
    # Set environment variable for LiteLLM if it relies on it globally
    # Note: LiteLLM usually picks it up automatically if set
    # os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    logging.info("GEMINI_API_KEY found.")


# Initialize Tools
logging.info("Initializing tools...")
try:
    visualization_tool = DataVisualizationTool(plot_save_dir=PLOT_SAVE_DIRECTORY)
    modeling_tool = ModelingTool(model_save_path=MODEL_SAVE_PATH)
    retriever_tool = RetrieverTool(pdf_directory=PDF_DIRECTORY)
    search_tool = DuckDuckGoSearchTool()
    logging.info("Tools initialized.")
except Exception as e:
     logging.error(f"Failed to initialize tools: {e}", exc_info=True)
     sys.exit("Exiting due to tool initialization failure.")

# Initialize Main Model and Agent (Primary Agent orchestrating tools)
logging.info("Initializing primary AI model and agent...")
try:
    # Ensure the model ID is correct and supported by LiteLLM/Gemini
    primary_model = OpenAIServerModel(
    model_id="models/gemini-2.0-flash",
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=os.getenv("GEMINI_API_KEY")) # Or your preferred orchestrator model
    primary_agent = CodeAgent( 
        tools=[visualization_tool, modeling_tool, retriever_tool, search_tool],
        model=primary_model
        # Removed CodeAgent specific params like additional_authorized_imports/add_base_tools
        # ToolCallingAgent focuses on selecting and running tools based on description/inputs
    )
    logging.info("Primary AI model and agent initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize primary AI model or agent: {e}", exc_info=True)
    sys.exit("Exiting due to AI setup failure.")


logging.info("--- Application Initialization Complete ---")

# --- Helper Function ---
def get_conversation_context_from_list(history_list, num_messages=10):
    """Takes a list of {'role': 'user'/'assistant', 'content': 'message'}"""
    context = ""
    for msg in history_list[-num_messages:]:
        role = msg.get("role", "unknown").lower()
        content = msg.get("content", "")
        prefix = "User: " if role == "user" else "Bot: "
        context += prefix + content + "\n"
    return context.strip()

# --- API Endpoints ---

# Endpoint to serve generated plots
@app.route('/api/plots/<path:filename>')
def serve_plot(filename):
    """Serves a file from the plot directory."""
    logging.info(f"Request received for plot: {filename}")
    try:
        return send_from_directory(PLOT_SAVE_DIRECTORY, filename, as_attachment=False)
    except FileNotFoundError:
        logging.warning(f"Plot file not found: {filename}")
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logging.error(f"Error serving plot {filename}: {e}", exc_info=True)
        return jsonify({"error": "Error serving file"}), 500


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Receives chat message and history, returns bot response."""
    global main_dataframe  # Declare the global variable

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    user_input = data.get('message')
    history = data.get('history', []) # Expecting [{'role': 'user'/'assistant', 'content': '...'}, ...]

    if not user_input:
        return jsonify({"error": "Missing 'message' in request"}), 400
    if not isinstance(history, list):
         return jsonify({"error": "'history' must be a list of {'role': ..., 'content': ...} objects"}), 400

    # Optional: Basic validation for history format
    for item in history:
        if not isinstance(item, dict) or 'role' not in item or 'content' not in item:
            logging.warning(f"Invalid history item format received: {item}")
            # Decide whether to reject or proceed carefully
            # return jsonify({"error": "Invalid item format in 'history'"}), 400

    logging.info(f"Received message: {user_input}")

    # Simple exit command handling (optional - better handled by frontend state)
    if user_input.lower() in ["exit", "quit", "bye"]:
        return jsonify({"response": "Goodbye!", "plot_urls": []})

    # --- Prepare for Agent ---
    # ToolCallingAgent typically works best with structured history
    # Convert our simple history list to the format expected by the agent if needed
    # Or construct a prompt manually if ToolCallingAgent doesn't take history directly
    # For now, assume ToolCallingAgent can handle the prompt or takes messages list

    # Construct a system prompt or context for the ToolCallingAgent
    # This might be less detailed than the CodeAgent prompts within tools
    # Focus on instructing the agent *which tool to call*
    system_prompt = f"""

You are a Credit Risk Intelligence Platform assistant. Your primary goal is to accurately interpret user requests and utilize the appropriate specialized tool ONLY when explicitly requested. Maintain a helpful and conversational tone.

**CRITICAL GREETING INSTRUCTIONS:**
When a user greets you with "hi", "hello", "hey", or similar greetings, you MUST respond with ONLY a greeting and a brief introduction. DO NOT CALL ANY TOOLS for greetings.

**TOOL USAGE INSTRUCTIONS:**
- User: "Hi there" or "Hello" or "How are you?" 
  - DO NOT USE ANY TOOLS! Just respond with a friendly greeting.
Example good response to "hi": "Hello! I'm your Credit Risk Assistant. I can help you analyze credit risk data, create visualizations, build predictive models, and search for relevant information. How can I assist you today?"
**Available Tools:**
- 'data_visualizer': ONLY use when the user EXPLICITLY asks for visualization using words like "visualize", "plot", "chart", "graph", "show me". Never trigger for general questions or greetings.
- 'modeling_tool': ONLY use when the user EXPLICITLY asks for modeling/prediction using words like "predict", "model", "forecast", "machine learning". Never trigger for general questions or greetings.
- 'retriever': ONLY use when the user EXPLICITLY asks for research information.
- 'DuckDuckGoSearch': ONLY use when the user EXPLICITLY asks for web search.

**Tool Selection Steps:**
1. For ANY user message, FIRST determine if it's a greeting ("hi", "hello", etc.). If it IS a greeting, respond ONLY with a greeting - DO NOT CALL ANY TOOLS.
2. If it's NOT a greeting, then check if the message EXPLICITLY requests a specific tool function.
3. ONLY if there is an EXPLICIT request matching a tool's purpose should you call that tool.
4. If no tool is explicitly requested, respond conversationally or ask for clarification.

**Follow these steps for EVERY message:**
1. Is this a greeting? → If YES → Respond with greeting ONLY, NO TOOLS
2. Not a greeting → Does message explicitly request visualization? → If YES → Use data_visualizer ONLY
3. Not a greeting → Does message explicitly request modeling? → If YES → Use modeling_tool ONLY
4. Not a greeting → Does message explicitly request research? → If YES → Use retriever ONLY
5. Not a greeting → Does message explicitly request web search? → If YES → Use DuckDuckGoSearch ONLY
6. None of the above → Respond conversationally, NO TOOLS

Remember: The user MUST explicitly indicate which tool they want to use. Simple questions or greetings should NEVER trigger tools.
   
    """

    conversation_history_str = get_conversation_context_from_list(history)
    full_user_query = f"Conversation History:\n{conversation_history_str}\n\nUser's Latest Message: {user_input}"


    logging.info("Running primary agent...")
    try:
        if main_dataframe is None:
            return jsonify({"error": "No dataset has been uploaded."}), 400  # Handle case where no dataset is available

        agent_response_text = primary_agent.run(
            full_user_query,
            additional_args={"dataset": main_dataframe.copy()}  # Use the global variable
        )

        # Check if the response is a DataFrame
        if isinstance(agent_response_text, pd.DataFrame):
            agent_response_text = agent_response_text.to_string()  # Convert DataFrame to string

        logging.info(f"Agent raw response: {agent_response_text}")

        # --- Post-process agent response ---
        # Check for visualization indicators
        viz_indicators = [
            "visualization tool",
            "created plot",
            "generated plot",
            "plotted",
            "plot showing",
            "plots",
            "saved png:",
            "successfully saved",
            ".png"
        ]
        
        # Check for visualization indicators in the response (case-insensitive)
        is_viz_response = any(indicator.lower() in str(agent_response_text).lower() for indicator in viz_indicators)
        logging.info(f"Is visualization response: {is_viz_response}")
        
        # Handle visualization responses
        if is_viz_response:
            # Ensure the plot directory exists
            os.makedirs(PLOT_SAVE_DIRECTORY, exist_ok=True)
            
            # Get all PNG files from plot directory
            plot_files = []
            try:
                plot_files = [f for f in os.listdir(PLOT_SAVE_DIRECTORY) if f.endswith('.png')]
                # Sort by creation time (newest first)
                plot_files.sort(key=lambda x: os.path.getmtime(os.path.join(PLOT_SAVE_DIRECTORY, x)), reverse=True)
                logging.info(f"Found {len(plot_files)} plot files in {PLOT_SAVE_DIRECTORY}")
            except Exception as e:
                logging.error(f"Error listing plot directory: {e}")
            
            # Generate URLs for the plot files
            plot_urls = []
            for fname in plot_files:
                file_path = os.path.join(PLOT_SAVE_DIRECTORY, fname)
                if os.path.exists(file_path):
                    # Include plots created/modified in the last 5 minutes (more generous timeframe)
                    file_mtime = os.path.getmtime(file_path)
                    if time.time() - file_mtime < 300:  # 300 seconds = 5 minutes
                        plot_urls.append(f"/api/plots/{fname}")
                        logging.info(f"Added recent plot: {fname}")
                    else:
                        logging.debug(f"Skipping older plot: {fname}")
                else:
                    logging.warning(f"Plot file '{fname}' not found in {PLOT_SAVE_DIRECTORY}")
            
            # Clean up the response text for visualization responses
            if plot_urls:
                # Extract insights from the response while removing code blocks and technical details
                cleaned_response = ""
                
                # Get lines that don't contain code markers or technical output
                lines = agent_response_text.split('\n')
                in_code_block = False
                for line in lines:
                    # Skip code blocks and technical output
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    
                    if in_code_block:
                        continue
                        
                    # Skip lines with technical information
                    skip_patterns = [
                        "SAVED PNG:", 
                        "✓ Successfully saved", 
                        "Visualization completed",
                        "Dataset shape:",
                        "Columns:",
                        "Will save plots",
                        "Total PNG plots created:",
                        "Summary of PNG files",
                        "Visualization tool",
                        "visualization agent",
                        "import",
                        "matplotlib",
                        "plt.",
                        "Code:"
                    ]
                    
                    if any(pattern in line for pattern in skip_patterns):
                        continue
                    
                    # Keep meaningful insights
                    stripped_line = line.strip()
                    if stripped_line and not stripped_line.startswith('#'):
                        cleaned_response += line + "\n"
                
                # Create a more user-friendly response with plot descriptions
                plot_descriptions = {
                    'target_distribution.png': "distribution of the target variable (default payments)",
                    'correlation_heatmap.png': "correlation heatmap showing relationships between features",
                    'feature_distributions.png': "distributions of key numerical features",
                    'boxplots_by_target.png': "boxplots of features grouped by default status",
                    'feature_pairplot.png': "pairwise relationships between key features"
                }
                
                # Build a friendly response with plot descriptions
                friendly_response = "I've created the following visualizations for you:\n\n"
                for plot_url in plot_urls:
                    plot_name = os.path.basename(plot_url)
                    description = plot_descriptions.get(plot_name, plot_name)
                    friendly_response += f"- A {description}\n"
                
                # Add any extracted insights from the original response
                if cleaned_response.strip():
                    friendly_response += "\nKey insights:\n" + cleaned_response.strip()
                
                # Return the cleaned response and plot URLs
                return jsonify({
                    "response": friendly_response,
                    "plot_urls": plot_urls
                })
            else:
                # No plots found, return a simple error message
                return jsonify({
                    "response": "I tried to create visualizations but couldn't generate any plots. Please try a different request.",
                    "plot_urls": []
                })
        else:
            # Non-visualization response
            return jsonify({
                "response": agent_response_text,
                "plot_urls": []
            })
    except Exception as e:
        logging.error(f"Unhandled exception in /api/chat endpoint: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please check the backend logs."}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network if needed
    logging.info(f"Starting Flask development server on http://0.0.0.0:{FLASK_PORT}")
    logging.info(f"Accepting requests from frontend origin: {FRONTEND_URL}")
    # Turn off Flask's default reloader if it causes issues with H2O/Agent init
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True, use_reloader=False)