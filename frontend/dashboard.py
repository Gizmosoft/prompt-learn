import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import List, Dict, Any
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
# Default API URL (can be overridden via .env file or sidebar)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="PromptLearn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Override Streamlit's default styles for main-header */
    .main-header,
    .st-emotion-cache-467cry .main-header,
    [class*="st-emotion-cache"] .main-header {
        font-size: 35px !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Override Streamlit's paragraph styles for main-header */
    .main-header p,
    .st-emotion-cache-467cry .main-header p,
    [class*="st-emotion-cache"] .main-header p {
        font-size: 35px !important;
        font-weight: bold !important;
        color: #1f77b4 !important;
        margin-bottom: 1rem !important;
    }
    .create-button {
        font-size: 0.9rem;
        padding: 0.4rem 0.8rem;
    }
    .experiment-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-running {
        color: #ffc107;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=5)  # Cache for 5 seconds
def fetch_experiments(api_base_url: str) -> List[Dict[str, Any]]:
    """Fetch all experiments from the API"""
    try:
        response = requests.get(f"{api_base_url}/api/experiments", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch experiments: {e}")
        return []


@st.cache_data(ttl=5)
def fetch_experiment_details(experiment_id: str, api_base_url: str) -> Dict[str, Any]:
    """Fetch detailed information for a specific experiment"""
    try:
        response = requests.get(
            f"{api_base_url}/api/experiments/{experiment_id}",
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch experiment details: {e}")
        return None


def create_experiment(api_base_url: str, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new experiment via POST API"""
    try:
        response = requests.post(
            f"{api_base_url}/api/experiments/run",
            json=experiment_data,
            timeout=300  # 5 minutes timeout for long-running experiments
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to create experiment: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                st.error(f"Error details: {error_detail}")
            except:
                st.error(f"Error response: {e.response.text}")
        return None


def format_datetime(dt_str: str) -> str:
    """Format datetime string for display"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return dt_str


def get_status_color(status: str) -> str:
    """Get color class for status"""
    status_map = {
        "success": "status-success",
        "error": "status-error",
        "running": "status-running"
    }
    return status_map.get(status.lower(), "")


def display_experiment_summary(experiment: Dict[str, Any]):
    """Display summary metrics for an experiment"""
    results = experiment.get("results", [])
    if not results:
        st.info("No results available for this experiment.")
        return
    
    # Calculate summary metrics
    total_variants = len(results)
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = sum(1 for r in results if r.get("status") == "error")
    
    # Token and cost metrics
    total_input_tokens = sum(
        r.get("tokens", {}).get("input", 0) or 0 
        for r in results if r.get("tokens")
    )
    total_output_tokens = sum(
        r.get("tokens", {}).get("output", 0) or 0 
        for r in results if r.get("tokens")
    )
    total_cost = sum(
        r.get("cost_usd", 0) or 0 
        for r in results
    )
    
    # Latency metrics
    latencies = [
        r.get("latency_ms") 
        for r in results 
        if r.get("latency_ms") is not None
    ]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    min_latency = min(latencies) if latencies else 0
    max_latency = max(latencies) if latencies else 0
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Variants", total_variants)
    
    with col2:
        st.metric("Successful", successful, delta=None if failed == 0 else f"-{failed} failed")
    
    with col3:
        st.metric("Avg Latency", f"{avg_latency:.0f} ms" if avg_latency > 0 else "N/A")
    
    with col4:
        st.metric("Total Cost", f"${total_cost:.6f}" if total_cost > 0 else "$0.00")
    
    with col5:
        st.metric("Total Tokens", f"{total_input_tokens + total_output_tokens:,}" if (total_input_tokens + total_output_tokens) > 0 else "0")


def display_variant_comparison(results: List[Dict[str, Any]]):
    """Display comparison charts for variants"""
    if not results:
        return
    
    # Prepare data for charts
    variant_data = []
    for r in results:
        variant_data.append({
            "Variant": r.get("variant_label", "Unknown"),
            "Latency (ms)": r.get("latency_ms") or 0,
            "Cost (USD)": r.get("cost_usd") or 0,
            "Total Tokens": (
                r.get("tokens", {}).get("total", 0) or 0
                if r.get("tokens") else 0
            ),
            "Status": r.get("status", "unknown")
        })
    
    df = pd.DataFrame(variant_data)
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        if df["Latency (ms)"].sum() > 0:
            fig_latency = px.bar(
                df,
                x="Variant",
                y="Latency (ms)",
                title="Latency Comparison",
                color="Status",
                color_discrete_map={
                    "success": "#28a745",
                    "error": "#dc3545",
                    "running": "#ffc107"
                }
            )
            fig_latency.update_layout(showlegend=False)
            st.plotly_chart(fig_latency, use_container_width=True, config={"displayModeBar": False})
    
    with col2:
        if df["Cost (USD)"].sum() > 0:
            fig_cost = px.bar(
                df,
                x="Variant",
                y="Cost (USD)",
                title="Cost Comparison",
                color="Status",
                color_discrete_map={
                    "success": "#28a745",
                    "error": "#dc3545",
                    "running": "#ffc107"
                }
            )
            fig_cost.update_layout(showlegend=False)
            st.plotly_chart(fig_cost, use_container_width=True, config={"displayModeBar": False})
    
    # Token usage chart
    if df["Total Tokens"].sum() > 0:
        st.subheader("Token Usage")
        col1, col2 = st.columns(2)
        
        with col1:
            # Input vs Output tokens
            input_tokens = [
                r.get("tokens", {}).get("input", 0) or 0 
                for r in results if r.get("tokens")
            ]
            output_tokens = [
                r.get("tokens", {}).get("output", 0) or 0 
                for r in results if r.get("tokens")
            ]
            variants = [
                r.get("variant_label", "Unknown") 
                for r in results if r.get("tokens")
            ]
            
            if input_tokens or output_tokens:
                fig_tokens = go.Figure(data=[
                    go.Bar(name='Input Tokens', x=variants, y=input_tokens),
                    go.Bar(name='Output Tokens', x=variants, y=output_tokens)
                ])
                fig_tokens.update_layout(
                    barmode='stack',
                    title="Input vs Output Tokens",
                    xaxis_title="Variant",
                    yaxis_title="Tokens"
                )
                st.plotly_chart(fig_tokens, use_container_width=True, config={"displayModeBar": False})
        
        with col2:
            # Total tokens bar chart
            if df["Total Tokens"].sum() > 0:
                fig_total = px.bar(
                    df,
                    x="Variant",
                    y="Total Tokens",
                    title="Total Token Usage",
                    color="Status",
                    color_discrete_map={
                        "success": "#28a745",
                        "error": "#dc3545",
                        "running": "#ffc107"
                    }
                )
                fig_total.update_layout(showlegend=False)
                st.plotly_chart(fig_total, use_container_width=True, config={"displayModeBar": False})


def display_variant_details(results: List[Dict[str, Any]]):
    """Display detailed information for each variant"""
    st.subheader("Variant Details")
    
    for idx, result in enumerate(results):
        with st.expander(
            f"**{result.get('variant_label', 'Unknown')}** - "
            f"Status: {result.get('status', 'unknown').upper()}",
            expanded=False
        ):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Metrics**")
                if result.get("latency_ms"):
                    st.write(f"⏱️ Latency: {result.get('latency_ms')} ms")
                if result.get("cost_usd"):
                    st.write(f"💰 Cost: ${result.get('cost_usd'):.6f}")
                if result.get("metrics_confidence"):
                    st.write(f"📊 Confidence: {result.get('metrics_confidence')}")
            
            with col2:
                st.write("**Token Usage**")
                tokens = result.get("tokens")
                if tokens:
                    if tokens.get("input"):
                        st.write(f"📥 Input: {tokens.get('input'):,}")
                    if tokens.get("output"):
                        st.write(f"📤 Output: {tokens.get('output'):,}")
                    if tokens.get("total"):
                        st.write(f"📊 Total: {tokens.get('total'):,}")
                else:
                    st.write("No token data available")
            
            with col3:
                st.write("**Links**")
                if result.get("trace_url"):
                    st.markdown(f"[🔗 View Trace]({result.get('trace_url')})")
                else:
                    st.write("No trace URL available")
            
            # Response text
            if result.get("response_text"):
                st.write("**Response:**")
                st.text_area(
                    "Response Text",
                    value=result.get("response_text"),
                    height=150,
                    key=f"response_{idx}",
                    disabled=True
                )
            
            # Error message if any
            if result.get("status") == "error" and result.get("error_message"):
                st.error(f"❌ Error: {result.get('error_message')}")


def render_create_experiment_page(current_api_url: str):
    """Render the create experiment page"""
    st.markdown('<p class="main-header">📊 PromptLearn</p>', unsafe_allow_html=True)
    st.markdown("### Create New Experiment")
    
    # Initialize num_variants in session state if not exists
    if "num_variants" not in st.session_state:
        st.session_state.num_variants = 1
    
    with st.form("create_experiment_form"):
        # Basic experiment info
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_name = st.text_input(
                "Experiment Name",
                placeholder="e.g., Summarization Test",
                help="Optional: A human-readable name for this experiment"
            )
        
        with col2:
            model = st.selectbox(
                "Model",
                options=["gemini-2.5-flash", "gemini-2.5-pro"],
                index=0,
                help="The LLM model to use"
            )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Sampling temperature (0.0 = deterministic, higher = more creative)"
        )
        
        st.divider()
        st.markdown("### Input Text")
        
        input_text = st.text_area(
            "Input Text",
            placeholder="Enter the text that will be used in your prompt templates",
            height=150,
            help="This text will be available as {text} in your prompt templates"
        )
        
        # Always use "text" as the variable key
        variables = {"text": input_text} if input_text else {}
        
        st.divider()
        st.markdown("### Prompt Templates")
        st.caption("Define one or more prompt templates to test. Use {text} to reference the input text.")
        
        # Number of variants input (inside form, uses session state for persistence)
        num_variants = st.number_input(
            "Number of Variants",
            min_value=1,
            max_value=10,
            value=st.session_state.num_variants,
            step=1,
            key="num_variants_input"
        )
        
        # Update session state when number changes
        if num_variants != st.session_state.num_variants:
            st.session_state.num_variants = num_variants
        
        variants = []
        for i in range(st.session_state.num_variants):
            with st.expander(f"Variant {i+1}", expanded=True):
                col1, col2 = st.columns([1, 3])
                with col1:
                    variant_label = st.text_input(
                        "Label",
                        value=f"P{i+1}",
                        key=f"variant_label_{i}",
                        placeholder="e.g., P1, P2"
                    )
                with col2:
                    variant_notes = st.text_input(
                        "Notes (Optional)",
                        key=f"variant_notes_{i}",
                        placeholder="Optional notes about this variant"
                    )
                
                variant_template = st.text_area(
                    "Prompt Template",
                    key=f"variant_template_{i}",
                    placeholder="Enter your prompt template here. Use {text} to reference the input text.",
                    height=150,
                    help="Example: Summarize the following text:\n{text}"
                )
                
                # Always collect the variant data (even if empty, we'll validate later)
                variants.append({
                    "label": variant_label,
                    "template": variant_template,
                    "notes": variant_notes if variant_notes else None
                })
        
        submitted = st.form_submit_button("🚀 Create Experiment", use_container_width=True)
        
        if submitted:
            # Validation
            if not input_text or not input_text.strip():
                st.error("Please enter input text.")
                return
            
            # Filter out empty variants and validate
            valid_variants = [
                v for v in variants 
                if v.get("label") and v.get("label").strip() and v.get("template") and v.get("template").strip()
            ]
            
            if not valid_variants:
                st.error("Please define at least one prompt variant with both label and template.")
                return
            
            # Check if templates use {text} correctly
            invalid_vars = []
            for variant in valid_variants:
                template_vars = re.findall(r'\{(\w+)\}', variant["template"])
                for var in template_vars:
                    if var != "text":
                        invalid_vars.append(f"{var} (used in {variant['label']}) - only {{text}} is supported")
            
            if invalid_vars:
                st.error(f"Invalid variables in templates: {', '.join(set(invalid_vars))}")
                return
            
            # Prepare experiment data
            experiment_data = {
                "name": experiment_name if experiment_name else None,
                "model": model,
                "temperature": temperature,
                "variables": variables,
                "variants": valid_variants
            }
            
            # Show progress
            with st.spinner("Creating experiment... This may take a while as we run all variants."):
                result = create_experiment(current_api_url, experiment_data)
            
            if result:
                st.success("✅ Experiment created successfully!")
                st.json(result)
                
                # Clear cache and redirect
                st.cache_data.clear()
                st.session_state.page = "home"
                st.rerun()


def render_home_page(current_api_url: str):
    """Render the main home page with experiment list"""
    # Header with create button
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown('<p class="main-header">📊 PromptLearn</p>', unsafe_allow_html=True)
    with col2:
        if st.button("➕ Create an Experiment", use_container_width=False, type="primary"):
            st.session_state.page = "create"
            st.rerun()
    
    # Fetch experiments
    experiments = fetch_experiments(current_api_url)
    
    if not experiments:
        st.warning("No experiments found. Make sure the API is running and accessible.")
        st.info(f"Trying to connect to: {current_api_url}")
        return
    
    # Experiment selection
    st.subheader("Select an Experiment")
    
    # Create experiment selector
    experiment_options = {
        f"{exp.get('name', 'Unnamed')} ({exp.get('experiment_id', '')[:8]}...) - {format_datetime(exp.get('created_at', ''))}": exp.get('experiment_id')
        for exp in experiments
    }
    
    selected_label = st.selectbox(
        "Choose an experiment to view details:",
        options=list(experiment_options.keys()),
        index=0
    )
    
    selected_id = experiment_options[selected_label]
    
    # Fetch and display experiment details
    experiment_details = fetch_experiment_details(selected_id, current_api_url)
    
    if not experiment_details:
        st.error("Failed to load experiment details.")
        return
    
    # Display experiment header
    st.divider()
    st.markdown("### Experiment Details")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.write("**Experiment ID**")
        st.code(experiment_details.get("experiment_id", "N/A"), language=None)
    
    with col2:
        st.write("**Name**")
        st.write(experiment_details.get("name") or "Unnamed")
    
    with col3:
        st.write("**Model**")
        st.write(experiment_details.get("model", "N/A"))
    
    with col4:
        st.write("**Temperature**")
        st.write(experiment_details.get("temperature", "N/A"))
    
    st.write("**Created At:**", format_datetime(experiment_details.get("created_at", "")))
    
    # Display summary metrics
    st.divider()
    st.markdown("### Summary Metrics")
    display_experiment_summary(experiment_details)
    
    # Display comparison charts
    st.divider()
    st.markdown("### Variant Comparison")
    display_variant_comparison(experiment_details.get("results", []))
    
    # Display detailed variant information
    st.divider()
    display_variant_details(experiment_details.get("results", []))
    
    # All experiments table
    st.divider()
    st.markdown("### All Experiments")
    
    # Create a summary table
    experiments_df = pd.DataFrame([
        {
            "Name": exp.get("name") or "Unnamed",
            "ID": exp.get("experiment_id", "")[:8] + "...",
            "Model": exp.get("model", "N/A"),
            "Temperature": exp.get("temperature", "N/A"),
            "Created": format_datetime(exp.get("created_at", ""))
        }
        for exp in experiments
    ])
    
    st.dataframe(
        experiments_df,
        use_container_width=True,
        hide_index=True
    )


def main():
    # Initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "home"
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Initialize session state for API URL
        if "api_base_url" not in st.session_state:
            st.session_state.api_base_url = API_BASE_URL
        
        api_url = st.text_input(
            "API Base URL",
            value=st.session_state.api_base_url,
            help="Base URL for the FastAPI backend (e.g., http://localhost:8000)"
        )
        
        if api_url != st.session_state.api_base_url:
            st.session_state.api_base_url = api_url
            st.cache_data.clear()
            st.rerun()
        
        st.divider()
        st.info(f"📡 Connecting to:\n`{st.session_state.api_base_url}`")
        st.divider()
        
        # Navigation
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = "home"
            st.cache_data.clear()
            st.rerun()
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Use session state API URL
    current_api_url = st.session_state.api_base_url
    
    # Render appropriate page
    if st.session_state.page == "create":
        render_create_experiment_page(current_api_url)
    else:
        render_home_page(current_api_url)


if __name__ == "__main__":
    main()

