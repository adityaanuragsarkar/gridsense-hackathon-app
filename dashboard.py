# Save this as dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import time
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="GridSense AI - Neural Grid Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ELITE CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;500;700;900&display=swap');
    
    .main > div {
        padding-top: 2rem;
    }
    
    .main {
        background: #0a0e27;
        background-image: 
            radial-gradient(at 20% 30%, rgba(0, 242, 254, 0.08) 0px, transparent 50%),
            radial-gradient(at 80% 70%, rgba(79, 172, 254, 0.08) 0px, transparent 50%),
            radial-gradient(at 40% 80%, rgba(138, 43, 226, 0.05) 0px, transparent 50%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f35 100%);
        border-right: 2px solid #00f2fe;
        box-shadow: 5px 0 20px rgba(0, 242, 254, 0.3);
    }
    
    [data-testid="stSidebar"] h2 {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1.4rem;
        background: linear-gradient(120deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 1.5rem;
    }
    
    h1 {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 4rem !important;
        background: linear-gradient(120deg, #00f2fe 0%, #4facfe 50%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 4px;
        text-shadow: 0 0 40px rgba(0, 242, 254, 0.5);
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        color: #8b9dc3;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 2rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 300;
    }
    
    h2 {
        font-family: 'Orbitron', sans-serif;
        color: #00f2fe;
        font-weight: 700;
        font-size: 1.6rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 2rem;
    }
    
    h3 {
        font-family: 'Rajdhani', sans-serif;
        color: #4facfe;
        font-weight: 600;
        font-size: 1.3rem !important;
    }
    
    [data-testid="stMetricValue"] {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .tech-card {
        background: linear-gradient(135deg, rgba(15, 20, 25, 0.9) 0%, rgba(26, 31, 53, 0.9) 100%);
        border: 1px solid rgba(0, 242, 254, 0.3);
        border-radius: 16px;
        padding: 1.8rem;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.05),
            0 0 0 1px rgba(0, 242, 254, 0.1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    
    .tech-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f2fe, transparent);
        animation: scan 3s ease-in-out infinite;
    }
    
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .info-box {
        background: rgba(79, 172, 254, 0.1);
        border-left: 4px solid #4facfe;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Rajdhani', sans-serif;
        color: #8b9dc3;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .info-box strong {
        color: #00f2fe;
    }
    
    .stAlert {
        font-family: 'Rajdhani', sans-serif;
        border-radius: 12px;
        border-left: 4px solid #00f2fe;
        background: rgba(0, 242, 254, 0.05);
        backdrop-filter: blur(10px);
        font-size: 1rem;
    }
    
    .stButton > button {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(79, 172, 254, 0.6);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(120deg, #4facfe 0%, #00f2fe 100%);
    }
    
    [data-testid="stSidebar"] .stAlert {
        background: rgba(0, 242, 254, 0.08);
        border-left: 3px solid #00f2fe;
        font-size: 0.9rem;
    }
    
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.5), transparent);
        margin: 2rem 0;
    }
    
    .grid-container {
        background: linear-gradient(135deg, rgba(10, 14, 39, 0.95) 0%, rgba(15, 20, 25, 0.95) 100%);
        border: 2px solid rgba(0, 242, 254, 0.3);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 
            0 20px 60px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .grid-container::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 242, 254, 0.03) 1px, transparent 1px);
        background-size: 30px 30px;
        animation: grid-move 20s linear infinite;
        pointer-events: none;
    }
    
    @keyframes grid-move {
        0% { transform: translate(0, 0); }
        100% { transform: translate(30px, 30px); }
    }
    
    .terminal-log {
        background: #0a0e1a;
        border: 1px solid #00f2fe;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'Courier New', monospace;
        color: #00f2fe;
        font-size: 0.85rem;
        max-height: 150px;
        overflow-y: auto;
        box-shadow: inset 0 0 20px rgba(0, 242, 254, 0.1);
    }
    
    .help-icon {
        background: rgba(79, 172, 254, 0.2);
        border: 1px solid #4facfe;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: #00f2fe;
        cursor: help;
        margin-left: 8px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0f1419;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00f2fe 0%, #4facfe 100%);
        border-radius: 4px;
    }
    
    .expander-header {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        color: #00f2fe;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model ---
try:
    with open('grid_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ SYSTEM ERROR: Neural network model not found. Initialize training sequence.")
    st.stop()

# --- Grid Functions ---
def create_grid():
    G = nx.Graph()
    nodes = {
        'Substation': {'pos': (0, 3), 'icon': '⚡', 'type': 'source', 'critical': 'high'},
        'Hospital': {'pos': (2, 5), 'icon': '🏥', 'type': 'critical', 'critical': 'high'},
        'Factory': {'pos': (2, 1), 'icon': '🏭', 'type': 'industrial', 'critical': 'medium'},
        'DataCenter': {'pos': (4, 3), 'icon': '💾', 'type': 'tech', 'critical': 'high'},
        'Homes': {'pos': (6, 3), 'icon': '🏘️', 'type': 'residential', 'critical': 'medium'}
    }
    for node, attrs in nodes.items():
        G.add_node(node, **attrs, status='Healthy', load=50, voltage=230)
    
    edges = [
        ('Substation', 'Hospital', 100), ('Substation', 'Factory', 100),
        ('Hospital', 'DataCenter', 80), ('Factory', 'DataCenter', 80),
        ('DataCenter', 'Homes', 120)
    ]
    for u, v, capacity in edges:
        G.add_edge(u, v, capacity=capacity, status='ok', current_flow=0)
    return G

def create_advanced_3d_grid_viz(G, predictions, reroute_path=None):
    """Elite 3D power flow visualization"""
    
    pos = {node: G.nodes[node]['pos'] for node in G.nodes()}
    prediction_map = {p['Node']: p['Probability'] for p in predictions}
    
    # Create 3D coordinates with height based on load
    node_x, node_y, node_z = [], [], []
    node_text, node_color, node_size = [], [], []
    
    status_colors = {
        'Healthy': '#10b981',
        'Under Stress': '#f59e0b',
        'Failure Predicted!': '#ef4444'
    }
    
    for node in G.nodes():
        x, y = pos[node]
        load = G.nodes[node]['load']
        z = load / 20  # Height represents load intensity
        
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        status = G.nodes[node]['status']
        prob = prediction_map.get(node, 0)
        icon = G.nodes[node]['icon']
        voltage = G.nodes[node]['voltage']
        
        # Friendly hover text
        node_text.append(
            f"<b>{icon} {node}</b><br>"
            f"━━━━━━━━━━━━━━━<br>"
            f"<b>Status:</b> {status}<br>"
            f"<b>Power Load:</b> {load} MW<br>"
            f"<b>Voltage:</b> {voltage}V<br>"
            f"<b>Failure Risk:</b> {prob:.1%}<br>"
            f"<b>Priority:</b> {G.nodes[node]['critical'].upper()}"
        )
        
        node_color.append(status_colors[status])
        node_size.append(15 if status == 'Failure Predicted!' else (12 if status == 'Under Stress' else 10))
    
    # Node trace
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=node_size,
            color=node_color,
            symbol='diamond',
            line=dict(color='rgba(255, 255, 255, 0.6)', width=2),
            opacity=0.95
        ),
        text=[G.nodes[node]['icon'] for node in G.nodes()],
        textfont=dict(size=20, color='white'),
        textposition='middle center',
        hovertext=node_text,
        hoverinfo='text',
        name='Grid Nodes'
    )
    
    # Edge traces with power flow
    edge_traces = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        z0 = G.nodes[u]['load'] / 20
        z1 = G.nodes[v]['load'] / 20
        
        capacity = G.edges[u, v]['capacity']
        line_load = (G.nodes[u]['load'] + G.nodes[v]['load']) / 2
        load_factor = line_load / capacity
        
        # Color based on line stress
        if load_factor > 0.8:
            color = 'rgba(239, 68, 68, 0.8)'  # Red - Critical
            width = 6
        elif load_factor > 0.6:
            color = 'rgba(245, 158, 11, 0.8)'  # Yellow - Warning
            width = 4
        else:
            color = 'rgba(93, 173, 226, 0.6)'  # Blue - Normal
            width = 3
        
        edge_trace = go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color=color, width=width),
            hoverinfo='text',
            hovertext=f"{u} ↔ {v}<br>Capacity: {capacity} MW<br>Load: {load_factor*100:.1f}%",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Highlight reroute path
    if reroute_path:
        for i in range(len(reroute_path) - 1):
            x0, y0 = pos[reroute_path[i]]
            x1, y1 = pos[reroute_path[i + 1]]
            z0 = G.nodes[reroute_path[i]]['load'] / 20
            z1 = G.nodes[reroute_path[i + 1]]['load'] / 20
            
            reroute_trace = go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode='lines',
                line=dict(color='#00f2fe', width=8, dash='dash'),
                hoverinfo='text',
                hovertext=f"⚡ REROUTE: {reroute_path[i]} → {reroute_path[i+1]}",
                showlegend=False
            )
            edge_traces.append(reroute_trace)
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor='rgba(0, 242, 254, 0.1)', showticklabels=False, title=''),
            yaxis=dict(showgrid=True, gridcolor='rgba(0, 242, 254, 0.1)', showticklabels=False, title=''),
            zaxis=dict(showgrid=True, gridcolor='rgba(0, 242, 254, 0.1)', showticklabels=False, title='Load (MW)'),
            bgcolor='rgba(10, 14, 39, 0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600
    )
    
    return fig

def create_heatmap_viz(G, predictions):
    """Real-time threat heatmap"""
    prediction_map = {p['Node']: p['Probability'] for p in predictions}
    
    nodes = list(G.nodes())
    risks = [prediction_map.get(node, 0) * 100 for node in nodes]
    
    fig = go.Figure(data=go.Bar(
        x=nodes,
        y=risks,
        marker=dict(
            color=risks,
            colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']],
            line=dict(color='rgba(0, 242, 254, 0.5)', width=2)
        ),
        text=[f"{r:.1f}%" for r in risks],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Failure Risk: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='AI THREAT ASSESSMENT',
            font=dict(family='Orbitron', size=16, color='#00f2fe'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='', 
            showgrid=False, 
            color='#6b7280',
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title='Failure Risk (%)', 
            showgrid=True, 
            gridcolor='rgba(0, 242, 254, 0.1)',
            color='#6b7280',
            range=[0, 100]
        ),
        plot_bgcolor='rgba(10, 14, 39, 0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=50, r=20, t=60, b=40),
        font=dict(family='Rajdhani', color='#8b9dc3')
    )
    
    return fig

def find_reroute_path(G, failed_node):
    temp_G = G.copy()
    if temp_G.has_node(failed_node): 
        temp_G.remove_node(failed_node)
    
    if nx.has_path(temp_G, 'Substation', 'Homes'):
        return nx.shortest_path(temp_G, 'Substation', 'Homes')
    return None

# --- Session State ---
if 'grid' not in st.session_state:
    st.session_state.grid = create_grid()
if 'total_failures_prevented' not in st.session_state:
    st.session_state.total_failures_prevented = 0
if 'energy_saved' not in st.session_state:
    st.session_state.energy_saved = 0
if 'uptime' not in st.session_state:
    st.session_state.uptime = 100.0
if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'show_tutorial' not in st.session_state:
    st.session_state.show_tutorial = True

G = st.session_state.grid
nodes_list = list(G.nodes())

# --- HEADER ---
st.markdown("# GRIDSENSE NEURAL CONTROL")
st.markdown("<p class='subtitle'>⚡ Autonomous Power Grid Management System</p>", unsafe_allow_html=True)

# --- TUTORIAL/HELP SECTION ---
if st.session_state.show_tutorial:
    with st.expander("📚 **HOW TO USE THIS DASHBOARD** - Click to expand", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🎮 **Controls (Left Sidebar)**
            
            **⛈️ Storm Mode**  
            Toggle this to simulate extreme weather conditions that stress the power grid.
            
            **🌡️ Temperature Slider**  
            Adjust ambient temperature. Higher temps = higher AC load = more grid stress.
            
            **⏱️ Refresh Rate**  
            Control how fast the simulation updates (1 = fastest, 5 = slowest).
            """)
        
        with col2:
            st.markdown("""
            ### 🗺️ **Understanding the 3D Grid**
            
            **Node Colors:**
            - 🟢 Green = Healthy (safe)
            - 🟡 Yellow = Under Stress (warning)
            - 🔴 Red = Failure Predicted (critical)
            
            **Line Colors:**
            - Blue = Normal load
            - Yellow = High load (60-80%)
            - Red = Critical load (>80%)
            
            **Dashed Cyan Line = AI Reroute Path** (auto-healing in action!)
            """)
        
        with col3:
            st.markdown("""
            ### 💡 **What's Happening?**
            
            The AI constantly monitors all power nodes and predicts failures **before they happen**.
            
            When failure risk exceeds 70%, the system automatically reroutes power through alternate paths to prevent blackouts.
            
            **Green AI Impact:**  
            - Saves energy by preventing outages
            - Reduces carbon emissions
            - Runs on CPU (no GPU needed!)
            """)
        
        if st.button("✓ Got it! Hide this tutorial", use_container_width=True):
            st.session_state.show_tutorial = False
            st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## ⚙️ CONTROL PANEL")
    st.markdown("---")
    
    st.markdown("### 🎛️ Simulation Settings")
    is_storm = st.checkbox("⛈️ Storm Mode", value=False, help="Simulates extreme weather conditions")
    manual_temp = st.slider("🌡️ Temperature (°C)", 15.0, 50.0, 25.0, 0.5, 
                            help="Higher temperature increases power demand")
    simulation_speed = st.select_slider("⏱️ Refresh Rate", options=[1, 2, 3, 5], value=2,
                                       help="How fast the dashboard updates")
    
    st.markdown("---")
    st.markdown("## 📊 LIVE STATISTICS")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Failures Prevented", st.session_state.total_failures_prevented, 
                 delta="AI Active", help="Number of blackouts prevented by AI")
    with col2:
        st.metric("System Uptime", f"{st.session_state.uptime:.1f}%", 
                 delta="Optimal", help="Percentage of nodes functioning normally")
    
    st.metric("Energy Saved", f"{st.session_state.energy_saved:.1f} MWh", 
             delta="+2.3 MWh", help="Energy conserved through intelligent routing")
    st.metric("AI Predictions Made", st.session_state.total_predictions, 
             delta="Real-time", help="Total number of AI predictions run")
    
    st.markdown("---")
    st.markdown("## 🎯 HACKATHON INFO")
    st.info("""
    **GreenMind 2025**
    
    🌱 Sustainable AI Solution  
    💻 CPU-Only Inference  
    ⚡ Edge-Ready Deployment  
    🏆 Smart Cities Track
    
    This dashboard demonstrates how AI can make power grids more efficient and reduce carbon emissions.
    """)
    
    if st.button("🔄 Reset Statistics", use_container_width=True):
        st.session_state.total_failures_prevented = 0
        st.session_state.energy_saved = 0
        st.session_state.log_messages = []
        st.session_state.total_predictions = 0
        st.success("Statistics reset!")

# --- MAIN DASHBOARD ---
viz_col, stats_col = st.columns([2, 1])

with viz_col:
    graph_placeholder = st.empty()

with stats_col:
    heatmap_placeholder = st.empty()
    log_placeholder = st.empty()

st.markdown("---")

# Quick legend
legend_col1, legend_col2, legend_col3 = st.columns(3)
with legend_col1:
    st.markdown("🟢 **Healthy** = Operating normally")
with legend_col2:
    st.markdown("🟡 **Under Stress** = Warning - monitoring closely")
with legend_col3:
    st.markdown("🔴 **Failure Predicted** = Critical - AI rerouting power")

st.markdown("---")
metrics_row = st.container()
st.markdown("---")
insights_row = st.container()

# --- SIMULATION LOOP ---
iteration = 0
while True:
    iteration += 1
    st.session_state.total_predictions += 1
    failed_node_this_loop = None
    reroute_path = None
    
    # 1. Grid simulation
    for node in G.nodes():
        base_load = np.random.randint(30, 90)
        if is_storm:
            base_load = min(100, base_load + np.random.randint(10, 30))
        G.nodes[node]['load'] = base_load
        G.nodes[node]['voltage'] = np.random.randint(220, 240)

    # 2. AI Predictions
    predictions = []
    max_risk_node = None
    max_risk = 0
    
    for i, node in enumerate(nodes_list):
        features = pd.DataFrame([[
            datetime.now().hour, i, G.nodes[node]['load'],
            manual_temp, 1 if is_storm else 0
        ]], columns=['hour', 'node_id', 'load', 'temp', 'storm_active'])
        
        prob = model.predict_proba(features)[0][1]
        predictions.append({'Node': node, 'Probability': prob})
        
        if prob > max_risk:
            max_risk = prob
            max_risk_node = node
        
        if prob > 0.7:
            G.nodes[node]['status'] = 'Failure Predicted!'
            failed_node_this_loop = node
        elif prob > 0.4:
            G.nodes[node]['status'] = 'Under Stress'
        else:
            G.nodes[node]['status'] = 'Healthy'

    # 3. Self-healing logic
    if failed_node_this_loop:
        reroute_path = find_reroute_path(G, failed_node_this_loop)
        if reroute_path:
            st.session_state.total_failures_prevented += 1
            st.session_state.energy_saved += np.random.uniform(1.5, 3.5)
            log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ⚡ ALERT: Power rerouted around {failed_node_this_loop}"
            st.session_state.log_messages.insert(0, log_msg)
            if len(st.session_state.log_messages) > 5:
                st.session_state.log_messages.pop()
            st.toast(f"✅ BLACKOUT PREVENTED: AI rerouted power around {failed_node_this_loop}", icon="⚡")