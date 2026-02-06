import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from supabase import create_client, Client

# --- CONFIGURATION ---
load_dotenv()
URL = os.environ.get("PROJECT_URL")
KEY = os.environ.get("API_KEY")

st.set_page_config(
    page_title="Bio-D-Scan Dashboard",
    page_icon="🐞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stDataFrame {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Supabase
@st.cache_resource
def init_connection():
    return create_client(URL, KEY)

supabase = init_connection()

# --- SIDEBAR ---
st.sidebar.title("🐞 Bio-D-Scan")
st.sidebar.markdown("---")

# Session filter
sessions = supabase.table('sessions').select('*').order('started_at', desc=True).limit(10).execute()
session_options = ["All Sessions"]
if sessions.data:
    for s in sessions.data:
        started = datetime.fromisoformat(s['started_at'].replace('Z', '+00:00'))
        label = f"{started.strftime('%b %d, %H:%M')} - {s['device_id']}"
        session_options.append((s['id'], label))

selected_session = st.sidebar.selectbox(
    "📅 Filter by Session",
    options=range(len(session_options)),
    format_func=lambda x: session_options[x] if x == 0 else session_options[x][1]
)

# Time range filter
time_range = st.sidebar.selectbox(
    "⏰ Time Range",
    ["Last Hour", "Last 24 Hours", "Last 7 Days", "All Time"]
)

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Powered by YOLO + Hailo AI")

# --- MAIN CONTENT ---
st.title("🐝 Real-Time Insect Monitoring Dashboard")

# Build query based on filters
query = supabase.table('tracks').select('*')

if selected_session != 0:
    query = query.eq('session_id', session_options[selected_session][0])

time_filter = datetime.now()
if time_range == "Last Hour":
    time_filter = datetime.now() - timedelta(hours=1)
elif time_range == "Last 24 Hours":
    time_filter = datetime.now() - timedelta(days=1)
elif time_range == "Last 7 Days":
    time_filter = datetime.now() - timedelta(days=7)
else:
    time_filter = None

if time_filter:
    query = query.gte('timestamp', time_filter.isoformat())

response = query.order('timestamp', desc=True).limit(100).execute()
data = response.data

if data:
    df = pd.DataFrame(data)
    
    # --- STATISTICS ROW ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_count = len(df)
        st.metric("🦋 Total Detections", total_count)
    
    with col2:
        unique_types = df['type'].nunique()
        st.metric("🏷️ Species Detected", unique_types)
    
    with col3:
        avg_confidence = df['confidence'].mean() * 100
        st.metric("📊 Avg. Confidence", f"{avg_confidence:.1f}%")
    
    with col4:
        if 'duration_seconds' in df.columns and df['duration_seconds'].notna().any():
            avg_duration = df['duration_seconds'].mean()
            st.metric("⏱️ Avg. Track Duration", f"{avg_duration:.1f}s")
        else:
            st.metric("⏱️ Avg. Track Duration", "N/A")
    
    st.markdown("---")
    
    # --- CHARTS ROW ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("🥧 Species Distribution")
        type_counts = df['type'].value_counts().reset_index()
        type_counts.columns = ['Species', 'Count']
        
        fig_pie = px.pie(
            type_counts,
            values='Count',
            names='Species',
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with chart_col2:
        st.subheader("📈 Detection Timeline")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.floor('H')
        timeline = df.groupby(['hour', 'type']).size().reset_index(name='count')
        
        fig_timeline = px.area(
            timeline,
            x='hour',
            y='count',
            color='type',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_timeline.update_layout(
            xaxis_title="Time",
            yaxis_title="Detections",
            height=350,
            legend_title="Species"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    st.markdown("---")
    
    # --- ENTRY/EXIT ANALYSIS ---
    if 'entry_point' in df.columns and df['entry_point'].notna().any():
        st.subheader("🧭 Movement Patterns")
        
        move_col1, move_col2 = st.columns(2)
        
        with move_col1:
            entry_counts = df['entry_point'].value_counts().reset_index()
            entry_counts.columns = ['Direction', 'Count']
            fig_entry = px.bar(
                entry_counts,
                x='Direction',
                y='Count',
                color='Direction',
                title="Entry Points"
            )
            st.plotly_chart(fig_entry, use_container_width=True)
        
        with move_col2:
            if 'exit_point' in df.columns and df['exit_point'].notna().any():
                exit_counts = df['exit_point'].value_counts().reset_index()
                exit_counts.columns = ['Direction', 'Count']
                fig_exit = px.bar(
                    exit_counts,
                    x='Direction',
                    y='Count',
                    color='Direction',
                    title="Exit Points"
                )
                st.plotly_chart(fig_exit, use_container_width=True)
    
    st.markdown("---")
    
    # --- RECENT DETECTIONS TABLE ---
    st.subheader("📋 Recent Detections")
    
    # Prepare display columns
    display_cols = ['image_url', 'type', 'confidence', 'timestamp']
    if 'entry_point' in df.columns:
        display_cols.append('entry_point')
    if 'exit_point' in df.columns:
        display_cols.append('exit_point')
    if 'distance_traveled' in df.columns:
        display_cols.append('distance_traveled')
    if 'duration_seconds' in df.columns:
        display_cols.append('duration_seconds')
    
    # Filter to available columns
    display_cols = [c for c in display_cols if c in df.columns]
    display_df = df[display_cols].copy()
    
    # Format columns
    if 'distance_traveled' in display_df.columns:
        display_df['distance_traveled'] = display_df['distance_traveled'].apply(
            lambda x: f"{x:.0f}px" if pd.notna(x) else "N/A"
        )
    if 'duration_seconds' in display_df.columns:
        display_df['duration_seconds'] = display_df['duration_seconds'].apply(
            lambda x: f"{x:.1f}s" if pd.notna(x) else "N/A"
        )
    
    st.dataframe(
        display_df,
        column_config={
            "image_url": st.column_config.ImageColumn(
                "Image", help="Captured image with trajectory"
            ),
            "type": st.column_config.TextColumn("Species"),
            "timestamp": st.column_config.DatetimeColumn(
                "Time", format="D MMM YYYY, h:mm a"
            ),
            "confidence": st.column_config.ProgressColumn(
                "Confidence", format="%.2f", min_value=0, max_value=1
            ),
            "entry_point": st.column_config.TextColumn("Entry"),
            "exit_point": st.column_config.TextColumn("Exit"),
            "distance_traveled": st.column_config.TextColumn("Distance"),
            "duration_seconds": st.column_config.TextColumn("Duration"),
        },
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # --- TRAJECTORY VIEWER ---
    st.markdown("---")
    st.subheader("🛤️ Trajectory Viewer")
    
    tracks_with_paths = df[df['path_points'].apply(lambda x: x is not None and len(x) > 1)] if 'path_points' in df.columns else pd.DataFrame()
    
    if not tracks_with_paths.empty:
        selected_track = st.selectbox(
            "Select a track to view trajectory",
            options=tracks_with_paths.index,
            format_func=lambda x: f"Track {tracks_with_paths.loc[x, 'tracker_id']} - {tracks_with_paths.loc[x, 'type']} ({tracks_with_paths.loc[x, 'timestamp'].strftime('%H:%M:%S')})"
        )
        
        if selected_track is not None:
            track_data = tracks_with_paths.loc[selected_track]
            path = track_data['path_points']
            
            # Create trajectory plot
            fig_traj = go.Figure()
            
            # Add path line
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            
            fig_traj.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                line=dict(color='yellow', width=3),
                marker=dict(size=6),
                name='Path'
            ))
            
            # Mark start (green) and end (red)
            fig_traj.add_trace(go.Scatter(
                x=[x_coords[0]],
                y=[y_coords[0]],
                mode='markers',
                marker=dict(size=15, color='green', symbol='circle'),
                name='Start'
            ))
            fig_traj.add_trace(go.Scatter(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='x'),
                name='End'
            ))
            
            fig_traj.update_layout(
                title=f"{track_data['type']} Trajectory",
                xaxis_title="X Position",
                yaxis_title="Y Position",
                yaxis=dict(autorange="reversed"),  # Match image coordinates
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_traj, use_container_width=True)
            
            # Show track details
            detail_col1, detail_col2, detail_col3 = st.columns(3)
            with detail_col1:
                st.info(f"**Entry:** {track_data.get('entry_point', 'N/A')}")
            with detail_col2:
                st.info(f"**Exit:** {track_data.get('exit_point', 'N/A')}")
            with detail_col3:
                dist = track_data.get('distance_traveled')
                st.info(f"**Distance:** {dist:.0f}px" if dist else "N/A")
    else:
        st.info("No trajectory data available. New detections will include paths.")

else:
    st.info("🔍 No insects detected yet. Start monitoring to see data here.")
    
    # Show empty state with instructions
    st.markdown("""
    ### Getting Started
    1. Start the detection script on your Raspberry Pi
    2. The dashboard will automatically update with new detections
    3. Use the sidebar to filter by session or time range
    """)