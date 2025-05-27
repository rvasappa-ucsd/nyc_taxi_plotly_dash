import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_mantine_components as dmc
from dash_iconify import DashIconify
import warnings
import os
warnings.filterwarnings('ignore')

# Initialize app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Global data storage
taxi_data = None
multi_year_data = None

# Zone mapping
def create_zone_mapping():
    zone_mapping = {}
    famous_locations = {
        1: "Newark Airport", 4: "Alphabet City", 12: "Battery Park", 
        13: "Battery Park City", 43: "Central Park", 45: "Chinatown",
        79: "East Village", 87: "Financial District North", 88: "Financial District South",
        90: "Flatiron", 100: "Garment District", 103: "Gramercy",
        107: "Greenwich Village North", 113: "Greenwich Village South",
        114: "Harlem", 116: "Hell's Kitchen North", 120: "Hell's Kitchen South",
        127: "Kips Bay", 148: "Little Italy/NoLiTa", 151: "Lower East Side",
        158: "Midtown Center", 161: "Midtown East", 162: "Midtown North",
        163: "Midtown South", 166: "Murray Hill", 186: "Soho",
        209: "Times Sq/Theatre District", 224: "Union Sq",
        229: "Upper East Side North", 230: "Upper East Side South",
        231: "Upper West Side North", 232: "Upper West Side South",
        237: "West Chelsea/Hudson Yards", 238: "West Village"
    }
    
    for i in range(1, 265):
        if i <= 68:
            borough = "Manhattan"
        elif i <= 108:
            borough = "Brooklyn"
        elif i <= 163:
            borough = "Queens"
        elif i <= 226:
            borough = "Bronx"
        else:
            borough = "Staten Island"
            
        zone_name = famous_locations.get(i, f"{borough} Zone {i}")
        zone_mapping[i] = {
            'zone': zone_name,
            'borough': borough,
            'service_zone': 'Yellow Zone'
        }
    
    return zone_mapping

ZONE_MAPPING = create_zone_mapping()

def safe_zone_lookup(location_id, key, default):
    try:
        location_id = int(location_id) if pd.notna(location_id) else 0
        zone_info = ZONE_MAPPING.get(location_id, {})
        return zone_info.get(key, default) if isinstance(zone_info, dict) else default
    except:
        return default

def safe_percentage(numerator, denominator):
    try:
        num = float(numerator) if pd.notna(numerator) else 0
        den = float(denominator) if pd.notna(denominator) else 0
        return round((num / den * 100), 2) if den > 0 else 0.0
    except:
        return 0.0

def generate_sample_data(years, months):
    try:
        np.random.seed(42)
        all_data = []
        
        seasonal_multipliers = {
            1: 0.85, 2: 0.80, 3: 0.95, 4: 1.05, 5: 1.10, 6: 1.15,
            7: 1.05, 8: 1.00, 9: 1.10, 10: 1.15, 11: 1.20, 12: 1.25
        }
        
        yoy_growth = {
            2015: 0.75, 2016: 0.80, 2017: 0.85, 2018: 0.90, 2019: 0.95,
            2020: 0.70, 2021: 0.80, 2022: 1.0, 2023: 1.08, 2024: 1.12
        }
        
        zones = list(range(1, 265))
        base_records = 4000
        
        for year in years:
            for month in months:
                seasonal_factor = seasonal_multipliers.get(month, 1.0)
                growth_factor = yoy_growth.get(year, 1.0)
                records_count = int(base_records * seasonal_factor * growth_factor)
                
                base_fare = 15 * growth_factor
                fare_amounts = np.random.normal(base_fare * seasonal_factor, 8, records_count).clip(4, 120)
                tip_amounts = fare_amounts * np.random.beta(2, 5, records_count) * 0.30
                extras = np.random.normal(2.5, 1.5, records_count).clip(0, 12)
                
                month_data = pd.DataFrame({
                    'PULocationID': np.random.choice(zones, records_count),
                    'DOLocationID': np.random.choice(zones, records_count),
                    'fare_amount': fare_amounts,
                    'tip_amount': tip_amounts,
                    'total_amount': fare_amounts + tip_amounts + extras,
                    'trip_distance': np.random.exponential(2.8, records_count).clip(0.1, 45),
                    'data_year': year,
                    'data_month': month,
                    'date': pd.to_datetime(f'{year}-{month:02d}-15')
                })
                
                all_data.append(month_data)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['fare_amount'] = combined_df['fare_amount'].clip(3, 150)
        combined_df['tip_amount'] = combined_df['tip_amount'].clip(0, 80)
        combined_df['total_amount'] = combined_df['total_amount'].clip(4, 200)
        
        return combined_df.sort_values('date').reset_index(drop=True)
        
    except Exception as e:
        print(f"Error generating data: {e}")
        return pd.DataFrame({
            'PULocationID': np.random.choice(range(1, 100), 3000),
            'DOLocationID': np.random.choice(range(1, 100), 3000),
            'fare_amount': np.random.normal(15, 8, 3000).clip(5, 100),
            'tip_amount': np.random.normal(3, 2, 3000).clip(0, 20),
            'total_amount': np.random.normal(20, 10, 3000).clip(8, 120),
            'trip_distance': np.random.exponential(2.5, 3000).clip(0.1, 30),
            'data_year': 2022,
            'data_month': 6,
            'date': pd.to_datetime('2022-06-01')
        })

def calculate_hotspots(df, year_filters=None, month_filters=None, location_type='pickup'):
    try:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        filtered_df = df.copy()
        
        if year_filters and len(year_filters) > 0:
            year_filters = [int(y) for y in year_filters if y is not None]
            if year_filters:
                filtered_df = filtered_df[filtered_df['data_year'].isin(year_filters)]
        
        if month_filters and len(month_filters) > 0:
            month_filters = [int(m) for m in month_filters if m is not None]
            if month_filters:
                filtered_df = filtered_df[filtered_df['data_month'].isin(month_filters)]
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        location_col = 'PULocationID' if location_type == 'pickup' else 'DOLocationID'
        
        hotspots = filtered_df.groupby(location_col).agg({
            'fare_amount': ['count', 'sum'],
            'total_amount': 'sum',
            'tip_amount': 'sum'
        }).round(2)
        
        hotspots.columns = ['ride_count', 'total_fare', 'total_revenue', 'total_tips']
        hotspots = hotspots.reset_index()
        hotspots.rename(columns={location_col: 'LocationID'}, inplace=True)
        
        hotspots['zone_name'] = hotspots['LocationID'].apply(lambda x: safe_zone_lookup(x, 'zone', f'Zone_{x}'))
        hotspots['borough'] = hotspots['LocationID'].apply(lambda x: safe_zone_lookup(x, 'borough', 'Unknown'))
        hotspots['demand_intensity'] = hotspots['ride_count'] / 30
        hotspots['tip_percentage'] = hotspots.apply(
            lambda row: safe_percentage(row['total_tips'], row['total_revenue']), axis=1
        )
        
        return hotspots.sort_values('ride_count', ascending=False)
        
    except Exception as e:
        print(f"Error in hotspot calculation: {e}")
        return pd.DataFrame()

def calculate_time_series(df):
    try:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        numeric_columns = ['total_amount', 'tip_amount', 'fare_amount', 'trip_distance']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['date'] = pd.to_datetime(df['date'])
        df['period'] = df['date'].dt.to_period('M')
        
        trends = df.groupby(['period', 'data_year', 'data_month']).agg({
            'total_amount': ['sum', 'mean', 'count'],
            'tip_amount': ['sum', 'mean'],
            'fare_amount': ['sum', 'mean'],
            'PULocationID': 'nunique'
        }).round(2)
        
        trends.columns = [
            'total_revenue', 'avg_revenue_per_ride', 'total_rides',
            'total_tips', 'avg_tip_per_ride',
            'total_fare', 'avg_fare_per_ride',
            'active_zones'
        ]
        
        trends = trends.reset_index()
        trends['tip_percentage'] = trends.apply(
            lambda row: safe_percentage(row['total_tips'], row['total_revenue']), axis=1
        )
        
        trends = trends.sort_values('period')
        trends['revenue_growth_rate'] = trends['total_revenue'].pct_change() * 100
        trends['revenue_growth_rate'] = trends['revenue_growth_rate'].fillna(0)
        
        return trends
        
    except Exception as e:
        print(f"Error calculating trends: {e}")
        return pd.DataFrame()

def calculate_top_zones(df, top_n=10):
    try:
        if df is None or len(df) == 0:
            return pd.DataFrame()
        
        numeric_columns = ['total_amount', 'tip_amount', 'fare_amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        zone_performance = df.groupby('PULocationID').agg({
            'total_amount': ['sum', 'mean', 'count'],
            'tip_amount': ['sum', 'mean'],
            'fare_amount': ['sum', 'mean']
        }).round(2)
        
        zone_performance.columns = [
            'total_revenue', 'avg_revenue_per_ride', 'total_rides',
            'total_tips', 'avg_tip_per_ride',
            'total_fare', 'avg_fare_per_ride'
        ]
        
        zone_performance = zone_performance.reset_index()
        zone_performance.rename(columns={'PULocationID': 'LocationID'}, inplace=True)
        
        zone_performance['zone_name'] = zone_performance['LocationID'].apply(
            lambda x: safe_zone_lookup(x, 'zone', f'Zone_{x}')
        )
        zone_performance['borough'] = zone_performance['LocationID'].apply(
            lambda x: safe_zone_lookup(x, 'borough', 'Unknown')
        )
        zone_performance['tip_percentage'] = zone_performance.apply(
            lambda row: safe_percentage(row['total_tips'], row['total_revenue']), axis=1
        )
        
        return zone_performance.sort_values('total_revenue', ascending=False).head(top_n)
        
    except Exception as e:
        print(f"Error calculating top zones: {e}")
        return pd.DataFrame()

def create_time_series_chart(trends_df):
    try:
        if trends_df is None or len(trends_df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, font=dict(color='white', size=16))
            fig.update_layout(height=600, paper_bgcolor='rgba(26,27,30,1)', showlegend=False)
            return fig
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Trends', 'Rides Trends', 'Tips Analysis', 'Growth Rates'),
            vertical_spacing=0.15
        )
        
        trends_df['date_plot'] = pd.to_datetime(trends_df['period'].astype(str))
        
        fig.add_trace(go.Scatter(
            x=trends_df['date_plot'], y=trends_df['total_revenue'],
            mode='lines+markers', line=dict(color='#ff6b6b', width=3),
            hovertemplate='Revenue: $%{y:,.0f}<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=trends_df['date_plot'], y=trends_df['total_rides'],
            mode='lines+markers', line=dict(color='#00d4aa', width=3),
            hovertemplate='Rides: %{y:,.0f}<extra></extra>'
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=trends_df['date_plot'], y=trends_df['total_tips'],
            mode='lines+markers', line=dict(color='#ffd43b', width=3),
            hovertemplate='Tips: $%{y:,.0f}<extra></extra>'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=trends_df['date_plot'], y=trends_df['revenue_growth_rate'],
            mode='lines+markers', line=dict(color='#9775fa', width=3),
            hovertemplate='Growth: %{y:.1f}%<extra></extra>'
        ), row=2, col=2)
        
        fig.update_layout(
            height=600, showlegend=False,
            paper_bgcolor='rgba(26,27,30,1)', plot_bgcolor='rgba(26,27,30,1)',
            font=dict(color='white', size=11)
        )
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='white'))
        fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', tickfont=dict(color='white'))
        
        return fig
        
    except Exception as e:
        print(f"Error creating chart: {e}")
        fig = go.Figure()
        fig.add_annotation(text="Error creating chart", x=0.5, y=0.5, font=dict(color='white'))
        fig.update_layout(height=600, paper_bgcolor='rgba(26,27,30,1)', showlegend=False)
        return fig

def create_top_zones_chart(top_zones_df):
    try:
        if top_zones_df is None or len(top_zones_df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, font=dict(color='white', size=16))
            fig.update_layout(height=500, paper_bgcolor='rgba(26,27,30,1)', showlegend=False)
            return fig
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Revenue Leaders', 'Ride Volume', 'Average Fare', 'Tip Percentage'),
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
            vertical_spacing=0.15
        )
        
        zone_names = [name[:12] + '...' if len(str(name)) > 12 else str(name) 
                     for name in top_zones_df['zone_name']]
        
        fig.add_trace(go.Bar(
            x=zone_names, y=top_zones_df['total_revenue'],
            marker_color='#ff6b6b', width=0.6,
            hovertemplate='Revenue: $%{y:,.0f}<extra></extra>'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=zone_names, y=top_zones_df['total_rides'],
            marker_color='#00d4aa', width=0.6,
            hovertemplate='Rides: %{y:,.0f}<extra></extra>'
        ), row=1, col=2)
        
        fig.add_trace(go.Bar(
            x=zone_names, y=top_zones_df['avg_fare_per_ride'],
            marker_color='#ffd43b', width=0.6,
            hovertemplate='Avg Fare: $%{y:.2f}<extra></extra>'
        ), row=2, col=1)
        
        fig.add_trace(go.Bar(
            x=zone_names, y=top_zones_df['tip_percentage'],
            marker_color='#9775fa', width=0.6,
            hovertemplate='Tip %: %{y:.1f}%<extra></extra>'
        ), row=2, col=2)
        
        fig.update_layout(
            height=500, showlegend=False,
            paper_bgcolor='rgba(26,27,30,1)', plot_bgcolor='rgba(26,27,30,1)',
            font=dict(color='white', size=10),
            bargap=0.3, bargroupgap=0.1
        )
        
        fig.update_xaxes(tickfont=dict(color='white', size=8), tickangle=45)
        fig.update_yaxes(tickfont=dict(color='white', size=9))
        
        return fig
        
    except Exception as e:
        print(f"Error creating histogram: {e}")
        fig = go.Figure()
        fig.add_annotation(text="Error", x=0.5, y=0.5, font=dict(color='white'))
        fig.update_layout(height=500, paper_bgcolor='rgba(26,27,30,1)', showlegend=False)
        return fig

def create_hotspot_map(hotspots_df, title="NYC Taxi Hotspots"):
    try:
        if len(hotspots_df) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, font=dict(color='white'))
            fig.update_layout(height=500, paper_bgcolor='rgba(26,27,30,1)', showlegend=False)
            return fig
        
        np.random.seed(42)
        borough_centers = [
            (40.7831, -73.9712), (40.6782, -73.9442), (40.7282, -73.7949),
            (40.8448, -73.8648), (40.5795, -74.1502)
        ]
        
        lats, lons = [], []
        for i in range(len(hotspots_df)):
            center_lat, center_lon = borough_centers[i % len(borough_centers)]
            lat = center_lat + np.random.normal(0, 0.02)
            lon = center_lon + np.random.normal(0, 0.02)
            lats.append(lat)
            lons.append(lon)
        
        if hotspots_df['ride_count'].max() > 0:
            max_rides = hotspots_df['ride_count'].max()
            marker_sizes = 10 + (hotspots_df['ride_count'] / max_rides) * 25
        else:
            marker_sizes = [15] * len(hotspots_df)
        
        hover_texts = []
        for _, row in hotspots_df.iterrows():
            text = f"<b>{row.get('zone_name', 'Unknown')}</b><br>"
            text += f"Borough: {row.get('borough', 'Unknown')}<br>"
            text += f"Rides: {row.get('ride_count', 0):,}<br>"
            text += f"Revenue: ${row.get('total_revenue', 0):,.0f}"
            hover_texts.append(text)
        
        fig = go.Figure(go.Scattermapbox(
            lat=lats, lon=lons, mode='markers',
            marker=dict(
                size=marker_sizes, color=hotspots_df['demand_intensity'].values,
                colorscale='Plasma', opacity=0.8
            ),
            text=hover_texts, hovertemplate='%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=16, color='white')),
            mapbox=dict(style='carto-darkmatter', center=dict(lat=40.7549, lon=-73.9840), zoom=9.5),
            height=500, paper_bgcolor='rgba(26,27,30,1)', font=dict(color='white'),
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating map: {e}")
        fig = go.Figure()
        fig.add_annotation(text="Error", x=0.5, y=0.5, font=dict(color='white'))
        fig.update_layout(height=500, paper_bgcolor='rgba(26,27,30,1)', showlegend=False)
        return fig

app.layout = dmc.MantineProvider(
    theme={"colorScheme": "dark"},
    children=[
        dmc.Container([
            dmc.Paper([
                dmc.Group([
                    dmc.Group([
                        DashIconify(icon="mdi:taxi", width=35, height=35, color="#00d4aa"),
                        dmc.Stack([
                            dmc.Title("🚕 NYC Taxi Analytics", order=1, c="white"),
                            dmc.Text("10-Year Data Analysis", size="sm", c="dimmed")
                        ], gap=2)
                    ]),
                    dmc.Group([
                        dmc.Badge("☁️ Cloud", color="blue", variant="filled"),
                        dmc.Badge("📊 10-Year", color="teal", variant="filled")
                    ])
                ], justify="space-between", align="center")
            ], p="md", mb="lg", style={"backgroundColor": "#1a1b1e"}),
            
            dmc.Paper([
                dmc.SegmentedControl(
                    id="page-selector", value="page1",
                    data=[
                        {"value": "page1", "label": "🗺️ Hotspots & Maps"},
                        {"value": "page2", "label": "📈 Analytics"}
                    ],
                    size="lg", color="teal", style={"width": "100%"}
                )
            ], p="md", mb="lg", style={"backgroundColor": "#1a1b1e"}),
            
            dmc.Paper([
                dmc.Group([
                    dmc.Title("🎛️ Controls", order=3, c="white"),
                    dmc.Group([
                        dmc.Button("📊 Load 10-Year Data", id="load-multi-year-btn", variant="light", color="blue", size="sm"),
                        dmc.Button("⚡ Quick Demo", id="sample-data-btn", variant="light", color="teal", size="sm")
                    ])
                ], justify="space-between", align="center", mb="md"),
                
                dmc.SimpleGrid([
                    dmc.MultiSelect(
                        id="year-selector", label="📅 Years",
                        data=[{"value": str(y), "label": f"{y}"} for y in range(2015, 2025)],
                        value=["2022"], size="md", placeholder="Select years"
                    ),
                    dmc.MultiSelect(
                        id="month-selector", label="📆 Months",
                        data=[{"value": str(m), "label": f"Month {m}"} for m in range(1, 13)],
                        value=["6"], size="md", placeholder="Select months"
                    ),
                    dmc.Select(
                        id="location-type", label="🎯 Type",
                        data=[
                            {"value": "pickup", "label": "🚖 Pickup"},
                            {"value": "dropoff", "label": "📍 Dropoff"}
                        ],
                        value="pickup", size="md"
                    ),
                    dmc.Button("🚀 Analyze", id="load-button", variant="filled", color="teal", fullWidth=True, size="lg")
                ], cols=4, spacing="md")
            ], p="md", mb="lg", style={"backgroundColor": "#1a1b1e"}),
            
            dcc.Store(id="data-store"),
            dcc.Store(id="multi-year-data-store"),
            html.Div(id="page-content")
            
        ], fluid=True, style={"backgroundColor": "#0c0d0f", "minHeight": "100vh", "padding": "1rem"})
    ]
)

@app.callback(
    Output("page-content", "children"),
    [Input("page-selector", "value")]
)
def display_page(page):
    if page == "page1":
        return dmc.Stack([
            dmc.SimpleGrid([
                dmc.Paper([
                    dmc.Group([
                        dmc.ThemeIcon(DashIconify(icon="mdi:car", width=20), size="lg", color="teal"),
                        dmc.Stack([
                            dmc.Text("Rides", size="sm", c="dimmed"),
                            dmc.Text("0", id="total-rides", size="lg", fw=700, c="white")
                        ], gap=1)
                    ])
                ], p="sm", style={"backgroundColor": "#1a1b1e"}),
                dmc.Paper([
                    dmc.Group([
                        dmc.ThemeIcon(DashIconify(icon="mdi:currency-usd", width=20), size="lg", color="red"),
                        dmc.Stack([
                            dmc.Text("Revenue", size="sm", c="dimmed"),
                            dmc.Text("$0", id="total-revenue", size="lg", fw=700, c="white")
                        ], gap=1)
                    ])
                ], p="sm", style={"backgroundColor": "#1a1b1e"}),
                dmc.Paper([
                    dmc.Group([
                        dmc.ThemeIcon(DashIconify(icon="mdi:cash-multiple", width=20), size="lg", color="yellow"),
                        dmc.Stack([
                            dmc.Text("Tips", size="sm", c="dimmed"),
                            dmc.Text("$0", id="total-tips", size="lg", fw=700, c="white")
                        ], gap=1)
                    ])
                ], p="sm", style={"backgroundColor": "#1a1b1e"}),
                dmc.Paper([
                    dmc.Group([
                        dmc.ThemeIcon(DashIconify(icon="mdi:map-marker", width=20), size="lg", color="blue"),
                        dmc.Stack([
                            dmc.Text("Zones", size="sm", c="dimmed"),
                            dmc.Text("0", id="active-zones", size="lg", fw=700, c="white")
                        ], gap=1)
                    ])
                ], p="sm", style={"backgroundColor": "#1a1b1e"})
            ], cols=4, spacing="md", mb="lg"),
            
            dmc.SimpleGrid([
                dmc.Paper([
                    dcc.Graph(id="hotspot-map", style={"height": "500px"})
                ], p="md", style={"backgroundColor": "#1a1b1e"}),
                dmc.Paper([
                    dmc.Title("🏆 Top Hotspots", order=4, mb="md", c="white"),
                    html.Div(id="hotspot-table")
                ], p="md", style={"backgroundColor": "#1a1b1e"})
            ], cols=2, spacing="md")
        ])
    else:
        return dmc.Stack([
            dmc.Paper([
                dmc.Title("📈 Time Series Analysis", order=3, c="white", mb="md"),
                dcc.Graph(id="time-series-charts")
            ], p="md", mb="lg", style={"backgroundColor": "#1a1b1e"}),
            
            dmc.Paper([
                dmc.Title("🏆 Top Zones", order=3, c="white", mb="md"),
                dcc.Graph(id="top-zones-histogram")
            ], p="md", mb="lg", style={"backgroundColor": "#1a1b1e"}),
            
            dmc.Paper([
                dmc.Title("📊 Summary", order=3, c="white", mb="md"),
                html.Div(id="multi-year-summary")
            ], p="md", style={"backgroundColor": "#1a1b1e"})
        ])

@app.callback(
    [Output("data-store", "data"), Output("multi-year-data-store", "data"),
     Output("total-rides", "children"), Output("total-revenue", "children"),
     Output("total-tips", "children"), Output("active-zones", "children")],
    [Input("load-button", "n_clicks"), Input("sample-data-btn", "n_clicks"), Input("load-multi-year-btn", "n_clicks")],
    [dash.dependencies.State("year-selector", "value"), dash.dependencies.State("month-selector", "value")]
)
def load_data(load_clicks, sample_clicks, multi_year_clicks, years, months):
    global taxi_data, multi_year_data
    
    ctx = dash.callback_context
    if not ctx.triggered:
        return {}, {}, "0", "$0", "$0", "0"
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if button_id == "load-multi-year-btn" and multi_year_clicks:
            multi_year_data = generate_sample_data(years=range(2015, 2025), months=range(1, 13))
            
            year_list = years if isinstance(years, list) else [years] if years else ["2022"]
            month_list = months if isinstance(months, list) else [months] if months else ["6"]
            year_ints = [int(y) for y in year_list if y is not None]
            month_ints = [int(m) for m in month_list if m is not None]
            
            current_selection = multi_year_data[
                (multi_year_data['data_year'].isin(year_ints)) &
                (multi_year_data['data_month'].isin(month_ints))
            ]
            
            if len(current_selection) == 0:
                current_selection = multi_year_data.head(5000)
            
            taxi_data = current_selection
            
            total_rides = len(current_selection)
            total_revenue = current_selection['total_amount'].sum()
            total_tips = current_selection['tip_amount'].sum()
            active_zones = current_selection['PULocationID'].nunique()
            
            return (
                {"loaded": True, "years": year_ints, "months": month_ints},
                {"loaded": True, "records": len(multi_year_data)},
                f"{total_rides:,}", f"${total_revenue:,.0f}", f"${total_tips:,.0f}", f"{active_zones}"
            )
            
        elif button_id in ["load-button", "sample-data-btn"] and (load_clicks or sample_clicks):
            year_list = years if isinstance(years, list) else [years] if years else ["2022"]
            month_list = months if isinstance(months, list) else [months] if months else ["6"]
            year_ints = [int(y) for y in year_list if y is not None]
            month_ints = [int(m) for m in month_list if m is not None]
            
            if button_id == "sample-data-btn":
                selected_data = generate_sample_data(years=range(2015, 2025), months=range(1, 13))
                multi_year_data = selected_data
                current_display = selected_data[
                    (selected_data['data_year'].isin(year_ints)) &
                    (selected_data['data_month'].isin(month_ints))
                ]
                if len(current_display) == 0:
                    current_display = selected_data.head(5000)
                taxi_data = current_display
            else:
                selected_data = generate_sample_data(year_ints, month_ints)
                taxi_data = selected_data
                multi_year_data = selected_data
            
            total_rides = len(taxi_data)
            total_revenue = taxi_data['total_amount'].sum()
            total_tips = taxi_data['tip_amount'].sum()
            active_zones = taxi_data['PULocationID'].nunique()
            
            return (
                {"loaded": True, "years": year_ints, "months": month_ints},
                {"loaded": True, "records": len(selected_data)},
                f"{total_rides:,}", f"${total_revenue:,.0f}", f"${total_tips:,.0f}", f"{active_zones}"
            )
            
    except Exception as e:
        print(f"Error in data loading: {e}")
        return {}, {}, "Error", "Error", "Error", "Error"
    
    return {}, {}, "0", "$0", "$0", "0"

@app.callback(
    [Output("hotspot-map", "figure"), Output("hotspot-table", "children")],
    [Input("data-store", "data"), Input("year-selector", "value"), 
     Input("month-selector", "value"), Input("location-type", "value")]
)
def update_hotspots(data_store, years, months, location_type):
    global taxi_data
    
    empty_fig = go.Figure()
    empty_fig.add_annotation(text="🚀 Load data to see analysis", x=0.5, y=0.5, font=dict(color='white', size=14))
    empty_fig.update_layout(height=500, paper_bgcolor='rgba(26,27,30,1)', showlegend=False)
    
    if not data_store or not data_store.get("loaded") or taxi_data is None:
        return empty_fig, "Load data to see hotspots"
    
    try:
        year_list = years if isinstance(years, list) else [years] if years else None
        month_list = months if isinstance(months, list) else [months] if months else None
        
        hotspots = calculate_hotspots(taxi_data, year_list, month_list, location_type)
        
        if len(hotspots) == 0:
            return empty_fig, "No hotspots found"
        
        map_fig = create_hotspot_map(hotspots.head(15), f"{location_type.title()} Hotspots")
        
        top_hotspots = hotspots.head(8)
        table_rows = []
        
        for i, (_, row) in enumerate(top_hotspots.iterrows()):
            table_rows.append(
                html.Div([
                    html.Strong(f"#{i+1} {row.get('zone_name', 'Unknown')[:20]}", 
                              style={"color": "white", "fontSize": "12px"}),
                    html.Br(),
                    html.Span(f"🚗 {row.get('ride_count', 0):,} rides", 
                            style={"color": "#00d4aa", "fontSize": "11px"}),
                    html.Br(),
                    html.Span(f"💰 ${row.get('total_revenue', 0):,.0f}", 
                            style={"color": "#ff6b6b", "fontSize": "11px"})
                ], style={"marginBottom": "10px", "padding": "8px", 
                         "backgroundColor": "#2C2E33", "borderRadius": "6px"})
            )
        
        return map_fig, html.Div(table_rows)
        
    except Exception as e:
        return empty_fig, f"Error: {str(e)[:50]}"

@app.callback(
    [Output("time-series-charts", "figure"), Output("top-zones-histogram", "figure"), 
     Output("multi-year-summary", "children")],
    [Input("multi-year-data-store", "data")]
)
def update_analytics(multi_year_store):
    global multi_year_data
    
    empty_fig = go.Figure()
    empty_fig.add_annotation(text="📊 Load 10-year data for analytics", x=0.5, y=0.5, font=dict(color='white', size=14))
    empty_fig.update_layout(height=400, paper_bgcolor='rgba(26,27,30,1)', showlegend=False)
    
    if not multi_year_store or not multi_year_store.get("loaded") or multi_year_data is None:
        return empty_fig, empty_fig, "Load 10-year data to see analytics"
    
    try:
        trends = calculate_time_series(multi_year_data)
        top_zones = calculate_top_zones(multi_year_data, top_n=10)
        
        time_series_fig = create_time_series_chart(trends)
        top_zones_fig = create_top_zones_chart(top_zones)
        
        total_records = len(multi_year_data)
        total_revenue = multi_year_data['total_amount'].sum()
        years_covered = sorted(multi_year_data['data_year'].unique())
        
        summary = dmc.SimpleGrid([
            dmc.Paper([
                dmc.Text("📊 Records", size="sm", c="dimmed"),
                dmc.Text(f"{total_records:,}", size="lg", fw=700, c="white")
            ], p="sm", style={"backgroundColor": "#2C2E33"}),
            dmc.Paper([
                dmc.Text("💰 Revenue", size="sm", c="dimmed"),
                dmc.Text(f"${total_revenue:,.0f}", size="lg", fw=700, c="white")
            ], p="sm", style={"backgroundColor": "#2C2E33"}),
            dmc.Paper([
                dmc.Text("📅 Years", size="sm", c="dimmed"),
                dmc.Text(f"{years_covered[0]}-{years_covered[-1]}", size="lg", fw=700, c="white")
            ], p="sm", style={"backgroundColor": "#2C2E33"})
        ], cols=3, spacing="md")
        
        return time_series_fig, top_zones_fig, summary
        
    except Exception as e:
        return empty_fig, empty_fig, f"Error: {str(e)[:50]}"

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8050))
    app.run_server(host='0.0.0.0', port=port, debug=False)