import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
from prophet import Prophet
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
import pycountry
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from textblob import TextBlob

st.set_page_config(page_title="H&J Academy YouTube Dashboard", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Overall Background */
    .reportview-container {
        background: linear-gradient(135deg, #ece9e6, #ffffff);
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #ece9e6, #ffffff);
        color: #333;
    }
    
    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .header-banner h1 {
        margin: 0;
        font-size: 3em;
        font-weight: 700;
    }
    .header-banner p {
        font-size: 1.5em;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: scale(1.03);
    }
    .metric-card h2 {
        font-size: 2.5em;
        margin: 0;
        color: #667eea;
    }
    .metric-card p {
        font-size: 1.2em;
        margin: 0;
        color: #764ba2;
    }
    
    /* Custom Tabs Styling */
    div[data-baseweb="tab-list"] > button {
        background: #ffffff;
        border: none;
        border-bottom: 4px solid transparent;
        border-radius: 0;
        padding: 10px 20px;
        font-size: 1.1em;
        transition: border-color 0.3s ease, color 0.3s ease;
        color: #333;
    }
    div[data-baseweb="tab-list"] > button:hover {
        color: #667eea;
    }
    div[data-baseweb="tab-list"] > button[aria-selected="true"] {
        border-bottom: 4px solid #667eea;
        color: #667eea;
    }
    
    /* Enhanced Plotly Chart Containers */
    .plotly-graph-div {
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True
)


st.markdown(
    """
    <div class="header-banner">
        <h1>H&amp;J Academy YouTube Dashboard</h1>
        <p>Data-Driven Insights for Maximum Impact</p>
    </div>
    """, unsafe_allow_html=True
)

# ---------------------------
# 1. Loading Data from PostgreSQL
# ---------------------------
db_user = st.secrets["postgresql"]["user"]
db_password = st.secrets["postgresql"]["password"]
db_host = st.secrets["postgresql"]["host"]
db_port = st.secrets["postgresql"]["port"]
db_name = st.secrets["postgresql"]["dbname"]

connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(connection_string)

daily_video_metrics = pd.read_sql('SELECT * FROM daily_video_metrics', engine)
comments = pd.read_sql('SELECT * FROM comments', engine)
province_metrics = pd.read_sql('SELECT * FROM province_metrics', engine)
daily_annotation_metrics = pd.read_sql('SELECT * FROM daily_annotation_metrics', engine)
traffic_source_metrics = pd.read_sql('SELECT * FROM traffic_source_metrics', engine)
revenue_metrics = pd.read_sql('SELECT * FROM revenue_metrics', engine)
ad_type_metrics = pd.read_sql('SELECT * FROM ad_type_metrics', engine)
sharing_metrics = pd.read_sql('SELECT * FROM sharing_metrics', engine)
daily_country_specific_metrics = pd.read_sql('SELECT * FROM daily_country_specific_metrics', engine)

# ---------------------------
# 2. Data Preprocessing & Calculations
# ---------------------------
daily_video_metrics['day'] = pd.to_datetime(daily_video_metrics['day'])
revenue_metrics['day'] = pd.to_datetime(revenue_metrics['day'])
daily_video_metrics['views_growth'] = daily_video_metrics['views'].pct_change() * 100
daily_video_metrics['views_7_day_avg'] = daily_video_metrics['views'].rolling(window=7).mean()

# ---------------------------
# 3. Sidebar Filters
# ---------------------------
st.sidebar.header("Filters")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))
filtered_data = daily_video_metrics[
    (daily_video_metrics['day'] >= pd.to_datetime(start_date)) &
    (daily_video_metrics['day'] <= pd.to_datetime(end_date))
]

# ---------------------------
# 4. Dashboard Tabs
# ---------------------------
tabs = st.tabs([
    "Overview 🔍", 
    "Video Trends 📈", 
    "Revenue Analysis 💰", 
    "Geographic Analysis 🌍", 
    "Clustering 🔢", 
    "Forecasting 🔮", 
    "Comments Analysis 💬",
    "Advanced Engagement 🚀",
    "Revenue Prediction Model 🤖"
])

# ----- Overview Tab -----
with tabs[0]:
    st.header("Overview")
    
    
    if st.button("Show Revenue Recommendations"):
        st.markdown("""
        **Revenue Optimization Recommendations:**
        - **Focus on CPM, Views, and Watch Time:** Analysis shows these are the most influential factors for revenue.
        - **Improve Content Quality & Ad Placements:** Enhance your videos to boost engagement and ad performance.
        - **Upload Videos Consistently:** Channels with more uploads generally achieve higher view counts.
        - **Study Top Performers:** Analyze your best-performing videos and replicate their strategies.
        """)
    
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><p>Total Views</p><h2>{:,}</h2></div>'.format(
                daily_video_metrics["views"].sum()
            ),
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><p>Total Minutes Watched</p><h2>{:,}</h2></div>'.format(
                daily_video_metrics["estimatedMinutesWatched"].sum()
            ),
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><p>Total Subscribers Gained</p><h2>{:,}</h2></div>'.format(
                daily_video_metrics["subscribersGained"].sum()
            ),
            unsafe_allow_html=True
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><p>Avg. View Duration (s)</p><h2>{:.2f}</h2></div>'.format(
                daily_video_metrics["averageViewDuration"].mean()
            ),
            unsafe_allow_html=True
        )
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Performance Over Time")
    
    
    if not filtered_data.empty:
        avg_daily_views = filtered_data['views'].mean()
        peak_day_data = filtered_data.loc[filtered_data['views'].idxmax()]
        peak_day = peak_day_data['day'].strftime('%Y-%m-%d')
        peak_views = peak_day_data['views']
        min_day_data = filtered_data.loc[filtered_data['views'].idxmin()]
        min_day = min_day_data['day'].strftime('%Y-%m-%d')
        min_views = min_day_data['views']
    else:
        avg_daily_views = peak_day = peak_views = min_day = min_views = 0
    
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown(
            '<div class="metric-card"><p>Avg. Daily Views</p><h2>{:,.0f}</h2></div>'.format(avg_daily_views),
            unsafe_allow_html=True
        )
    with colB:
        st.markdown(
            '<div class="metric-card"><p>Peak Day</p><h2>{} ({:,} views)</h2></div>'.format(peak_day, peak_views),
            unsafe_allow_html=True
        )
    with colC:
        st.markdown(
            '<div class="metric-card"><p>Lowest Day</p><h2>{} ({:,} views)</h2></div>'.format(min_day, min_views),
            unsafe_allow_html=True
        )
    
    
    target_duration = 300  
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=daily_video_metrics["averageViewDuration"].mean(),
        delta={'reference': target_duration, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, target_duration * 1.5], 'tickfont': {'color': '#333'}},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, target_duration], 'color': "#e0eafc"},
                {'range': [target_duration, target_duration * 1.5], 'color': "#cfdef3"}
            ]},
        title={"text": "Avg. View Duration (s)", "font": {"size": 20, "color": "#333"}}
    ))
    gauge_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#333"))
    st.plotly_chart(gauge_fig, use_container_width=True)
    
    
    daily_video_metrics['day_name'] = daily_video_metrics['day'].dt.day_name()
    avg_views_by_day = daily_video_metrics.groupby('day_name')['views'].mean()
    best_day = avg_views_by_day.idxmax()
    best_avg = avg_views_by_day.max()
    is_weekend = daily_video_metrics['day_name'].isin(['Saturday', 'Sunday'])
    avg_views_weekend = daily_video_metrics[is_weekend]['views'].mean()
    avg_views_weekday = daily_video_metrics[~is_weekend]['views'].mean()
    if avg_views_weekend > avg_views_weekday:
        recommendation = (
            f"Data suggests that weekends have higher average views "
            f"({avg_views_weekend:,.0f}) compared to weekdays ({avg_views_weekday:,.0f}). "
            f"Consider uploading on weekends. The best day overall is {best_day} "
            f"with an average of {best_avg:,.0f} views."
        )
    else:
        recommendation = (
            f"Data suggests that weekdays have higher average views "
            f"({avg_views_weekday:,.0f}) compared to weekends ({avg_views_weekend:,.0f}). "
            f"Consider uploading on weekdays. The best day overall is {best_day} "
            f"with an average of {best_avg:,.0f} views."
        )
    st.markdown("### Upload Recommendation")
    st.info(recommendation)

# ----- Video Trends Tab -----
with tabs[1]:
    st.header("Video Trends")

    if not filtered_data.empty:
        
        sorted_data = filtered_data.sort_values('day')

        if len(sorted_data) >= 2:
            
            current_views = sorted_data['views'].iloc[-1]
            current_day = sorted_data['day'].iloc[-1].strftime("%Y-%m-%d")

            
            previous_views = sorted_data['views'].iloc[-2]
            previous_day = sorted_data['day'].iloc[-2].strftime("%Y-%m-%d")

            
            if previous_views != 0:
                pct_change = (current_views - previous_views) / previous_views * 100
            else:
                pct_change = 0

            
            if pct_change >= 0:
                arrow_symbol = "↑"
                arrow_color = "blue"
            else:
                arrow_symbol = "↓"
                arrow_color = "red"

            
            st.markdown(
                f"<h4>Views on {current_day}: {current_views:.0f}</h4>",
                unsafe_allow_html=True
            )

            
            st.markdown(
                f"<p style='color:{arrow_color}; font-weight:bold;'>"
                f"{arrow_symbol} {abs(pct_change):.2f}% from {previous_day}"
                "</p>",
                unsafe_allow_html=True
            )
        else:
            st.write("Not enough data to calculate % change.")

        
        advanced_fig = px.line(
            sorted_data,
            x='day',
            y='views',
            title="Amazing Views Trend Over Time",
            line_shape="spline",
            markers=True,
            color_discrete_sequence=["#667eea"]
        )
        advanced_fig.update_traces(line=dict(width=4), marker=dict(size=6))
        advanced_fig.update_traces(fill='tozeroy', fillcolor='rgba(102,126,234,0.2)')
        advanced_fig.add_trace(
            go.Scatter(
                x=sorted_data['day'],
                y=sorted_data['views_7_day_avg'],
                mode='lines',
                name="7-Day Rolling Average",
                line=dict(dash="dot", width=3, color="#764ba2")
            )
        )
        advanced_fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Views",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(size=14, color="#333"),
            hovermode="x unified"
        )
        st.plotly_chart(advanced_fig, use_container_width=True)

    else:
        st.write("No data available for metric comparison.")

    st.subheader("Featured Video")
    st.video("https://www.youtube.com/watch?v=QtfqiBUDVGs&t=647s")

    fig2 = px.line(
        sorted_data,
        x='day',
        y='views_7_day_avg',
        title="7-Day Rolling Average of Views",
        template="plotly_white"
    )
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.line(
        sorted_data,
        x='day',
        y='views_growth',
        title="Daily Views Growth (%)",
        template="plotly_white"
    )
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)


# ----- Revenue Analysis Tab -----
with tabs[2]:
    st.header("Revenue Analysis")
    fig4 = px.line(revenue_metrics, x='day', y='estimatedRevenue', title="Estimated Revenue Over Time", template="plotly_white")
    fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)
    
    merged_rev = pd.merge(daily_video_metrics, revenue_metrics, on='day', how='inner')
    fig5 = px.scatter(merged_rev, x='views_x', y='estimatedRevenue', 
                      title="Views vs. Estimated Revenue", trendline="ols", template="plotly_white")
    fig5.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig5, use_container_width=True)

# ----- Geographic Analysis Tab -----
with tabs[3]:
    st.header("Geographic Analysis")
    def convert_to_iso3(two_letter_code):
        try:
            country = pycountry.countries.get(alpha_2=two_letter_code)
            return country.alpha_3 if country else None
        except:
            return None
    if daily_country_specific_metrics['country'].str.len().iloc[0] == 2:
        daily_country_specific_metrics['iso_alpha'] = daily_country_specific_metrics['country'].apply(convert_to_iso3)
    else:
        daily_country_specific_metrics['iso_alpha'] = daily_country_specific_metrics['country']
    
    fig6 = px.choropleth(
        daily_country_specific_metrics,
        locations='iso_alpha',
        locationmode='ISO-3',
        color='views',
        color_continuous_scale='Viridis',
        title='Views by Country',
        template="plotly_white"
    )
    fig6.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig6, use_container_width=True)
    
    st.subheader("Top 10 Countries by Views")
    top_countries = (
        daily_country_specific_metrics
        .groupby('country', as_index=False)['views']
        .sum()
        .sort_values('views', ascending=False)
        .head(10)
    )
    st.dataframe(top_countries)
    fig_top_countries = px.bar(
        top_countries,
        x='views',
        y='country',
        orientation='h',
        color='views',
        color_discrete_sequence=px.colors.sequential.Blues,
        title='Top 10 Countries by Total Views',
        labels={'country': 'Country', 'views': 'Total Views'},
        template="plotly_white"
    )
    fig_top_countries.update_layout(
        xaxis=dict(title='Total Views', showgrid=False),
        yaxis=dict(title='', showgrid=False),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_top_countries, use_container_width=True)
    
    st.subheader("Viewer Demographics (Age & Gender)")
    df_demo = pd.read_sql("SELECT * FROM demographic_metrics", engine)
    df_demo['viewerPercentage'] = pd.to_numeric(df_demo['viewerPercentage'], errors='coerce')
    fig_demo = px.bar(
        df_demo,
        x='ageGroup',
        y='viewerPercentage',
        color='gender',
        barmode='group',
        title="Viewer Percentage by Age Group & Gender",
        template="plotly_white"
    )
    st.plotly_chart(fig_demo, use_container_width=True)

# ----- Clustering Tab -----
with tabs[4]:
    st.header("Clustering Analysis")
    features = daily_video_metrics[['views', 'estimatedMinutesWatched', 'subscribersGained']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    daily_video_metrics['cluster'] = kmeans.fit_predict(scaled_features)
    fig7 = px.scatter(daily_video_metrics, x='views', y='estimatedMinutesWatched', color='cluster',
                      title="K-Means Clustering of Videos", template="plotly_white")
    st.plotly_chart(fig7, use_container_width=True)

# ----- Forecasting Tab -----
with tabs[5]:
    st.header("Forecasting Views with Prophet")
    forecast_period = st.number_input("Forecast Period (days)", min_value=1, value=30, step=1)
    df_forecast = daily_video_metrics[['day', 'views']].copy()
    df_forecast.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df_forecast)
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    fig8 = model.plot(forecast)
    ax = fig8.gca()
    last_date = df_forecast['ds'].max()
    ax.axvline(x=last_date, color='red', linestyle='--', label='Forecast Start')
    ax.legend()
    st.pyplot(fig8)
    st.markdown("### Forecasted Values for the Next {} Days".format(forecast_period))
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_period))
    st.markdown("### Forecast Components")
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)

# ----- Comments Analysis Tab -----
with tabs[6]:
    st.header("Comments Analysis")
    st.subheader("Sentiment Analysis on Comments")
    def get_sentiment_label(comment):
        polarity = TextBlob(comment).sentiment.polarity
        if polarity > 0.1:
            return "POSITIVE"
        elif polarity < 0.0:
            return "NEGATIVE"
        else:
            return "NEUTRAL"
    if "sentiment_label" not in comments.columns:
        comments['sentiment_label'] = comments['comment'].apply(get_sentiment_label)
    sentiment_counts = comments['sentiment_label'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']
    
    fig_sentiment = px.pie(
        sentiment_counts, 
        names='sentiment', 
        values='count',
        title="Sentiment Distribution of Comments",
        template="plotly_white",
        hole=0.4
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)
    st.write("Sentiment Counts:", comments['sentiment_label'].value_counts())
    
    st.subheader("Word Cloud for Positive Comments")
    positive_comments = " ".join(comments[comments['sentiment_label'] == "POSITIVE"]['comment'].tolist())
    if positive_comments:
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_comments)
        buf_pos = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        plt.title("Positive Comments", fontsize=20)
        plt.savefig(buf_pos, format="png")
        st.image(buf_pos, caption="Positive Comments", use_container_width=True)
        plt.clf()
    else:
        st.write("No positive comments to display.")
    
    st.subheader("Word Cloud for Negative Comments")
    negative_comments = " ".join(comments[comments['sentiment_label'] == "NEGATIVE"]['comment'].tolist())
    if negative_comments:
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(negative_comments)
        buf_neg = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        plt.title("Negative Comments", fontsize=20)
        plt.savefig(buf_neg, format="png")
        st.image(buf_neg, caption="Negative Comments", use_container_width=True)
        plt.clf()
    else:
        st.write("No negative comments to display.")

# ----- Advanced Engagement Analysis Tab -----
with tabs[7]:
    st.header("Advanced Metrics & Engagement Analysis")
    if len(daily_video_metrics) > 300:
        df_sample = daily_video_metrics.sample(n=300, random_state=42)
    else:
        df_sample = daily_video_metrics.copy()
    features_for_parallel = ['views', 'estimatedMinutesWatched', 'averageViewDuration', 'subscribersGained']
    df_scaled = df_sample.copy()
    for col in features_for_parallel:
        col_min = df_scaled[col].min()
        col_max = df_scaled[col].max()
        if col_max != col_min:
            df_scaled[col] = (df_scaled[col] - col_min) / (col_max - col_min)
    st.subheader("Parallel Coordinates Plot")
    fig9 = px.parallel_coordinates(
        df_scaled,
        dimensions=features_for_parallel,
        color='views',
        color_continuous_scale=px.colors.diverging.Tealrose,
        title="Parallel Coordinates: Scaled Video Performance Metrics",
        template="plotly_white"
    )
    st.plotly_chart(fig9, use_container_width=True)
    
    st.subheader("Box Plot: Average View Duration by Cluster")
    fig10 = px.box(daily_video_metrics, x='cluster', y='averageViewDuration',
                   title="Average View Duration by Cluster",
                   labels={'cluster': 'Cluster', 'averageViewDuration': 'Avg. View Duration'},
                   template="plotly_white")
    st.plotly_chart(fig10, use_container_width=True)
    
    st.subheader("Sharing Metrics by Service")
    sharing_agg = (
        sharing_metrics
        .groupby('sharingService')['shares']
        .sum()
        .reset_index()
        .sort_values('shares', ascending=False)
    )
    fig11 = px.bar(
        sharing_agg,
        x='sharingService',
        y='shares',
        title="Total Shares by Sharing Service (Sorted)",
        labels={'sharingService': 'Sharing Service', 'shares': 'Total Shares'},
        template="plotly_white"
    )
    st.plotly_chart(fig11, use_container_width=True)
    
    st.subheader("Ad Type Metrics Treemap")
    fig12 = px.treemap(ad_type_metrics, path=['adType'], 
                       values='adImpressions', color='cpm',
                       color_continuous_scale='RdBu',
                       title="Ad Type Metrics: Ad Impressions and CPM",
                       template="plotly_white")
    st.plotly_chart(fig12, use_container_width=True)
    
    st.subheader("Traffic Source Distribution")
    traffic_agg = traffic_source_metrics.groupby('insightTrafficSourceType')['views'].sum().reset_index()
    # Create a donut chart for traffic source distribution
    fig13 = px.pie(
        traffic_agg, 
        names='insightTrafficSourceType', 
        values='views',
        title="Traffic Source Distribution (by Views)",
        template="plotly_white",
        hole=0.4
    )
    st.plotly_chart(fig13, use_container_width=True)

# ----- Revenue Prediction Model Tab -----
with tabs[8]:
    st.header("Revenue Prediction Model")
    st.markdown("## Data Preparation")
    ml_data = pd.merge(daily_video_metrics, revenue_metrics, on='day', how='inner')
    if 'views_y' in ml_data.columns:
        ml_data = ml_data.drop(columns=['views_y'])
    ml_data = ml_data.rename(columns={'views_x': 'views'})
    annotation_data = daily_annotation_metrics.copy()
    if 'views' in annotation_data.columns:
        annotation_data = annotation_data.drop(columns=['views'])
    annotation_data['day'] = pd.to_datetime(annotation_data['day'])
    ml_data = pd.merge(ml_data, annotation_data, on='day', how='inner')
    if 'day_name' in ml_data.columns:
        dummies = pd.get_dummies(ml_data['day_name'], prefix='day')
        ml_data = pd.concat([ml_data, dummies], axis=1)
        ml_data = ml_data.drop(columns=['day_name'])
    
    st.write("### Data Sample")
    st.dataframe(ml_data.tail())
    
    st.markdown("## Model Training and Evaluation")
    selected_features = ['views', 'estimatedMinutesWatched','averageViewDuration', 'cpm','subscribersGained']
    additional_columns = ['likes']
    for col in additional_columns:
        if col in ml_data.columns:
            selected_features.append(col)
    day_columns = [col for col in ml_data.columns if col.startswith('day_')]
    if day_columns:
        selected_features.extend(day_columns)
    
    st.write("### Features Used for Prediction:")
    st.write(selected_features)
    
    features = ml_data[selected_features]
    target = ml_data['estimatedRevenue']
    
    st.markdown("### Correlation Analysis")
    corr_data = ml_data[selected_features + ['estimatedRevenue']].corr()
    st.write("#### Numerical Correlation Matrix")
    st.write(corr_data)
    
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("**Model Performance:**")
    st.write(f"R² Score: {r2:.2f}")
    
    st.markdown("## Feature Importance")
    feature_importances = pd.DataFrame({
        'Feature': features.columns,
        'Importance': model_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.write(feature_importances)
    
    fig_importance = px.bar(feature_importances, x='Feature', y='Importance',
                            title="Feature Importance for Revenue Prediction",
                            labels={'Importance': 'Importance Score'},
                            template="plotly_white")
    st.plotly_chart(fig_importance, use_container_width=True)
