# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import calendar
import numpy as np
import plotly.graph_objects as go

# Function to load data
def load_data():
    data_path = 'Dataset/transformed_data.csv'
    threshold_data_path = 'Dataset/Threshold.csv'
    data = pd.read_csv(data_path)
    threshold_data = pd.read_csv(threshold_data_path)
    melted_data = data.melt(id_vars=['Location', 'Year'], var_name='Month', value_name='Water Level')
    return melted_data, threshold_data

# Dashboard Page
def show_dashboard(): 

    
    st.title("Flood Monitoring Dashboard")
    
    data, threshold_data = load_data()


    month_mapping = {str(i): calendar.month_abbr[i] for i in range(1, 13)}
    data['Month'] = data['Month'].map(month_mapping)

    #location and year selection
    col1, col2 = st.columns([1, 3])
    with col1:
        location = st.selectbox('', data['Location'].unique())

    # Add year range slider in the sidebar
    with col2:
        min_year, max_year = int(data['Year'].min()), int(data['Year'].max())
        selected_year_range = st.slider('', min_year, max_year, (min_year, max_year))

    # Filter data based on selected location and year range
    location_data = data[(data['Location'] == location) & (data['Year'] >= selected_year_range[0]) & (data['Year'] <= selected_year_range[1])]
    # Convert Year to int to remove decimal, then to str for concatenation
    location_data['Year'] = location_data['Year'].astype(int).astype(str)
    # Concatenate Year and Month, then convert to datetime
    location_data['Date'] = pd.to_datetime(location_data['Year'] + '-' + location_data['Month'])
    location_data = location_data.sort_values('Date')

    # Update max_water_levels to reflect the selected year
    max_water_levels = location_data.groupby('Month')['Water Level'].max().reset_index()
    max_water_levels_sorted = max_water_levels.sort_values('Water Level', ascending=True)


    # Line Chart and Histogram
    col1, col2 = st.columns([3, 1])
    with col1:
        # Update line chart for selected year range
        fig_line = px.line(location_data, x='Date', y='Water Level', title=f"Water Levels over Time for {location} ({selected_year_range[0]}-{selected_year_range[1]})", color_discrete_sequence=['#00C59F'])
        fig_line.update_layout(xaxis_title='Date', yaxis_title='Water Level (m)', legend_title='Location')

        # Get thresholds for the selected location
        thresholds = threshold_data[threshold_data['Location'] == location].iloc[0]
        normal = thresholds['Normal']
        alert = thresholds['Alert']
        warning = thresholds['Warning']
        danger = thresholds['Danger']

        # Add threshold lines to the plot
        fig_line.add_trace(go.Scatter(x=location_data['Date'], y=[normal]*len(location_data), mode='lines', name='Normal', line=dict(color='blue', dash='dash')))
        fig_line.add_trace(go.Scatter(x=location_data['Date'], y=[alert]*len(location_data), mode='lines', name='Alert', line=dict(color='green', dash='dash')))
        fig_line.add_trace(go.Scatter(x=location_data['Date'], y=[warning]*len(location_data), mode='lines', name='Warning', line=dict(color='orange', dash='dash')))
        fig_line.add_trace(go.Scatter(x=location_data['Date'], y=[danger]*len(location_data), mode='lines', name='Danger', line=dict(color='red', dash='dash')))

        # Display the chart
        st.plotly_chart(fig_line, use_container_width=True)


    # Histogram for Months with Highest Water Level
    with col2:
        # Calculate max water levels for each month
        max_water_levels = location_data.groupby('Month')['Water Level'].max().reset_index()

        # Sort and select the top 5 months with the highest water levels
        top_months = max_water_levels.sort_values('Water Level', ascending=True).head(5)

        # Create a histogram/bar chart for these months, with different colors
        fig_hist = px.bar(top_months, x='Water Level', y='Month', orientation='h', 
                        title="Months Highest Water Level", 
                        color='Water Level',  # Color by water level
                        color_continuous_scale=px.colors.sequential.Viridis)  # Use a color scale
        fig_hist.update_layout(xaxis_title='Max Water Level', yaxis_title='Month', bargap=0.2)

        # Display the chart
        st.plotly_chart(fig_hist, use_container_width=True)


    # Area Charts in a 2x2 Grid
    st.write("## Detailed Water Level Analysis")
    col1, col2 = st.columns(2)

    # Histogram: Monthly Average Water Levels Across All Years
    with col1:
        avg_monthly_data = data.groupby('Month')['Water Level'].mean().reset_index()
        avg_monthly_data['Month'] = pd.Categorical(avg_monthly_data['Month'], categories=list(calendar.month_abbr)[1:], ordered=True)
        avg_monthly_data = avg_monthly_data.sort_values('Month')

        # Create a bar chart with custom color
        fig_avg_monthly = px.bar(avg_monthly_data, x='Month', y='Water Level', title="Avg Monthly Water Levels Across All Years",
                                color_discrete_sequence=["#00C59F"])  # Set the bar color

        st.plotly_chart(fig_avg_monthly, use_container_width=True)

    # Area Chart 2: Yearly Total Water Levels for Each Location
    # Yearly Total Water Levels for Each Location
    with col2:
        yearly_data = data.groupby(['Year', 'Location'])['Water Level'].sum().reset_index()

        # Calculate the rank based on total water level
        yearly_total_rank = yearly_data.groupby('Year')['Water Level'].rank(method='dense', ascending=False)
        yearly_data['Rank'] = yearly_total_rank

        # Sort the data by Year and Rank
        yearly_data_sorted = yearly_data.sort_values(by=['Year', 'Rank'])

        # Plotting
        fig_yearly = px.area(yearly_data_sorted, x='Year', y='Water Level', color='Location', title="Yearly Total Water Levels")
        st.plotly_chart(fig_yearly, use_container_width=True)


    col1, col2 = st.columns(2)

    # Area Chart 3: Cumulative Water Levels for a Selected Year
    # Cumulative Water Levels for a Selected Year
    with col1:
        selected_year = st.selectbox('Select Year for Cumulative Chart:', sorted(data['Year'].unique()), key='cumulative_year')
        cum_data = data[data['Year'] == selected_year].copy()
        cum_data['Cumulative Water Level'] = cum_data.groupby('Location')['Water Level'].cumsum()

        # Determine the total cumulative water level at the end of the year
        final_totals = cum_data.groupby('Location')['Cumulative Water Level'].max()

        # Rank the locations based on these totals
        rank_totals = final_totals.rank(method='dense', ascending=False)
        cum_data = cum_data.join(rank_totals, on='Location', rsuffix='_rank')

        # Sort the data based on this rank
        cum_data_sorted = cum_data.sort_values(by='Cumulative Water Level_rank')

        # Plotting
        fig_cumulative = px.area(cum_data_sorted, x='Month', y='Cumulative Water Level', color='Location', title=f"Cumulative Water Levels in {selected_year}")
        st.plotly_chart(fig_cumulative, use_container_width=True)

    # Area Chart 4: Histogram of Avg Monthly Water Levels Across Selected Years
    with col2:
        selected_years = st.multiselect('Select Years for Comparison:', sorted(data['Year'].unique()), key='compare_years')
        if selected_years:
            compare_data = data[data['Year'].isin(selected_years)].groupby(['Year', 'Month'])['Water Level'].mean().reset_index()
            compare_data['Month'] = pd.Categorical(compare_data['Month'], categories=list(calendar.month_abbr)[1:], ordered=True)
            compare_data = compare_data.sort_values(['Year', 'Month'])
            
            # Create a histogram for these months
            fig_compare = px.bar(compare_data, x='Month', y='Water Level', color='Year', barmode='group', title="Avg Monthly Water Levels Comparison")
            fig_compare.update_layout(xaxis_title='Month', yaxis_title='Avg Water Level', legend_title='Year')
            
            # Display the chart
            st.plotly_chart(fig_compare, use_container_width=True)

    # Section for displaying raw data with additional calculated columns
    st.write("## Summary")
    st.write("Quick Insights")

    # Filters for the data table
    col1, col2 = st.columns(2)

    with col1:
        table_location = st.selectbox('', data['Location'].unique(), key='table_location')

    with col2:
        table_year = st.selectbox('', sorted(data['Year'].unique()), key='table_year')

    # Filter the data based on selected location and year
    filtered_table_data = data[(data['Location'] == table_location) & (data['Year'] == table_year)]

    # Calculate monthly change in water level
    filtered_table_data['Monthly Change'] = filtered_table_data['Water Level'].diff().fillna(0)

    # Determine the water level status based on thresholds
    thresholds = threshold_data[threshold_data['Location'] == table_location].iloc[0]
    def determine_status(level):
        if level >= thresholds['Danger']:
            return 'Danger'
        elif level >= thresholds['Warning']:
            return 'Warning'
        elif level >= thresholds['Alert']:
            return 'Alert'
        else:
            return 'Normal'

    filtered_table_data['Status'] = filtered_table_data['Water Level'].apply(determine_status)

    # Display the data table with full width and height
    filtered_table_data.reset_index(drop=True, inplace=True)  # Reset the index
    filtered_table_data.index += 1  # Start index from 1
    st.dataframe(filtered_table_data[['Location', 'Year', 'Month', 'Water Level', 'Monthly Change', 'Status']], width=2000, height=400)


# Main execution
if __name__ == "__main__":
    
    show_dashboard()
