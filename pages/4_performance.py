import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score

def yearly_average(data):
    monthly_columns = [str(i) for i in range(1, 13)]
    data['Yearly_Average'] = data[monthly_columns].mean(axis=1)
    return data

def calculate_accuracy(actual_data, predicted_data):
    return r2_score(actual_data, predicted_data) * 100

def plot_comparison(actual_data, predicted_data, title):
    fig = px.line(title=title)
    fig.add_scatter(x=actual_data.index, y=actual_data, name='Actual', line=dict(color='blue'))  # Actual data in blue
    fig.add_scatter(x=predicted_data.index, y=predicted_data, name='Predicted', line=dict(color='green'))  # Predicted data in green
    st.plotly_chart(fig, use_container_width=True)

def show_model_accuracy(value, model_name):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': model_name},
        gauge = {'axis': {'range': [None, 100]},
                 'bar': {'color': "#00C59F"},
                 'steps' : [{'range': [0, value], 'color': "lightgray"}],
                 'threshold' : {'line': {'color': "#FF2283", 'width': 4}, 'thickness': 0.75, 'value': value}}))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Load actual and predicted data
actual_data = pd.read_csv('Dataset/transformed_data.csv')
predicted_data_lr = pd.read_csv('Dataset/lr_predicted_data.csv')
predicted_data_svr = pd.read_csv('Dataset/svr_predicted_data.csv')
predicted_data_lstm = pd.read_csv('Dataset/predicted_data.csv')  # LSTM predictions

# Preprocess the data
actual_data_avg = yearly_average(actual_data)
predicted_data_lr_avg = yearly_average(predicted_data_lr)
predicted_data_svr_avg = yearly_average(predicted_data_svr)
predicted_data_lstm_avg = yearly_average(predicted_data_lstm)  # LSTM

# Merge the datasets on 'Year' and 'Location'
def merge_datasets(actual, predicted, suffix):
    return pd.merge(
        actual[['Location', 'Year', 'Yearly_Average']],
        predicted[['Location', 'Year', 'Yearly_Average']],
        on=['Location', 'Year'],
        suffixes=('_Actual', suffix)
    )

merged_data_lr = merge_datasets(actual_data_avg, predicted_data_lr_avg, '_Predicted_LR')
merged_data_svr = merge_datasets(actual_data_avg, predicted_data_svr_avg, '_Predicted_SVR')
merged_data_lstm = merge_datasets(actual_data_avg, predicted_data_lstm_avg, '_Predicted_LSTM')  # LSTM

def show_performance():
    st.title('Model Performance')

    # Display performance for each model
    for model_data, model_name in [(merged_data_lr, "Linear Regression"), 
                                   (merged_data_svr, "Support Vector Regression"), 
                                   (merged_data_lstm, "LSTM")]:  # LSTM
        st.subheader(f"{model_name} Model Performance")
        actual_col = model_data.columns[2]  # 'Yearly_Average_Actual'
        predicted_col = model_data.columns[3]  # 'Yearly_Average_Predicted_*'
        
        # Plot comparison and show model accuracy
        plot_comparison(model_data[actual_col], model_data[predicted_col], f'Actual vs Predicted Yearly Averages ({model_name})')
        model_accuracy = calculate_accuracy(model_data[actual_col], model_data[predicted_col])
        show_model_accuracy(model_accuracy, f"{model_name} Model Accuracy")
        
        # Display comparison data table
        display_data_table(model_data)

        #line
        st.write("-----")

def display_data_table(merged_data):
    # Identify the predicted column name (assuming it's the fourth column)
    predicted_col_name = merged_data.columns[3]

    # Create a DataFrame with 'Location', 'Year', 'Actual', 'Predicted', and 'Difference'
    comparison_df = pd.DataFrame({
        'Location': merged_data['Location'],
        'Year': merged_data['Year'],
        'Actual': merged_data['Yearly_Average_Actual'],
        'Predicted': merged_data[predicted_col_name]
    })
    comparison_df['Difference'] = comparison_df['Actual'] - comparison_df['Predicted']

    # Reset the index to hide it from display
    comparison_df.reset_index(drop=True, inplace=True)

    # Display the DataFrame without the index
    st.dataframe(comparison_df, width=2000, height=400)


if __name__ == "__main__":
    show_performance()
