import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import autoreload
import sys
import os
import csv
import prophet
import time
import folium 


from branca.element import Template, MacroElement
from folium.plugins import MarkerCluster, HeatMap, HeatMapWithTime
from streamlit_folium import folium_static
from streamlit_folium import st_folium
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import precision_score, f1_score, accuracy_score, classification_report, recall_score, mean_absolute_error, mean_squared_error, root_mean_squared_error, confusion_matrix, ConfusionMatrixDisplay


page_config = {"page_title":"EchoMinds Innovation", "page_icon":":computer:", "layout":"wide"}
st.set_page_config(**page_config)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 400px;
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)

try:
    # check if the key exists in session state
    _ = st.session_state.keep_graphics
except AttributeError:
    # otherwise set it to false
    st.session_state.keep_graphics = False


with st.sidebar:
	selected = option_menu(
		menu_title = 'Menu',
		options = ['Home', 'Analysis', 'Mapping', 'Health', ''],
		icons = ['speedometer', 'activity', 'hospital', 'ambulance', 'house'],
		menu_icon = 'cast',
		default_index = 0
		)

# ------------------------------------------------- Home section ----------------------------------------------------------#

if selected == 'Home':
	st.write( '''
		# :earth_africa: NASA SPACE APPS CHALLENGE 2024

		## Gender and Climate 

		### About the Challenge
		The United Nations Sustainable Development Goals are a series of targets designed to promote human development and protect the environment. Two of the goals concern gender equality and climate action. While most people might consider these two topics to be quite disparate, they are actually closely related. Your challenge is to model the relationship between climate change and gender inequality, and propose a solution that promotes equality and action to support sustainable development for all.''')

	members = {
    'Member name': ['Faith Waithera', 'Veronica', 'Victor', 'Laurent', 'Faith Chemutai', 'Antonie'],
    'Gender': ['Female', 'Female', 'Male', 'Male', 'Female', 'Male'],
    'Tech Stack': ['Javascript, React', 'Python, ML, Blockchain, DS, Flutter', 'Python, Django, ML, Cyber Sec', 
                   'Python, Django, Cyber Sec', 'Python, ML, Blockchain, DS', 'Python, Django, ML & DS, IoT'],
    'Team name': ['EchoMinds Innovation', 'EchoMinds Innovation', 'EchoMinds Innovation', 'EchoMinds Innovation',
    				'EchoMinds Innovation', 'EchoMinds Innovation']
                   }
	st.dataframe(members)

	st.write('''
		## :milky_way: AIR POLLUTION
		
		''')
	
	image_file = '../source/air1.jpg'

	st.image(image_file)

	# st.video(video_url, format="pollution/mp4", start_time=0, *, subtitles=None, end_time=None, loop=False, autoplay=False, muted=False)
	
	st.write('''
		#### Understanding Air Pollution
		Air pollution consists of harmful or poisonous substances in outdoor or indoor air. It is harmful to people even if they do not have lung disease, but it is particularly dangerous for people living with asthma, COPD, and other respiratory ailments.

		Despite progress in recent years, air pollution continues to be a serious environmental and health problem. The Chicago metropolitan region continues to violate federal air quality standards for harmful ozone and still faces threats from particulate matter pollution. The U.S. Environmental Protection Agency also ranks poor indoor air quality among the top five environmental risks to public health.

		#### What is the air quality forecast for Kenya?
		In early 2021, the capital city of Nairobi was classed as the dirtiest city in Kenya with a US AQI reading of 73. With a figure such as this, the air quality can be classed as “Moderate” according to recommendations by the World Health Organisation (WHO). The city which recorded the second worse level was Ngong in the Kajiado region.

		The cleanest city was Lodwar in the Turkana region with a “Good” air quality.

		#### Where does the information come from that records Kenya’s polluted air?
		Gases such as sulphur dioxide, nitrogen dioxide and carbon monoxide are a result of burning fossil fuels. If inhaled by humans, these gases can significantly affect the amount of oxygen entering the bloodstream, which can have devastating effects.

		Methane is a gas produced by the burning of oil and the decomposition of organic materials at waste filling sites. It can also come from the digestion of cattle and other livestock.

		Additionally, the satellite can monitor aerosol particles that are formed by oil combustion, forest fires, desert dust, or volcanic eruptions. This new data on air pollution access has the potential to improve air pollution coverage and enable governments, especially in developing countries, and other stakeholders to make the impact of poor air quality on human health and the environment a priority better to solve the problem.

		Currently, the USA and the UK are assisting Kenya in its fight with air pollution but through the use of relatively low-tech devices. Using data available from such satellites all governments will have access to data which will allow then to devise new policies aimed at making cleaner air available to its citizens.

		#### Impact of our Solution
		we hope to create a system that provides realtime information of air quality data across Kenya and Africa at large, the soltion focuses on clean air initiatives to empower women, protect their health, and create opportunities for sustainable livelihoods, while also helping mitigate climate change impacts.
		Also having access to realtime air quality data captured and analyzed by our IoT sensors for the purpose of information access
		''')


	image_file = '../source/clean1.jpg'

	st.image(image_file)


elif selected == 'Analysis':
	st.write('''
			# :chart_with_upwards_trend: EXPLORATORY DATA ANALYSIS ON AIR QUALITY
		''')

	st.image('../source/air2.jpg')

	st.write('''
			### Dataset Overview
			 1. Air Quality Indicators: AQI, PM10, PM2_5, NO2, SO2, O3 (measure of pollutants in the air).

			     - AQI (Air Quality Index): A composite index representing overall air quality.
			     - PM10 (Particulate Matter ≤ 10 microns): Larger particulate matter that can be inhaled into the lungs.
			     - PM2.5 (Particulate Matter ≤ 2.5 microns): Finer particulate matter that can penetrate deep into the lungs and bloodstream.
			     - NO2 (Nitrogen Dioxide): A harmful gas primarily from fossil fuel combustion.
			     - SO2 (Sulfur Dioxide): A toxic gas produced by burning fossil fuels and volcanic activity.
			     - O3 (Ozone): A pollutant formed at ground level, harmful to respiratory health.
			     
			 2. Environmental Factors: Temperat, Humidity, WindSpee (weather conditions that influence air quality).

			 3. Health-Related Data: Respiratio, Cardiovas, HospitalA (medical conditions and admissions).

			 4. Health Impact Measures: HealthImp, HealthImpactClass (quantified measures of health impact).
		''')

	# Data Date Dataset Download
	df1 = pd.read_csv(r'../source/data_date.csv', encoding='utf-8')

	df1_data = df1.to_csv(index=False).encode('utf-8')
	st.dataframe(df1.head())

	with open("../source/data_date.csv", "rb") as file:
		st.download_button(label = 'download csv file', data = df1_data, file_name = "data_date.csv", mime='text/csv')

	# Air Quality Index for each country
	df2 = pd.read_csv(r'../source/AQI and Lat Long of Countries.csv', encoding='utf-8')
	df2_data = df2.to_csv(index=False).encode('utf-8')
	st.dataframe(df2.head())

	with open("../source/AQI and Lat Long of Countries.csv", "rb") as file:
		st.download_button(label = 'download csv file', data = df2_data, file_name = "AQI and Lat Long of Countries.csv", mime='text/csv')

	# Air Quality vs Health Impact Dataset
	df3 = pd.read_csv(r'../source/air_quality_health_impact_data.csv', encoding='utf-8')
	df3_data = df3.to_csv(index=False).encode('utf-8')
	st.dataframe(df3.head())

	with open("../source/air_quality_health_impact_data.csv", "rb") as file:
		st.download_button(label = 'download csv file', data = df3_data, file_name = "air_quality_health_impact_data.csv", mime='text/csv')



	#---------------------- Analysis on first Dataset ---------------------------#

	df3.drop(columns=['RecordID'], inplace=True)

	# Overall heatmap
	df3_heatmap = plt.figure(figsize=(14, 7))
	sb.heatmap(df3.corr(), annot=True, linewidth=0.5)
	plt.title('Pearsons Correlation of Columns for the \nImpact of Air Quality on Health', fontsize=14)
	plt.savefig('Pearsons Correlation for air quality')
	st.pyplot(df3_heatmap)

	# Target Column Heatmap
	target_heatmap = plt.figure(figsize=(14, 7))
	df3.select_dtypes(include=['number']).corr()['HealthImpactScore'].sort_values(ascending=False).plot(kind='bar')
	plt.title('Target Variable correlation with other series', fontsize=14)
	plt.savefig('Health Impact Score Correlation')
	st.pyplot(target_heatmap)


	hist_plot_option = st.selectbox(
	    "Select a column to plot Histogram",
	    (df3.columns.unique()),)

	hist_fig, ax = plt.subplots()
	ax.hist(df3[hist_plot_option], bins=30, color='skyblue', edgecolor='black')
	ax.set_title(f'Histogram of {hist_plot_option} \nwith a mean of {df3[hist_plot_option].mean().round(2)}')
	ax.set_xlabel(hist_plot_option)
	ax.set_ylabel('Frequency')

	# Display the histogram in Streamlit
	st.pyplot(hist_fig)

	with st.expander("See explanation"):
	    st.write('''
	    	### Heatmaps
	        Include explanation here.................................
	        .........................................................
	        ...........................................................
	    ''')


	st.write('''
			### Health Impact Classification
		''')


	df3_x_1 = df3.drop(columns=['Temperature', 'Humidity', 'WindSpeed', 'RespiratoryCases', 'CardiovascularCases', 'HospitalAdmissions', 'HealthImpactScore', 'HealthImpactClass'])
	df3_y_1 = df3[['HealthImpactClass', 'HealthImpactScore']]

	df3_x_train_1, df3_x_test_1, df3_y_train_1, df3_y_test_1 = train_test_split(df3_x_1, df3_y_1, test_size=0.2, random_state=42)

	gbr_model = GradientBoostingRegressor()
	mor_model = MultiOutputRegressor(gbr_model)

	mor_model.fit(df3_x_train_1, df3_y_train_1)
	df3_y_pred_1 = mor_model.predict(df3_x_test_1).round(0)

	mean_ab_er = mean_absolute_error(df3_y_pred_1, df3_y_test_1)
	mean_sq_er = mean_squared_error(df3_y_pred_1, df3_y_test_1)


	with st.form('Health Impact Prediction'):
		col1, col2, col3 = st.columns(3)

		with col1:

			aqi_input = st.number_input(
			    label="Air Quality Index [AQI]", 
			    min_value=0.0,   
			    max_value=1000.0, 
			    value=0.0,       
			    step=0.01,       
			    format="%.2f",    
				placeholder="Enter AQI value...")

			no2_input = st.number_input(
			    label="Nitrogen Dioxide [NO2]", 
			    min_value=0.0,   
			    max_value=1000.0, 
			    value=0.0,       
			    step=0.01,       
			    format="%.2f",    
				placeholder="Enter NO2 value...")


		with col2:

			pm10_input = st.number_input(
			    label="Particulate Matter ≤ 10 microns [PM10]", 
			    min_value=0.0,   
			    max_value=1000.0, 
			    value=0.0,       
			    step=0.01,       
			    format="%.2f",    
				placeholder="Enter PM10 value...")

			so2_input = st.number_input(
			    label="Sulphur Dioxide [SO2]", 
			    min_value=0.0,   
			    max_value=1000.0, 
			    value=0.0,       
			    step=0.01,       
			    format="%.2f",    
				placeholder="Enter SO2 value...")


		with col3:

			pm25_input = st.number_input(
			    label="Particulate Matter ≤ 2.5 microns ", 
			    min_value=0.0,   
			    max_value=1000.0, 
			    value=0.0,       
			    step=0.01,       
			    format="%.2f",    
				placeholder="Enter PM2.5 value...")

			o3_input = st.number_input(
			    label="Ozone [O3]", 
			    min_value=0.0,   
			    max_value=1000.0, 
			    value=0.0,       
			    step=0.01,       
			    format="%.2f",    
				placeholder="Enter O3 value...")

		submitted = st.form_submit_button("Submit")

		if submitted:
			new_data = { 
				'AQI': [aqi_input], 
				'NO2': [no2_input], 
				'PM10': [pm10_input], 
				'SO2': [so2_input], 
				'PM2.5': [pm25_input], 
				'O3': [o3_input],
				}
			
			new_data_list = [aqi_input, no2_input, pm10_input, so2_input, pm25_input, o3_input]
			
			new_df_vals = pd.DataFrame(new_data_list)


			def manual_experiment(model, data):
			    input_data = data
			    
			    input_data_to_array = np.asarray(input_data)
			    
			    reshape_input_data = input_data_to_array.reshape(1, -1)
			    
			    model.fit(df3_x_train_1, df3_y_train_1)
			    
			    pred = model.predict(reshape_input_data)

			    pred_df = pd.DataFrame(pred, columns=['Predicted Health Impact Class', 'Predicted Health Impact Score'])
			    
			    st.dataframe(pred_df)

			st.dataframe(new_data)
			
			manual_experiment(mor_model, new_data_list)
			
			with st.spinner('analyzing...'):
			    time.sleep(5)
			    st.success('Successful...', icon="✅")

			

	if submitted:
		with st.expander("See explanation"):
		    st.write('''
		        From the data input above, the system has successfully analyzed and 
		        come up with the following engineered output
		        Considering the higher RMSE we advice that you dont trust it 100%.
		    ''')
		    st.info(f'Root mean squared error: {mean_squared_error(df3_y_test_1, df3_y_pred_1)**0.5}', icon="ℹ️")
		    st.info(f'Mean squared error: {mean_sq_er}', icon="ℹ️")
		    st.info(f'Mean absolute error: {mean_ab_er}', icon="ℹ️")

		    st.image("../source/imppact.jpg")




#----------------- Analysis on Second Dataset -------------------------------#
	st.write('''
		### Global Air Quality Indexing
		''')

	country_plot_input = st.number_input(
			    label="entries..", 
			    min_value=5,   
			    max_value=100, 
			    value=10,    
				placeholder="Enter value...")

	country_group = df1.groupby('Country')['AQI Value'].sum().sort_values(ascending=False).head(country_plot_input)
	

	fig = go.Figure()

	fig.add_trace(go.Bar(x=country_group.index, y=country_group.values))

	fig.update_layout(
		        title=f'Top {country_plot_input} Countries with poor Air Quality \nbetween Aug 2022 to OCT 2024',
		        yaxis_title='AQI',
		        legend_title='AQI'
		    )
	
	# Update the plot in the placeholder
	placeholder = st.empty()
	placeholder.plotly_chart(fig, use_container_width=True)

	with st.expander("See explanation"):
	    st.write('''
	    	### Global AQI Value
	        Include explanation here.................................
	        .........................................................
	        ...........................................................
	    ''')

				
	df1['Date'] = pd.to_datetime(df1['Date'])

	df1.set_index('Date', inplace=True)

	country_select = st.selectbox("Select a country",(df1['Country'].unique()))

	country_group_2 = df1.groupby('Country').get_group(country_select)

	country_fig = go.Figure()

	st.dataframe(country_group_2.head())

	country_fig.add_trace(go.Bar(x=country_group_2.index, y=country_group_2['AQI Value']))
	

	country_fig.update_layout(
		        title=f'Air Quality for {country_select} projection \nbetween Aug 2022 to OCT 2024',
		        yaxis_title='AQI',
		        legend_title='AQI'
		    )
	st.plotly_chart(country_fig, use_container_width=True)

	


	resample_fig = go.Figure()
	
	series = {
		'hourly':'h',
		'daily': 'd',
		'weekly': 'w',
		'monhtly': 'm',
		'yearly': 'y',
	}

	time_series_filter = st.selectbox('Select Time series', (series.keys()))

	# Resample AQI Value based on the selected time series filter
	resampled_data = df1['AQI Value'].resample(series[time_series_filter]).sum()

	# Create a new figure for the resampled time series data
	resample_fig = go.Figure()

	# Add the resampled data to the plot as a line chart
	resample_fig.add_trace(go.Scatter(x=resampled_data.index, y=resampled_data, mode='lines', name='AQI Value'))


	resample_fig.update_layout(
		        title=f'Global Air Quality {time_series_filter.capitalize()} Resample projection \nover the period of Aug 2022 to Oct 2024',
		        yaxis_title='AQI',
		        legend_title='AQI'
		    )
	
	st.plotly_chart(resample_fig, use_container_width=True)

	with st.expander("See explanation"):
	    st.write('''
	    	### Time Series Projection
	        Include explanation here.................................
	        .........................................................
	        ...........................................................
	    ''')

	st.write('''
			### Predict Future Trend
		''')

	# freq_popover = st.popover("Select frequency")
	
	# col1, col2 = st.columns(2)

	# with col1:
	# 	future_days = freq_popover.slider('Pick a day to predict (0-100)', 0, 100)
	
	# with col2:
	# 	future_months = freq_popover.slider('Pick a month to predict (1-3)', 1, 3)

	freq = st.number_input(
		label='Enter Frequency in days',
		max_value=365,
		min_value=10,
		value=100,
		placeholder='Enter Frequency in days....')

	df1.reset_index(inplace=True)
	df1.rename(columns=
	           {
	        'Date':'ds', 'AQI Value':'y'
	            }, 
	           inplace=True)



	forecast_model = prophet.Prophet()
	forecast_model.fit(df1)

	future = forecast_model.make_future_dataframe(periods=freq, freq='D', include_history=False)
	forecast = forecast_model.predict(future)
	# forecast_trends = forecast_model.predict_trend(future)


	fig = go.Figure()


	fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat']))
	# fig.add_trace(go.Scatter(x=forecast_trends['ds'], y=forecast_trends['yhat']))

	fig.update_layout(
		        title=f'Global Air Quality {time_series_filter.capitalize()} Resample projection \nover the period of Aug 2022 to Oct 2024',
		        yaxis_title='AQI',
		        legend_title='AQI'
		    )

	st.plotly_chart(fig, use_container_width=True)



elif selected == 'Mapping':

	st.write('''
		# World Air Pollution 

		### Mapping Air Quality Index after GIS Analysis
		''')

	df2 = pd.read_csv(r'../source/AQI and Lat Long of Countries.csv', encoding='utf-8')

	df2.dropna(axis=0, inplace=True)

	# -------------------- Heatmap ------------------- #
	m = folium.Map(location=[df2['lat'].mean(), df2['lng'].mean()], zoom_start=2, width='100%', height='100')

	heat_data = [[df2['lat'], df2['lng'], df2['AQI Value']] for index, df2 in df2.iterrows()]

	# Add heatmap to the map
	HeatMap(heat_data, radius=15, max_zoom=12).add_to(m)

	# Display the map in Streamlit
	folium_static(m)




	col1, col2 = st.columns(2)

	with col1:
		# Filter the data by Country
		country_select = st.selectbox("Select a country", df2['Country'].sort_values(ascending=True).unique())
		get_grouped_country = df2[df2['Country'] == country_select]

	# with col2:
	# 	city_list = st.selectbox("Select a city", get_grouped_country['City'].sort_values())
	# 	group_city = get_grouped_country[get_grouped_country['City'] == city_list]

	if any(get_grouped_country) == True:
		# Create a base map
		m = folium.Map(location=[get_grouped_country['lat'].mean(), get_grouped_country['lng'].mean()], zoom_start=6)

		# Add marker cluster to handle multiple points
		marker_cluster = MarkerCluster().add_to(m)

		# Loop through each row in the dataframe and plot the points
		for idx, row in get_grouped_country.iterrows():
		    # Information to display in the popup
		    popup_info = (f"City: {row['City']}<br>"
		                  f"AQI Value: {row['AQI Value']}<br>"
		                  f"AQI Category: {row['AQI Category']}<br>"
		                  f"Ozone AQI Value: {row['Ozone AQI Value']}<br>"
		                  f"NO2 AQI Value: {row['NO2 AQI Value']}<br>"
		                  f"PM2.5 AQI Value: {row['PM2.5 AQI Value']}")

		    # Set circle marker size proportional to the AQI value
		    folium.CircleMarker(
		        location=[row['lat'], row['lng']],
		        radius=row['AQI Value'] / 10,  
		        popup=folium.Popup(popup_info, max_width=300),
		        color='blue',
		        fill=True,
		        fill_color='blue',
		        fill_opacity=0.7
		    ).add_to(marker_cluster)

		# Display the map in Streamlit
		folium_static(m)

		city_list = st.selectbox("Select a city", get_grouped_country['City'].sort_values())
		group_city = get_grouped_country[get_grouped_country['City'] == city_list]
		if any(group_city) == True:
			# Create a base map
			m = folium.Map(location=[group_city['lat'].mean(), group_city['lng'].mean()], zoom_start=6)

			# Add marker cluster to handle multiple points
			marker_cluster = MarkerCluster().add_to(m)

			# Loop through each row in the dataframe and plot the points
			for idx, row in group_city.iterrows():
			    # Information to display in the popup
			    popup_info = (f"City: {row['City']}<br>"
			                  f"AQI Value: {row['AQI Value']}<br>"
			                  f"AQI Category: {row['AQI Category']}<br>"
			                  f"Ozone AQI Value: {row['Ozone AQI Value']}<br>"
			                  f"NO2 AQI Value: {row['NO2 AQI Value']}<br>"
			                  f"PM2.5 AQI Value: {row['PM2.5 AQI Value']}")

			    # Set circle marker size proportional to the AQI value
			    folium.CircleMarker(
			        location=[row['lat'], row['lng']],
			        radius=row['AQI Value'] / 10,  
			        popup=folium.Popup(popup_info, max_width=300),
			        color='blue',
			        fill=True,
			        fill_color='blue',
			        fill_opacity=0.7
			    ).add_to(marker_cluster)

			# Display the map in Streamlit
			# folium_static(m)
		folium_static(m)


	


	
	


