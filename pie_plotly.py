import plotly.express as px
import webbrowser

# Step 1: Load built-in gapminder dataset
df = px.data.gapminder()

# Step 2: Filter for the year 2007
df_2007 = df[df['year'] == 2007]

# Step 3: Group by continent and sum population
continent_pop = df_2007.groupby('continent', as_index=False)['pop'].sum()

# Step 4: Create the pie chart
fig = px.pie(continent_pop,
             names='continent',
             values='pop',
             title='Population Distribution by Continent (2007)',
             template='plotly_dark')

# Step 5: Save and open in browser
fig.write_html("continent_population_pie.html")
webbrowser.open("continent_population_pie.html")
