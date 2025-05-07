import plotly.express as px
import webbrowser

# Step 1: Load Plotly's built-in gapminder dataset
df = px.data.gapminder()

# Step 2: Filter for a single year (e.g., 2007)
df_2007 = df[df['year'] == 2007]

# Step 3: Create the scatter plot
fig = px.scatter(df_2007,
                 x='gdpPercap', 
                 y='lifeExp',
                 size='pop', 
                 color='continent',
                 hover_name='country',
                 log_x=True,
                 title='GDP per Capita vs Life Expectancy (2007)',
                 labels={'gdpPercap': 'GDP per Capita', 'lifeExp': 'Life Expectancy'},
                 template='plotly_white')

# Step 4: Save and open in browser
fig.write_html("gdp_life_scatter.html")
webbrowser.open("gdp_life_scatter.html")
