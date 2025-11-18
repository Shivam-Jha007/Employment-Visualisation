'''from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Load dataset
    df = pd.read_csv('data/employment.csv')

    # --- Data Cleaning ---
    # Keep only useful columns
    df = df[['ref_area.label', 'indicator.label', 'sex.label', 'time', 'obs_value']]

    # Drop missing values and duplicates
    df = df.dropna().drop_duplicates()

    # Convert column types
    df['time'] = df['time'].astype(int)
    df['obs_value'] = df['obs_value'].astype(float)

    # --- Filtering Options ---
    # Get unique values for dropdowns
    indicators = sorted(df['indicator.label'].unique())
    years = sorted(df['time'].unique())

    # Get user-selected indicator and year (default = first in list)
    selected_indicator = request.args.get('indicator', indicators[0])
    selected_year = int(request.args.get('year', years[-1]))

    # Filter based on selection
    filtered_df = df[
        (df['indicator.label'] == selected_indicator) &
        (df['time'] == selected_year)
    ]

    # --- Create Chart ---
    fig = px.bar(
        filtered_df,
        x='ref_area.label',
        y='obs_value',
        color='sex.label',
        title=f'{selected_indicator} Across Countries ({selected_year})',
        labels={'ref_area.label': 'Country', 'obs_value': 'Employment Rate (%)', 'sex.label': 'Gender'}
    )

    fig.update_layout(template='plotly_white', xaxis_tickangle=-45)

    chart_html = fig.to_html(full_html=False)

    # Render template
    return render_template('index.html',
                           chart=chart_html,
                           indicators=indicators,
                           years=years,
                           selected_indicator=selected_indicator,
                           selected_year=selected_year)

if __name__ == '__main__':
    app.run(debug=True) 
'''

''' from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Load dataset
    df = pd.read_csv('data/employment.csv')

    # --- Data Cleaning ---
    df = df[['ref_area.label', 'indicator.label', 'sex.label', 'time', 'obs_value']]
    df = df.dropna().drop_duplicates()
    df['time'] = df['time'].astype(int)
    df['obs_value'] = df['obs_value'].astype(float)

    # --- Dropdown Options ---
    indicators = sorted(df['indicator.label'].unique())
    years = sorted(df['time'].unique())
    genders = sorted(df['sex.label'].unique())

    # --- User Selections ---
    selected_indicator = request.args.get('indicator', indicators[0])
    selected_year = int(request.args.get('year', years[-1]))
    selected_gender = request.args.get('gender', 'Total')  # Default to 'Total' if available

    # --- Filter Data ---
    filtered_df = df[
        (df['indicator.label'] == selected_indicator) &
        (df['time'] == selected_year) &
        (df['sex.label'] == selected_gender)
    ]

    # --- Simplify Visualization: Top 15 countries ---
    top_countries = (
        filtered_df.sort_values('obs_value', ascending=False)
        .head(15)
    )

    # --- Create Chart ---
    fig = px.bar(
        top_countries,
        x='ref_area.label',
        y='obs_value',
        color='ref_area.label',
        title=f'{selected_indicator} ({selected_gender}, {selected_year}) - Top 15 Countries',
        labels={'ref_area.label': 'Country', 'obs_value': 'Employment Rate (%)'}
    )
    fig.update_layout(template='plotly_white', showlegend=False, xaxis_tickangle=-45, height=600)

    chart_html = fig.to_html(full_html=False)

    # --- Insight Summary ---
    avg_value = round(filtered_df['obs_value'].mean(), 2)
    max_row = filtered_df.loc[filtered_df['obs_value'].idxmax()]
    min_row = filtered_df.loc[filtered_df['obs_value'].idxmin()]

    insight = (
        f"In {selected_year}, the average {selected_indicator.lower()} "
        f"for {selected_gender.lower()}s was {avg_value}%. "
        f"The highest rate was in {max_row['ref_area.label']} "
        f"({max_row['obs_value']}%), while the lowest was in {min_row['ref_area.label']} "
        f"({min_row['obs_value']}%)."
    )

    # Render template
    return render_template(
        'index.html',
        chart=chart_html,
        indicators=indicators,
        years=years,
        genders=genders,
        selected_indicator=selected_indicator,
        selected_year=selected_year,
        selected_gender=selected_gender,
        insight=insight
    )

if __name__ == '__main__':
    app.run(debug=True)'''

'''from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # --- Load dataset ---
    df = pd.read_csv('data/employment.csv')

    # --- Data Cleaning ---
    df = df[['ref_area.label', 'indicator.label', 'sex.label', 'time', 'obs_value']]
    df = df.dropna(subset=['obs_value']).drop_duplicates()
    df['time'] = df['time'].astype(int)
    df['obs_value'] = df['obs_value'].astype(float)

    # --- Dropdown Options ---
    indicators = sorted(df['indicator.label'].unique())
    years = sorted(df['time'].unique())
    genders = sorted(df['sex.label'].unique())

    # --- User Selections ---
    selected_indicator = request.args.get('indicator', indicators[0])
    selected_year = int(request.args.get('year', years[-1]))
    selected_gender = request.args.get('gender', 'Total')

    # --- Filter Data ---
    filtered_df = df[
        (df['indicator.label'] == selected_indicator) &
        (df['time'] == selected_year) &
        (df['sex.label'] == selected_gender)
    ]

    # --- Now collapse multiple entries per country by averaging ---
    country_summary = (
        filtered_df.groupby('ref_area.label', as_index=False)
        .agg({'obs_value': 'mean'})
        .sort_values('obs_value', ascending=False)
        .head(15)
    )

    # --- Create Bar Chart ---
    fig = px.bar(
        country_summary,
        x='ref_area.label',
        y='obs_value',
        color='ref_area.label',
        title=f'{selected_indicator} ({selected_gender}, {selected_year}) - Top 15 Countries (Average)',
        labels={'ref_area.label': 'Country', 'obs_value': 'Employment Rate (%)'}
    )
    fig.update_layout(template='plotly_white', showlegend=False, xaxis_tickangle=-45, height=600)

    chart_html = fig.to_html(full_html=False)

    # --- Insight Summary ---
    avg_value = round(country_summary['obs_value'].mean(), 2)
    max_row = country_summary.iloc[0]
    min_row = country_summary.iloc[-1]

    insight = (
        f"In {selected_year}, the average {selected_indicator.lower()} "
        f"for {selected_gender.lower()}s was {avg_value}%. "
        f"The highest rate was in {max_row['ref_area.label']} "
        f"({round(max_row['obs_value'],2)}%), while the lowest was in {min_row['ref_area.label']} "
        f"({round(min_row['obs_value'],2)}%)."
    )

    # --- Render Template ---
    return render_template(
        'index.html',
        chart=chart_html,
        indicators=indicators,
        years=years,
        genders=genders,
        selected_indicator=selected_indicator,
        selected_year=selected_year,
        selected_gender=selected_gender,
        insight=insight
    )

if __name__ == '__main__':
    app.run(debug=True)
'''

from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # --- Load dataset ---
    df = pd.read_csv('data/employment.csv')

    # --- Data Cleaning ---
    df = df[['ref_area.label', 'indicator.label', 'sex.label', 'time', 'obs_value']]
    df = df.dropna().drop_duplicates()
    df['time'] = df['time'].astype(int)
    df['obs_value'] = df['obs_value'].astype(float)

    # --- Dropdown Options ---
    indicators = sorted(df['indicator.label'].unique())
    years = sorted(df['time'].unique())
    genders = sorted(df['sex.label'].unique())

    # --- User Selections ---
    selected_indicator = request.args.get('indicator', indicators[0])
    selected_year = int(request.args.get('year', years[-1]))
    selected_gender = request.args.get('gender', 'Total')
    selected_country = request.args.get('country', '').strip()
    view_mode = request.args.get('mode', 'Top15')  # 'Top15' or 'Trend'

    # --- Filter Data ---
    filtered_df = df[
        (df['indicator.label'] == selected_indicator) &
        (df['sex.label'] == selected_gender)
    ]

    if selected_country:
        filtered_df = filtered_df[
            filtered_df['ref_area.label'].str.contains(selected_country, case=False, na=False)
        ]

    # --- Create Chart ---
    if view_mode == 'Trend':
        # Show time trend for filtered countries
        trend_df = (
            filtered_df.groupby(['time', 'ref_area.label'], as_index=False)['obs_value']
            .mean()
        )

        fig = px.line(
            trend_df,
            x='time',
            y='obs_value',
            color='ref_area.label',
            title=f'{selected_indicator} Trend ({selected_gender})',
            labels={'obs_value': 'Employment Rate (%)', 'time': 'Year', 'ref_area.label': 'Country'}
        )
        fig.update_layout(template='plotly_white', height=600)
    else:
        # Show Top 15 countries for selected year
        year_df = filtered_df[filtered_df['time'] == selected_year]
        country_summary = (
            year_df.groupby('ref_area.label', as_index=False)
            .agg({'obs_value': 'mean'})
            .sort_values('obs_value', ascending=False)
            .head(15)
        )

        fig = px.bar(
            country_summary,
            x='ref_area.label',
            y='obs_value',
            color='ref_area.label',
            title=f'{selected_indicator} ({selected_gender}, {selected_year}) - Top 15 Countries',
            labels={'ref_area.label': 'Country', 'obs_value': 'Employment Rate (%)'}
        )
        fig.update_layout(template='plotly_white', showlegend=False, xaxis_tickangle=-45, height=600)

    # --- Chart to HTML ---
    chart_html = fig.to_html(full_html=False)

    # --- Insight Summary ---
    if not filtered_df.empty:
        avg_value = round(filtered_df['obs_value'].mean(), 2)
        max_row = filtered_df.loc[filtered_df['obs_value'].idxmax()]
        min_row = filtered_df.loc[filtered_df['obs_value'].idxmin()]

        insight = (
            f"In {selected_year}, the average {selected_indicator.lower()} "
            f"for {selected_gender.lower()}s was {avg_value}%. "
            f"The highest rate was in {max_row['ref_area.label']} "
            f"({max_row['obs_value']}%), while the lowest was in {min_row['ref_area.label']} "
            f"({min_row['obs_value']}%)."
        )
    else:
        insight = "No data available for the selected filters."

    # --- Render Template ---
    return render_template(
        'index.html',
        chart=chart_html,
        indicators=indicators,
        years=years,
        genders=genders,
        selected_indicator=selected_indicator,
        selected_year=selected_year,
        selected_gender=selected_gender,
        selected_country=selected_country,
        view_mode=view_mode,
        insight=insight
    )

if __name__ == '__main__':
    app.run(debug=True)
