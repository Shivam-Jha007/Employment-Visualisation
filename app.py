import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, jsonify # <--- Added jsonify here
import pandas as pd
import plotly.express as px
import google.generativeai as genai
import os

# --- API CONFIGURATION ---
# Replace this with your actual key if needed, but be careful sharing it!
genai.configure(api_key="AIzaSyCROvc4PCRdkiwQkKdtS1jYETLxkxEcZV0")

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # --- Load dataset ---
    # Ensure the path 'data/employment.csv' is correct relative to this script
    try:
        df = pd.read_csv('data/employment.csv')
    except FileNotFoundError:
        return "Error: 'data/employment.csv' not found. Please check your file path."

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
    # Handle case where years list might be empty if CSV is empty
    default_year = years[-1] if years else 2024 
    selected_year = int(request.args.get('year', default_year))
    selected_gender = request.args.get('gender', 'Total')
    selected_country = request.args.get('country', '').strip()
    view_mode = request.args.get('mode', 'Top15')

    # --- Base & filtered frames ---
    base_df = df[df['indicator.label'] == selected_indicator].copy() if selected_indicator else df.copy()

    # --- Filter Data ---
    filtered_df = base_df[base_df['sex.label'] == selected_gender].copy()

    if selected_country:
        filtered_df = filtered_df[
            filtered_df['ref_area.label'].str.contains(selected_country, case=False, na=False)
        ]

    # --- Create Chart ---
    if filtered_df.empty:
        # Handle empty data case to prevent chart crash
        chart_html = "<p>No data available for chart.</p>"
    elif view_mode == 'Trend':
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
        chart_html = fig.to_html(full_html=False)
    else:
        year_df = filtered_df[filtered_df['time'] == selected_year]
        if not year_df.empty:
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
            chart_html = fig.to_html(full_html=False)
        else:
             chart_html = "<p>No data for selected year.</p>"


    # --- Insight Summary ---
    if not filtered_df.empty:
        avg_value = round(filtered_df['obs_value'].mean(), 2)
        # Safe check for max/min
        max_idx = filtered_df['obs_value'].idxmax()
        min_idx = filtered_df['obs_value'].idxmin()
        max_row = filtered_df.loc[max_idx]
        min_row = filtered_df.loc[min_idx]

        insight = (
            f"In {selected_year}, the average {selected_indicator.lower()} "
            f"for {selected_gender.lower()}s was {avg_value}%. "
            f"The highest rate was in {max_row['ref_area.label']} "
            f"({max_row['obs_value']}%), while the lowest was in {min_row['ref_area.label']} "
            f"({min_row['obs_value']}%)."
        )
    else:
        insight = "No data available for the selected filters."

    # --- Consistency / Volatility ---
    try:
        consistency_df = (
            df[df['indicator.label'] == selected_indicator]
            .groupby('ref_area.label')['obs_value']
            .std()
            .reset_index()
            .rename(columns={'obs_value': 'std_dev'})
            .dropna()
        )

        if not consistency_df.empty:
            most_stable = consistency_df.sort_values('std_dev').iloc[0]
            most_volatile = consistency_df.sort_values('std_dev', ascending=False).iloc[0]
            consistency_insight = (
                f"Most stable country over time: {most_stable['ref_area.label']} "
                f"(std = {round(most_stable['std_dev'], 2)}). "
                f"Most volatile: {most_volatile['ref_area.label']} "
                f"(std = {round(most_volatile['std_dev'], 2)})."
            )
        else:
            consistency_insight = "Not enough data."
    except:
        consistency_insight = "Not enough data to measure consistency."

    # --- Year-on-Year Change (THE FIXED PART) ---
    prev_year = selected_year - 1
    yoy_insight = "Year-on-year comparison unavailable." # Default value
    
    if prev_year in years:
        prev_df = df[
            (df['indicator.label'] == selected_indicator) &
            (df['sex.label'] == selected_gender) &
            (df['time'] == prev_year)
        ]

        yoy_df = filtered_df.merge(
            prev_df[['ref_area.label', 'obs_value']],
            on='ref_area.label',
            suffixes=('_curr', '_prev')
        )

        if not yoy_df.empty:
            yoy_df['change'] = yoy_df['obs_value_curr'] - yoy_df['obs_value_prev']
            yoy_df = yoy_df.sort_values('change', ascending=False)

            # THESE LINES MUST BE INSIDE THE IF BLOCK
            top_improver = yoy_df.iloc[0]
            top_decliner = yoy_df.iloc[-1]

            yoy_insight = (
                f"Compared to {prev_year}, {selected_year} saw a "
                f"{round(yoy_df['change'].mean(), 2)}% global {'increase' if yoy_df['change'].mean() > 0 else 'decrease'} "
                f"in {selected_indicator.lower()} for {selected_gender.lower()}s. "
                f"Top improvement: {top_improver['ref_area.label']} (+{round(top_improver['change'], 2)}%). "
                f"Top decline: {top_decliner['ref_area.label']} ({round(top_decliner['change'], 2)}%)."
            )
        else:
            yoy_insight = f"Data available for {selected_year}, but no matching records found in {prev_year}."

    # --- Gender Gap ---
    gender_gap_insight = ""
    if 'Male' in genders and 'Female' in genders:
        male_df = df[(df['indicator.label'] == selected_indicator) & (df['sex.label'] == 'Male') & (df['time'] == selected_year)]
        female_df = df[(df['indicator.label'] == selected_indicator) & (df['sex.label'] == 'Female') & (df['time'] == selected_year)]

        gap_df = male_df.merge(
            female_df[['ref_area.label', 'obs_value']],
            on='ref_area.label',
            suffixes=('_male', '_female')
        )
        if not gap_df.empty:
            gap_df['gap'] = gap_df['obs_value_male'] - gap_df['obs_value_female']
            largest_gap = gap_df.iloc[gap_df['gap'].abs().idxmax()]
            gender_gap_insight = (
                f"The largest gender employment gap is in {largest_gap['ref_area.label']} "
                f"with a {round(largest_gap['gap'], 2)}% difference between males and females."
            )

    # --- Trend Overview ---
    trend_insight = ""
    if not filtered_df.empty:
        trend_all = (
            filtered_df.groupby('time')['obs_value']
            .mean()
            .reset_index()
            .sort_values('time')
        )
        if len(trend_all) >= 2:
            trend_change = trend_all['obs_value'].iloc[-1] - trend_all['obs_value'].iloc[0]
            trend_insight = (
                f"From {trend_all['time'].iloc[0]} to {trend_all['time'].iloc[-1]}, "
                f"{selected_indicator.lower()} for {selected_gender.lower()}s "
                f"changed by {round(trend_change, 2)}% overall."
            )

    # --- Predictions (CLEANED UP) ---
    prediction_insight = ""
    prediction_points = []
    prediction_series_name = "Global mean"

    def simple_prediction_from_series(series_years, series_vals, n_steps=3):
        # Simple linear regression fallback
        X = np.array(series_years).reshape(-1, 1)
        y = np.array(series_vals).astype(float)
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next n_steps
        last_year = int(series_years[-1])
        preds_years = [last_year + i for i in range(1, n_steps+1)]
        X_pred = np.array(preds_years).reshape(-1, 1)
        y_pred = model.predict(X_pred)
        
        # Calculate confidence interval (simplified)
        residuals = y - model.predict(X)
        resid_std = residuals.std(ddof=1) if len(residuals) > 1 else 0.0
        z = 1.96 # 95% CI
        
        out = []
        for i, yr in enumerate(preds_years):
            val = float(y_pred[i])
            out.append({
                'year': int(yr), 
                'pred': round(val, 2),
                'lower': round(val - z * resid_std, 2),
                'upper': round(val + z * resid_std, 2)
            })
        return out

    try:
        # 1. Identify the time series to predict on
        years_series = []
        vals_series = []
        
        if selected_country:
            # Match country from BASE_DF, not the undefined _df
            country_mask = base_df['ref_area.label'].str.contains(selected_country, case=False, na=False)
            country_ts = base_df[country_mask].groupby('time')['obs_value'].mean().reset_index().sort_values('time')
            
            if not country_ts.empty:
                years_series = country_ts['time'].tolist()
                vals_series = country_ts['obs_value'].tolist()
                prediction_series_name = selected_country
            else:
                prediction_insight = f"No time series found for '{selected_country}'; falling back to global mean."
        
        # Fallback to global if country failed or wasn't selected
        if not years_series:
            global_ts = filtered_df.groupby('time')['obs_value'].mean().reset_index().sort_values('time')
            years_series = global_ts['time'].tolist()
            vals_series = global_ts['obs_value'].tolist()
            prediction_series_name = "Global mean"

        # 2. Generate Prediction
        n_points = len(years_series)
        if n_points >= 3:
            preds = simple_prediction_from_series(years_series, vals_series, n_steps=3)
            prediction_points = preds
            prediction_insight = (
                f"Using linear regression on {prediction_series_name} ({n_points} years), "
                f"estimated next values: {preds[0]['year']} ≈ {preds[0]['pred']}%."
            )
        else:
            prediction_insight = f"Not enough historical years ({n_points}) to predict reliably."

    except Exception as e:
        print("DEBUG prediction error:", e)
        prediction_insight = "Prediction unavailable due to data limitations."

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
        insight=insight,
        yoy_insight=yoy_insight,
        trend_insight=trend_insight,
        consistency_insight=consistency_insight,
        gender_gap_insight=gender_gap_insight,
        prediction_insight=prediction_insight,
        prediction_points=prediction_points,
        prediction_series_name=prediction_series_name,
    )

# --- AI ROUTE ---
@app.route('/get_ai_insight', methods=['POST'])
def get_ai_insight():
    try:
        data = request.get_json()
        country = data.get('country', 'Global')
        year = data.get('year')
        indicator = data.get('indicator')
        
        prompt = f"""
        Act as a senior labour economist. Analyze the following data point:
        Context:
        - Country: {country}
        - Year: {year}
        - Indicator: {indicator}
        
        Task:
        Provide a concise, 3-sentence professional explanation for why this trend might be happening. 
        Mention specific economic factors (recession, policy changes) relevant to {country} around {year}.
        """
        
        model = genai.GenerativeModel('models/gemini-pro-latest')
        response = model.generate_content(prompt)
        return jsonify({'success': True, 'content': response.text})
        
    except Exception as e:
        print(f"AI Error: {e}")
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)