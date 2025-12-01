from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import plotly.graph_objects as go
import io
import os
import joblib

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change in production

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

users = {}

class User(UserMixin):
    def __init__(self, id, email, password):
        self.id = id
        self.email = email
        self.password = password

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

def get_user_data_path(user_id):
    return f'user_data/{user_id}.csv'

def get_user_model_path(user_id):
    return f'models/{user_id}_model.pkl'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email in [u.email for u in users.values()]:
            flash('Email already exists!', 'danger')
            return redirect(url_for('signup'))
        user_id = str(len(users) + 1)
        users[user_id] = User(user_id, email, generate_password_hash(password))
        os.makedirs('user_data', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = next((u for u in users.values() if u.email == email), None)
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials!', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    data_path = get_user_data_path(current_user.id)
    model_path = get_user_model_path(current_user.id)
    df = pd.DataFrame()
    graph_html = ''
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    
    if request.method == 'POST':
        if 'year' in request.form and 'month' in request.form and 'sales' in request.form:
            try:
                year = int(request.form['year'])
                month = int(request.form['month'])
                sales = float(request.form['sales'])
                new_row = pd.DataFrame({'year': [year], 'month': [month], 'sales': [sales]})
                df = pd.concat([df, new_row], ignore_index=True)
                flash('Data added successfully!', 'success')
            except ValueError:
                flash('Invalid input values. Please check year, month, and sales.', 'danger')
                return redirect(url_for('dashboard'))
        
        if 'file' in request.files:
            file = request.files['file']
            if file and file.filename.endswith('.csv'):
                try:
                    uploaded_df = pd.read_csv(file)
                    if not all(col in uploaded_df.columns for col in ['year', 'month', 'sales']):
                        flash('CSV must have columns: year, month, sales.', 'danger')
                        return redirect(url_for('dashboard'))
                    df = pd.concat([df, uploaded_df], ignore_index=True)
                    flash('CSV uploaded successfully!', 'success')
                except Exception as e:
                    flash(f'Error uploading CSV: {str(e)}', 'danger')
                    return redirect(url_for('dashboard'))
        
        if not df.empty:
            df.to_csv(data_path, index=False)
        
        if not df.empty and len(df) > 1:
            try:
                df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
                df = df.sort_values('date')
                X = df[['year', 'month']].values
                y = df['sales'].values
                model = Ridge(alpha=0.1)
                model.fit(X, y)
                joblib.dump(model, model_path)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['date'], y=df['sales'], mode='lines+markers', name='Historical Sales', line=dict(color='blue')))
                fig.update_layout(title='Historical Sales Data', xaxis_title='Date', yaxis_title='Sales ($)', template='plotly_white')
                graph_html = fig.to_html(full_html=False)
            except Exception as e:
                flash(f'Error training model: {str(e)}', 'danger')
    
    elif not df.empty and os.path.exists(model_path):
        df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['date'], y=df['sales'], mode='lines+markers', name='Historical Sales', line=dict(color='blue')))
        fig.update_layout(title='Historical Sales Data', xaxis_title='Date', yaxis_title='Sales ($)', template='plotly_white')
        graph_html = fig.to_html(full_html=False)
    
    return render_template('dashboard.html', graph_html=graph_html)

@app.route('/predicted_sales', methods=['GET', 'POST'])
@login_required
def predicted_sales():
    model_path = get_user_model_path(current_user.id)
    data_path = get_user_data_path(current_user.id)
    graph_html = ''
    predicted_data = []
    
    if request.method == 'POST':
        print(f"DEBUG: POST request received for user {current_user.id}")  # DEBUG: Check if POST is triggered
        if not os.path.exists(model_path):
            print("DEBUG: Model path does not exist")  # DEBUG
            flash('No trained model found. Please add historical data first.', 'warning')
            return render_template('predicted_sales.html', graph_html=graph_html, predicted_data=predicted_data)
        if not os.path.exists(data_path):
            print("DEBUG: Data path does not exist")  # DEBUG
            flash('No historical data found. Please add data first.', 'warning')
            return render_template('predicted_sales.html', graph_html=graph_html, predicted_data=predicted_data)
        
        try:
            years = int(request.form['years'])
            print(f"DEBUG: Predicting for {years} years")  # DEBUG
            model = joblib.load(model_path)
            df = pd.read_csv(data_path)
            print(f"DEBUG: Loaded data: {df.head()}")  # DEBUG: Show sample data
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            last_year = df['year'].max()
            last_month = df['month'].max()
            print(f"DEBUG: Last year: {last_year}, Last month: {last_month}")  # DEBUG
            
            future_data = []
            for y in range(last_year + 1, last_year + years + 1):
                for m in range(1, 13):
                    future_data.append({'year': int(y), 'month': int(m)})
            future_df = pd.DataFrame(future_data)
            X_future = future_df[['year', 'month']].values
            predictions = model.predict(X_future)
            predictions = np.round(predictions).astype(int)  # Round to whole numbers
            future_df['sales'] = predictions
            predicted_data = future_df.to_dict('records')
            print(f"DEBUG: Predicted data sample: {predicted_data[:3]}")  # DEBUG: Show first 3 predictions
            
            current_user.predicted_data = predicted_data
            
            # Generate graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=future_df.apply(lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1), y=future_df['sales'], mode='lines+markers', name='Predicted Sales', line=dict(color='green')))
            fig.update_layout(title='Predicted Sales Data', xaxis_title='Year-Month', yaxis_title='Sales ($)', template='plotly_white')
            graph_html = fig.to_html(full_html=False)
            print(f"DEBUG: Graph HTML length: {len(graph_html)}")  # DEBUG: Check if graph is generated
        except Exception as e:
            print(f"DEBUG: Error during prediction: {str(e)}")  # DEBUG
            flash(f'Error generating predictions: {str(e)}', 'danger')
    
    return render_template('predicted_sales.html', graph_html=graph_html, predicted_data=predicted_data)

@app.route('/export_csv')
@login_required
def export_csv():
    if hasattr(current_user, 'predicted_data'):
        df = pd.DataFrame(current_user.predicted_data)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode()), mimetype='text/csv', as_attachment=True, download_name='predicted_sales.csv')
    flash('No data to export!', 'danger')
    return redirect(url_for('predicted_sales'))

if __name__ == '__main__':
    app.run(debug=True)