from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from bson.objectid import ObjectId
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import json
import requests
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from datetime import datetime
from mongodb_setup import setup_mongodb, import_sample_data, MONGODB_URI, DB_NAME

# Create Flask application
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ailoancalculator2024secretkey'
app.config['MONGODB_URI'] = 'mongodb+srv://admin:hyperX@loancalculator.mnccqo8.mongodb.net/?retryWrites=true&w=majority&appName=LoanCalculator'
app.config['DB_NAME'] = DB_NAME

# Initialize Flask extensions
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'

# In-memory storage for fallback when MongoDB is not available
memory_db = {
    'users': {},  # user_id -> user_data
    'calculations': [],  # list of calculation objects
    'chat_logs': [],  # list of chat log objects
    'next_id': 1  # simple auto-increment id
}

# Add a default admin user to memory_db
default_admin_password = bcrypt.generate_password_hash('admin123').decode('utf-8')
memory_db['users']['1'] = {
    '_id': '1',
    'name': 'Admin User',
    'email': 'admin@example.com',
    'password': default_admin_password,
    'date_registered': datetime.now()
}
memory_db['next_id'] = 2  # Set next_id to 2 since we already used 1

# Connect to MongoDB
try:
    # Attempt to set up MongoDB with our utility
    mongo_setup_success = setup_mongodb()
    
    if mongo_setup_success:
        client = MongoClient(os.environ.get('mongodb+srv://admin:hyperX@loancalculator.mnccqo8.mongodb.net/?retryWrites=true&w=majority&appName=LoanCalculator'))
        db = client['LoanCalculator']
        print("Connected to MongoDB successfully!")
        
        # Check if we need to import sample data (can be controlled with environment variable)
        if os.environ.get('IMPORT_SAMPLE_DATA', '').lower() in ('true', '1', 'yes'):
            import_sample_data(db)
            print("Sample data imported successfully!")
    else:
        # If setup failed, db will be None
        print("MongoDB setup failed")
        db = None
except Exception as e:
    print(f"Failed to connect to MongoDB: {e}")
    print("Using in-memory storage instead.")
    db = None
    
    # Make sure MongoDB Compass info is displayed to user
    print("\nTo use MongoDB Compass:")
    print("1. Make sure MongoDB service is running")
    print("2. Open MongoDB Compass")
    print("3. Connect to: mongodb://localhost:27017/")
    print("4. Create database: loan_calculator_db")
    print("5. Create collections: users, calculations, chat_logs")

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.name = user_data['name']
        self.email = user_data['email']
        self.password = user_data['password']
        self.date_registered = user_data.get('date_registered', datetime.now())

# Load user callback for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    global db
    
    if db is not None:
        try:
            user_data = db.users.find_one({'_id': ObjectId(user_id)})
            if user_data:
                return User(user_data)
        except Exception as e:
            print(f"Error loading user from MongoDB: {e}")
            db = None  # Reset db since it's not working
    
    # Fallback to memory storage
    if user_id in memory_db['users']:
        user_data = memory_db['users'][user_id]
        user_data['_id'] = user_id  # Ensure _id is set
        return User(user_data)
    
    return None

# Initialize the chatbot
class LoanChatbot:
    def __init__(self):
        self.responses = {
            'greeting': ['hello', 'hi', 'hey'],
            'loan_info': ['loan', 'interest', 'rate', 'payment'],
            'eligibility': ['eligible', 'qualify', 'requirements'],
            'process': ['process', 'apply', 'application'],
        }
        
    def get_response(self, user_input):
        user_input = user_input.lower()
        tokens = word_tokenize(user_input)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        
        # Check for keywords and generate response
        for category, keywords in self.responses.items():
            if any(keyword in tokens for keyword in keywords):
                return self.generate_response(category)
        
        return "I'm not sure about that. Would you like to know about loan eligibility, interest rates, or the application process?"

    def generate_response(self, category):
        responses = {
            'greeting': "Hello! I'm your AI loan assistant. How can I help you today?",
            'loan_info': "Our loans have competitive interest rates starting from 5.99% APR. The exact rate depends on your credit score and loan amount.",
            'eligibility': "To be eligible, you need to be at least 18 years old, have a stable income, and a credit score of 650 or higher.",
            'process': "The application process is simple: 1) Fill out the online form, 2) Submit required documents, 3) Get approved within 24 hours."
        }
        return responses.get(category, "I'm here to help! What would you like to know about our loans?")

# Initialize the rule-based chatbot
chatbot = LoanChatbot()

# Free Hugging Face API integration
def get_huggingface_response(user_message):
    API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    headers = {"Authorization": "Bearer hf_dummy_key_for_free_tier"}  # Replace with your API key if you have one
    
    # Set a system message for context
    payload = {
        "inputs": {
            "past_user_inputs": ["I want to know about loans"],
            "generated_responses": ["I can help you with loan-related questions. What would you like to know?"],
            "text": user_message
        },
        "parameters": {
            "temperature": 0.7,
            "max_length": 100
        }
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json().get("generated_text", "Sorry, I couldn't process that request.")
        else:
            print(f"API error: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        print(f"Error calling Hugging Face API: {str(e)}")
        return None

def calculate_loan_payment(principal, annual_rate, years):
    monthly_rate = annual_rate / 12 / 100
    num_payments = years * 12
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    return monthly_payment

def generate_amortization_schedule(principal, annual_rate, years):
    monthly_rate = annual_rate / 12 / 100
    num_payments = years * 12
    monthly_payment = calculate_loan_payment(principal, annual_rate, years)
    
    schedule = []
    balance = principal
    
    for month in range(1, num_payments + 1):
        interest_payment = balance * monthly_rate
        principal_payment = monthly_payment - interest_payment
        balance -= principal_payment
        
        schedule.append({
            'month': month,
            'payment': monthly_payment,
            'principal': principal_payment,
            'interest': interest_payment,
            'balance': max(0, balance)
        })
    
    return schedule

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    global db
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        user_found = False
        
        if db is not None:
            try:
                user_data = db.users.find_one({'email': email})
                if user_data and bcrypt.check_password_hash(user_data['password'], password):
                    user = User(user_data)
                    login_user(user, remember=remember)
                    next_page = request.args.get('next')
                    flash('Login successful!', 'success')
                    return redirect(next_page if next_page else url_for('dashboard'))
                user_found = user_data is not None
            except Exception as e:
                print(f"Error during MongoDB login: {e}")
                db = None  # Reset db since it's not working
        
        # Fallback to memory storage if not found in MongoDB or if MongoDB is not available
        if not user_found:
            for user_id, user_data in memory_db['users'].items():
                if user_data.get('email') == email:
                    if bcrypt.check_password_hash(user_data['password'], password):
                        user_data['_id'] = user_id  # Ensure _id is set
                        user = User(user_data)
                        login_user(user, remember=remember)
                        next_page = request.args.get('next')
                        flash('Login successful!', 'success')
                        return redirect(next_page if next_page else url_for('dashboard'))
        
        flash('Invalid email or password', 'danger')
    
    return render_template('login.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not name or not email or not password:
            flash('Please fill in all fields', 'danger')
            return render_template('register.html')
        
        global db
        
        if db is not None:
            try:
                if db.users.find_one({'email': email}):
                    flash('Email already registered', 'danger')
                    return render_template('register.html')
                
                hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
                user_data = {
                    'name': name,
                    'email': email,
                    'password': hashed_password,
                    'date_registered': datetime.now()
                }
                
                result = db.users.insert_one(user_data)
                user_data['_id'] = result.inserted_id
                user = User(user_data)
                login_user(user)
                flash('Registration successful!', 'success')
                return redirect(url_for('dashboard'))
            except Exception as e:
                print(f"Error during MongoDB registration: {e}")
                db = None  # Reset db since it's not working
        
        # Fallback to memory storage if MongoDB is not available
        for user_data in memory_db['users'].values():
            if user_data.get('email') == email:
                flash('Email already registered', 'danger')
                return render_template('register.html')
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user_id = str(memory_db['next_id'])
        user_data = {
            '_id': user_id,
            'name': name,
            'email': email,
            'password': hashed_password,
            'date_registered': datetime.now()
        }
        
        memory_db['users'][user_id] = user_data
        memory_db['next_id'] += 1
        
        user = User(user_data)
        login_user(user)
        flash('Registration successful!', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('register.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Dashboard route (main application)
@app.route('/')
@app.route('/dashboard')
@login_required
def dashboard():
    global db
    loan_history = []
    if db is not None:
        try:
            # Fetch user's calculation history from MongoDB
            history = list(db.calculations.find({'user_id': current_user.id}).sort('date_created', -1))
            for item in history:
                item['id'] = str(item['_id'])
                loan_history.append(item)
        except Exception as e:
            print(f"Error fetching calculations from MongoDB: {e}")
            db = None  # Reset db since it's not working
    
    # If MongoDB failed or no history found, fallback to memory storage
    if not loan_history:
        for item in memory_db['calculations']:
            if item.get('user_id') == current_user.id:
                loan_history.append(item)
        # Sort by date (newest first)
        loan_history = sorted(loan_history, key=lambda x: x.get('date_created', datetime.now()), reverse=True)
    
    return render_template('index.html', loan_history=loan_history)

# Messages page route
@app.route('/messages_page')
@login_required
def messages_page():
    return render_template('messages.html')

# Update profile route
@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    global db
    
    if request.method == 'POST':
        name = request.form.get('name')
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        
        if db is not None:
            try:
                user_data = db.users.find_one({'_id': ObjectId(current_user.id)})
                if not current_password or not bcrypt.check_password_hash(user_data['password'], current_password):
                    flash('Current password is incorrect.', 'danger')
                    return redirect(url_for('dashboard'))
                
                update_data = {'name': name}
                
                if new_password:
                    hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
                    update_data['password'] = hashed_password
                
                db.users.update_one({'_id': ObjectId(current_user.id)}, {'$set': update_data})
                flash('Profile updated successfully!', 'success')
            except Exception as e:
                print(f"Error updating profile in MongoDB: {e}")
                db = None  # Reset db since it's not working
                # Fall through to memory storage
        
        # If MongoDB failed or is not available, use memory storage
        if db is None:
            user_id = current_user.id
            if user_id in memory_db['users']:
                if not current_password or not bcrypt.check_password_hash(memory_db['users'][user_id]['password'], current_password):
                    flash('Current password is incorrect.', 'danger')
                    return redirect(url_for('dashboard'))
                
                memory_db['users'][user_id]['name'] = name
                
                if new_password:
                    hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
                    memory_db['users'][user_id]['password'] = hashed_password
                
                flash('Profile updated successfully!', 'success')
            else:
                flash('User not found.', 'danger')
    
    return redirect(url_for('dashboard'))

# Calculate loan endpoint
@app.route('/calculate', methods=['POST'])
@login_required
def calculate():
    data = request.json
    principal = float(data['principal'])
    annual_rate = float(data['rate'])
    years = int(data['years'])
    
    monthly_payment = calculate_loan_payment(principal, annual_rate, years)
    total_payment = monthly_payment * years * 12
    total_interest = total_payment - principal
    
    # Create loan breakdown pie chart
    labels = ['Principal', 'Interest']
    values = [principal, total_interest]
    colors = ['#4a90e2', '#e74c3c']
    
    fig = px.pie(
        names=labels, 
        values=values, 
        title='Loan Breakdown',
        color_discrete_sequence=colors,
        hole=0.4
    )
    
    # Customize the pie chart
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label',
        hoverinfo='label+value+percent',
        marker=dict(line=dict(color='#FFFFFF', width=2))
    )
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add monthly payment chart
    monthly_data = {
        'Category': ['Monthly Payment'],
        'Amount': [monthly_payment]
    }
    
    fig2 = px.bar(
        monthly_data, 
        x='Category', 
        y='Amount',
        title='Monthly Payment',
        color_discrete_sequence=['#2ecc71'],
        text='Amount'
    )
    
    fig2.update_traces(
        texttemplate='$%{text:.2f}',
        textposition='outside'
    )
    
    fig2.update_layout(
        xaxis=dict(title=''),
        yaxis=dict(title='Amount ($)', showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Convert both figures to JSON
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    monthly_graph_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'monthly_payment': round(monthly_payment, 2),
        'total_payment': round(total_payment, 2),
        'total_interest': round(total_interest, 2),
        'breakdown_graph': graph_json,
        'monthly_graph': monthly_graph_json
    })

# Save calculation to database
@app.route('/save_calculation', methods=['POST'])
@login_required
def save_calculation():
    global db
    
    data = request.json
    
    calculation = {
        'user_id': current_user.id,
        'principal': float(data['principal']),
        'interest_rate': float(data['rate']),
        'term_years': int(data['years']),
        'monthly_payment': float(data['monthly_payment']),
        'total_payment': float(data['total_payment']),
        'total_interest': float(data['total_interest']),
        'date_created': datetime.now()
    }
    
    if db is not None:
        try:
            result = db.calculations.insert_one(calculation)
            calculation['id'] = str(result.inserted_id)
        except Exception as e:
            print(f"Error saving calculation to MongoDB: {e}")
            db = None  # Reset db since it's not working
    
    # If MongoDB failed or is not available, use memory storage
    if db is None:
        calculation['id'] = str(memory_db['next_id'])
        calculation['_id'] = memory_db['next_id']
        memory_db['next_id'] += 1
        memory_db['calculations'].append(calculation)
    
    return jsonify({'success': True})

# Load calculation from history
@app.route('/load_calculation/<calculation_id>', methods=['GET'])
@login_required
def load_calculation(calculation_id):
    global db
    
    if db is not None:
        try:
            calculation = db.calculations.find_one({
                '_id': ObjectId(calculation_id),
                'user_id': current_user.id
            })
            
            if calculation:
                return jsonify({
                    'principal': calculation['principal'],
                    'interest_rate': calculation['interest_rate'],
                    'term_years': calculation['term_years']
                })
        except Exception as e:
            print(f"Error loading calculation from MongoDB: {e}")
            db = None  # Reset db since it's not working
    
    # Fallback to memory storage if MongoDB failed or calculation not found
    for calc in memory_db['calculations']:
        if calc.get('id') == calculation_id and calc.get('user_id') == current_user.id:
            return jsonify({
                'principal': calc['principal'],
                'interest_rate': calc['interest_rate'],
                'term_years': calc['term_years']
            })
    
    return jsonify({'error': 'Calculation not found'}), 404

# Delete calculation from history
@app.route('/delete_calculation/<calculation_id>', methods=['DELETE'])
@login_required
def delete_calculation(calculation_id):
    global db
    
    if db is not None:
        try:
            result = db.calculations.delete_one({
                '_id': ObjectId(calculation_id),
                'user_id': current_user.id
            })
            
            if result.deleted_count > 0:
                return jsonify({'success': True})
        except Exception as e:
            print(f"Error deleting calculation from MongoDB: {e}")
            db = None  # Reset db since it's not working
    
    # Fallback to memory storage if MongoDB failed or calculation not found
    for i, calc in enumerate(memory_db['calculations']):
        if calc.get('id') == calculation_id and calc.get('user_id') == current_user.id:
            memory_db['calculations'].pop(i)
            return jsonify({'success': True})
    
    return jsonify({'error': 'Calculation not found or could not be deleted'}), 404

# Chat endpoint
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    global db
    
    try:
        user_message = request.json['message']
        # Check if custom mode is enabled (default to False if not provided)
        custom_mode = request.json.get('custom_mode', False)
        print(f"Received message: '{user_message}' (Custom mode: {custom_mode})")
        
        # If custom mode is enabled, try the Hugging Face API first
        if custom_mode:
            # Try to get a response from Hugging Face API for custom questions
            hf_response = get_huggingface_response(user_message)
            if hf_response:
                print(f"Using AI response for custom question")
                
                # Log the chat in the database if connected
                chat_log = {
                    'user_id': current_user.id,
                    'message': user_message,
                    'response': hf_response,
                    'mode': 'custom',
                    'timestamp': datetime.now()
                }
                
                if db is not None:
                    try:
                        db.chat_logs.insert_one(chat_log)
                    except Exception as e:
                        print(f"Error saving chat log to MongoDB: {e}")
                        db = None  # Reset db since it's not working
                
                if db is None:
                    # Fallback to memory storage
                    chat_log['id'] = str(memory_db['next_id'])
                    memory_db['next_id'] += 1
                    memory_db['chat_logs'].append(chat_log)
                
                return jsonify({'response': hf_response})
        
        # If not in custom mode or if API call failed, use rule-based responses
        user_message_lower = user_message.lower()
        
        # Check if it's a standard loan question
        if any(word in user_message_lower for word in ['hi', 'hello', 'hey']):
            response = f"Hello {current_user.name}! I'm your AI loan assistant. How can I help you today?"
        
        # General loan information
        elif any(word in user_message_lower for word in ['what is', 'what are']) and 'loan' in user_message_lower:
            response = "Loans are financial products that allow you to borrow money and repay it over time with interest. Common types include personal loans, mortgages, auto loans, and student loans."
        
        # Interest rate information
        elif 'interest' in user_message_lower or 'rate' in user_message_lower:
            response = "Our loans have competitive interest rates starting from 5.99% APR. The exact rate depends on your credit score and loan amount. Higher credit scores generally qualify for lower rates."
        
        # Eligibility criteria
        elif any(word in user_message_lower for word in ['eligible', 'qualify', 'requirements', 'criteria']):
            response = "To be eligible for our loans, you need to be at least 18 years old, have a stable income, and a credit score of 650 or higher. We also consider your debt-to-income ratio and employment history."
        
        # Application process
        elif any(word in user_message_lower for word in ['process', 'apply', 'application', 'how to']):
            response = "Our application process is simple: 1) Fill out the online form with your personal and financial information, 2) Submit required documents like ID and proof of income, 3) Get approved within 24 hours, 4) Receive funds in your account in 1-3 business days."
        
        # Loan types
        elif 'type' in user_message_lower and 'loan' in user_message_lower:
            response = "We offer several types of loans: Personal loans (for general expenses), Home loans (for purchasing property), Auto loans (for vehicles), Education loans (for tuition), and Business loans (for entrepreneurs)."
        
        # Loan terms
        elif 'term' in user_message_lower or 'duration' in user_message_lower:
            response = "Our loan terms range from 1 to 30 years depending on the loan type. Personal loans typically have terms of 1-7 years, while mortgages can extend to 15-30 years."
        
        # Loan amounts
        elif 'amount' in user_message_lower or 'how much' in user_message_lower:
            response = "Our loan amounts range from $1,000 to $500,000 depending on the loan type, your creditworthiness, and your financial situation. Use our calculator to estimate monthly payments for different loan amounts."
        
        # Payment questions
        elif 'payment' in user_message_lower or 'repay' in user_message_lower or 'pay back' in user_message_lower:
            response = "You can make payments monthly via direct debit, online banking, or through our app. We offer flexible repayment options and you can make additional payments anytime without penalties."
        
        # Early repayment
        elif 'early' in user_message_lower and ('repay' in user_message_lower or 'pay' in user_message_lower):
            response = "Yes, you can repay your loan early without any prepayment penalties. This can save you money on interest over the loan term."
            
        # Credit score questions
        elif 'credit' in user_message_lower and 'score' in user_message_lower:
            if 'improve' in user_message_lower or 'increase' in user_message_lower or 'boost' in user_message_lower:
                response = "To improve your credit score: 1) Pay all bills on time, 2) Reduce credit card balances, 3) Don't apply for too many new accounts, 4) Keep old accounts open, 5) Regularly check your credit report for errors."
            else:
                response = "Your credit score is a number between 300-850 that represents your creditworthiness. Scores above 700 are considered good. We typically require a minimum score of 650 for loan approval, though higher scores will qualify you for better rates."
        
        # If no rule matches and we're not in custom mode, or if we are but API failed
        else:
            # Try one more time with Hugging Face if not already tried
            if not custom_mode:
                hf_response = get_huggingface_response(user_message)
                if hf_response:
                    chat_log = {
                        'user_id': current_user.id,
                        'message': user_message,
                        'response': hf_response,
                        'mode': 'fallback',
                        'timestamp': datetime.now()
                    }
                    
                    if db is not None:
                        try:
                            db.chat_logs.insert_one(chat_log)
                        except Exception as e:
                            print(f"Error saving chat log to MongoDB: {e}")
                            db = None  # Reset db since it's not working
                    
                    if db is None:
                        # Fallback to memory storage
                        chat_log['id'] = str(memory_db['next_id'])
                        memory_db['next_id'] += 1
                        memory_db['chat_logs'].append(chat_log)
                    
                    return jsonify({'response': hf_response})
            
            # Last resort fallback
            response = "I understand you're asking about loans, but I'm not sure about the specifics. You can ask about loan types, interest rates, eligibility, or application process. Or you can try switching to Custom AI mode for more complex questions."
        
        # Log the chat in the database if connected
        chat_log = {
            'user_id': current_user.id,
            'message': user_message,
            'response': response,
            'mode': 'standard',
            'timestamp': datetime.now()
        }
        
        if db is not None:
            try:
                db.chat_logs.insert_one(chat_log)
            except Exception as e:
                print(f"Error saving chat log to MongoDB: {e}")
                db = None  # Reset db since it's not working
        
        if db is None:
            # Fallback to memory storage
            chat_log['id'] = str(memory_db['next_id'])
            memory_db['next_id'] += 1
            memory_db['chat_logs'].append(chat_log)
        
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error in chat processing: {e}")
        return jsonify({'response': 'Sorry, there was an error processing your message. Please try again.'}), 500

# Get all users for messaging
@app.route('/users')
@login_required
def get_users():
    global db
    users_list = []
    
    if db is not None:
        try:
            # Exclude the current user and get all other users
            users = list(db.users.find({'_id': {'$ne': ObjectId(current_user.id)}}, 
                                      {'_id': 1, 'name': 1, 'email': 1}))
            for user in users:
                user['id'] = str(user['_id'])
                # Check if there are unread messages
                unread_count = db.messages.count_documents({
                    'sender_id': str(user['_id']), 
                    'recipient_id': current_user.id,
                    'read': False
                })
                user['unread_count'] = unread_count
                users_list.append(user)
        except Exception as e:
            print(f"Error fetching users from MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        for user_id, user_data in memory_db['users'].items():
            if user_id != current_user.id:
                user = {
                    'id': user_id,
                    'name': user_data.get('name', 'Unknown User'),
                    'email': user_data.get('email', 'unknown@example.com'),
                }
                # Count unread messages from memory DB
                unread_count = 0
                for msg in memory_db.get('messages', []):
                    if msg.get('sender_id') == user_id and msg.get('recipient_id') == current_user.id and not msg.get('read', False):
                        unread_count += 1
                user['unread_count'] = unread_count
                users_list.append(user)
    
    return jsonify(users_list)

# Get messages between current user and another user
@app.route('/messages/<user_id>')
@login_required
def get_messages(user_id):
    global db
    messages = []
    
    if db is not None:
        try:
            # Get messages where either current user is sender and user_id is recipient
            # or current user is recipient and user_id is sender
            query = {
                '$or': [
                    {'sender_id': current_user.id, 'recipient_id': user_id},
                    {'sender_id': user_id, 'recipient_id': current_user.id}
                ]
            }
            
            msgs = list(db.messages.find(query).sort('timestamp', 1))
            
            # Format messages for display
            for msg in msgs:
                messages.append({
                    'id': str(msg['_id']),
                    'sender_id': msg['sender_id'],
                    'recipient_id': msg['recipient_id'],
                    'content': msg['content'],
                    'timestamp': msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'read': msg.get('read', False),
                    'is_self': msg['sender_id'] == current_user.id
                })
                
            # Mark messages as read if current user is recipient
            db.messages.update_many(
                {'sender_id': user_id, 'recipient_id': current_user.id, 'read': False},
                {'$set': {'read': True}}
            )
        except Exception as e:
            print(f"Error fetching messages from MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        for msg in memory_db.get('messages', []):
            if ((msg.get('sender_id') == current_user.id and msg.get('recipient_id') == user_id) or
                (msg.get('sender_id') == user_id and msg.get('recipient_id') == current_user.id)):
                messages.append({
                    'id': msg.get('id', '0'),
                    'sender_id': msg.get('sender_id'),
                    'recipient_id': msg.get('recipient_id'),
                    'content': msg.get('content', ''),
                    'timestamp': msg.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                    'read': msg.get('read', False),
                    'is_self': msg.get('sender_id') == current_user.id
                })
                
                # Mark as read if current user is recipient
                if msg.get('recipient_id') == current_user.id and not msg.get('read', False):
                    msg['read'] = True
        
        # Sort messages by timestamp
        messages.sort(key=lambda x: x['timestamp'])
    
    return jsonify(messages)

# Send a message to another user
@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    global db
    data = request.json
    recipient_id = data.get('recipient_id')
    content = data.get('content')
    
    if not recipient_id or not content:
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Create message object
    message = {
        'sender_id': current_user.id,
        'recipient_id': recipient_id,
        'content': content,
        'timestamp': datetime.now(),
        'read': False
    }
    
    if db is not None:
        try:
            result = db.messages.insert_one(message)
            message_id = str(result.inserted_id)
        except Exception as e:
            print(f"Error sending message to MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        if 'messages' not in memory_db:
            memory_db['messages'] = []
            
        message['id'] = str(memory_db['next_id'])
        memory_db['next_id'] += 1
        memory_db['messages'].append(message)
        message_id = message['id']
    
    return jsonify({
        'id': message_id,
        'sender_id': message['sender_id'],
        'recipient_id': message['recipient_id'],
        'content': message['content'],
        'timestamp': message['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
        'read': message['read'],
        'is_self': True
    })

# Get conversations (message threads) for the current user
@app.route('/conversations')
@login_required
def get_conversations():
    global db
    conversations = []
    
    if db is not None:
        try:
            # Find all users that have exchanged messages with the current user
            pipeline = [
                # Match messages where current user is either sender or recipient
                {'$match': {'$or': [
                    {'sender_id': current_user.id},
                    {'recipient_id': current_user.id}
                ]}},
                # Sort by timestamp descending to get latest messages
                {'$sort': {'timestamp': -1}},
                # Group by the other user to get latest message for each conversation
                {'$group': {
                    '_id': {
                        '$cond': [
                            {'$eq': ['$sender_id', current_user.id]},
                            '$recipient_id',
                            '$sender_id'
                        ]
                    },
                    'last_message': {'$first': '$$ROOT'},
                    'unread_count': {
                        '$sum': {
                            '$cond': [
                                {'$and': [
                                    {'$eq': ['$recipient_id', current_user.id]},
                                    {'$eq': ['$read', False]}
                                ]},
                                1, 0
                            ]
                        }
                    }
                }}
            ]
            
            result = list(db.messages.aggregate(pipeline))
            
            # Get user details for each conversation
            for convo in result:
                other_user_id = convo['_id']
                user_data = db.users.find_one({'_id': ObjectId(other_user_id)}, {'name': 1, 'email': 1})
                
                if user_data:
                    last_msg = convo['last_message']
                    conversations.append({
                        'user_id': other_user_id,
                        'name': user_data.get('name', 'Unknown User'),
                        'email': user_data.get('email', 'unknown@example.com'),
                        'last_message': last_msg.get('content', ''),
                        'timestamp': last_msg.get('timestamp').strftime('%Y-%m-%d %H:%M:%S'),
                        'unread_count': convo['unread_count'],
                        'is_sender': last_msg.get('sender_id') == current_user.id
                    })
        except Exception as e:
            print(f"Error fetching conversations from MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        # Create lookup of user_id to user info
        users_lookup = {user_id: user_data for user_id, user_data in memory_db['users'].items()}
        
        # Get all messages involving current user
        user_messages = []
        for msg in memory_db.get('messages', []):
            if msg.get('sender_id') == current_user.id or msg.get('recipient_id') == current_user.id:
                user_messages.append(msg)
        
        # Group messages by other user
        convo_map = {}
        for msg in user_messages:
            other_user_id = msg.get('recipient_id') if msg.get('sender_id') == current_user.id else msg.get('sender_id')
            
            if other_user_id not in convo_map:
                convo_map[other_user_id] = {
                    'messages': [],
                    'unread_count': 0
                }
            
            convo_map[other_user_id]['messages'].append(msg)
            
            # Count unread messages
            if msg.get('recipient_id') == current_user.id and not msg.get('read', False):
                convo_map[other_user_id]['unread_count'] += 1
        
        # Create conversations list
        for other_user_id, data in convo_map.items():
            if other_user_id in users_lookup:
                # Sort messages to get the latest one
                data['messages'].sort(key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
                last_msg = data['messages'][0] if data['messages'] else None
                
                if last_msg:
                    conversations.append({
                        'user_id': other_user_id,
                        'name': users_lookup[other_user_id].get('name', 'Unknown User'),
                        'email': users_lookup[other_user_id].get('email', 'unknown@example.com'),
                        'last_message': last_msg.get('content', ''),
                        'timestamp': last_msg.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                        'unread_count': data['unread_count'],
                        'is_sender': last_msg.get('sender_id') == current_user.id
                    })
    
    # Sort conversations by timestamp (newest first)
    conversations.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify(conversations)

# Check for new messages (for notifications)
@app.route('/check_messages')
@login_required
def check_new_messages():
    global db
    unread_count = 0
    
    if db is not None:
        try:
            # Count unread messages where current user is recipient
            unread_count = db.messages.count_documents({
                'recipient_id': current_user.id,
                'read': False
            })
        except Exception as e:
            print(f"Error checking messages from MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        for msg in memory_db.get('messages', []):
            if msg.get('recipient_id') == current_user.id and not msg.get('read', False):
                unread_count += 1
    
    return jsonify({'unread_count': unread_count})

# Check if an email exists in the database
@app.route('/check_email', methods=['POST'])
@login_required
def check_email():
    global db
    data = request.json
    email = data.get('email')
    
    if not email:
        return jsonify({'exists': False, 'error': 'No email provided'}), 400
    
    print(f"Checking if email exists: {email}")
    
    if db is not None:
        try:
            # Debug: print all users
            all_users = list(db.users.find({}, {'email': 1, 'name': 1}))
            print(f"All registered users in MongoDB: {len(all_users)} users found:")
            for user in all_users:
                print(f"  - {user.get('name', 'Unknown')}: {user.get('email', 'No email')} (ID: {user['_id']})")
            
            # Find user with this email (case-insensitive search)
            import re
            email_pattern = re.compile(f"^{re.escape(email)}$", re.IGNORECASE)
            user = db.users.find_one({'email': {'$regex': email_pattern}})
            if user:
                print(f"User found with email: {email}, ID: {user['_id']}")
                return jsonify({
                    'exists': True,
                    'user_id': str(user['_id']),
                    'name': user.get('name', 'Unknown User')
                })
            else:
                print(f"No user found with email: {email}")
                return jsonify({'exists': False})
        except Exception as e:
            print(f"Error checking email in MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        print("Falling back to memory DB for email check")
        print("Memory DB users:")
        for user_id, user_data in memory_db.get('users', {}).items():
            print(f"  - {user_data.get('name', 'Unknown')}: {user_data.get('email', 'No email')} (ID: {user_id})")
        for user_id, user_data in memory_db.get('users', {}).items():
            if user_data.get('email', '').lower() == email.lower():
                print(f"User found in memory DB with email: {email}, ID: {user_id}")
                return jsonify({
                    'exists': True,
                    'user_id': user_id,
                    'name': user_data.get('name', 'Unknown User')
                })
    
    print(f"Email not found anywhere: {email}")
    return jsonify({'exists': False})

# Request a loan
@app.route('/request_loan', methods=['POST'])
@login_required
def request_loan():
    global db
    data = request.json
    
    lender_id = data.get('lender_id')
    amount = float(data.get('amount', 0))
    term_months = int(data.get('term_months', 12))
    reason = data.get('reason', '')
    interest_rate = float(data.get('interest_rate', 5.0))
    
    if not lender_id or amount <= 0 or term_months <= 0:
        return jsonify({'success': False, 'error': 'Missing or invalid loan details'}), 400
    
    # Create loan request object
    loan_request = {
        'borrower_id': current_user.id,
        'lender_id': lender_id,
        'amount': amount,
        'remaining_amount': amount,
        'term_months': term_months,
        'reason': reason,
        'interest_rate': interest_rate,
        'status': 'pending',  # pending, approved, rejected, paid, defaulted
        'date_requested': datetime.now(),
        'date_approved': None,
        'date_completed': None,
        'payments': [],
        'monthly_payment': round((amount * (1 + interest_rate/100)) / term_months, 2)
    }
    
    if db is not None:
        try:
            result = db.loans.insert_one(loan_request)
            loan_id = str(result.inserted_id)
            loan_request['id'] = loan_id
        except Exception as e:
            print(f"Error creating loan request in MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        if 'loans' not in memory_db:
            memory_db['loans'] = []
            
        loan_request['id'] = str(memory_db['next_id'])
        loan_request['_id'] = memory_db['next_id']
        memory_db['next_id'] += 1
        memory_db['loans'].append(loan_request)
        loan_id = loan_request['id']
    
    return jsonify({
        'success': True,
        'loan_id': loan_id,
        'amount': amount,
        'term_months': term_months,
        'monthly_payment': loan_request['monthly_payment'],
        'status': 'pending'
    })

# Get loan details
@app.route('/loans/<loan_id>', methods=['GET'])
@login_required
def get_loan(loan_id):
    global db
    
    if db is not None:
        try:
            # Get the loan document from MongoDB
            if loan_id.isdigit():  # Memory DB ID
                loan = None
                # Fall through to memory DB
            else:  # MongoDB ObjectId
                loan = db.loans.find_one({'_id': ObjectId(loan_id)})
            
            if loan:
                # Check if current user is either borrower or lender
                if loan['borrower_id'] == current_user.id or loan['lender_id'] == current_user.id:
                    loan['id'] = str(loan['_id'])
                    
                    # Format dates
                    loan['date_requested'] = loan['date_requested'].strftime('%Y-%m-%d %H:%M:%S')
                    if loan.get('date_approved'):
                        loan['date_approved'] = loan['date_approved'].strftime('%Y-%m-%d %H:%M:%S')
                    if loan.get('date_completed'):
                        loan['date_completed'] = loan['date_completed'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Get borrower and lender details
                    borrower = db.users.find_one({'_id': ObjectId(loan['borrower_id'])})
                    lender = db.users.find_one({'_id': ObjectId(loan['lender_id'])})
                    
                    if borrower:
                        loan['borrower_name'] = borrower.get('name', 'Unknown User')
                        loan['borrower_email'] = borrower.get('email', 'unknown@example.com')
                    
                    if lender:
                        loan['lender_name'] = lender.get('name', 'Unknown User')
                        loan['lender_email'] = lender.get('email', 'unknown@example.com')
                    
                    # Remove _id which is not JSON serializable
                    loan.pop('_id', None)
                    
                    return jsonify(loan)
                else:
                    return jsonify({'error': 'Not authorized to view this loan'}), 403
            
        except Exception as e:
            print(f"Error fetching loan from MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed or if using memory DB ID
    for loan in memory_db.get('loans', []):
        if loan.get('id') == loan_id or str(loan.get('_id')) == loan_id:
            # Check if current user is either borrower or lender
            if loan['borrower_id'] == current_user.id or loan['lender_id'] == current_user.id:
                # Create a copy to modify
                loan_copy = loan.copy()
                
                # Format dates (if they're datetime objects)
                if isinstance(loan_copy.get('date_requested'), datetime):
                    loan_copy['date_requested'] = loan_copy['date_requested'].strftime('%Y-%m-%d %H:%M:%S')
                if isinstance(loan_copy.get('date_approved'), datetime):
                    loan_copy['date_approved'] = loan_copy['date_approved'].strftime('%Y-%m-%d %H:%M:%S')
                if isinstance(loan_copy.get('date_completed'), datetime):
                    loan_copy['date_completed'] = loan_copy['date_completed'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Get borrower and lender details
                for user_id, user_data in memory_db['users'].items():
                    if user_id == loan['borrower_id']:
                        loan_copy['borrower_name'] = user_data.get('name', 'Unknown User')
                        loan_copy['borrower_email'] = user_data.get('email', 'unknown@example.com')
                    if user_id == loan['lender_id']:
                        loan_copy['lender_name'] = user_data.get('name', 'Unknown User')
                        loan_copy['lender_email'] = user_data.get('email', 'unknown@example.com')
                
                return jsonify(loan_copy)
            else:
                return jsonify({'error': 'Not authorized to view this loan'}), 403
    
    return jsonify({'error': 'Loan not found'}), 404

# Get user's loans
@app.route('/user_loans', methods=['GET'])
@login_required
def get_user_loans():
    global db
    loans = []
    
    if db is not None:
        try:
            # Get all loans where current user is either borrower or lender
            query = {
                '$or': [
                    {'borrower_id': current_user.id},
                    {'lender_id': current_user.id}
                ]
            }
            
            cursor = db.loans.find(query).sort('date_requested', -1)
            
            # Process each loan
            for loan in cursor:
                loan_obj = {
                    'id': str(loan['_id']),
                    'borrower_id': loan['borrower_id'],
                    'lender_id': loan['lender_id'],
                    'amount': loan['amount'],
                    'remaining_amount': loan.get('remaining_amount', loan['amount']),
                    'term_months': loan['term_months'],
                    'reason': loan.get('reason', ''),
                    'interest_rate': loan.get('interest_rate', 0),
                    'status': loan['status'],
                    'date_requested': loan['date_requested'].strftime('%Y-%m-%d %H:%M:%S'),
                    'monthly_payment': loan.get('monthly_payment', 0),
                    'is_borrower': loan['borrower_id'] == current_user.id,
                    'payments_count': len(loan.get('payments', []))
                }
                
                # Add formatted dates if they exist
                if loan.get('date_approved'):
                    loan_obj['date_approved'] = loan['date_approved'].strftime('%Y-%m-%d %H:%M:%S')
                if loan.get('date_completed'):
                    loan_obj['date_completed'] = loan['date_completed'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Get counterparty details (other user's name)
                counterparty_id = loan['lender_id'] if loan['borrower_id'] == current_user.id else loan['borrower_id']
                try:
                    counterparty = db.users.find_one({'_id': ObjectId(counterparty_id)})
                    if counterparty:
                        loan_obj['counterparty_name'] = counterparty.get('name', 'Unknown User')
                except Exception as e:
                    print(f"Error fetching counterparty details: {e}")
                    loan_obj['counterparty_name'] = 'Unknown User'
                
                loans.append(loan_obj)
        except Exception as e:
            print(f"Error fetching loans from MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        for loan in memory_db.get('loans', []):
            if loan.get('borrower_id') == current_user.id or loan.get('lender_id') == current_user.id:
                loan_obj = {
                    'id': loan.get('id'),
                    'borrower_id': loan.get('borrower_id'),
                    'lender_id': loan.get('lender_id'),
                    'amount': loan.get('amount'),
                    'remaining_amount': loan.get('remaining_amount', loan.get('amount')),
                    'term_months': loan.get('term_months'),
                    'reason': loan.get('reason', ''),
                    'interest_rate': loan.get('interest_rate', 0),
                    'status': loan.get('status'),
                    'monthly_payment': loan.get('monthly_payment', 0),
                    'is_borrower': loan.get('borrower_id') == current_user.id,
                    'payments_count': len(loan.get('payments', []))
                }
                
                # Format date
                if isinstance(loan.get('date_requested'), datetime):
                    loan_obj['date_requested'] = loan['date_requested'].strftime('%Y-%m-%d %H:%M:%S')
                else:
                    loan_obj['date_requested'] = str(loan.get('date_requested', ''))
                
                if isinstance(loan.get('date_approved'), datetime):
                    loan_obj['date_approved'] = loan['date_approved'].strftime('%Y-%m-%d %H:%M:%S')
                    
                if isinstance(loan.get('date_completed'), datetime):
                    loan_obj['date_completed'] = loan['date_completed'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Get counterparty details
                counterparty_id = loan.get('lender_id') if loan.get('borrower_id') == current_user.id else loan.get('borrower_id')
                if counterparty_id in memory_db['users']:
                    loan_obj['counterparty_name'] = memory_db['users'][counterparty_id].get('name', 'Unknown User')
                else:
                    loan_obj['counterparty_name'] = 'Unknown User'
                
                loans.append(loan_obj)
    
    # Sort by date requested (newest first)
    loans.sort(key=lambda x: x.get('date_requested', ''), reverse=True)
    
    return jsonify(loans)

# Approve or reject a loan request
@app.route('/respond_to_loan', methods=['POST'])
@login_required
def respond_to_loan():
    global db
    data = request.json
    
    loan_id = data.get('loan_id')
    action = data.get('action')  # 'approve' or 'reject'
    
    if not loan_id or not action:
        return jsonify({'success': False, 'error': 'Missing loan_id or action'}), 400
    
    if action not in ['approve', 'reject']:
        return jsonify({'success': False, 'error': 'Invalid action'}), 400
    
    # Determine updated status
    new_status = 'approved' if action == 'approve' else 'rejected'
    
    if db is not None:
        try:
            # Get the loan document from MongoDB
            loan = db.loans.find_one({'_id': ObjectId(loan_id)})
            
            if not loan:
                return jsonify({'success': False, 'error': 'Loan not found'}), 404
            
            # Check if current user is the lender
            if loan['lender_id'] != current_user.id:
                return jsonify({'success': False, 'error': 'Not authorized to respond to this loan'}), 403
            
            # Check if loan is already approved or rejected
            if loan['status'] != 'pending':
                return jsonify({'success': False, 'error': f'Loan is already {loan["status"]}'}), 400
            
            # Update the loan status
            update_data = {
                'status': new_status,
                'date_approved': datetime.now() if action == 'approve' else None
            }
            
            result = db.loans.update_one(
                {'_id': ObjectId(loan_id)},
                {'$set': update_data}
            )
            
            if result.modified_count == 0:
                return jsonify({'success': False, 'error': 'Failed to update loan status'}), 500
            
        except Exception as e:
            print(f"Error updating loan in MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        found_loan = False
        for loan in memory_db.get('loans', []):
            if loan.get('id') == loan_id:
                found_loan = True
                
                # Check if current user is the lender
                if loan.get('lender_id') != current_user.id:
                    return jsonify({'success': False, 'error': 'Not authorized to respond to this loan'}), 403
                
                # Check if loan is already approved or rejected
                if loan.get('status') != 'pending':
                    return jsonify({'success': False, 'error': f'Loan is already {loan.get("status")}'}), 400
                
                # Update the loan status
                loan['status'] = new_status
                if action == 'approve':
                    loan['date_approved'] = datetime.now()
                break
        
        if not found_loan:
            return jsonify({'success': False, 'error': 'Loan not found'}), 404
    
    return jsonify({
        'success': True,
        'status': new_status,
        'message': f'Loan has been {new_status}'
    })

# Make a payment on a loan
@app.route('/make_payment', methods=['POST'])
@login_required
def make_payment():
    global db
    data = request.json
    
    loan_id = data.get('loan_id')
    amount = float(data.get('amount', 0))
    
    if not loan_id or amount <= 0:
        return jsonify({'success': False, 'error': 'Missing loan_id or invalid amount'}), 400
    
    payment = {
        'amount': amount,
        'date': datetime.now(),
        'payer_id': current_user.id
    }
    
    if db is not None:
        try:
            # Get the loan document from MongoDB
            loan = db.loans.find_one({'_id': ObjectId(loan_id)})
            
            if not loan:
                return jsonify({'success': False, 'error': 'Loan not found'}), 404
            
            # Check if current user is the borrower
            if loan['borrower_id'] != current_user.id:
                return jsonify({'success': False, 'error': 'Only the borrower can make payments'}), 403
            
            # Check if loan is approved
            if loan['status'] != 'approved':
                return jsonify({'success': False, 'error': f'Cannot make payment on a loan with status: {loan["status"]}'}), 400
            
            # Add payment to the payments array
            result = db.loans.update_one(
                {'_id': ObjectId(loan_id)},
                {
                    '$push': {'payments': payment},
                    '$inc': {'remaining_amount': -amount}
                }
            )
            
            if result.modified_count == 0:
                return jsonify({'success': False, 'error': 'Failed to add payment'}), 500
            
            # Get the updated loan to check if it's fully paid
            updated_loan = db.loans.find_one({'_id': ObjectId(loan_id)})
            
            # Check if loan is fully paid
            remaining = updated_loan.get('remaining_amount', 0)
            if remaining <= 0:
                # Mark loan as paid
                db.loans.update_one(
                    {'_id': ObjectId(loan_id)},
                    {
                        '$set': {
                            'status': 'paid',
                            'date_completed': datetime.now(),
                            'remaining_amount': 0  # Ensure it's exactly 0
                        }
                    }
                )
                
                return jsonify({
                    'success': True,
                    'payment_amount': amount,
                    'status': 'paid',
                    'message': 'Payment made successfully. Loan is now fully paid!'
                })
            
        except Exception as e:
            print(f"Error making payment in MongoDB: {e}")
            db = None
    
    # Fall back to memory DB if MongoDB failed
    if db is None:
        found_loan = False
        for loan in memory_db.get('loans', []):
            if loan.get('id') == loan_id:
                found_loan = True
                
                # Check if current user is the borrower
                if loan.get('borrower_id') != current_user.id:
                    return jsonify({'success': False, 'error': 'Only the borrower can make payments'}), 403
                
                # Check if loan is approved
                if loan.get('status') != 'approved':
                    return jsonify({'success': False, 'error': f'Cannot make payment on a loan with status: {loan.get("status")}'}), 400
                
                # Add payment to the payments array
                if 'payments' not in loan:
                    loan['payments'] = []
                loan['payments'].append(payment)
                
                # Update remaining amount
                if 'remaining_amount' not in loan:
                    loan['remaining_amount'] = loan.get('amount', 0)
                loan['remaining_amount'] -= amount
                
                # Check if loan is fully paid
                if loan['remaining_amount'] <= 0:
                    loan['status'] = 'paid'
                    loan['date_completed'] = datetime.now()
                    loan['remaining_amount'] = 0  # Ensure it's exactly 0
                    
                    return jsonify({
                        'success': True,
                        'payment_amount': amount,
                        'status': 'paid',
                        'message': 'Payment made successfully. Loan is now fully paid!'
                    })
                break
        
        if not found_loan:
            return jsonify({'success': False, 'error': 'Loan not found'}), 404
    
    return jsonify({
        'success': True,
        'payment_amount': amount,
        'remaining_amount': loan.get('remaining_amount') if db is None else updated_loan.get('remaining_amount'),
        'message': 'Payment made successfully'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True) 