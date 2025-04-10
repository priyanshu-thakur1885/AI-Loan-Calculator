from pymongo import MongoClient
import os
import sys
from datetime import datetime
from flask_bcrypt import Bcrypt

# Initialize bcrypt for password hashing
bcrypt = Bcrypt()

# MongoDB connection settings
MONGODB_URI = 'mongodb://localhost:27017/'
DB_NAME = 'loan_calculator_db'

def setup_mongodb():
    """Initialize MongoDB database, create collections and indexes."""
    try:
        # Connect to MongoDB
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        
        # Print connection info
        print("===========================================")
        
        # Set up collections if they don't exist
        if 'users' not in db.list_collection_names():
            db.create_collection('users')
            print("Created 'users' collection")
        
        if 'calculations' not in db.list_collection_names():
            db.create_collection('calculations')
            print("Created 'calculations' collection")
        
        if 'chat_logs' not in db.list_collection_names():
            db.create_collection('chat_logs')
            print("Created 'chat_logs' collection")

        if 'messages' not in db.list_collection_names():
            db.create_collection('messages')
            print("Created 'messages' collection")
        
        if 'loans' not in db.list_collection_names():
            db.create_collection('loans')
            print("Created 'loans' collection")
        
        # Create indexes for efficient queries
        db.users.create_index([('email', 1)], unique=True)
        print("Created index on users.email")
        
        db.calculations.create_index([('user_id', 1), ('date_created', -1)])
        print("Created index on calculations.user_id and date_created")
        
        db.chat_logs.create_index([('user_id', 1), ('timestamp', -1)])
        print("Created index on chat_logs.user_id and timestamp")

        db.messages.create_index([('sender_id', 1), ('recipient_id', 1), ('timestamp', -1)])
        print("Created index on messages.sender_id, recipient_id and timestamp")
        
        db.loans.create_index([('lender_id', 1), ('borrower_id', 1), ('status', 1)])
        print("Created index on loans.lender_id, borrower_id and status")
        
        print("\nMongoDB setup completed successfully!")
        
        # Print connection details for MongoDB Compass
        print("\nMongoDB Compass Connection Info:")
        print(f"Connection String: {MONGODB_URI}")
        print(f"Database Name: {DB_NAME}")
        print("Collections: users, calculations, chat_logs, messages, loans")
        
        # Optionally create admin user if not exists
        admin = db.users.find_one({'email': 'admin@example.com'})
        if not admin:
            from datetime import datetime
            import bcrypt
            
            # Create admin user with hashed password
            hashed_password = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
            admin_user = {
                'name': 'Admin User',
                'email': 'admin@example.com',
                'password': hashed_password.decode('utf-8'),
                'date_registered': datetime.now(),
                'is_admin': True
            }
            db.users.insert_one(admin_user)
            print("\nCreated admin user (email: admin@example.com, password: admin123)")
        
        return True
        
    except Exception as e:
        print(f"Error setting up MongoDB: {e}")
        return False

def import_sample_data(db=None):
    """Import sample data for testing purposes."""
    try:
        # If db is not provided, connect to database
        if db is None:
            client = MongoClient(MONGODB_URI)
            db = client[DB_NAME]
        
        # Get admin user
        admin = db.users.find_one({'email': 'admin@example.com'})
        if not admin:
            print("Admin user not found. Please run setup_mongodb() first.")
            return False
        
        admin_id = str(admin['_id'])
        
        # Add sample calculations if collection is empty
        if db.calculations.count_documents({}) == 0:
            sample_calculations = [
                {
                    'user_id': admin_id,
                    'principal': 250000,
                    'interest_rate': 4.5,
                    'term_years': 30,
                    'monthly_payment': 1267.05,
                    'total_payment': 456138.00,
                    'total_interest': 206138.00,
                    'date_created': datetime.now()
                },
                {
                    'user_id': admin_id,
                    'principal': 25000,
                    'interest_rate': 6.5,
                    'term_years': 5,
                    'monthly_payment': 489.27,
                    'total_payment': 29356.20,
                    'total_interest': 4356.20,
                    'date_created': datetime.now()
                }
            ]
            db.calculations.insert_many(sample_calculations)
            print(f"Added {len(sample_calculations)} sample calculations")
        
        # Add sample chat logs if collection is empty
        if db.chat_logs.count_documents({}) == 0:
            sample_chats = [
                {
                    'user_id': admin_id,
                    'message': 'What are your interest rates?',
                    'response': 'Our loans have competitive interest rates starting from 5.99% APR. The exact rate depends on your credit score and loan amount.',
                    'mode': 'standard',
                    'timestamp': datetime.now()
                },
                {
                    'user_id': admin_id,
                    'message': 'How do I qualify for a loan?',
                    'response': 'To be eligible for our loans, you need to be at least 18 years old, have a stable income, and a credit score of 650 or higher.',
                    'mode': 'standard',
                    'timestamp': datetime.now()
                }
            ]
            db.chat_logs.insert_many(sample_chats)
            print(f"Added {len(sample_chats)} sample chat logs")
        
        print("\nSample data import completed successfully!")
        return True
    
    except Exception as e:
        print(f"Error importing sample data: {e}")
        return False

if __name__ == "__main__":
    print("MongoDB Setup Utility for AI Loan Calculator")
    print("===========================================")
    
    # Setup MongoDB database
    if setup_mongodb():
        # Ask if user wants to import sample data
        response = input("\nDo you want to import sample data? (y/n): ")
        if response.lower() == 'y':
            import_sample_data()
    
    print("\nSetup completed. You can now run the application with: python app.py") 