# AI Loan Calculator

A modern web application that combines loan calculation capabilities with an AI-powered chatbot assistant. The application features an intuitive interface, interactive graphs, and real-time loan calculations.

## Features

- Loan amount, interest rate, and term calculation
- Monthly payment and total cost breakdown
- Interactive amortization schedule graph
- AI-powered chatbot assistant with two modes:
  - Standard Mode: Rule-based responses to common loan questions
  - Custom AI Mode: Advanced AI responses to complex questions (using Hugging Face API)
- Modern, responsive design
- Real-time calculations and updates

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository or download the files
2. Navigate to the project directory
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. (Optional) Get a free Hugging Face API key:
   - Sign up at [Hugging Face](https://huggingface.co/join)
   - Go to your profile → Settings → Access Tokens
   - Create a new token with read permissions
   - Replace the dummy key in `app.py` with your API key:
     ```python
     headers = {"Authorization": "Bearer YOUR_API_KEY"}
     ```

## Usage

1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Enter your loan details in the calculator
4. Interact with the AI assistant for additional information:
   - Toggle between Standard and Custom AI modes using the switch
   - Standard mode gives fast, pre-programmed responses to common questions
   - Custom AI mode handles complex or unusual loan-related questions

## Project Structure

```
ai_loan_calculator/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/          # HTML templates
│   └── index.html     # Main application template
└── README.md          # Project documentation
```

## Technologies Used

- Flask (Python web framework)
- Bootstrap 5 (CSS framework)
- jQuery (JavaScript library)
- Plotly (Interactive graphs)
- NLTK (Natural Language Processing)
- scikit-learn (Machine Learning)

## Contributing

Feel free to submit issues and enhancement requests! 