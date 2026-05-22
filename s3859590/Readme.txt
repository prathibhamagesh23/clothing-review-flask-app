# Clothing Review and Recommendation Web App

This is a Flask-based web application that allows users to browse clothing items, search by keyword, view details and past reviews, and submit new reviews. The application uses a machine learning model to predict whether a user would recommend a product based on their written review.


## Project Structure

The project includes the following files and directories:

- app.py : Main Flask application file
- data/
  - assignment3_II.csv : Dataset of clothing items and reviews
  - reviews.json : File to store newly submitted reviews
- templates/
  - base.html : Base layout template
  - index.html : Homepage and item search
  - item.html : Individual item details with reviews
  - new_review.html : Review submission form
  - confirm_review.html : Confirmation page with prediction
  - about.html : Static about page
- static/ (optional)
  - style.css : Custom styles if needed
- utils.py : Model loading and vectorization functions
- README.md : Project documentation

How to Run the App

1. Clone or download the project files.
2. Install the required Python packages. You can use pip:
3. Make sure the following files exist in the `data/` folder:
- assignment3_II.csv
- reviews.json (can be empty initially)

4. Run the Flask application:
python app.py

5. Open a web browser and go to:
http://127.0.0.1:5000/


## How It Works

- The home page displays a searchable list of clothing items.
- Users can search for clothing titles using a live search bar with suggestions.
- Clicking an item title opens the detail page for that clothing item.
- Users can write a new review on the item page.
- On submission, the application uses a FastText-based machine learning model to predict if the review indicates a recommendation or not.
- Users can accept or override the prediction and then submit the review.
- Submitted reviews are stored in memory and also written to `data/reviews.json`.


## Machine Learning Details

- The model is loaded using `utils.py` and includes:
- FastText word embeddings
- TF-IDF vectorization
- Logistic Regression classifier
- Only the review title and text are used for prediction.
- Ratings are displayed but not used in the prediction model.


## Search Function

- The search bar uses fuzzy string matching to suggest and filter results based on partial matches in the clothing title or description.
- The result count is displayed live as users type.
- The backend supports flexible matching using Python's difflib library.


## Recommendation Display

- In item pages and review results, the label "Recommended" or "Not Recommended" is shown.
- These labels are color-coded (green for recommended, red for not recommended) to make them easily identifiable.


## Notes

- This app is intended for academic or demonstration use only.
- No authentication or database is used. Reviews are stored in memory and optionally written to a JSON file.
- The project was built as part of the Advanced Programming for Data Science (COSC2820/2815) course.


## Author

Prathibha Magesh  
Master of Data Science  
RMIT University  
8 June, 2025
