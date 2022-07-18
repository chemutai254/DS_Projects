import joblib
import streamlit as st

# Define functions
with open('svc_model1.pkl', 'rb') as model:
    classifier = joblib.load(model)
    
def predictor(sepal_length, sepal_width, petal_length, petal_width):
    global classifier
    prediction = classifier.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if prediction == 0:
        return 'setosa'
    elif prediction == 1:
        return 'versicolor'
    else:
        return 'virginica'
    
# Create web app
def main():
    # Title
    st.title('Iris Prediciton App')
    
    # Body Input
    sepal_length = st.number_input('Sepal Length')
    sepal_width = st.number_input('Sepal Width')
    petal_length = st.number_input('Petal Length')
    petal_width = st.number_input('Petal Width')
       
    # Predict
    if st.button('Predict'):
        prediction = predictor(sepal_length, sepal_width, petal_length, petal_width)
        st.success(f'The flower is an iris {prediction}')
        
if __name__ == '__main__':
        main()