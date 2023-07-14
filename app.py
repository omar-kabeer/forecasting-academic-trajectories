import gradio as gr
from sklearn.linear_model import LogisticRegression

# Load the logistic regressor model
model = LogisticRegression()
model.load_model('/src/pickles/model.pkl')

# Create a gradio app
app = gr.Interface(model, inputs="numerical", outputs="classification")

# Run the app
app.launch()
