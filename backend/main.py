import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from frontend.ui_gradio import user_interface

if __name__ == "__main__":
    app = user_interface()
    app.launch()
