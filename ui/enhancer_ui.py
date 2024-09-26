import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from streamlit_image_comparison import image_comparison
from utils import upscale_image, enhanced_image


def ui():
    img_selected = None
    input_text = None
    uploaded_file = None

    input_image_area = st.columns(2)

    with input_image_area[0]:
        option = st.selectbox(
            "How do you want to provide the image?",
            ("Fetch from URL", "Upload from local machine"),
            key="option_enhanced"
        )
        if option == "Upload from local machine":
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"], key='file_enhanced')
        elif option == "Fetch from URL":
            input_text = st.text_input(
                "Enter the image URL", key='input_enhanced')

        if st.button("Submit", key="submit_enhanced"):
            if option == "Upload from local machine" and uploaded_file is not None:
                try:
                    img_selected = Image.open(uploaded_file)
                    # st.image(image, caption="Uploaded Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error opening image: {e}")
            elif option == "Fetch from URL" and input_text:
                try:
                    response = requests.get(input_text)
                    response.raise_for_status()
                    img_selected = Image.open(BytesIO(response.content))
                    # st.image(image, caption="Image from URL", use_column_width=True)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching image: {e}")

            if img_selected:
                width, height = img_selected.size
                if width > 1000 or height > 1000:
                    st.error(
                        "Unable to upscale. The size of upscaled image should be less than 1000x1000")
                    img_selected = None


    with input_image_area[1]:
        option_model = st.selectbox(
            "Which model do you want to use?",
            ('SRUNET_x2', 'SRUNET_x3', 'SRUNET_x4', 'SRUNET_x234'),
            key="option_model_enhanced"
        )

    
    if img_selected:
        st.header('Results')
        st.text(f'Model: {option_model}')
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_selected, caption="Original",
                        use_column_width=True)
        with col2:
            img_enhanced = enhanced_image(img_selected, option_model)
            # img_enhanced = img_selected.resize((64, 64))
            col2.image(img_enhanced, caption="Enhanced",
                    use_column_width=True)

        image_comparison(
            img1=img_selected,
            img2=img_enhanced,
        )
        
       
       