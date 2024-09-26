import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from streamlit_image_comparison import image_comparison
from utils import upscale_image


def ui():
    image = None
    input_text = None
    uploaded_file = None

    input_area = st.columns([2, 1, 1])

    with input_area[0]:
        option = st.selectbox(
            "How do you want to provide the image?",
            ("Fetch from URL", "Upload from local machine")
        )

    with input_area[2]:
        option_scale = st.selectbox(
            "Which factors do you want to upscale?",
            (2, 3, 4)
        )

    with input_area[1]:
        # , 'SRUNET_interpolation', 'SRUNET_x234_interpolation'
        option_model = st.selectbox(
            "Which model do you want to use?",
            ('SRUNET_x2', 'SRUNET_x3', 'SRUNET_x4', 'SRUNET_x234')
        )

    picture_url_area = st.columns([2, 2], vertical_alignment="top")
    with picture_url_area[0]:
        if option == "Upload from local machine":
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"])
        elif option == "Fetch from URL":
            input_text = st.text_input("Enter the image URL")

        if st.button("Submit"):
            if option == "Upload from local machine" and uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    # st.image(image, caption="Uploaded Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Error opening image: {e}")
            elif option == "Fetch from URL" and input_text:
                try:
                    response = requests.get(input_text)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content))
                    # st.image(image, caption="Image from URL", use_column_width=True)
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching image: {e}")

            if image:
                width, height = image.size
                if width * int(option_scale) > 1000 or height * int(option_scale) > 1000:
                    st.error(
                        "Unable to upscale. The size of upscaled image should be less than 1000x1000")
                    image = None

    if image:
        st.header('Results')
        # image_resize = image.resize((1024, 1024))
        width, height = image.size
        picture_url_area[1].text(
            f"Image size: {width}x{height} --> {width*int(option_scale)}x{height*int(option_scale)}")
        picture_url_area[1].image(
            image, caption="Original", use_column_width=True)

        img_1 = image.resize(
            (width*int(option_scale), height*int(option_scale)), Image.BICUBIC)
        img_2 = upscale_image(image, option_model, int(option_scale))

        image_comparison(
            img1=img_1,
            img2=img_2,
        )
