from openai import OpenAI
from pathlib import Path

import base64
import requests
import streamlit as st

# Image to text with OpenAI
def image_to_text(image_path: Path, open_ai_key: str) -> str:
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    client = OpenAI(api_key=open_ai_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe what is in this image in 30 words or less"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    text = response.choices[0].message.content
    return text

def generate_story(scenario: str, open_ai_key: str) -> str:
    client = OpenAI(api_key=open_ai_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a story teller. You can generate a short story based on a simple narrative, the story should be no more then 60 words."
            },
            {
                "role": "user",
                "content": scenario
            }
        ],
        max_tokens=300
    )
    story = response.choices[0].message.content
    return story


def story_to_speech(story: str, open_ai_key: str) -> Path:
    client = OpenAI(api_key=open_ai_key)
    speech_file_path = Path(__file__).parent / "audio" / "story.mp3"
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=story
    ) as response:
        response.stream_to_file(speech_file_path)

    return speech_file_path


def main():
    st.set_page_config(page_title="Image to Story", page_icon="📖")
    st.title("Picstory")
    st.header("Turn an image into a short audio story")
    
    open_ai_key = "sk-proj-NGqCNDPPpA3iIjHP5s9FT3BlbkFJM6efNjhzZHAj6G5I1Rzx"
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        img = uploaded_image.getvalue()
        upload_path = Path(__file__).parent / "images" / uploaded_image.name
        with open(upload_path, "wb") as f:
            f.write(img)

        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Processing image..."):
            scenario = image_to_text(upload_path, open_ai_key)
        
        with st.expander("Scenario"):
            st.write(scenario)
        
        with st.spinner("Generating story..."):
            story = generate_story(scenario, open_ai_key)

        with st.expander("Story"):
            st.write(story)

        with st.spinner("Generating audio..."):
            speech_file_path = story_to_speech(story, open_ai_key)

        st.audio(str(speech_file_path), format="audio/mp3", autoplay=True)


if __name__ == "__main__":
    main()

