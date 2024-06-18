import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType

st.set_page_config(
    page_title="AI Movie Script Autodubbing",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

api = st.sidebar.text_input("Enter Your OPENAI API KEY HERE", type="password")

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("AI Movie Script Autodubbing💻")
st.sidebar.markdown("## Welcome to the AI Movie Script Autodubbing!")
st.sidebar.markdown(
    "This App Harnesses power of Lyzr Automata to Translate Movie script to many languages. You Need to input Your script and Language, this app Translate Your movie script to your desired language.")

if api:
    openai_model = OpenAIModel(
        api_key=api,
        parameters={
            "model": "gpt-4-turbo-preview",
            "temperature": 0.2,
            "max_tokens": 1500,
        },
    )
else:
    st.sidebar.error("Please Enter Your OPENAI API KEY")


def script_translator(lang1, lang2, script):
    translator_agent = Agent(
        prompt_persona=f"You are a Script translator with over 10 years of experience in the film industry.You have a deep understanding of both {lang1} and {lang2} languages and is well-versed in the nuances of movie scripts.",
        role="Script Translation",
    )

    translation_task = Task(
        name="Script Translation Task",
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
        model=openai_model,
        agent=translator_agent,
        log_output=True,
        instructions=f"""Translate the provided movie script from {lang1} to {lang2} while maintaining the original tone, style, and cultural context.
        Follow Below Instructions:
            - Ensure that the translation is accurate and conveys the original meaning and emotions of the dialogues and descriptions.
            - Adapt cultural references appropriately to make sense to a {lang2}-speaking audience.
            - Maintain the natural flow of conversations and descriptions, ensuring that the translated text sounds natural to native {lang2} speakers.
            - Keep the characters' personalities and voices consistent with the original script.
            - Preserve the formatting of the script, including scene headings, action lines, and dialogues.
            
        Script: {script}
       
        """,
    )

    output = LinearSyncPipeline(
        name="Script Translation",
        completion_message="Script Translation Done!",
        tasks=[
            translation_task
        ],
    ).run()
    return output[0]['task_output']


language1 = st.text_input("Enter Your Script Language", placeholder="English")
language2 = st.text_input("Enter Translating language", placeholder="Hindi")
scripts = st.text_area("Enter Your Script", height=300)

if st.button("Translate"):
    solution = script_translator(language1, language2, scripts)
    st.markdown(solution)
