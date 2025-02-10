import streamlit as st
from streamlit_chat import message
import re
import sys
import os
import schedule
import time

sys.path.append(".")
import constant as const
from src.data_loader.data_loader_main import process_vector_store, remove_vector_store
from src.llm.llm import final_chain
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.environ["APP_NAME"]
APP_LOGO_URL = os.environ["APP_LOGO_URL"]

APP_USERNAME=os.environ["APP_USERNAME"]
APP_PASSWORD=os.environ["APP_PASSWORD"]

USERNAME = APP_USERNAME or "admin"
PASSWORD = APP_PASSWORD or "admin"

def process_folder(folder_path):
    src_path = os.path.join(folder_path, "pending_doc")
    for filename in os.listdir(src_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(src_path, filename)
            print(file_path)
            process_vector_store(file_path, None)

def scheduled_job():
    folder_path = "./source_data/"
    print("running")
    process_folder(folder_path)

def run_scheduler():
    schedule.every().minute.do(scheduled_job)

    while True:
        schedule.run_pending()
        time.sleep(1)

# threading.Thread(target=run_scheduler).start()

def authenticate(username, password):
    return True
    
def set_page_configuration():
    st.set_page_config(
        "RAG Demo",
        initial_sidebar_state="expanded",
        page_icon="logo.png",
    )

def generate_response(prompt_input):
    output = final_chain(prompt_input, const.FAISS_DB1)
    return output

def clear_vector_db(db_name):
    remove_vector_store(db_name)

set_page_configuration()

logo_image = APP_LOGO_URL
name = APP_NAME or 'RAG Demo'

logo_width = 150

title_text_size = "2.5rem"

st.markdown(
    f"""
    <div style='display: flex; justify-content: center;'>
        <img src='{logo_image}' alt='Logo' style='max-width: {logo_width}px;'>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"<h1 style='text-align: center; font-size: {title_text_size}'>{name}</h1>",
    unsafe_allow_html=True
)

if "login_status" not in st.session_state:
    st.session_state.login_status = False

if not st.session_state.login_status:
    login_form = st.form("login_form")
    login_username = login_form.text_input("Username")
    login_password = login_form.text_input("Password", type="password")
    login_button = login_form.form_submit_button("Login")

    if login_button:
        if authenticate(login_username, login_password):
            st.session_state.login_status = True
            login_form.empty()
            st.rerun()
        else:
            st.warning("Incorrect username or password. Please try again.")

if st.session_state.login_status:
    section = st.radio("Select Section:", ["Query Section", "Data Input Section"])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


    if section == "Query Section":
        st.text("Provide your query below. We are happy to help!")

        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "How may I assist you today?"}
            ]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        print("--" * 25)
                        print("\nUser question: \n", prompt)
                        response = generate_response(prompt)
                        placeholder = st.empty()
                        full_response = ""
                        print("\n\n")
                        print("RESPONSE KEYS: \n\n", response.keys(), "\n\n")

                        ans = ""
                        if response["answer"] != "":
                            ans = response["answer"]
                            ans = (
                                ans.replace("ANSWER:", "")
                                .replace("More Information:", "\n\n **More Information:**")
                                .strip()
                            )
                            full_response += full_response + f"\n\n{ans}"
                            placeholder.markdown(full_response)

                        refs_lst = ""
                        if response["sources"] == "":
                            pass
                        else:
                            refs = response["sources"].split(",")
                            for ref in refs:
                                refs_lst += f"\n- {ref}"

                            full_response = full_response + f"\n\n **References**: \n{refs_lst}"
                            placeholder.markdown(full_response)
                        print("--" * 25)
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)

    elif section == "Data Input Section":
        st.text("Input data by uploading a PDF or providing a web URL below.")

        if st.button("Clear Vector Database"):
            clear_vector_db(const.FAISS_DB1_PATH)
            st.success("Vector database cleared successfully!")

        pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
        if pdf_file:
            st.success("PDF file uploaded successfully!")

        web_url = st.text_input("Enter web URL:")
        if st.button("Submit URL"):
            url_pattern = re.compile(r"https?://\S+|www\.\S+")
            if not url_pattern.match(web_url):
                st.error("Invalid URL format. Please provide a valid URL.")
            else:
                st.success(f"Web URL submitted successfully: {web_url}")
            web_url = ""

        if pdf_file or web_url:
            process_vector_store(pdf_file, web_url, const.FAISS_DB1)
