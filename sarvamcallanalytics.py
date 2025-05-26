import streamlit as st
import requests
import json
import asyncio
import aiofiles
import os
from urllib.parse import urlparse
from azure.storage.filedatalake.aio import DataLakeDirectoryClient, FileSystemClient
from azure.storage.filedatalake import ContentSettings
import mimetypes
import logging
from datetime import datetime
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = 'https://api.sarvam.ai/call-analytics/'

class SarvamClient:
    def __init__(self, url: str):
        self.account_url, self.file_system_name, self.directory_name, self.sas_token = (
            self._extract_url_components(url)
        )
        self.lock = asyncio.Lock()

    def update_url(self, url: str):
        self.account_url, self.file_system_name, self.directory_name, self.sas_token = (
            self._extract_url_components(url)
        )

    def _extract_url_components(self, url: str):
        parsed_url = urlparse(url)
        account_url = f"{parsed_url.scheme}://{parsed_url.netloc}".replace(
            ".blob.", ".dfs."
        )
        path_components = parsed_url.path.strip("/").split("/")
        file_system_name = path_components[0]
        directory_name = "/".join(path_components[1:])
        sas_token = parsed_url.query
        return account_url, file_system_name, directory_name, sas_token

    async def upload_files(self, files):
        async with DataLakeDirectoryClient(
            account_url=f"{self.account_url}?{self.sas_token}",
            file_system_name=self.file_system_name,
            directory_name=self.directory_name,
            credential=None,
        ) as directory_client:
            for file in files:
                await self._upload_file(directory_client, file)

    async def _upload_file(self, directory_client, file):
        try:
            file_name = file.name
            mime_type = mimetypes.guess_type(file_name)[0] or "audio/wav"
            file_client = directory_client.get_file_client(file_name)
            
            # Convert StreamlitUploadedFile to bytes
            content = file.getvalue()
            
            await file_client.upload_data(
                content,
                overwrite=True,
                content_settings=ContentSettings(content_type=mime_type),
            )
            st.success(f"✅ File uploaded successfully: {file_name}")
            return True
        except Exception as e:
            st.error(f"❌ Upload failed for {file_name}: {str(e)}")
            return False

    async def list_files(self):
        file_names = []
        async with FileSystemClient(
            account_url=f"{self.account_url}?{self.sas_token}",
            file_system_name=self.file_system_name,
            credential=None,
        ) as file_system_client:
            async for path in file_system_client.get_paths(self.directory_name):
                file_name = path.name.split("/")[-1]
                async with self.lock:
                    file_names.append(file_name)
        return file_names

    async def download_file(self, file_name):
        async with DataLakeDirectoryClient(
            account_url=f"{self.account_url}?{self.sas_token}",
            file_system_name=self.file_system_name,
            directory_name=self.directory_name,
            credential=None,
        ) as directory_client:
            file_client = directory_client.get_file_client(file_name)
            download = await file_client.download_file()
            content = await download.readall()
            return content

async def initialize_job(api_key):
    url = BASE_URL + 'job/init'
    headers = {'API-Subscription-Key': api_key}
    response = requests.post(url, headers=headers)
    if response.status_code == 202:
        return response.json()
    st.error(f"Failed to initialize job: {response.text}")
    return None

async def start_job(api_key, job_params):
    url = BASE_URL + 'job'
    headers = {
        'API-Subscription-Key': api_key,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json=job_params)
    if response.status_code == 200:
        return response.json()
    st.error(f"Failed to start job: {response.text}")
    return None

async def check_job_status(api_key, job_id):
    url = BASE_URL + f'job/{job_id}/status'
    headers = {'API-Subscription-Key': api_key}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    return None

async def process_batch_job(api_key, uploaded_files, questions, num_speakers, with_diarization):
    try:
        # Initialize job
        init_response = await initialize_job(api_key)
        if not init_response:
            return

        job_id = init_response['job_id']
        input_storage_path = init_response['input_storage_path']
        output_storage_path = init_response['output_storage_path']

        # Upload files
        client = SarvamClient(input_storage_path)
        await client.upload_files(uploaded_files)

        # Start job
        job_params = {
            "job_id": job_id,
            "job_parameters": {
                "model": "saaras:v2",
                "with_diarization": with_diarization,
                "num_speakers": num_speakers,
                "questions": questions
            }
        }
        
        start_response = await start_job(api_key, job_params)
        if not start_response:
            return

        # Monitor job status
        status_placeholder = st.empty()
        while True:
            status = await check_job_status(api_key, job_id)
            if not status:
                st.error("Failed to get job status")
                break

            current_state = status['job_state']
            status_placeholder.write(f"Current status: {current_state}")

            if current_state == 'Completed':
                st.success("Job completed successfully!")
                
                # Download and display results
                client.update_url(output_storage_path)
                files = await client.list_files()
                results = []
                for file in files:
                    if file.endswith('.json'):
                        content = await client.download_file(file)
                        results.append(json.loads(content))
                return results
            elif current_state == 'Failed':
                st.error("Job failed!")
                break
            
            await asyncio.sleep(10)

    except Exception as e:
        st.error(f"Error processing batch job: {str(e)}")
        return None

def display_results(results):
    for result in results:
        st.subheader("Transcript")
        st.write(result.get("transcript", "No transcript available"))
        
        st.subheader("Diarized Transcript")
        st.write(result.get("diarized_transcript", "No diarized transcript available"))
        
        st.subheader("Analysis Results")
        answers = result.get("answers", [])
        for answer in answers:
            st.write("Question:", answer.get("question"))
            st.write("Response:", answer.get("response"))
            st.write("Reasoning:", answer.get("reasoning"))
            st.write("---")


def main():
    # Set page config at the very beginning
    st.set_page_config(
        page_title="Call Analytics - Sarvam AI",
        page_icon="🎙️",
        layout="wide"
    )

    # Add dark theme CSS
    st.markdown("""
        <style>
        .dark-container { background-color: #1E1E1E; color: #E0E0E0; padding: 1rem; border: 1px solid #333; border-radius: 4px; margin-bottom: 1rem; }
        .dark-transcript { max-height: 400px; overflow-y: auto; padding: 1rem; background-color: #2D2D2D; border: 1px solid #333; border-radius: 4px; color: #E0E0E0; font-family: monospace; }
        .dark-response { background-color: #2D2D2D; padding: 0.5rem; border-radius: 4px; margin-bottom: 1rem; color: #E0E0E0; }
        .dark-header { color: #E0E0E0; border-bottom: 1px solid #333; padding-bottom: 0.5rem; margin-bottom: 1rem; }
        </style>
    """, unsafe_allow_html=True)

    # Configuration section
    st.subheader("1️⃣ Configuration")
    api_key = st.text_input(
        "API Key",
        type="password",
        help="Enter your Sarvam AI API key",
        placeholder="Enter your API key here..."
    )

    # File upload section
    st.subheader("2️⃣ Upload Files")
    uploaded_files = st.file_uploader(
        "Upload audio files",
        type=["wav", "mp3"],
        accept_multiple_files=True,
        help="Drag and drop your audio files here"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        for file in uploaded_files:
            st.text(f"📄 {file.name} ({file.size/1024:.1f} KB)")

    # Diarization settings
    st.subheader("3️⃣ Diarization Settings")
    col1, col2 = st.columns(2)
    with col1:
        with_diarization = st.toggle(
            "Enable Speaker Diarization",
            value=True,
            help="Split the transcript by different speakers"
        )
    with col2:
        num_speakers = st.number_input(
            "Number of Speakers",
            min_value=2,
            max_value=10,
            value=2,
            disabled=not with_diarization,
            help="Specify the number of speakers"
        )

    # Questions section
    st.subheader("4️⃣ Questions")
    
    # Question controls
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("➕ Add Question", use_container_width=True):
            st.session_state.num_questions += 1
    with col2:
        if st.button("➖ Remove Question", use_container_width=True) and st.session_state.num_questions > 1:
            st.session_state.num_questions -= 1

    # Initialize session state
    if 'num_questions' not in st.session_state:
        st.session_state.num_questions = 1

    # Questions input
    questions = []
    for i in range(st.session_state.num_questions):
        with st.container():
            st.markdown(
                f"""<div class="question-container">
                    <h4>Question {i+1}</h4>
                """,
                unsafe_allow_html=True
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                question_text = st.text_input(
                    "Question",
                    key=f"q_text_{i}",
                    placeholder="Enter your question here...",
                    help="Enter the question you want to ask about the audio"
                )
            
            with col2:
                question_type = st.selectbox(
                    "Type",
                    ["short answer", "long answer", "boolean", "enum", "number"],
                    key=f"q_type_{i}",
                    help="Select the type of answer you expect"
                )
            
            question_desc = st.text_input(
                "Description (Optional)",
                key=f"q_desc_{i}",
                placeholder="Add additional context or instructions...",
                help="Provide additional context or instructions for this question"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            if question_text:
                questions.append({
                    "id": f"q{i+1}",
                    "text": question_text,
                    "type": question_type,
                    "description": question_desc if question_desc else ""
                })

    # Analysis button
    st.subheader("5️⃣ Start Analysis")
    start_disabled = not (api_key and uploaded_files and questions)
    
    if start_disabled:
        st.warning("Please complete all required fields before starting the analysis.")
        
    start_button = st.button(
        "🚀 Start Analysis",
        disabled=start_disabled,
        use_container_width=True,
    )

    # Results display
    if start_button and not start_disabled:
        with st.spinner("Processing your audio files..."):
            results = asyncio.run(process_batch_job(
                api_key,
                uploaded_files,
                questions,
                num_speakers,
                with_diarization
            ))
            
            if results:
                st.subheader("6️⃣ Analysis Results")
                for idx, result in enumerate(results):
                    st.markdown(f"### 📄 Result File {idx + 1}")
                    
                    result_tabs = st.tabs([
                        "📝 Transcript", 
                        "👥 Diarized Transcript", 
                        "🔍 Analysis"
                    ])
                    
                    # Transcript Tab
                    with result_tabs[0]:
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            transcript = str(result.get("transcript", "No transcript available"))
                            html_content = (
                                '<div class="dark-transcript">'
                                + transcript.replace('\n', '<br>')
                                + '</div>'
                            )
                            st.markdown(html_content, unsafe_allow_html=True)
                        with col2:
                            st.download_button(
                                "📥 Download",
                                transcript,
                                file_name=f"transcript_{idx+1}.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key=f"download_transcript_{idx}"
                            )
                    
                    # Diarized Transcript Tab
                    with result_tabs[1]:
                        if with_diarization:
                            col1, col2 = st.columns([5, 1])
                            with col1:
                                diarized = str(result.get("diarized_transcript", "No diarized transcript available"))
                                html_content = (
                                    '<div class="dark-transcript">'
                                    + diarized.replace('\n', '<br>')
                                    + '</div>'
                                )
                                st.markdown(html_content, unsafe_allow_html=True)
                            with col2:
                                st.download_button(
                                    "📥 Download",
                                    diarized,
                                    file_name=f"diarized_{idx+1}.txt",
                                    mime="text/plain",
                                    use_container_width=True,
                                    key=f"download_diarized_{idx}"
                                )
                        else:
                            st.info("Diarization was not enabled for this analysis.")
                    
                    # Analysis Tab
                    with result_tabs[2]:
                        answers = result.get("answers", [])
                        for answer_idx, answer in enumerate(answers):
                            with st.expander(f"❓ {answer.get('question')}", expanded=False):
                                response = str(answer.get('response', 'No response available'))
                                reasoning = str(answer.get('reasoning', 'No reasoning available'))
                                html_content = (
                                    '<div class="dark-container">'
                                    + '<p><strong>Response:</strong></p>'
                                    + '<div class="dark-response">' + response + '</div>'
                                    + '<p><strong>Reasoning:</strong></p>'
                                    + '<div class="dark-response">' + reasoning + '</div>'
                                    + '</div>'
                                )
                                st.markdown(html_content, unsafe_allow_html=True)
                    
                    st.markdown('<hr style="border-color: #333;">', unsafe_allow_html=True)
            else:
                st.error("No results received from the API. Please try again.")

if __name__ == "__main__":
    main()
