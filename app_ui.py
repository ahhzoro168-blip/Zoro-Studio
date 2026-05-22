import os
import shutil
import io
import zipfile
import streamlit as st  # type: ignore
from gradio_client import Client, handle_file  # type: ignore
from google import genai  # type: ignore

# 1. Initialize Page Configuration Settings 
st.set_page_config(page_title="Zoro's Studio", layout="wide")

def load_css(file_name="style.css"):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ External style sheet asset '{file_name}' not found.")

load_css()

st.title("Zoro's Studio")

# Initialize Session States
if "generated_chunks" not in st.session_state:
    st.session_state.generated_chunks = []

if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = ""

if "voice_profiles_list" not in st.session_state:
    base_ref_dir = "references"
    scanned_profiles = []
    if os.path.exists(base_ref_dir):
        for filename in sorted(os.listdir(base_ref_dir)):
            if filename.lower().endswith(('.wav', '.mp3')):
                scanned_profiles.append({
                    "name": os.path.splitext(filename)[0],
                    "path": os.path.join(base_ref_dir, filename).replace("\\", "/"),
                    "mode": "Controllable Cloning", "instruction": "", "is_default": True  
                })
    if not scanned_profiles:
        scanned_profiles = [
            {"name": "Female_1", "path": "references/Female_1.wav", "mode": "Controllable Cloning", "instruction": "", "is_default": True},
            {"name": "Male_1", "path": "references/Male_1.wav", "mode": "Controllable Cloning", "instruction": "", "is_default": True}
        ]
    st.session_state.voice_profiles_list = scanned_profiles

# Connect to running VoxCPM2 Backend
with st.sidebar:
    st.header("Backend Settings")
    VOXCPM_SERVER_URL = st.text_input("VoxCPM2 Server URL", value="http://localhost:8808/")
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
@st.cache_resource
def get_voxcpm_client():
    try: return Client(VOXCPM_SERVER_URL)
    except Exception as e: return None

client = get_voxcpm_client()

def generate_single_line(text, character, tone, line_idx, cfg_val, use_denoise, use_normalize, dit_steps_val, output_dir="ui_output_chunks"):
    profiles_dict = {p["name"]: p for p in st.session_state.voice_profiles_list}
    profile = profiles_dict.get(character, st.session_state.voice_profiles_list[0] if st.session_state.voice_profiles_list else None)
    if not profile: return None
    final_chunk_path = os.path.join(output_dir, f"chunk_{line_idx:03d}_{character}.wav")
    try:
        result = client.predict(
            text=text, control_instruction=tone if tone.strip() else profile.get("instruction", ""), 
            ref_wav=handle_file(profile["path"]), use_prompt_text=False, prompt_text_value="",                      
            cfg_value=cfg_val, do_normalize=use_normalize, denoise=use_denoise, dit_steps=dit_steps_val, api_name="/generate"                        
        )
        temp_audio_path = result if isinstance(result, str) else result.get("name")
        if temp_audio_path:
            shutil.copy(temp_audio_path, final_chunk_path)
            return final_chunk_path
    except Exception: pass
    return None

# Sidebar Configuration Layout
with st.sidebar:
    st.header("Voice Profiles")
    if st.button("Add Voice Profile", use_container_width=True):
        new_id = len(st.session_state.voice_profiles_list) + 1
        st.session_state.voice_profiles_list.append({
            "name": f"Custom_Voice_{new_id}", "path": "", "mode": "Controllable Cloning", "instruction": "", "is_default": False
        })
        st.rerun()
    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    
    to_delete = None
    for i, profile in enumerate(st.session_state.voice_profiles_list):
        st.markdown(f"<p class='character-title'>Character {i+1}</p>", unsafe_allow_html=True)
        st.markdown("</br>", unsafe_allow_html=True)
        st.session_state.voice_profiles_list[i]["name"] = st.text_input("Character Name", value=profile["name"], key=f"side_name_val_{i}")
        
        if profile.get("is_default", False):
            st.session_state.voice_profiles_list[i]["path"] = st.text_input("Audio Path", value=profile["path"], key=f"side_ref_val_{i}")
        else:
            uploaded_file = st.file_uploader("Upload Voice File", type=["wav", "mp3"], key=f"side_upload_val_{i}")
            if uploaded_file:
                upload_dir = "uploaded_profiles"
                os.makedirs(upload_dir, exist_ok=True)
                saved_target_path = os.path.join(upload_dir, f"voice_{profile['name'].lower()}.wav")
                with open(saved_target_path, "wb") as f: f.write(uploaded_file.getbuffer())
                st.session_state.voice_profiles_list[i]["path"] = saved_target_path
        if not profile.get("is_default", False):
            if st.button("Delete Profile", key=f"side_del_btn_{i}", use_container_width=True): to_delete = i
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    if to_delete is not None:
        st.session_state.voice_profiles_list.pop(to_delete)
        st.rerun()

# Workspace Tabs Design Layout
tab_studio, tab_transcribe = st.tabs(["Multi-Voice Script", "AI Media Transcriber"])

# --- TAB 1: STUDIO GENERATOR ---
with tab_studio:
    # Large Studio Subheader
    st.markdown("## Multi-Voice Script Workspace")
    
    # Normal body text explanation line
    st.markdown("<p style='color: #FFFFFF; font-weight: 500; font-size:15px; margin-top:5px;'>Type your multi-character script using tags like <span style='color:#FF4B4B; font-weight:600;'>[Character] [Tone of Voice] Dialogue</span>.</p>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)
    
    # Clean input label using standard markdown size
    st.markdown("### Script Input Box")
    user_script = st.text_area("Script Input Box", value="", placeholder="Example: [Female_1] [Quiet, polite tone] អ្នកលួចនិយាយជាមួយនាងពីក្រោយខ្នងខ្ញុំមែនទេ?", height=180, label_visibility="collapsed")

    # --- ADVANCED SETTINGS WITH CUSTOM LAYOUT ---
    with st.expander("Advanced Settings", expanded=False):
        opt_denoise = st.toggle("Reference audio enhancement", value=False)
        opt_normalize = st.toggle("Text normalization", value=False)

        st.markdown("<span class='cfg-badge-lbl'>CFG (Guidance Scale)</span>", unsafe_allow_html=True)
        opt_cfg = st.slider("CFG Value", min_value=1.0, max_value=7.0, value=2.0, step=0.1, label_visibility="collapsed")
        st.markdown("<span class='cfg-badge-lbl'>LocDiT Flow-Matching Steps</span>", unsafe_allow_html=True)
        opt_steps = st.slider("LocDiT Steps", min_value=1, max_value=50, value=10, step=1, label_visibility="collapsed")

    # --- PROCESSED SCRIPT FULL WIDTH ACTION BUTTON ---
    col_btn_wrap = st.container()
    with col_btn_wrap:
        generate_clicked = st.button("Overwrite & Process Script" if st.session_state.generated_chunks else "Process Full Script", type="primary", use_container_width=True)

    if generate_clicked:
        if not client: st.error("Backend server connection offline.")
        elif not user_script.strip(): st.warning("Script workspace cannot be blank.")
        else:
            output_dir = "ui_output_chunks"
            if os.path.exists(output_dir): shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
            st.session_state.generated_chunks = []
            for idx, line in enumerate(user_script.strip().split("\n")):
                if not line.strip(): continue
                character, tone, text = "Narrator", "", line.strip()
                if line.startswith("[") and "]" in line:
                    c_end = line.find("]")
                    character = line[1:c_end].strip()
                    rem = line[c_end+1:].strip()
                    if rem.startswith("[") and "]" in rem:
                        t_end = rem.find("]")
                        tone = rem[1:t_end].strip()
                        text = rem[t_end+1:].strip()
                    else: text = rem
                f_path = generate_single_line(text, character, tone, idx, opt_cfg, opt_denoise, opt_normalize, opt_steps, output_dir)
                if f_path: st.session_state.generated_chunks.append({"line_idx": idx, "character": character, "tone": tone, "text": text, "file_path": f_path})
            st.success("Batch script processing successfully completed!")
            st.rerun()

    # --- AUDIO PLAYER BLOCK WITH CUSTOM CORAL LAYOUT ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 class='custom-accent-header' style='width:240px;'>Audio Review Player</h4>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    
    col_dl_all, col_clear_btn, _ = st.columns([1.5, 1.8, 4.7])
    with col_dl_all:
        if st.session_state.generated_chunks:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for item in st.session_state.generated_chunks:
                    if os.path.exists(item["file_path"]):
                        zf.write(item["file_path"], arcname=f"line_{item['line_idx']:03d}_{item['character']}.wav")
            st.download_button(label="Download All Voices", data=zip_buffer.getvalue(), file_name="studio_batch_render.zip", mime="application/zip", use_container_width=True)
        else: st.button("Download All Voices", disabled=True, use_container_width=True)
    with col_clear_btn:
        if st.button("Clear All Generated Voices", type="secondary", use_container_width=True):
            st.session_state.generated_chunks = []
            st.rerun()

    if st.session_state.generated_chunks:
        available_characters = [p["name"] for p in st.session_state.voice_profiles_list]
        for index, item in enumerate(st.session_state.generated_chunks):
            st.markdown(f"<div class='studio-row-card'>", unsafe_allow_html=True)
            col_char, col_tone, col_text, col_actions = st.columns([1.2, 1.5, 3.5, 1.5])
            with col_char:
                st.session_state.generated_chunks[index]['character'] = st.selectbox("Select Character", options=available_characters, index=available_characters.index(item['character']) if item['character'] in available_characters else 0, key=f"edit_char_{index}", label_visibility="collapsed")
            with col_tone:
                st.session_state.generated_chunks[index]['tone'] = st.text_input("Edit Tone", value=item['tone'], key=f"edit_tone_{index}", label_visibility="collapsed")
            with col_text:
                st.session_state.generated_chunks[index]['text'] = st.text_input("Edit Script Text", value=item['text'], key=f"edit_text_{index}", label_visibility="collapsed")
            with col_actions:
                col_regen, col_dl = st.columns(2)
                with col_regen:
                    if st.button("Renew", key=f"regen_btn_{index}", use_container_width=True):
                        with st.spinner("..."):
                            new_path = generate_single_line(item['text'], item['character'], item['tone'], item['line_idx'], opt_cfg, opt_denoise, opt_normalize, opt_steps)
                            if new_path: st.session_state.generated_chunks[index]['file_path'] = new_path; st.rerun()
                with col_dl:
                    if os.path.exists(item["file_path"]):
                        with open(item["file_path"], "rb") as fa: st.download_button(label="DL", data=fa, file_name=f"line_{item['line_idx']:03d}_{item['character']}.wav", mime="audio/wav", key=f"dl_btn_{index}", use_container_width=True)
            st.audio(item["file_path"], format="audio/wav")
            st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 2: TRANSCRIPTION & TRANSLATION ENGINE ---

with tab_transcribe:
    st.markdown("## AI Media Transcriber")

    st.markdown("<p style='color: #FFFFFF; font-weight: 500; font-size:15px; margin-top:5px;'>Upload localized video or audio assets to <span style='color:#FF4B4B; font-weight:600;'>Transcribe</span> and <span style='color:#FF4B4B; font-weight:600;'>Translate</span> into production-ready scripts.</p>", unsafe_allow_html=True)
    st.markdown("<div style='margin-top:15px;'></div>", unsafe_allow_html=True)

    # Secure API Key Entry Field
    st.markdown("### Gemini API Key")
    gemini_key = st.text_input("Enter Gemini API Key:", type="password", help="Grab an API key from Google AI Studio")
    
    uploaded_media = st.file_uploader("Choose a video or audio file from your PC", type=["mp3", "wav", "m4a", "mp4", "mkv", "mov"])
    
    if uploaded_media:
        st.info(f"📂 Selected asset loaded: {uploaded_media.name} ({uploaded_media.size / (1024*1024):.2f} MB)")
        
        col_run_tx, _ = st.columns([1.5, 3])
        with col_run_tx:
            if st.button("Run AI Translation Breakdown", type="primary", use_container_width=True):
                if not gemini_key:
                    st.error("Please add your Gemini API Key first.")
                else:
                    with st.spinner("AI is processing media file... This may take a minute."):
                        try:
                            # 1. Initialize the new 2026 GenAI SDK Client
                            ai_client = genai.Client(api_key=gemini_key)
                            
                            # 2. Convert memory upload into a byte pass payload asset
                            media_bytes = uploaded_media.read()
                            mime_type = uploaded_media.type
                            
                            # 3. Assemble your exact multi-modal structural prompt engineering blueprint
                            structural_prompt = (
                                "You are an expert media transcriber and script supervisor. "
                                "Analyze the uploaded audio or video file carefully and break down the dialogue line-by-line. "
                                "You must output the final text strictly using the following formatting syntax on each line:\n\n"
                                "* [Speaker_Label] [Tone of Voice in English] Khmer Translation Text\n\n"
                                "CRITICAL RULES FOR SPEAKER LABELS:\n"
                                "Identify the speaker age/gender and use these exact snake_case labels (add numbers like _1 if there are multiples):\n"
                                "- For an old man: [Old_Man_1]\n"
                                "- For an old woman: [Old_Woman_1]\n"
                                "- For a young girl: [Young_Girl_1]\n"
                                "- For a young boy: [Young_Boy_1]\n"
                                "- For a child girl: [Child_Girl_1]\n"
                                "- For a child boy: [Child_Boy_1]\n\n"
                                "FORMATTING RULES:\n"
                                "1. In the second brackets, describe the precise emotional context/tone of voice in English (e.g., [Sad, crying tone], [Angry, shouting tone], [Happy, laughing tone]).\n"
                                "2. The text outside the brackets must be a direct, natural translation into Khmer, preserving the intense emotional weight of the scene.\n"
                                "3. Use a clean list format with circle bullet points (*) for each line of dialogue to exactly match the reference image style.\n"
                                "4. Do not include introductory text or conclusions—provide only the formatted bullet points."
                            )
                            
                            # 4. Dispatch natively using the blazing fast gemini-2.5-flash model
                            response = ai_client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[
                                    genai.types.Part.from_bytes(
                                        data=media_bytes,
                                        mime_type=mime_type,
                                    ),
                                    structural_prompt
                                ]
                            )
                            st.session_state.transcription_result = response.text
                            st.success("Analysis complete!")
                        except Exception as error:
                            st.error(f"Gemini processing error: {error}")

    if st.session_state.transcription_result:
        st.markdown("### 📋 AI Breakdown Output")
        
        # Display the output nicely in a scrollable, selectable workspace block
        st.text_area("Copy-Pasteable Results", value=st.session_state.transcription_result, height=400)
    