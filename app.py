import streamlit as st
import imaplib
import email
from email.header import decode_header
import os
import io
import time
import zipfile
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()
DEFAULT_API_KEY = os.getenv("GEMINI_API_KEY", "")

st.set_page_config(page_title="Data Science Mail Agent", page_icon="📥", layout="centered")

st.title("📥 Data Science Mail Document Agent")
st.write("Connect your Gmail inbox, scan your recent emails, and automatically collect all Data Science & AI documents.")

MODELS_TO_TRY = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.0-flash",
]


class EmailCheck(BaseModel):
    is_data_science: bool
    reason: str


def clean_text(raw_header):
    if not raw_header:
        return ""
    decoded_parts = decode_header(raw_header)
    result = []
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            result.append(part.decode(encoding or "utf-8", errors="ignore"))
        else:
            result.append(str(part))
    return "".join(result)


def is_ds_email(client, subject: str, filenames: list, body_snippet: str) -> EmailCheck:
    prompt = f"""
    Decide if this email is related to Data Science, Machine Learning, AI, 
    Data Analytics, SQL, Python for data, or Data Science job roles/study material.

    Subject: {subject}
    Attached Filenames: {", ".join(filenames)}
    Body Snippet: {body_snippet[:1000]}
    """
    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": EmailCheck,
                },
            )
            return response.parsed
        except Exception as e:
            last_error = e
            time.sleep(1.5)
    raise Exception(f"All models unavailable. Last error: {last_error}")


with st.form("mail_form"):
    email_user = st.text_input("Gmail Address", placeholder="you@gmail.com")
    email_pass = st.text_input(
        "Gmail 16-Character App Password",
        type="password",
        help="Generate this in Google Account -> Security -> 2-Step Verification -> App Passwords",
    )
    pages = st.slider(
        "Number of inbox pages to scan (50 emails per page — includes Primary, Promotions & Updates)",
        min_value=1,
        max_value=10,
        value=5,
    )

    with st.expander("Optional: Use custom Gemini API Key"):
        custom_api_key = st.text_input("Gemini API Key", value=DEFAULT_API_KEY, type="password")

    submitted = st.form_submit_button("Scan Inbox & Collect Documents", use_container_width=True)

if submitted:
    if not email_user or not email_pass:
        st.error("Please enter both your Gmail address and your 16-character App Password.")
    elif not custom_api_key:
        st.error("Missing Gemini API Key. Please add it in the optional section above or in .env.")
    else:
        client = genai.Client(api_key=custom_api_key)
        saved_files = []
        emails_with_files_found = 0

        try:
            with st.status("Connecting to Gmail...", expanded=True) as status:
                mail = imaplib.IMAP4_SSL("imap.gmail.com")
                mail.login(email_user.strip(), email_pass.replace(" ", "").strip())
                mail.select("inbox")

                _, search_data = mail.search(None, "ALL")
                email_ids = search_data[0].split()
                total_to_scan = pages * 50
                target_ids = email_ids[-total_to_scan:][::-1]

                status.update(label=f"Scanning latest {len(target_ids)} emails ({pages} pages)...", state="running")
                progress_bar = st.progress(0)

                for idx, num in enumerate(target_ids, start=1):
                    progress_bar.progress(idx / len(target_ids))

                    _, msg_data = mail.fetch(num, "(RFC822)")
                    msg = email.message_from_bytes(msg_data[0][1])

                    subject = clean_text(msg.get("Subject"))
                    body_text = ""
                    attachments = []

                    # Catch ALL attached or inline files that have a filename
                    for part in msg.walk():
                        if part.get_content_maintype() == "multipart":
                            continue
                        raw_filename = part.get_filename()
                        if raw_filename:
                            fname = clean_text(raw_filename)
                            payload = part.get_payload(decode=True)
                            if fname and payload:
                                attachments.append((fname, payload))
                        elif part.get_content_type() == "text/plain":
                            payload = part.get_payload(decode=True)
                            if payload:
                                body_text += payload.decode(errors="ignore")

                    if not attachments:
                        continue

                    emails_with_files_found += 1
                    file_names_list = [f[0] for f in attachments]
                    st.write(f"Checking email #{idx}: **{subject[:60]}** *(Files: {', '.join(file_names_list)})*")

                    try:
                        decision = is_ds_email(client, subject, file_names_list, body_text)
                        if decision.is_data_science:
                            st.success(f"Matched: {subject[:60]} — {decision.reason}")
                            for fname, fbytes in attachments:
                                unique_name = f"{num.decode()}_{fname}"
                                saved_files.append((unique_name, fbytes, subject, decision.reason))
                        else:
                            st.caption(f"Skipped: {decision.reason}")
                        time.sleep(2)
                    except Exception as e:
                        st.warning(f"Skipped '{subject[:40]}' due to temporary API issue: {e}")

                mail.logout()
                status.update(
                    label=f"Scan complete! Checked {emails_with_files_found} emails with files.",
                    state="complete",
                    expanded=False,
                )

            if saved_files:
                st.subheader(f"✅ Found {len(saved_files)} Data Science Document(s)")

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname, fbytes, _, _ in saved_files:
                        zf.writestr(fname, fbytes)

                st.download_button(
                    label="📦 Download All Documents (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="data_science_documents.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

                for fname, fbytes, subj, reason in saved_files:
                    with st.expander(f"📄 {fname}"):
                        st.write(f"**Email Subject:** {subj}")
                        st.write(f"**Why it matched:** {reason}")
                        st.download_button(
                            label=f"Download {fname}",
                            data=fbytes,
                            file_name=fname,
                            key=f"dl_{fname}",
                        )
            else:
                st.info(
                    f"Scan finished! Found {emails_with_files_found} total emails with files in the latest "
                    f"{len(target_ids)} messages, and none matched Data Science. Try increasing the pages slider!"
                )

        except imaplib.IMAP4.error:
            st.error("Login failed! Double-check your Gmail address and make sure you used a 16-character App Password.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")