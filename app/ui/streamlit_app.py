import streamlit as st
import requests

# Define the API endpoints and API key
process_url = "http://127.0.0.1:8001/process-document/"
query_url = "http://127.0.0.1:8001/query-rag/"
api_key = "new_secret_key"

st.title("Document Chat Interface")

# File uploader for document processing
uploaded_file = st.file_uploader("Upload a document (PDF, PNG, JPG):")
if uploaded_file:
    with st.spinner("Processing document..."):
        files = {"file": uploaded_file}
        headers = {"x-api-key": api_key}
        response = requests.post(process_url, headers=headers, files=files)

        if response.status_code == 200:
            st.success("Document processed successfully!")
            st.write("You can now ask questions about the document.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

# Input for user query
query = st.text_input("Ask a question about your document:")

if st.button("Submit Query"):
    if query:
        with st.spinner("Fetching answer..."):
            # Send the POST request to the /query-rag/ endpoint
            headers = {"x-api-key": api_key, "Content-Type": "application/json"}
            data = {"query": query}
            response = requests.post(query_url, headers=headers, json=data)

            # Display the response
            if response.status_code == 200:
                result = response.json()
                st.success("Answer:")
                st.write(result.get("answer", "No answer found."))

                st.subheader("Retrieved Contexts:")
                retrieved_contexts = result.get("retrieved_contexts", [])
                if retrieved_contexts:
                    for context in retrieved_contexts:
                        metadata = context.get("metadata", {})
                        st.write(f"**Text:** {metadata.get('text', 'No text available.')}")
                        st.write(f"**Filename:** {metadata.get('filename', 'Unknown')}")
                        st.write(f"**Distance:** {context.get('distance', 'N/A')}")
                        st.write("---")
                else:
                    st.warning("No relevant contexts were retrieved.")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter a query.")