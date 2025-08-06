# Standard Library Imports
import os
import re
import json
import time
import random
import sqlite3
from collections import Counter

# Third-party Imports
import streamlit as st
import requests
import numpy as np
import nltk
import tiktoken
import pdfplumber
import docx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Constants and Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment variables!")
    st.info("Please set your GROQ_API_KEY in GitHub Secrets or environment variables.")
    st.stop()

# NLTK Data Download
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Model Loading
@st.cache_resource
def load_embedding_model():
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Error loading embedding model: {e}")
        return None

# Initialize resources
download_nltk_data()
embedding_model = load_embedding_model()

class DocumentProcessor:
    """Handles document processing including text extraction and chunking"""
    
    @staticmethod
    def extract_clean_text_from_pdf(pdf_path, header_lines=2, footer_lines=2):
        """Extract clean text from PDF by removing headers and footers"""
        all_page_texts = []
        headers = []
        footers = []

        # Step 1: Collect header/footer candidates
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                lines = page.extract_text().splitlines()
                if len(lines) >= (header_lines + footer_lines):
                    headers.extend(lines[:header_lines])
                    footers.extend(lines[-footer_lines:])

        # Step 2: Find most common headers/footers
        header_counts = Counter(headers)
        footer_counts = Counter(footers)
        common_headers = {line for line, count in header_counts.items() if count > 1}
        common_footers = {line for line, count in footer_counts.items() if count > 1}

        st.write(f"🔍 **Common Headers Found:** {len(common_headers)}")
        st.write(f"🔍 **Common Footers Found:** {len(common_footers)}")

        # Step 3: Extract cleaned text from each page
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                lines = page.extract_text().splitlines()
                clean_lines = [
                    line for line in lines
                    if line not in common_headers and line not in common_footers
                ]
                all_page_texts.append('\n'.join(clean_lines))

        return '\n\n'.join(all_page_texts)

    @staticmethod
    def extract_text_from_docx(docx_path):
        """Extract text from Word document"""
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])

    @staticmethod
    def count_tokens(text, model="gpt-3.5-turbo"):
        """Count tokens using tiktoken"""
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

    @staticmethod
    def chunk_text(text, max_tokens=500):
        """Chunk text into ~500 token chunks"""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sent in sentences:
            test_chunk = current_chunk + " " + sent if current_chunk else sent
            if DocumentProcessor.count_tokens(test_chunk) <= max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    @classmethod
    def process_document(cls, filepath):
        """Process document and return chunks"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == ".pdf":
            st.write("📄 **Processing PDF document...**")
            full_text = cls.extract_clean_text_from_pdf(filepath)
        elif ext == ".docx":
            st.write("📄 **Processing Word document...**")
            full_text = cls.extract_text_from_docx(filepath)
        else:
            raise ValueError("Unsupported file type")

        st.write(f"📏 **Document length:** {len(full_text)} characters")
        st.write("🔧 **Creating chunks...**")
        
        chunks = cls.chunk_text(full_text, max_tokens=500)
        st.write(f"✅ **Created {len(chunks)} chunks**")
        
        return chunks

class LLMProcessor:
    """Handles interactions with the LLM API"""
    
    @staticmethod
    def call_llm(chunk_text):
        """Call Groq API to generate metadata for insurance policy chunk"""
        prompt = f"""You must respond with ONLY a valid JSON object. No explanations, no markdown, no additional text.

STRICT REQUIREMENTS:
1. Return ONLY raw JSON - no ```json``` blocks, no extra text
2. Use exactly these 4 fields: heading, type, keywords, applicable_conditions
3. All values must be strings (even if empty, use "")

Example format:
{{"heading": "Medical Coverage Benefits", "type": "coverage", "keywords": "medical, benefits, coverage", "applicable_conditions": "subject to deductible"}}

Insurance clause to analyze:
{chunk_text[:1000]}

JSON response:"""

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 300
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            raise Exception(f"API call failed: {response.status_code}")

        except Exception as e:
            st.error(f"❌ API call error: {e}")
            raise

    @staticmethod
    def parse_llm_response(response_text, chunk_num):
        """Parse LLM response with multiple fallback strategies"""
        
        # Strategy 1: Direct JSON parsing
        try:
            response_text = response_text.strip()
            meta = json.loads(response_text)

            # Validate required fields
            required_fields = ['heading', 'type', 'keywords', 'applicable_conditions']
            for field in required_fields:
                if field not in meta:
                    meta[field] = ""

            # Ensure all values are strings
            for key, value in meta.items():
                if not isinstance(value, str):
                    meta[key] = str(value)

            return meta

        except json.JSONDecodeError:
            pass

        # Strategy 2: Clean markdown and common formatting
        try:
            clean_response = response_text.strip()

            # Remove markdown code blocks
            if '```json' in clean_response:
                clean_response = clean_response.split('```json')[1].split('```')[0].strip()
            elif '```' in clean_response:
                clean_response = clean_response.split('```')[1].split('```')[0].strip()

            # Remove common prefixes
            prefixes_to_remove = [
                "Here is the summarized clause in JSON format:",
                "Here's the metadata JSON:",
                "JSON response:",
                "Metadata:",
                "Here is the JSON:",
                "Response:"
            ]

            for prefix in prefixes_to_remove:
                if clean_response.startswith(prefix):
                    clean_response = clean_response[len(prefix):].strip()

            # Try to extract JSON from within text
            start_idx = clean_response.find('{')
            end_idx = clean_response.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_part = clean_response[start_idx:end_idx]
                meta = json.loads(json_part)

                # Validate and fix fields
                required_fields = ['heading', 'type', 'keywords', 'applicable_conditions']
                for field in required_fields:
                    if field not in meta:
                        meta[field] = ""
                    elif not isinstance(meta[field], str):
                        meta[field] = str(meta[field])

                return meta

        except (json.JSONDecodeError, IndexError):
            pass

        # Strategy 3: Manual key-value extraction
        try:
            meta = {
                'heading': f'Chunk {chunk_num}',
                'type': 'unknown',
                'keywords': 'insurance, policy',
                'applicable_conditions': ''
            }

            # Look for quoted values after field names
            heading_match = re.search(r'"heading":\s*"([^"]*)"', response_text, re.IGNORECASE)
            if heading_match:
                meta['heading'] = heading_match.group(1)[:50]

            type_match = re.search(r'"type":\s*"([^"]*)"', response_text, re.IGNORECASE)
            if type_match:
                meta['type'] = type_match.group(1)

            keywords_match = re.search(r'"keywords":\s*"([^"]*)"', response_text, re.IGNORECASE)
            if keywords_match:
                meta['keywords'] = keywords_match.group(1)

            conditions_match = re.search(r'"applicable_conditions":\s*"([^"]*)"', response_text, re.IGNORECASE)
            if conditions_match:
                meta['applicable_conditions'] = conditions_match.group(1)

            return meta

        except Exception:
            pass

        # Strategy 4: Complete fallback
        return {
            'heading': f'Policy Clause {chunk_num}',
            'type': 'policy_text',
            'keywords': 'insurance, policy, clause',
            'applicable_conditions': 'Refer to original document'
        }

class DatabaseManager:
    """Handles database operations"""
    
    @staticmethod
    def setup_sqlite():
        """Initialize SQLite database for storing policy chunks and metadata"""
        try:
            conn = sqlite3.connect("policy_chunks.db")
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    heading TEXT,
                    type TEXT,
                    keywords TEXT,
                    applicable_conditions TEXT,
                    chunk TEXT,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

            # Check existing records
            cursor.execute("SELECT COUNT(*) FROM metadata_chunks")
            existing_count = cursor.fetchone()[0]

            return conn, existing_count

        except Exception as e:
            st.error(f"❌ Database setup failed: {e}")
            raise

    @staticmethod
    def process_and_store_chunks(chunks, delay_c=2, delay_random=3):
        """Process insurance policy chunks and store them with metadata"""
        if not chunks:
            st.error("⚠️ No chunks provided to process")
            return

        conn, existing_count = DatabaseManager.setup_sqlite()
        cursor = conn.cursor()

        st.write(f"📊 **Existing records in database:** {existing_count}")
        st.write(f"🚀 **Starting processing of {len(chunks)} chunks...**")
        st.write(f"⏱️ **Delay settings:** base={delay_c}s, random=0-{delay_random}s")

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        successful_processes = 0
        failed_processes = 0

        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            status_text.text(f"📋 Processing Chunk {chunk_num}/{len(chunks)}")

            try:
                st.write(f"**Chunk {chunk_num}/{len(chunks)}**")
                st.write(f"📏 Chunk length: {len(chunk)} characters")

                # Call LLM for metadata
                st.write("🤖 Making API call to Groq...")
                metadata_response = LLMProcessor.call_llm(chunk)

                # Parse JSON response
                st.write("🔍 Parsing response...")
                meta = LLMProcessor.parse_llm_response(metadata_response, chunk_num)

                if meta:
                    st.write("✅ Metadata parsed successfully")
                    st.write(f"🏷️ **Heading:** {meta.get('heading', 'N/A')}")
                    st.write(f"🔖 **Type:** {meta.get('type', 'N/A')}")
                    st.write(f"🏷️ **Keywords:** {meta.get('keywords', 'N/A')}")

                # Compute embedding
                st.write("🧮 Computing embedding...")
                heading_text = meta.get('heading', f'Chunk {chunk_num}')
                heading_embedding = embedding_model.encode(heading_text)
                st.write(f"✅ Embedding computed: shape {heading_embedding.shape}")

                # Store in database
                st.write("💾 Storing in database...")
                cursor.execute('''
                    INSERT INTO metadata_chunks (heading, type, keywords, applicable_conditions, chunk, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    meta.get('heading', ''),
                    meta.get('type', ''),
                    meta.get('keywords', ''),
                    meta.get('applicable_conditions', ''),
                    chunk,
                    heading_embedding.tobytes()
                ))
                conn.commit()

                successful_processes += 1
                st.write(f"✅ Chunk {chunk_num} processed and stored successfully")

                # Update progress
                progress = chunk_num / len(chunks)
                progress_bar.progress(progress)

            except Exception as e:
                failed_processes += 1
                st.error(f"❌ Error processing chunk {chunk_num}: {e}")
                continue

            # Rate limiting delay (except for last chunk)
            if chunk_num < len(chunks):
                delay = delay_c + random.uniform(0, delay_random)
                st.write(f"⏳ Sleeping for {delay:.2f} seconds...")
                time.sleep(delay)

        conn.close()

        # Final summary
        st.success(f"🎉 Processing Complete!")
        st.write(f"✅ **Successfully processed:** {successful_processes}/{len(chunks)} chunks")
        st.write(f"❌ **Failed:** {failed_processes}/{len(chunks)} chunks")
        st.write(f"📊 **Success rate:** {(successful_processes/len(chunks)*100):.1f}%")

class PolicyQuerySystem:
    """Handles the query processing pipeline"""
    
    def __init__(self, db_path="policy_chunks.db"):
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.groq_api_key = GROQ_API_KEY

    def preprocess_query(self, raw_query):
        """Preprocess query and extract structured information"""
        st.write(f"🔄 **Preprocessing query:** '{raw_query}'")

        extracted_info = {
            'age': None,
            'gender': None,
            'medical_condition': None,
            'location': None,
            'policy_duration': None,
            'other_terms': []
        }

        parts = [part.strip() for part in raw_query.split(',')]
        st.write(f"📋 **Query parts:** {parts}")

        search_terms = []

        for part in parts:
            part_lower = part.lower()

            # Extract age and gender
            age_gender_match = re.search(r'(\d+)\s*([MFmf])', part)
            if age_gender_match:
                extracted_info['age'] = int(age_gender_match.group(1))
                extracted_info['gender'] = age_gender_match.group(2).upper()
                search_terms.append(f"age {extracted_info['age']}")
                search_terms.append(f"gender {extracted_info['gender']}")
                continue

            # Extract age only
            age_match = re.search(r'(\d+)\s*(?:years?|yr|y\.o\.?)', part_lower)
            if age_match:
                extracted_info['age'] = int(age_match.group(1))
                search_terms.append(f"age {extracted_info['age']}")
                continue

            # Extract policy duration
            duration_match = re.search(r'(\d+)[-\s]*(?:month|year|yr|day)s?\s*(?:policy|coverage|plan)', part_lower)
            if duration_match:
                extracted_info['policy_duration'] = part
                search_terms.append("policy duration")
                search_terms.append(part)
                continue

            # Medical conditions
            medical_keywords = [
                'surgery', 'operation', 'procedure', 'treatment', 'therapy',
                'diabetes', 'hypertension', 'heart', 'cardiac', 'knee', 'hip',
                'cancer', 'tumor', 'fracture', 'accident', 'injury',
                'pregnancy', 'maternity', 'dental', 'eye', 'vision'
            ]

            if any(keyword in part_lower for keyword in medical_keywords):
                extracted_info['medical_condition'] = part
                search_terms.append(part)
                search_terms.append("medical treatment")
                continue

            # Location detection
            indian_locations = [
                'mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata',
                'hyderabad', 'ahmedabad', 'surat', 'jaipur', 'lucknow',
                'maharashtra', 'karnataka', 'tamil nadu', 'gujarat', 'rajasthan'
            ]

            if any(location in part_lower for location in indian_locations):
                extracted_info['location'] = part
                search_terms.append("location coverage")
                continue

            extracted_info['other_terms'].append(part)
            search_terms.append(part)

        processed_query = " ".join(search_terms)

        # Add contextual terms
        if extracted_info['medical_condition']:
            processed_query += " coverage benefits eligibility treatment"
        if extracted_info['age']:
            processed_query += " age limit qualification"
        if extracted_info['policy_duration']:
            processed_query += " waiting period term condition"

        st.write(f"✅ **Extracted info:** {extracted_info}")
        st.write(f"🔍 **Processed search query:** '{processed_query}'")

        return processed_query, extracted_info

    def search_similar_chunks(self, query, top_k=5, similarity_threshold=0.3):
        """Search for similar chunks using cosine similarity"""
        st.write(f"🔍 **Searching for similar chunks** (top_k={top_k})")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT id, heading, type, keywords, chunk, embedding FROM metadata_chunks")
            results = cursor.fetchall()

            if not results:
                st.warning("⚠️ No chunks found in database")
                return []

            st.write(f"📊 **Found {len(results)} chunks in database**")

            # Generate query embedding
            st.write("🧮 **Generating query embedding...**")
            query_embedding = self.embedding_model.encode(query).reshape(1, -1)

            # Calculate similarities
            similarities = []
            for row in results:
                chunk_id, heading, chunk_type, keywords, chunk_text, embedding_blob = row

                chunk_embedding = np.frombuffer(embedding_blob, dtype=np.float32).reshape(1, -1)
                similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]

                similarities.append({
                    'id': chunk_id,
                    'heading': heading,
                    'type': chunk_type,
                    'keywords': keywords,
                    'chunk': chunk_text,
                    'similarity': similarity
                })

            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            filtered_results = [item for item in similarities if item['similarity'] >= similarity_threshold]
            top_results = filtered_results[:top_k]

            st.write(f"✅ **Found {len(filtered_results)} chunks above threshold {similarity_threshold}**")
            st.write(f"📋 **Returning top {len(top_results)} results:**")

            for i, result in enumerate(top_results, 1):
                st.write(f"   {i}. Similarity: {result['similarity']:.3f} | Type: {result['type']} | Heading: {result['heading'][:50]}...")

            conn.close()
            return top_results

        except Exception as e:
            st.error(f"❌ Error searching chunks: {e}")
            return []

    def generate_response(self, original_query, extracted_info, hit_chunks):
        """Generate final response using LLM"""
        st.write(f"🤖 **Generating response using {len(hit_chunks)} relevant chunks...**")

        if not hit_chunks:
            return "❌ No relevant policy information found for your query. Please try rephrasing or contact customer support."

        context_chunks = []
        for i, chunk in enumerate(hit_chunks, 1):
            context_chunks.append(f"--- Relevant Policy Section {i} ---\n{chunk['chunk']}\n")

        context = "\n".join(context_chunks)

        prompt = f"""You are an insurance policy expert helping a customer understand their coverage.
        CUSTOMER QUERY: "{original_query}"

EXTRACTED CUSTOMER INFO:
- Age: {extracted_info.get('age', 'Not specified')}
- Gender: {extracted_info.get('gender', 'Not specified')}
- Medical Condition: {extracted_info.get('medical_condition', 'Not specified')}
- Location: {extracted_info.get('location', 'Not specified')}
- Policy Duration: {extracted_info.get('policy_duration', 'Not specified')}

RELEVANT POLICY SECTIONS:
{context}

INSTRUCTIONS:
1. Provide a clear, helpful answer based ONLY on the policy sections above
2. Address the specific customer's situation (age, condition, location, etc.)
3. Mention any relevant waiting periods, exclusions, or conditions
4. Be specific about coverage amounts, limits, or percentages if mentioned
5. If information is insufficient, clearly state what additional details are needed
6. Use a friendly, professional tone
7. Structure your response with clear sections if multiple topics are covered

RESPONSE:"""

        try:
            st.write("📤 **Sending request to Groq API...**")

            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama3-8b-8192",
                    "messages": [
                        {"role": "system", "content": "You are a helpful insurance policy expert. Provide clear, accurate information based only on the policy documents provided. Just respond all what you have don't ask any follow up questions."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "temperature": 0.3,
                    "max_tokens": 800
                },
                timeout=30
            )

            if response.status_code == 200:
                answer = response.json()['choices'][0]['message']['content'].strip()
                st.write("✅ **Response generated successfully**")
                return answer
            st.error(f"❌ API call failed: {response.status_code}")
            return f"❌ Error generating response. Please try again or contact support."

        except Exception as e:
            st.error(f"❌ Error generating response: {e}")
            return f"❌ Error processing your query. Please try again or contact support."

    def process_query(self, raw_query, top_k=5, similarity_threshold=0.3):
        """Complete query processing pipeline"""
        st.write(f"🚀 **Processing query:** '{raw_query}'")

        # Step 1: Preprocess query
        processed_query, extracted_info = self.preprocess_query(raw_query)

        # Step 2: Search for similar chunks
        hit_chunks = self.search_similar_chunks(processed_query, top_k, similarity_threshold)

        # Step 3: Generate response
        final_response = self.generate_response(raw_query, extracted_info, hit_chunks)

        return {
            'final_response': final_response
        }

# Streamlit App
def main():
    st.set_page_config(page_title="Insurance Policy RAG System", page_icon="📋", layout="wide")
    
    st.title("📋 Insurance Policy RAG System")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox("Choose Mode:", ["📄 Document Processing", "🔍 Query System"])
    
    if mode == "📄 Document Processing":
        st.header("📄 Document Processing & Database Creation")
        st.write("Upload insurance policy documents to process and store in the database.")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose PDF or Word files", 
            type=['pdf', 'docx'], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.write(f"📁 **Uploaded {len(uploaded_files)} file(s)**")
            
            # Processing parameters
            col1, col2 = st.columns(2)
            with col1:
                delay_base = st.slider("Base delay (seconds)", 1, 10, 2)
            with col2:
                delay_random = st.slider("Random delay range (seconds)", 1, 5, 3)
            
            if st.button("🚀 Process Documents", type="primary"):
                all_chunks = []
                
                for uploaded_file in uploaded_files:
                    st.subheader(f"Processing: {uploaded_file.name}")
                    
                    # Save uploaded file temporarily
                    with open(uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    
                    try:
                        # Process document
                        chunks = DocumentProcessor.process_document(uploaded_file.name)
                        all_chunks.extend(chunks)
                        
                        # Clean up temp file
                        os.remove(uploaded_file.name)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
                        if os.path.exists(uploaded_file.name):
                            os.remove(uploaded_file.name)
                
                if all_chunks:
                    st.write(f"📊 **Total chunks from all documents:** {len(all_chunks)}")
                    
                    # Process and store chunks
                    st.subheader("🗄️ Database Storage")
                    DatabaseManager.process_and_store_chunks(all_chunks, delay_base, delay_random)
                else:
                    st.error("❌ No chunks were created from the uploaded documents.")
    
    elif mode == "🔍 Query System":
        st.header("🔍 Query System")
        st.write("Ask questions about your insurance policy.")
        
        # Query input
        query = st.text_input(
            "Enter your query:", 
            placeholder="e.g., 46M, knee surgery, Pune, 3-month policy",
            help="Format: Age+Gender, Medical Condition, Location, Policy Duration"
        )
        
        # Query parameters
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of chunks to retrieve", 1, 10, 5)
        with col2:
            similarity_threshold = st.slider("Similarity threshold", 0.1, 0.8, 0.3)
        
        if query and st.button("🔍 Search", type="primary"):
            query_system = PolicyQuerySystem()
            
            # Create expander for detailed processing
            with st.expander("🔧 Processing Details", expanded=True):
                result = query_system.process_query(query, top_k, similarity_threshold)
            
            # Display final response prominently
            st.subheader("💬 Response")
            st.success(result['final_response'])
        
        # Database status
        if st.button("📊 Check Database Status"):
            try:
                conn = sqlite3.connect("policy_chunks.db")
                cursor = conn.cursor()

                cursor.execute("SELECT COUNT(*) FROM metadata_chunks")
                total_records = cursor.fetchone()[0]

                cursor.execute("SELECT type, COUNT(*) FROM metadata_chunks GROUP BY type")
                type_counts = cursor.fetchall()

                st.write(f"📊 **Total records:** {total_records}")
                if type_counts:
                    st.write("📋 **Records by type:**")
                    for type_name, count in type_counts:
                        st.write(f"   • {type_name}: {count}")

                conn.close()

            except Exception as e:
                st.error(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    main()
