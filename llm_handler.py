import os
import json
import re
import openai
import hashlib
from dotenv import load_dotenv
from content_processor import extract_topics_from_text, chunk_content_for_processing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import time
import random

# Download necessary NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

load_dotenv()  # Load environment variables from .env file

# Update OpenAI client initialization to use the current API format
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Initialize the OpenAI client with the newer format
from openai import OpenAI
client = OpenAI(api_key=API_KEY)

# Get model configuration from environment or use default
LLM_MODEL = os.getenv("MODEL", "gpt-3.5-turbo-16k")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

def generate_flashcards(content, config):
    """
    Generate flashcards from the content using an LLM with enhanced processing.
    
    Args:
        content (str): The educational content to convert into flashcards
        config (dict): Configuration options for flashcard generation
        
    Returns:
        list: List of flashcard dictionaries with question and answer pairs
    """
    try:
        print(f"Starting flashcard generation for {len(content)} characters of content...")
        
        # Validate minimum content length
        if len(content.strip()) < 100:
            raise ValueError("Content is too short to generate meaningful flashcards. Please provide at least 100 characters of educational content.")
        
        # Extract topics to understand content structure
        topics, topic_sections = extract_topics_from_text(content, num_topics=8)
        print(f"Detected topics: {topics}")
        
        # Chunk content for more effective processing
        content_chunks = chunk_content_for_processing(content, topic_sections, max_chunk_size=5000)
        print(f"Content divided into {len(content_chunks)} chunks")
        
        all_flashcards = []
        generation_metadata = {
            "method": "enhanced_primary",
            "chunks_processed": len(content_chunks),
            "topics_detected": topics,
            "content_length": len(content)
        }
        
        # Calculate cards per chunk more intelligently
        target_cards = config.get('num_cards', 15)
        if len(content_chunks) == 1:
            cards_per_chunk = target_cards
        else:
            # Distribute cards based on content length in each chunk
            cards_per_chunk = max(3, target_cards // len(content_chunks))
            # Ensure we don't go under minimum per chunk
            if cards_per_chunk < 3 and target_cards >= len(content_chunks) * 3:
                cards_per_chunk = 3
        
        print(f"Target: {cards_per_chunk} cards per chunk")
        
        # Process each content chunk with retry logic
        for chunk_idx, (chunk_text, chunk_topics) in enumerate(content_chunks):
            print(f"Processing chunk {chunk_idx + 1}/{len(content_chunks)}...")
            
            # Create config for this chunk
            chunk_config = config.copy()
            chunk_config['num_cards'] = cards_per_chunk
            chunk_config['detected_topics'] = chunk_topics
            chunk_config['chunk_index'] = chunk_idx
            chunk_config['total_chunks'] = len(content_chunks)
            
            # Generate flashcards for this chunk with retry
            chunk_cards = generate_chunk_flashcards_with_retry(chunk_text, chunk_config, max_retries=2)
            
            if chunk_cards:
                print(f"Generated {len(chunk_cards)} cards from chunk {chunk_idx + 1}")
                all_flashcards.extend(chunk_cards)
            else:
                print(f"Failed to generate cards from chunk {chunk_idx + 1}, trying fallback...")
                # Try a simpler approach for this chunk
                fallback_cards = generate_simple_chunk_cards(chunk_text, chunk_config)
                all_flashcards.extend(fallback_cards)
        
        print(f"Generated {len(all_flashcards)} cards before post-processing")
        
        # Post-process to ensure quality and meet requirements
        processed_cards = enhanced_post_process_flashcards(all_flashcards, config, topics)
        
        # Ensure minimum card count
        min_cards = max(10, config.get('min_cards', 10))
        if len(processed_cards) < min_cards:
            print(f"Need {min_cards - len(processed_cards)} additional cards...")
            additional_cards = generate_supplementary_cards(content, config, topics, 
                                                          min_cards - len(processed_cards))
            processed_cards.extend(additional_cards)
        
        # Final validation and truncation
        final_cards = validate_and_finalize_cards(processed_cards, config)
        
        generation_metadata["final_cards_count"] = len(final_cards)
        generation_metadata["success"] = True
        print(f"Final generation metadata: {generation_metadata}")
        
        return final_cards
        
    except Exception as e:
        print(f"Error in main flashcard generation: {str(e)}")
        # Enhanced fallback with better error handling
        return enhanced_fallback_generation(content, config)

def generate_chunk_flashcards_with_retry(chunk_text, config, max_retries=2):
    """Generate flashcards for a chunk with retry logic"""
    for attempt in range(max_retries + 1):
        try:
            return generate_chunk_flashcards(chunk_text, config, config.get('chunk_index', 0))
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries:
                # Add small delay and adjust parameters for retry
                time.sleep(1)
                # Reduce complexity for retry
                config['num_cards'] = max(3, config['num_cards'] // 2)
            else:
                print(f"All attempts failed for chunk")
                return []

def generate_chunk_flashcards(chunk_text, config, chunk_idx):
    """Generate flashcards for a specific content chunk with enhanced prompting"""
    # Create enhanced prompts
    system_prompt = build_enhanced_system_prompt(config, chunk_idx)
    user_prompt = build_enhanced_user_prompt(chunk_text, config, chunk_idx)
    
    try:
        # Update the API call to use the new client format
        print(f"Calling OpenAI API with {len(chunk_text)} characters of text...")
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        # Update to handle the new response format
        response_text = response.choices[0].message.content
        print(f"Raw API response length: {len(response_text)} characters")
        
        # Enhanced JSON extraction and parsing
        flashcards = parse_flashcard_response(response_text)
        
        # Add metadata to cards
        for card in flashcards:
            card['_chunk_index'] = chunk_idx
            card['_generation_method'] = 'primary'
        
        return flashcards
        
    except Exception as e:
        print(f"API call failed: {str(e)}")
        # Add more detailed error reporting
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        raise

def build_enhanced_system_prompt(config, chunk_idx=0):
    """Build an enhanced system prompt with better instructions"""
    subject = config.get('subject', 'General')
    num_cards = config.get('num_cards', 10)
    detected_topics = config.get('detected_topics', [])
    
    system_prompt = f"""You are an expert educational content creator specializing in {subject}. Your task is to create {num_cards} high-quality flashcards that will help students learn and retain key concepts.

FLASHCARD QUALITY STANDARDS:
1. Questions should be specific, clear, and test understanding (not just memorization)
2. Answers should be complete, accurate, and self-contained
3. Each card should focus on ONE key concept or fact
4. Use varied question types: definitions, applications, comparisons, cause-effect relationships
5. Avoid overly obvious or trivial questions
6. Ensure answers provide enough context to be understood independently

QUESTION VARIETY EXAMPLES:
- Definition: "What is [concept] and why is it important?"
- Application: "How would you apply [concept] in [scenario]?"
- Comparison: "What is the difference between [A] and [B]?"
- Analysis: "What are the key factors that influence [process]?"
- Cause-Effect: "What happens when [condition] occurs?"

ANSWER GUIDELINES:
- Start with a direct answer to the question
- Provide necessary context and explanation
- Include relevant examples when helpful
- Keep answers concise but comprehensive (50-150 words typically)
- Use clear, educational language appropriate for the subject level"""

    if detected_topics:
        topic_list = ", ".join(detected_topics)
        system_prompt += f"\n\nCONTENT FOCUS: This section covers: {topic_list}"
    
    if config.get('add_difficulty', False):
        system_prompt += """\n\nDIFFICULTY LEVELS:
- Easy: Basic definitions, simple facts, direct recall
- Medium: Application of concepts, relationships between ideas, analysis
- Hard: Complex analysis, synthesis, evaluation, problem-solving"""
    
    if config.get('add_topics', False):
        system_prompt += """\n\nTOPIC ASSIGNMENT:
Assign each card to the most relevant topic/category based on its main concept."""
    
    system_prompt += f"\n\nIMPORTANT: Generate exactly {num_cards} flashcards. Format your response as a valid JSON array."
    
    return system_prompt

def build_enhanced_user_prompt(content, config, chunk_idx=0):
    """Build an enhanced user prompt with better structure"""
    num_cards = config.get('num_cards', 10)
    
    # Build JSON schema description
    schema_parts = ["'question'", "'answer'"]
    
    if config.get('add_difficulty', False):
        schema_parts.append("'difficulty'")
    
    if config.get('add_topics', False):
        schema_parts.append("'topic'")
    
    schema_description = ", ".join(schema_parts)
    
    user_prompt = f"""Create {num_cards} educational flashcards from the following content.

FORMAT: Return a JSON array where each flashcard is an object with: {schema_description}

CONTENT ANALYSIS INSTRUCTIONS:
1. Identify the most important concepts, facts, and relationships
2. Create questions that test different levels of understanding
3. Ensure questions are answerable based solely on the provided content
4. Make answers self-contained and educational

CONTENT TO ANALYZE:
{content}

Remember: Generate exactly {num_cards} flashcards in valid JSON format."""
    
    return user_prompt

def parse_flashcard_response(response_text):
    """Enhanced JSON parsing with better error recovery"""
    try:
        # Clean up the response text
        cleaned_text = response_text.strip()
        
        # Extract JSON from markdown code blocks
        json_str = extract_json_from_response(cleaned_text)
        
        # Parse JSON
        flashcards = json.loads(json_str)
        
        # Validate structure
        if not isinstance(flashcards, list):
            if isinstance(flashcards, dict):
                # Single flashcard returned as object
                flashcards = [flashcards]
            else:
                raise ValueError("Response is not a list or object")
        
        # Validate each flashcard
        validated_cards = []
        for i, card in enumerate(flashcards):
            if validate_flashcard_structure(card):
                validated_cards.append(card)
            else:
                print(f"Skipping invalid card {i}: {card}")
        
        return validated_cards
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        # Try to fix common JSON issues
        fixed_json = attempt_json_repair(response_text)
        if fixed_json:
            try:
                return json.loads(fixed_json)
            except:
                pass
        
        # If all else fails, try to extract individual cards
        return extract_cards_from_broken_json(response_text)
    
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return []

def extract_json_from_response(text):
    """Extract JSON from various response formats"""
    # Check for markdown code blocks
    patterns = [
        r'```json\n(.*?)\n```',
        r'```\n(.*?)\n```',
        r'\[.*\]',  # JSON array
        r'\{.*\}'   # JSON object
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1) if '(' in pattern else match.group(0)
            # Quick validation
            candidate = candidate.strip()
            if candidate.startswith('[') or candidate.startswith('{'):
                return candidate
    
    # If no patterns match, return the original text
    return text.strip()

def validate_flashcard_structure(card):
    """Validate that a flashcard has the required structure"""
    if not isinstance(card, dict):
        return False
    
    # Check required fields
    if 'question' not in card or 'answer' not in card:
        return False
    
    # Check field quality
    question = card.get('question', '').strip()
    answer = card.get('answer', '').strip()
    
    if len(question) < 5 or len(answer) < 10:
        return False
    
    return True

def attempt_json_repair(text):
    """Attempt to repair common JSON formatting issues"""
    try:
        # Common fixes
        text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
        text = re.sub(r',\s*]', ']', text)  # Remove trailing commas in arrays
        text = re.sub(r'}\s*{', '},{', text)  # Add missing commas between objects
        
        # Try to wrap in array if it looks like multiple objects
        if text.count('{') > 1 and not text.strip().startswith('['):
            text = '[' + text + ']'
        
        return text
    except:
        return None

def extract_cards_from_broken_json(text):
    """Extract flashcards from malformed JSON using regex"""
    cards = []
    
    # Look for question-answer patterns
    patterns = [
        r'"question":\s*"([^"]+)"[^}]*"answer":\s*"([^"]+)"',
        r"'question':\s*'([^']+)'[^}]*'answer':\s*'([^']+)'",
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.DOTALL)
        for match in matches:
            question = match.group(1).strip()
            answer = match.group(2).strip()
            if len(question) > 5 and len(answer) > 10:
                cards.append({
                    "question": question,
                    "answer": answer
                })
    
    return cards[:10]  # Limit to prevent too many low-quality cards

def generate_simple_chunk_cards(chunk_text, config):
    """Simplified card generation for difficult chunks"""
    try:
        # Use a simpler, more direct prompt
        simple_prompt = f"""Create 3-5 simple flashcards from this text. 
Format: JSON array with 'question' and 'answer' fields.

Text: {chunk_text[:2000]}

Focus on key facts and concepts that are clearly stated in the text."""
        
        # Update to use new client format
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": simple_prompt}],
            temperature=0.5,
            max_tokens=1500
        )
        
        response_text = response.choices[0].message.content
        cards = parse_flashcard_response(response_text)
        
        # Add metadata
        for card in cards:
            card['_generation_method'] = 'simple_fallback'
        
        return cards[:5]  # Limit to 5 cards max
        
    except Exception as e:
        print(f"Simple generation failed: {str(e)}")
        return []

def enhanced_post_process_flashcards(flashcards, config, topics):
    """Enhanced post-processing with better quality control"""
    print(f"Post-processing {len(flashcards)} flashcards...")
    
    # Step 1: Remove duplicates more intelligently
    unique_cards = smart_deduplicate_cards(flashcards)
    print(f"After deduplication: {len(unique_cards)} cards")
    
    # Step 2: Enhanced quality validation
    quality_cards = [card for card in unique_cards if enhanced_card_quality_check(card)]
    print(f"After quality check: {len(quality_cards)} cards")
    
    # Step 3: Improve card content
    improved_cards = [improve_card_content(card) for card in quality_cards]
    
    # Step 4: Balance topics if enabled
    if config.get('add_topics', False) and topics:
        balanced_cards = smart_topic_balancing(improved_cards, topics, config.get('num_cards', 15))
    else:
        balanced_cards = improved_cards
    
    # Step 5: Sort by quality score
    scored_cards = [(card, calculate_card_quality_score(card)) for card in balanced_cards]
    scored_cards.sort(key=lambda x: x[1], reverse=True)
    final_cards = [card for card, score in scored_cards]
    
    print(f"Final post-processed cards: {len(final_cards)}")
    return final_cards

def smart_deduplicate_cards(cards):
    """Smarter deduplication using semantic similarity"""
    unique_cards = []
    question_variants = set()
    
    for card in cards:
        question = card['question'].lower().strip()
        
        # Normalize question for comparison
        normalized = re.sub(r'[^\w\s]', '', question)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Create multiple variants for checking
        variants = [
            normalized,
            re.sub(r'^(what|how|why|when|where|who|which)\s+', '', normalized),
            re.sub(r'\s+(is|are|was|were|do|does|did)\s+', ' ', normalized)
        ]
        
        # Check if any variant already exists
        is_duplicate = any(variant in question_variants for variant in variants)
        
        if not is_duplicate:
            unique_cards.append(card)
            question_variants.update(variants)
    
    return unique_cards

def enhanced_card_quality_check(card):
    """Enhanced quality checking with more criteria"""
    question = card.get('question', '').strip()
    answer = card.get('answer', '').strip()
    
    # Basic length checks
    if len(question) < 8 or len(answer) < 15:
        return False
    
    # Check for question words
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'define', 'explain', 'describe']
    has_question_word = any(word in question.lower() for word in question_words)
    has_question_mark = '?' in question
    
    if not (has_question_word or has_question_mark):
        return False
    
    # Check answer quality
    if answer.lower().startswith(('yes', 'no')) and len(answer) < 30:
        return False  # Avoid simple yes/no answers without explanation
    
    # Check for meaningful content
    if len(set(question.lower().split())) < 3:  # Too few unique words
        return False
    
    return True

def improve_card_content(card):
    """Improve individual card content"""
    improved_card = card.copy()
    
    # Improve question formatting
    question = improved_card['question'].strip()
    if not question.endswith('?') and not question.endswith(':'):
        if any(question.lower().startswith(word) for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
            question += '?'
    
    # Capitalize first letter
    if question and question[0].islower():
        question = question[0].upper() + question[1:]
    
    improved_card['question'] = question
    
    # Improve answer formatting
    answer = improved_card['answer'].strip()
    if answer and answer[0].islower():
        answer = answer[0].upper() + answer[1:]
    
    improved_card['answer'] = answer
    
    return improved_card

def calculate_card_quality_score(card):
    """Calculate a quality score for ranking cards"""
    score = 0
    
    question = card.get('question', '')
    answer = card.get('answer', '')
    
    # Length scores (optimal ranges)
    q_len = len(question)
    a_len = len(answer)
    
    if 15 <= q_len <= 100:
        score += 2
    elif 8 <= q_len <= 150:
        score += 1
    
    if 30 <= a_len <= 200:
        score += 2
    elif 15 <= a_len <= 300:
        score += 1
    
    # Question quality indicators
    question_words = ['what', 'how', 'why', 'explain', 'describe', 'analyze', 'compare']
    if any(word in question.lower() for word in question_words):
        score += 1
    
    # Answer quality indicators
    if not answer.lower().startswith(('yes', 'no', 'true', 'false')):
        score += 1
    
    # Check for educational value
    educational_words = ['because', 'therefore', 'however', 'although', 'example', 'such as']
    if any(word in answer.lower() for word in educational_words):
        score += 1
    
    return score

def smart_topic_balancing(cards, topics, target_count):
    """Intelligent topic balancing"""
    if not topics or len(topics) <= 1:
        return cards[:target_count]
    
    # Group cards by topic (assign if missing)
    topic_groups = {}
    unassigned_cards = []
    
    for card in cards:
        topic = card.get('topic')
        if topic and topic in topics:
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(card)
        else:
            # Try to assign topic based on content
            assigned_topic = assign_topic_to_card(card, topics)
            if assigned_topic:
                card['topic'] = assigned_topic
                if assigned_topic not in topic_groups:
                    topic_groups[assigned_topic] = []
                topic_groups[assigned_topic].append(card)
            else:
                unassigned_cards.append(card)
    
    # Balance selection
    cards_per_topic = max(1, target_count // len(topic_groups)) if topic_groups else 0
    balanced_cards = []
    
    # Take cards from each topic group
    for topic, topic_cards in topic_groups.items():
        # Sort by quality within topic
        topic_cards.sort(key=calculate_card_quality_score, reverse=True)
        balanced_cards.extend(topic_cards[:cards_per_topic])
    
    # Fill remaining slots
    remaining_slots = target_count - len(balanced_cards)
    if remaining_slots > 0:
        # Add remaining high-quality cards
        remaining_cards = []
        for topic_cards in topic_groups.values():
            remaining_cards.extend(topic_cards[cards_per_topic:])
        remaining_cards.extend(unassigned_cards)
        
        remaining_cards.sort(key=calculate_card_quality_score, reverse=True)
        balanced_cards.extend(remaining_cards[:remaining_slots])
    
    return balanced_cards

def assign_topic_to_card(card, topics):
    """Assign a topic to a card based on content similarity"""
    card_text = (card.get('question', '') + ' ' + card.get('answer', '')).lower()
    
    best_topic = None
    best_score = 0
    
    for topic in topics:
        topic_words = topic.lower().split()
        score = sum(1 for word in topic_words if word in card_text)
        if score > best_score:
            best_score = score
            best_topic = topic
    
    return best_topic if best_score > 0 else None

def generate_supplementary_cards(content, config, topics, num_needed):
    """Generate additional high-quality cards when not enough are produced"""
    print(f"Generating {num_needed} supplementary cards...")
    
    try:
        # Use a focused approach for supplementary cards
        system_prompt = f"""You are an expert educator. Generate {num_needed} additional HIGH-QUALITY flashcards from the content below.

REQUIREMENTS:
- Focus on important concepts that might have been missed
- Create diverse question types (not just definitions)
- Ensure each card tests understanding, not just recall
- Make answers complete and educational

Generate exactly {num_needed} flashcards as a JSON array."""

        user_prompt = f"""Content: {content[:6000]}

Create {num_needed} educational flashcards focusing on key concepts, relationships, and applications."""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.8,  # Higher temperature for more creative cards
            max_tokens=MAX_TOKENS
        )
        
        response_text = response.choices[0].message.content
        cards = parse_flashcard_response(response_text)
        
        # Add metadata
        for card in cards:
            card['_generation_method'] = 'supplementary'
            if config.get('add_topics') and topics:
                card['topic'] = assign_topic_to_card(card, topics) or topics[0]
        
        return cards[:num_needed]
        
    except Exception as e:
        print(f"Supplementary generation failed: {str(e)}")
        return []

def enhanced_fallback_generation(content, config):
    """Enhanced fallback generation with better content analysis"""
    print("Using enhanced fallback generation...")
    
    try:
        min_cards = max(10, config.get('min_cards', 10))
        
        # Use a simpler but more reliable approach
        simple_system = """You are an educational content expert. Create simple, clear flashcards from the given content.

Each flashcard should:
- Have a clear, specific question
- Have a complete, accurate answer based on the content
- Focus on key facts and concepts

Return a JSON array of flashcards with 'question' and 'answer' fields."""

        simple_user = f"""Create {min_cards} flashcards from this educational content:

{content[:4000]}

Format: JSON array with objects containing 'question' and 'answer' fields."""

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": simple_system},
                {"role": "user", "content": simple_user}
            ],
            temperature=0.5,
            max_tokens=3000
        )
        
        response_text = response.choices[0].message.content
        cards = parse_flashcard_response(response_text)
        
        # Add metadata
        for card in cards:
            card['_generation_method'] = 'enhanced_fallback'
            if config.get('add_difficulty'):
                card['difficulty'] = 'Medium'
            if config.get('add_topics'):
                card['topic'] = config.get('subject', 'General')
        
        return cards[:min_cards]
        
    except Exception as e:
        print(f"Enhanced fallback failed: {str(e)}")
        # Ultimate fallback - create basic cards from content structure
        return create_basic_structure_cards(content, config)

# Add a debug function to check API connectivity
def check_api_connection():
    """Test the OpenAI API connection and return status"""
    try:
        print("Testing OpenAI API connection...")
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",  # Using an instruction model for a simple test
            prompt="Say hello",
            max_tokens=5
        )
        print("API connection successful!")
        return True, "Connection successful"
    except Exception as e:
        error_message = f"API connection failed: {str(e)}"
        print(error_message)
        return False, error_message

def create_basic_structure_cards(content, config):
    """Create basic cards from content structure when all else fails"""
    cards = []
    min_cards = max(10, config.get('min_cards', 10))
    subject = config.get('subject', 'General')
    
    # Extract sentences and create basic cards
    sentences = sent_tokenize(content)
    meaningful_sentences = [s for s in sentences if len(s) > 30 and not s.startswith('Page')]
    
    # Use content directly to create specific cards rather than generic templates
    if len(meaningful_sentences) >= min_cards // 2:
        for i, sentence in enumerate(meaningful_sentences[:min_cards]):
            # Extract potential key terms or phrases
            important_terms = extract_important_terms(sentence)
            if important_terms:
                key_term = important_terms[0]
                question = f"What does the text explain about {key_term}?"
                answer = sentence.strip()
                
                cards.append({
                    "question": question,
                    "answer": answer,
                    "_generation_method": "content_extraction",
                    "difficulty": "Medium"
                })
                
                # If we've reached our minimum, stop
                if len(cards) >= min_cards:
                    break
    
    # If we don't have enough content-specific cards yet, use content structure
    if len(cards) < min_cards:
        # Try to extract key segments based on paragraph structure
        paragraphs = content.split('\n\n')
        meaningful_paragraphs = [p for p in paragraphs if len(p.strip()) > 100]
        
        for i, paragraph in enumerate(meaningful_paragraphs[:min_cards - len(cards)]):
            first_sentence = sent_tokenize(paragraph)[:1]
            if first_sentence:
                intro = first_sentence[0][:50].strip()
                
                # Create a relevant question based on paragraph content
                question = f"Based on the content about '{intro}...', what can be concluded?"
                
                # Use a summarization approach for the answer
                key_points = extract_key_points(paragraph)
                answer = key_points if key_points else paragraph[:300] + "..."
                
                cards.append({
                    "question": question,
                    "answer": answer,
                    "_generation_method": "paragraph_analysis",
                    "difficulty": "Medium"
                })
    
    # If we still need more cards, create focused subject-specific cards
    # that reference the content directly
    if len(cards) < min_cards:
        remaining_needed = min_cards - len(cards)
        content_focused_cards = create_content_focused_cards(content, subject, remaining_needed)
        cards.extend(content_focused_cards)
    
    # Add topics if configured
    if config.get('add_topics', False):
        # Try to derive meaningful topics from the content
        derived_topics = derive_topics_from_content(content, subject)
        for i, card in enumerate(cards):
            topic_index = i % len(derived_topics) if derived_topics else 0
            card['topic'] = derived_topics[topic_index] if derived_topics else f"{subject} Concepts"
    
    return cards[:min_cards]  # Ensure we don't exceed the minimum

def extract_important_terms(text):
    """Extract potentially important terms from a piece of text"""
    # Look for capitalized terms or phrases
    capitalized = re.findall(r'\b[A-Z][a-zA-Z]{2,}\b', text)
    
    # Look for terms in quotes
    quoted = re.findall(r'["\'](.*?)["\']', text)
    
    # Look for potentially important noun phrases
    words = word_tokenize(text)
    potential_terms = []
    
    for i in range(len(words) - 1):
        # Simple pattern: adjective + noun
        if words[i].isalpha() and words[i+1].isalpha() and len(words[i]) > 3 and len(words[i+1]) > 3:
            potential_terms.append(words[i] + " " + words[i+1])
    
    # Combine and prioritize results
    all_terms = []
    if quoted:
        all_terms.extend(quoted)
    if capitalized:
        all_terms.extend(capitalized)
    if potential_terms:
        all_terms.extend(potential_terms)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = [term for term in all_terms if not (term in seen or seen.add(term))]
    
    return unique_terms

def extract_key_points(text):
    """Extract key points from a paragraph of text"""
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    
    # Look for sentences with indicator phrases
    key_indicators = ['important', 'significant', 'key', 'crucial', 'essential', 'primary', 'fundamental']
    highlighted_sentences = []
    
    for sentence in sentences:
        lower_sent = sentence.lower()
        if any(indicator in lower_sent for indicator in key_indicators):
            highlighted_sentences.append(sentence)
    
    # If we found sentences with indicators, use those
    if highlighted_sentences:
        return " ".join(highlighted_sentences)
    
    # Otherwise use first and last sentence as they often contain key info
    if len(sentences) >= 2:
        return sentences[0] + " " + sentences[-1]
    elif sentences:
        return sentences[0]
    
    return ""

def create_content_focused_cards(content, subject, count):
    """Create cards focused on the content rather than generic templates"""
    cards = []
    
    # Extract potential analysis questions based on the content and subject
    text_sample = content[:3000]  # Work with a sample of the content
    
    # Extract any identifiable concepts or related terms
    related_concepts = []
    
    # Subject-specific concept extraction
    if subject == "Computer Science":
        # Look for potential programming concepts or technical terms
        programming_terms = re.findall(r'\b(algorithm|function|class|object|data structure|variable|loop|condition|database|API|interface|component|module|network|system|server|client)\b', 
                                     text_sample.lower())
        related_concepts.extend(programming_terms)
        
    elif subject == "Mathematics":
        # Look for math concepts
        math_terms = re.findall(r'\b(equation|function|theorem|proof|formula|calculation|integral|derivative|matrix|vector|set|probability|statistics|geometry|algebra)\b', 
                              text_sample.lower())
        related_concepts.extend(math_terms)
        
    elif subject == "Biology":
        biology_terms = re.findall(r'\b(cell|organism|species|evolution|genetics|protein|enzyme|tissue|organ|system|ecology|molecule|DNA|RNA|bacteria|virus)\b', 
                                 text_sample.lower())
        related_concepts.extend(biology_terms)
        
    elif subject == "History":
        history_terms = re.findall(r'\b(event|period|era|century|war|treaty|revolution|movement|civilization|culture|society|empire|nation|leader|government)\b', 
                                 text_sample.lower())
        related_concepts.extend(history_terms)
    
    # Deduplicate
    related_concepts = list(set(related_concepts))
    
    # Create content-specific questions
    content_based_questions = [
        f"Based on the content, how does {concept} relate to the main topic?" for concept in related_concepts
    ]
    
    # Add analysis questions
    analysis_questions = [
        "What is the primary focus of this content?",
        "How do the concepts in this content relate to each other?",
        "What evidence or examples support the main ideas in this content?",
        "What implications or applications arise from the concepts in this material?",
        "How might someone apply the information from this content in practice?"
    ]
    
    # Combine and shuffle questions for variety
    all_questions = content_based_questions + analysis_questions
    random.shuffle(all_questions)
    
    # Create cards from these questions, using content insights
    for i, question in enumerate(all_questions):
        if i >= count:
            break
            
        # For each question, extract a relevant part of the content as answer
        relevant_content = extract_relevant_content(content, question)
        
        cards.append({
            "question": question,
            "answer": relevant_content,
            "_generation_method": "content_analysis",
            "difficulty": "Medium"
        })
    
    return cards

def extract_relevant_content(content, question):
    """Extract content relevant to a specific question"""
    # Simple relevance algorithm:
    # 1. Extract key terms from the question
    question_words = set(question.lower().split())
    question_words = {word for word in question_words if len(word) > 3 and word not in stopwords.words('english')}
    
    # 2. Find paragraphs with the highest density of those terms
    paragraphs = content.split('\n\n')
    best_paragraph = ""
    best_score = 0
    
    for para in paragraphs:
        if len(para.strip()) < 50:  # Skip very short paragraphs
            continue
            
        para_lower = para.lower()
        score = sum(1 for word in question_words if word in para_lower)
        
        if score > best_score:
            best_score = score
            best_paragraph = para
    
    # If we found a relevant paragraph, use it
    if best_paragraph and len(best_paragraph) > 50:
        if len(best_paragraph) > 300:
            # Truncate long paragraphs
            return best_paragraph[:300] + "..."
        return best_paragraph
    
    # Fallback to first substantial paragraph
    for para in paragraphs:
        if len(para.strip()) >= 100:
            return para[:300] + "..." if len(para) > 300 else para
    
    # Ultimate fallback - first part of content
    return content[:200] + "..."

def derive_topics_from_content(content, subject):
    """Derive meaningful topic names from content"""
    # Start with subject-specific base topics
    base_topics = {
        "Computer Science": ["Algorithms", "Data Structures", "Programming", "Software Development", "Computer Systems"],
        "Mathematics": ["Algebra", "Calculus", "Geometry", "Statistics", "Number Theory"],
        "Biology": ["Cell Biology", "Genetics", "Ecology", "Physiology", "Evolution"],
        "History": ["Historical Events", "Cultural History", "Political History", "Social Movements", "Economic History"],
        "Physics": ["Mechanics", "Thermodynamics", "Electromagnetism", "Quantum Physics", "Relativity"],
        "General": ["Key Concepts", "Principles", "Applications", "Theory", "Analysis"]
    }
    
    # Get base topics for this subject
    topics = base_topics.get(subject, base_topics["General"])
    
    try:
        # Try to extract custom topics from content
        custom_topics = []
        
        # Look for capitalized multi-word phrases that might be topics
        topic_candidates = re.findall(r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', content)
        if topic_candidates:
            custom_topics.extend(topic_candidates[:3])  # Add up to 3 custom topics
        
        # Combine base and custom topics
        if custom_topics:
            topics = custom_topics + topics
            
        return topics
    except:
        # If anything goes wrong, return the base topics
        return topics

def extract_json(text):
    """Extract JSON string from text that might contain markdown or other content"""
    # Check for markdown code blocks
    patterns = [
        r'```json\n(.*?)\n```',
        r'```\n(.*?)\n```',
        r'\[.*\]',  # JSON array
        r'\{.*\}'   # JSON object
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            candidate = match.group(1) if '(' in pattern else match.group(0)
            # Quick validation
            candidate = candidate.strip()
            if candidate.startswith('[') or candidate.startswith('{'):
                return candidate
    
    # If no patterns match, return the original text
    return text.strip()

def validate_and_finalize_cards(cards, config):
    """Final validation and preparation of flashcards"""
    target_count = config.get('num_cards', 15)
    min_count = max(10, config.get('min_cards', 10))
    
    # Final quality check
    final_cards = []
    for card in cards:
        if enhanced_card_quality_check(card):
            # Add any missing fields
            if config.get('add_difficulty', False) and 'difficulty' not in card:
                card['difficulty'] = estimate_difficulty(card)
            
            final_cards.append(card)
    
    # Ensure we meet minimum requirements
    if len(final_cards) < min_count:
        print(f"Warning: Only {len(final_cards)} cards meet quality standards (minimum: {min_count})")
    
    # Truncate to target if we have too many
    if len(final_cards) > target_count:
        final_cards = final_cards[:target_count]
    
    return final_cards

def estimate_difficulty(card):
    """Estimate difficulty level for a card"""
    question = card.get('question', '').lower()
    answer = card.get('answer', '').lower()
    
    # Simple heuristics for difficulty estimation
    easy_indicators = ['what is', 'define', 'who is', 'when did']
    medium_indicators = ['how does', 'why does', 'explain how', 'describe the process']
    hard_indicators = ['analyze', 'evaluate', 'compare and contrast', 'what would happen if']
    
    if any(indicator in question for indicator in hard_indicators):
        return 'Hard'
    elif any(indicator in question for indicator in medium_indicators):
        return 'Medium'
    elif any(indicator in question for indicator in easy_indicators):
        return 'Easy'
    else:
        # Default based on answer complexity
        if len(answer) > 150:
            return 'Medium'
        else:
            return 'Easy'