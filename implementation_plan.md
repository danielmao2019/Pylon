# Knowledge Chat Bot Implementation Plan

## Project Overview

A chat bot that builds deep, layered knowledge from provided sources. Users provide sources (repositories, papers, databases) → the bot builds knowledge through iterative inference → users can then ask anything about those sources.

## Document Structure

**Part I: System Design**
1. **Key Innovation** - User interaction as an information source (unified source model)
2. **Core Architecture** - Information source interface and knowledge building process
3. **Knowledge Building** - Rigorous evidence-based inference (BFS/DFS strategies)

**Part II: Implementation**
4. **System Components** - Knowledge representation, inference engine, information processing
5. **Source Implementations** - GitHub repos, user interaction, PDFs, databases  
6. **User Interface** - Web interface and query processing

**Part III: Development** 
7. **Technical Stack** - Dependencies and implementation timeline
8. **Usage & Examples** - Practical workflows, detailed examples, and design decisions

---

# Part I: System Design

## Key Innovation: User Interaction as Information Source

### The Unified Source Insight

**Traditional chat bots**: User confirmation is handled as a special "interaction system"  
**Our approach**: User responses are just another **information source**, equal to files, databases, and web pages

```
// All information sources follow same interface:
sources = [
    GitHubRepo("/path/to/repo"),     // Static: files, structure, dependencies
    UserInteraction(),               // Interactive: confirmations, clarifications  
    PDFDocument("/path/to/paper"),   // Static: text, references, figures
    Database("connection_string")    // Dynamic: records, relationships
]

// Same processing pipeline for all sources:
for each source:
    raw_info = source.extract_information()     // Source-specific extraction
    knowledge = convert_to_knowledge(raw_info)  // Standardize format
    knowledge_base.add(knowledge)               // Store with source tracking
    inference_engine.infer_new_knowledge()     // Build derived knowledge
```

### Benefits of This Approach

- **🔄 Extensibility**: Adding new source types is trivial
- **🎯 Consistency**: Same rigorous processing for all information
- **🧠 Simplicity**: No special cases or complex interaction handling  
- **📊 Traceability**: All knowledge clearly shows its source
- **⚡ Scalability**: Unlimited source types with same architecture

### Information Source Equality Examples

| Traditional Approach | Our Unified Approach |
|---------------------|---------------------|
| Read GitHub repo → Extract facts | `github_source.extract_information()` |
| Read PDF → Extract content | `pdf_source.extract_information()` |
| Query database → Get records | `database_source.extract_information()` |
| **Special confirmation system** | `user_source.extract_information()` ✨ |

**The key insight**: Whether information comes from a file, database, or user response doesn't matter - it's all just `RawInformation` that gets processed the same way.

## Core Architecture: Unified Information Source Model

### Information Source Interface

All information sources implement the same contract:

```
interface InformationSource {
    get_source_type() -> string        // "github_repo", "pdf", "user_interaction"
    extract_information() -> [RawInfo] // Extract all available information
    is_available() -> boolean          // Check if source is ready
}

struct RawInformation {
    content: string           // The actual information content
    type: string             // "file_content", "user_statement", "database_record"
    source_metadata: dict    // Source-specific details (file path, timestamp, etc.)
}
```

### The Unified Knowledge Building Process

```
Multiple Information Sources → Extract RawInformation → Convert to Knowledge
     ↓                              ↓                        ↓
GitHub Repo                  File contents             Facts about code
PDF Documents               Text + structure          Research findings  
User Interaction            Confirmations             User clarifications
Database Records           Structured data            Domain facts
     ↓                              ↓                        ↓
                    All sources feed into same pipeline
                              ↓
                    Rigorous Inference Engine
                              ↓
                    Enhanced Knowledge Base
                              ↓
                    Ready for Questions
```

## Knowledge Building: Rigorous Evidence-Based Inference

### Knowledge Inference Strategies (BFS/DFS Analogies)

#### 1. **Breadth-First Knowledge Building**
*Explore all immediate inferences at each layer before going deeper*

```
BFS Strategy: Build wide knowledge first, then deep insights

Layer 0: [Facts from all sources] 
Layer 1: [All possible inferences from Layer 0]
Layer 2: [All possible inferences from Layer 1]
...continue until no new knowledge

Algorithm:
  current_layer = all_facts_and_relationships
  while current_layer has items and depth < MAX_DEPTH:
    next_layer = []
    for each pair (k1, k2) in current_layer:
      if can_infer_rigorously(k1, k2):
        new_knowledge = infer(k1, k2)
        next_layer.add(new_knowledge)
    current_layer = next_layer
    depth++
```

**BFS Result Example:**
```
Layer 0: [function_def(parse), function_def(analyze), class_def(Parser)]
Layer 1: [Parser_uses_parse, Parser_uses_analyze, parse_calls_analyze] 
Layer 2: [Parser_is_main_component, parse_analyze_pipeline_pattern]
Layer 3: [codebase_follows_modular_design]
```

#### 2. **Depth-First Knowledge Building** 
*Follow one reasoning chain as deep as possible before exploring alternatives*

```
DFS Strategy: Pick a domain and infer as deeply as possible

Focus areas: 'architecture', 'data_flow', 'error_handling', etc.

Algorithm:
  stack = [seed_knowledge_for_focus_area]
  while stack not empty and depth < MAX_DEPTH:
    current = stack.pop()
    best_next = find_strongest_inference_from(current, focus_area)
    if best_next and not already_known(best_next):
      knowledge_base.add(best_next)
      stack.push(best_next)  // Continue this reasoning chain
```

**DFS Result Example (focus_area='architecture'):**
```
Chain 1: class_def(FastAPI) → inherits(Starlette) → web_framework_pattern → 
         ASGI_architecture → async_request_handling → high_performance_design

Chain 2: function_def(@app.get) → decorator_pattern → routing_mechanism → 
         RESTful_API_design → HTTP_method_mapping → API_endpoint_architecture
```

#### 3. **Strategy Comparison**

| Aspect | Breadth-First | Depth-First |
|--------|---------------|-------------|
| **Coverage** | Comprehensive across all domains | Deep in specific domains |
| **Knowledge Type** | Broad understanding | Specialized expertise |
| **Use Case** | General Q&A about entire codebase | Focused analysis (architecture, security, etc.) |
| **Processing Time** | Longer (explores everything) | Faster (focused exploration) |
| **Result Quality** | Well-rounded knowledge | Deep domain insights |

**Strategy Selection Logic:**
```
Strategy Selection:
  if user_intent = "overview" or "general_understanding":
    use breadth_first  // Wide coverage across all domains
  elif user_intent = "architecture_analysis" or "security_review":
    use depth_first    // Deep dive into specific area
  elif source_type = "large_codebase":
    use hybrid         // BFS first, then DFS on key areas
  else:
    use breadth_first  // Default to comprehensive
```

---

# Part II: Implementation

## System Components

### Knowledge Representation System

#### Knowledge Confidence System

```
Knowledge Status Types:
  VERIFIED   - 100% certain from direct evidence (confidence = 1.0)
  DEDUCED    - Logically derived, no assumptions (confidence = 0.95)
  UNKNOWN    - Cannot determine from evidence (confidence = 0.0)
  CONFLICTED - Multiple contradictory sources (confidence = 0.0)

Knowledge Structure:
  {
    content: "Class Parser is defined in parser.py"
    type: "class_definition" 
    source: "github_repo:parser.py:15"
    status: VERIFIED
    evidence: [list of supporting knowledge items]
    confidence: 1.0
  }
```

#### Rigorous Inference Rules

```
Inference Rules (only build knowledge that can be PROVEN):

Rule 1: Function Usage
  if (k1 = function_definition AND k2 = function_call):
    if exact_name_match(k1.name, k2.name):
      create VERIFIED knowledge: "Function X is called at location Y"

Rule 2: Class Inheritance  
  if (k1 = class_definition AND k2 = inheritance_declaration):
    if exact_parent_match(k1.name, k2.parent):
      create VERIFIED knowledge: "Class X is base class for Y"

Rule 3: Import Usage
  if (k1 = import_statement AND k2 = symbol_usage):
    if import_provides_symbol(k1, k2.symbol):
      create VERIFIED knowledge: "Module X is used via symbol Y"

Handling Uncertainty:
  - If evidence insufficient → create UNKNOWN knowledge
  - If sources contradict → create CONFLICTED knowledge  
  - If no rigorous rule applies → create nothing (don't guess)
```

#### Rigorous Evidence Standards

```
Evidence Requirements by Inference Type:

function_usage:
  - Requires exactly 2 pieces of evidence
  - Must have function_definition + function_call  
  - Names must match exactly (no fuzzy matching)
  - Both must have valid location information

inheritance_relationship:
  - Requires exactly 2 pieces of evidence
  - Must have class_definition + inheritance_declaration
  - Parent class name must match exactly
  - Must use explicit "class Child(Parent):" syntax

dependency_usage:
  - Requires exactly 2 pieces of evidence  
  - Must have import_statement + symbol_usage
  - Import must actually provide the used symbol
  - No inferring imports from usage alone

General Rules:
  - Unknown inference types → not rigorous (return false)
  - Missing required evidence → not rigorous
  - Fuzzy/approximate matches → not rigorous
```

### Information Processing Flow

```python
class KnowledgeBase:
    """Unified knowledge storage treating all sources equally"""
    
    def __init__(self):
        self.verified_knowledge = []    # High confidence knowledge
        self.deduced_knowledge = []     # Logically derived knowledge  
        self.unknown_items = []         # Cannot determine from evidence
        self.conflicts = []             # Contradictory evidence
        self.source_tracker = {}        # Track which knowledge came from which sources
        
    def add_knowledge(self, knowledge: Knowledge):
        """Route knowledge to appropriate collection based on status"""
        # Track source
        source_type = knowledge.source.split(':')[0]
        if source_type not in self.source_tracker:
            self.source_tracker[source_type] = []
        self.source_tracker[source_type].append(knowledge)
        
        # Route by status
        if knowledge.status == KnowledgeStatus.VERIFIED:
            self.verified_knowledge.append(knowledge)
        elif knowledge.status == KnowledgeStatus.DEDUCED:
            self.deduced_knowledge.append(knowledge)
        elif knowledge.status == KnowledgeStatus.UNKNOWN:
            self.unknown_items.append(knowledge)
        elif knowledge.status == KnowledgeStatus.CONFLICTED:
            self.conflicts.append(knowledge)
            
    def get_knowledge_by_source(self, source_type: str) -> List[Knowledge]:
        """Get all knowledge from a specific source type"""
        return self.source_tracker.get(source_type, [])
        
    def query_with_uncertainty(self, question: str) -> QueryResult:
        """Return answer with explicit uncertainty information"""
        
        # Search all knowledge types
        verified_matches = self._search(question, self.verified_knowledge)
        deduced_matches = self._search(question, self.deduced_knowledge)  
        unknown_matches = self._search(question, self.unknown_items)
        conflict_matches = self._search(question, self.conflicts)
        
        return QueryResult(
            verified_facts=verified_matches,
            deduced_facts=deduced_matches,
            unknown_areas=unknown_matches,
            conflicts=conflict_matches
        )

class QueryResult:
    def __init__(self, verified_facts, deduced_facts, unknown_areas, conflicts):
        self.verified_facts = verified_facts      # 100% certain
        self.deduced_facts = deduced_facts        # Logically sound
        self.unknown_areas = unknown_areas        # Cannot determine
        self.conflicts = conflicts                # Contradictory evidence
        
    def has_uncertain_knowledge(self) -> bool:
        """Check if there are areas that need user clarification"""
        return len(self.unknown_areas) > 0 or len(self.conflicts) > 0 or len(self.deduced_facts) > 0
        
    def get_most_relevant_uncertain(self) -> Knowledge:
        """Get the most relevant uncertain knowledge for user clarification"""
        if self.conflicts:
            return self.conflicts[0]  # Conflicts are highest priority
        elif self.unknown_areas:
            return self.unknown_areas[0]  # Unknown areas next
        elif self.deduced_facts:
            return self.deduced_facts[0]  # Deductions can be confirmed
        return None
        
    def generate_response(self) -> str:
        """Generate response that clearly indicates certainty levels and sources"""
        response = []
        
        if self.verified_facts:
            response.append("VERIFIED FACTS:")
            for fact in self.verified_facts:
                source_type = fact.source.split(':')[0]
                response.append(f"✓ {fact.content} (Source: {source_type})")
                
        if self.deduced_facts:
            response.append("\nLOGICAL DEDUCTIONS:")
            for deduction in self.deduced_facts:
                response.append(f"→ {deduction.content} (Derived from: {len(deduction.evidence)} sources)")
                
        if self.unknown_areas:
            response.append("\nUNCLEAR AREAS:")
            for unknown in self.unknown_areas:
                response.append(f"? {unknown.content}")
                
        if self.conflicts:
            response.append("\nCONFLICTING INFORMATION:")
            for conflict in self.conflicts:
                response.append(f"⚠ {conflict.content}")
                
        return "\n".join(response)
```

### Unified Information Flow Example

```
User: "What's the purpose of the Parser class?"

Bot: "Based on analysis from multiple sources:
VERIFIED FACTS:
✓ Parser class is defined in parser.py (Source: github_repo)

LOGICAL DEDUCTIONS:
→ Parser appears to process input data (Derived from: 2 sources)

❓ I deduced that Parser appears to process input data. Is this correct?"

User: "Yes, but specifically it parses configuration files"

Bot: [Processes user response as new information source]
"Thank you! I've learned from your response. What else would you like to know?"

User: "How does the Parser work?"

Bot: "VERIFIED FACTS:
✓ Parser class is defined in parser.py (Source: github_repo)
✓ Parser parses configuration files (Source: user_interaction)
..."
```

## System Architecture

### Unified Information Source Model

The chat bot treats all information equally, whether from:
- **Static sources**: GitHub repos, PDFs, databases, web pages
- **Interactive sources**: User confirmations, clarifications, corrections

```python
class InformationSource(ABC):
    """Abstract base for all information sources"""
    
    @abstractmethod
    def get_source_type(self) -> str:
        """Return source type: 'github_repo', 'pdf', 'database', 'user_interaction'"""
        pass
        
    @abstractmethod
    def extract_information(self) -> List[RawInformation]:
        """Extract raw information from this source"""
        pass
        
    @abstractmethod
    def is_available(self) -> bool:
        """Check if source is ready to provide information"""
        pass

class RawInformation:
    """Single piece of raw information from any source"""
    def __init__(self, content: str, info_type: str, source_metadata: Dict):
        self.content = content
        self.info_type = info_type  # 'file_content', 'user_statement', 'database_record'
        self.source_metadata = source_metadata  # source-specific details
        self.timestamp = datetime.now()
```

### Information Source Types

```python
class GitHubRepoSource(InformationSource):
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        
    def get_source_type(self) -> str:
        return 'github_repo'
        
    def extract_information(self) -> List[RawInformation]:
        # Extract from files, README, dependencies...
        return raw_info_list

class UserInteractionSource(InformationSource):
    """User confirmations, clarifications, and corrections as information source"""
    
    def __init__(self):
        self.pending_questions = Queue()
        self.user_responses = []
        
    def get_source_type(self) -> str:
        return 'user_interaction'
        
    def extract_information(self) -> List[RawInformation]:
        """Convert user responses into raw information"""
        raw_info = []
        for response in self.user_responses:
            raw_info.append(RawInformation(
                content=response.content,
                info_type='user_statement',
                source_metadata={
                    'question_context': response.original_question,
                    'confidence': 1.0,
                    'user_provided': True
                }
            ))
        return raw_info
        
    def add_user_response(self, question: str, response: str):
        """Add user response as new information"""
        self.user_responses.append(UserResponse(question, response))
        
    def has_pending_questions(self) -> bool:
        return not self.pending_questions.empty()
        
    def get_next_question(self) -> str:
        return self.pending_questions.get()
        
    def ask_user(self, question: str, context: Knowledge):
        """Queue a question for the user"""
        self.pending_questions.put(UserQuestion(question, context))

class DatabaseSource(InformationSource):
    def get_source_type(self) -> str:
        return 'database'
        
class PDFSource(InformationSource):
    def get_source_type(self) -> str:
        return 'pdf_document'
```

### Core Components

```
Main ChatBot Architecture:

class KnowledgeChatBot:
  components:
    - knowledge_base: stores all knowledge with source tracking
    - inference_engine: builds new knowledge from existing knowledge
    - information_sources: list of all sources (github, pdf, user, etc.)
    - query_engine: answers questions based on knowledge

  key methods:
    add_information_source(source):
      - Add any source type to the system
      - Extract raw information from source  
      - Convert to structured knowledge
      - Run inference to build derived knowledge
      
    chat(user_input):
      - If user responding to bot question: process as new information source
      - If normal question: query knowledge base and generate response
      - If uncertain knowledge found: ask user for clarification
      - Return response with source citations

  information flow:
    Source → RawInformation → Knowledge → Inference → Enhanced Knowledge → Answers
```

### Knowledge Representation

*Note: The complete Knowledge and KnowledgeBase implementations with rigorous confidence system are defined in the "Rigorous Knowledge Building Design" section above.*

Key features of the knowledge representation:
- **Knowledge Status System**: VERIFIED, DEDUCED, UNKNOWN, CONFLICTED
- **Evidence Tracking**: Each knowledge item tracks its derivation sources
- **Confidence Scoring**: Automatic confidence assignment based on status
- **User Confirmation**: Knowledge can be upgraded to VERIFIED via user input

### Inference Engine

*Note: The rigorous inference implementation is defined as `RigorousInferenceEngine` in the "Rigorous Knowledge Building Design" section above.*

The inference engine implements both BFS and DFS strategies with rigorous evidence requirements:

```python
class InferenceEngine(RigorousInferenceEngine):
    """Builds new knowledge using rigorous evidence-based inference"""
    
    def expand(self, initial_knowledge: KnowledgeBase, 
               strategy: str = "breadth_first") -> KnowledgeBase:
        """Expand knowledge through rigorous inference with BFS/DFS strategies"""
        
        if strategy == "breadth_first":
            return breadth_first_inference(initial_knowledge)
        elif strategy == "depth_first":
            return depth_first_inference(initial_knowledge, focus_area="general")
        else:
            # Hybrid: BFS first for broad coverage, then DFS for deep insights
            broad_knowledge = breadth_first_inference(initial_knowledge)
            return depth_first_inference(broad_knowledge, focus_area="architecture")
```

Key principles:
- **Only rigorous inferences**: Uses exact matching, no guessing
- **Evidence tracking**: Every inference cites its source evidence  
- **Status management**: Properly categorizes as VERIFIED, DEDUCED, UNKNOWN, or CONFLICTED
- **Strategy support**: Both breadth-first and depth-first knowledge building

## Information Source Implementations

### GitHub Repository Source

```python
class GitHubRepoSource(InformationSource):
    """Extract information from GitHub repositories"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        
    def get_source_type(self) -> str:
        return 'github_repo'
        
    def is_available(self) -> bool:
        return os.path.exists(self.repo_path)
        
    def extract_information(self) -> List[RawInformation]:
        """Extract all information from repository"""
        raw_info = []
        
        # Extract repository structure
        for root, dirs, files in os.walk(self.repo_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    file_info = self._parse_python_file(file_path)
                    raw_info.extend(file_info)
                    
        # Extract documentation
        readme_path = os.path.join(self.repo_path, 'README.md')
        if os.path.exists(readme_path):
            raw_info.extend(self._parse_readme(readme_path))
            
        # Extract dependencies
        requirements_path = os.path.join(self.repo_path, 'requirements.txt')
        if os.path.exists(requirements_path):
            raw_info.extend(self._parse_requirements(requirements_path))
            
        return raw_info
        
    def _parse_python_file(self, file_path: str) -> List[RawInformation]:
        """Parse a Python file into raw information"""
        with open(file_path, 'r') as f:
            content = f.read()
            
        tree = ast.parse(content)
        raw_info = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                raw_info.append(RawInformation(
                    content=f"Class {node.name} defined",
                    info_type='class_definition',
                    source_metadata={
                        'location': f"{file_path}:{node.lineno}",
                        'class_name': node.name,
                        'file_path': file_path
                    }
                ))
                
                # Check for inheritance
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        raw_info.append(RawInformation(
                            content=f"{node.name} inherits from {base.id}",
                            info_type='inheritance_declaration',
                            source_metadata={
                                'location': f"{file_path}:{node.lineno}",
                                'child_class': node.name,
                                'parent_class': base.id
                            }
                        ))
                        
            elif isinstance(node, ast.FunctionDef):
                raw_info.append(RawInformation(
                    content=f"Function {node.name} defined",
                    info_type='function_definition',
                    source_metadata={
                        'location': f"{file_path}:{node.lineno}",
                        'function_name': node.name,
                        'file_path': file_path
                    }
                ))
                
        return raw_info

### User Interaction Source (Detailed Implementation)

class UserResponse:
    def __init__(self, question: str, response: str):
        self.question = question
        self.response = response
        self.timestamp = datetime.now()

class UserQuestion:
    def __init__(self, question: str, context: Knowledge):
        self.question = question
        self.context = context
        self.timestamp = datetime.now()

class UserInteractionSource(InformationSource):
    """User responses as an information source"""
    
    def __init__(self):
        self.pending_questions = Queue()
        self.user_responses = []
        
    def get_source_type(self) -> str:
        return 'user_interaction'
        
    def is_available(self) -> bool:
        return len(self.user_responses) > 0
        
    def extract_information(self) -> List[RawInformation]:
        """Convert user responses into raw information"""
        raw_info = []
        for response in self.user_responses:
            raw_info.append(RawInformation(
                content=response.response,
                info_type='user_statement',
                source_metadata={
                    'question_context': response.question,
                    'timestamp': response.timestamp.isoformat(),
                    'confidence': 1.0,
                    'user_provided': True
                }
            ))
        return raw_info
        
    def add_user_response(self, question: str, response: str):
        """Add user response as new information"""
        self.user_responses.append(UserResponse(question, response))
        
    def has_pending_questions(self) -> bool:
        return not self.pending_questions.empty()
        
    def get_next_question(self) -> UserQuestion:
        return self.pending_questions.get()
        
    def ask_user(self, question: str, context: Knowledge):
        """Queue a question for the user"""
        self.pending_questions.put(UserQuestion(question, context))
```

## Query Engine

```python
class QueryEngine:
    """Answer questions using built knowledge"""
    
    def __init__(self):
        self.llm = self._init_llm()
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        
    def answer(self, question: str, knowledge_base: KnowledgeBase) -> str:
        """Generate answer from knowledge base"""
        
        # Find relevant knowledge
        relevant_facts = self._find_relevant_knowledge(question, knowledge_base.facts)
        relevant_relationships = self._find_relevant_knowledge(question, knowledge_base.relationships)
        relevant_inferences = self._find_relevant_knowledge(question, knowledge_base.inferences)
        relevant_deep_insights = self._find_relevant_knowledge(question, knowledge_base.deep_insights)
        
        # Build context
        context = self._build_context(
            relevant_facts, 
            relevant_relationships,
            relevant_inferences,
            relevant_deep_insights
        )
        
        # Generate response
        response = self.llm.generate(
            prompt=self._build_prompt(question, context),
            max_tokens=500
        )
        
        return response
        
    def _build_prompt(self, question: str, context: str) -> str:
        return f"""You are a knowledge assistant that answers ONLY based on the provided knowledge.

Available Knowledge:
{context}

Question: {question}

Instructions:
1. Answer ONLY using the provided knowledge
2. If the knowledge doesn't contain the answer, say so
3. Cite specific sources when possible
4. Distinguish between facts, relationships, and inferences

Answer:"""
```

## Web Interface

```python
import streamlit as st

def main():
    st.title("🧠 Knowledge Chat Bot")
    st.markdown("I learn from sources you provide and answer questions about them!")
    
    # Initialize bot in session state
    if 'bot' not in st.session_state:
        st.session_state.bot = KnowledgeChatBot()
        st.session_state.messages = []
        
    # Sidebar for source management
    with st.sidebar:
        st.header("📚 Knowledge Sources")
        
        # Add new source
        source_type = st.selectbox(
            "Source Type",
            ["GitHub Repository", "PDF Paper", "Database (Coming Soon)"]
        )
        
        if source_type == "GitHub Repository":
            source_input = st.text_input("Repository URL or Path")
            
            inference_strategy = st.radio(
                "Knowledge Building Strategy",
                ["Breadth-First (Explore all connections)", 
                 "Depth-First (Follow reasoning chains)",
                 "Hybrid (Balanced approach)"]
            )
            
            if st.button("🔄 Build Knowledge"):
                with st.spinner("Building knowledge... This may take a few minutes."):
                    # Show progress
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    status.text("📖 Reading source files...")
                    progress_bar.progress(25)
                    
                    status.text("🧩 Extracting facts and relationships...")
                    progress_bar.progress(50)
                    
                    status.text("💡 Inferring new knowledge...")
                    progress_bar.progress(75)
                    
                    # Add source
                    source_id = st.session_state.bot.add_source(source_input)
                    
                    status.text("✅ Knowledge building complete!")
                    progress_bar.progress(100)
                    
                    st.success(f"Added source: {source_id}")
                    
        # Show loaded sources
        st.subheader("Loaded Sources")
        for source in st.session_state.bot.knowledge_base.list_sources():
            st.write(f"• {source['name']}")
            with st.expander("Knowledge Stats"):
                st.write(f"Facts: {source['fact_count']}")
                st.write(f"Relationships: {source['relationship_count']}")
                st.write(f"Inferences: {source['inference_count']}")
                st.write(f"Deep Insights: {source['deep_insight_count']}")
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    # Chat input
    if prompt := st.chat_input("Ask me anything about the loaded sources..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.bot.chat(prompt)
                st.write(response)
                
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
```

---

# Part III: Development

## Technical Stack

### Core Dependencies
```txt
# Knowledge storage and retrieval
chromadb>=0.4.0              # Vector database
sentence-transformers>=2.2.0  # Embeddings
networkx>=3.0                # Knowledge graph

# LLM integration  
openai>=1.0.0               # Or use local models
langchain>=0.1.0            # LLM orchestration

# Source parsing
gitpython>=3.1.0            # Git operations
ast                         # Python parsing
tree-sitter>=0.20.0         # Multi-language parsing

# Web interface
streamlit>=1.25.0           # Web UI
streamlit-chat>=0.1.0       # Chat components

# Future sources
PyMuPDF>=1.23.0            # PDF parsing
sqlalchemy>=2.0.0          # Database connections
```

## Implementation Timeline

### Week 1-2: Core Knowledge System
- Knowledge representation classes
- Basic inference engine with rules
- Source management system

### Week 3: GitHub Repository Parser  
- Python file parsing with AST
- Documentation extraction
- Dependency analysis

### Week 4: Inference Engine Enhancement
- Breadth-first inference
- Depth-first inference  
- Confidence scoring

### Week 5: Query System & UI
- LLM integration with prompts
- Streamlit chat interface
- Progress tracking for knowledge building

### Week 6: Testing & Optimization
- Knowledge validation
- Inference rule tuning
- Performance optimization

## Usage & Examples

```
User Workflow with Unified Sources:

1. Setup:
   bot = KnowledgeChatBot()
   bot.add_github_repo("/path/to/repo")    // Add static source
   bot.add_pdf("/path/to/paper.pdf")       // Add static source
   // User interaction source created automatically

2. Interactive Learning:
   user: "What is the architecture of this project?"
   bot: "Based on code analysis: [response] 
        ❓ I deduced X. Is this correct?"
   
   user: "Yes, but specifically it uses microservices"
   bot: [processes user response as new information source]
        "Thank you! I've learned from your response."
   
   user: "How does authentication work?"
   bot: "Based on github_repo and user_interaction sources: [enhanced response]"

3. Source Equality Demonstration:
   knowledge_by_source = {
     'github_repo': [facts from code analysis],
     'user_interaction': [facts from user responses],
     'pdf_document': [facts from paper content]
   }
   // All sources processed identically through same pipeline
```

### Information Source Equality

**Key insight**: All sources are just different ways to get information:

```
Information Extraction (all sources equal):
  GitHubRepo.extract()     → RawInformation about code
  UserInteraction.extract() → RawInformation about clarifications  
  PDFDocument.extract()    → RawInformation about research
  Database.extract()       → RawInformation about records
  
Processing Pipeline (identical for all):
  RawInformation → Knowledge → Inference → Enhanced Knowledge
```

### Detailed Knowledge Building Examples

### Example: Understanding FastAPI Repository

#### Initial Facts (Direct Extraction)
```
- Class FastAPI defined in main.py:15
- Function get() defined in routing.py:45
- Function post() defined in routing.py:67
- FastAPI inherits from Starlette
- Import statement: from pydantic import BaseModel
```

#### First-Level Inferences
```
- FastAPI is a web framework (from inheritance + HTTP method functions)
- Project uses Pydantic for data validation (from imports + usage patterns)
- Decorator pattern used for routing (from @app.get decorator analysis)
```

#### Deep Inferences (Multi-Level)
```
- FastAPI emphasizes type safety (Pydantic integration + type hints usage)
- Architecture follows dependency injection pattern (from decorator analysis + parameter inspection)
- Framework designed for API development with automatic documentation (OpenAPI integration detected)
```

### Knowledge Evolution Flow

```
Source Code → AST Parse → Extract Classes/Functions → Identify Patterns
                                    ↓
                            Build Relationships (imports, inheritance)
                                    ↓
                            Infer Purpose (web framework, API, etc.)
                                    ↓
                        Infer Design Philosophy (type safety, modern Python)
                                    ↓
                    Deep Architectural Understanding (DI, auto-docs, etc.)
```

## Key Design Decisions

1. **Unified Information Sources**: User interaction treated as information source, not special system - enables clean, extensible architecture

2. **Rigorous Inference Only**: System never guesses - builds knowledge only from provable evidence, creates UNKNOWN/CONFLICTED states for uncertainty

3. **Source Equality**: All information (files, user responses, databases) processed through identical pipeline - no special cases

4. **Iterative Knowledge Building**: System actively builds new knowledge through BFS/DFS inference strategies, not just storage

5. **Transparent Provenance**: Every piece of knowledge tracked to its source with confidence levels - users see reasoning chain

## Future Extensions

### Additional Source Types
- **PDF Papers**: Academic paper parsing with equation extraction
- **Databases**: Structured data with schema understanding
- **APIs**: Live data integration with endpoint discovery
- **Documentation Sites**: Web scraping with content understanding

### Advanced Inference
- **Cross-Source Synthesis**: Finding connections between different sources
- **Temporal Knowledge**: Understanding how knowledge evolves over time
- **Uncertainty Quantification**: Confidence levels for inferred knowledge
- **Knowledge Validation**: Consistency checking across sources

This design creates a true learning system that builds understanding layer by layer, exactly as envisioned!