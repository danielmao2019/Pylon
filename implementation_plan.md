# Knowledge Chat Bot Implementation Plan

## Project Overview

A chat bot that builds deep, layered knowledge from provided sources. Users provide sources (repositories, papers, databases) → the bot builds knowledge through iterative inference → users can then ask anything about those sources.

## Document Structure

1. **Core Concept** - Iterative knowledge building with BFS/DFS strategies
2. **Rigorous Knowledge Building** - Evidence-based system treating all sources equally
3. **System Architecture** - Unified information source model with user interaction as a source type
4. **Implementation Details** - Information source implementations, query engine, web interface  
5. **Technical Stack & Timeline** - Dependencies and 6-week development plan
6. **Examples & Design Decisions** - Practical usage showing source equality

## Core Concept: Iterative Knowledge Building

### The Knowledge Building Process

```
User provides source → Extract raw information → Build initial knowledge
                                                           ↓
                                                  Infer new knowledge
                                                           ↓
                                                  Infer deeper knowledge
                                                           ↓
                                                  ... (iterative process)
                                                           ↓
                                                  Ready for questions
```

### Knowledge Inference Strategies (BFS/DFS Analogies)

#### 1. **Breadth-First Knowledge Building**
*Explore all immediate inferences at each layer before going deeper*

```python
def breadth_first_inference(knowledge_base: KnowledgeBase) -> KnowledgeBase:
    """
    BFS Strategy: Build wide knowledge first, then deep insights
    
    Layer 1: All direct facts and relationships
    Layer 2: All immediate inferences from Layer 1  
    Layer 3: All inferences from Layer 2
    ...continue until no new knowledge can be inferred
    """
    current_layer = knowledge_base.get_facts() + knowledge_base.get_relationships()
    inference_depth = 0
    
    while current_layer and inference_depth < MAX_DEPTH:
        next_layer = []
        
        # Try to infer from ALL pairs in current layer
        for knowledge1 in current_layer:
            for knowledge2 in current_layer:
                if knowledge1 != knowledge2:
                    inference = try_rigorous_inference(knowledge1, knowledge2)
                    if inference and not knowledge_base.contains(inference):
                        next_layer.append(inference)
                        knowledge_base.add_inference(inference, layer=inference_depth+1)
        
        # Move to next layer
        current_layer = next_layer
        inference_depth += 1
        
    return knowledge_base
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

```python
def depth_first_inference(knowledge_base: KnowledgeBase, focus_area: str) -> KnowledgeBase:
    """
    DFS Strategy: Pick a domain/concept and infer as deeply as possible
    
    Focus areas could be: 'architecture', 'data_flow', 'error_handling', etc.
    """
    
    # Start with facts related to focus area
    seed_knowledge = knowledge_base.get_facts_by_domain(focus_area)
    inference_stack = [(k, 0) for k in seed_knowledge]  # (knowledge, depth)
    
    while inference_stack:
        current_knowledge, depth = inference_stack.pop()
        
        if depth >= MAX_DEPTH:
            continue
            
        # Find the BEST next inference for this reasoning chain
        best_inference = find_strongest_inference_from(
            current_knowledge, 
            focus_area, 
            knowledge_base
        )
        
        if best_inference and not knowledge_base.contains(best_inference):
            knowledge_base.add_inference(best_inference, chain_depth=depth+1)
            # Continue this specific reasoning chain
            inference_stack.append((best_inference, depth+1))
            
    return knowledge_base
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
```python
def choose_strategy(source_type: str, user_intent: str) -> str:
    """Choose BFS vs DFS based on source and intended use"""
    
    if user_intent in ['overview', 'general_understanding', 'exploration']:
        return 'breadth_first'
    elif user_intent in ['architecture_analysis', 'security_review', 'performance_analysis']:
        return 'depth_first'
    elif source_type == 'large_codebase':
        return 'hybrid'  # BFS first, then DFS on key areas
    else:
        return 'breadth_first'  # Default to comprehensive
```

## Rigorous Knowledge Building Design

### Core Rigorous System

#### Knowledge Confidence System

```python
from enum import Enum
from typing import List, Optional, Union

class KnowledgeStatus(Enum):
    VERIFIED = "verified"      # 100% certain from direct evidence
    DEDUCED = "deduced"        # Logically derived with no assumptions  
    UNKNOWN = "unknown"        # Cannot be determined from evidence
    CONFLICTED = "conflicted"  # Multiple contradictory evidence sources

class Knowledge:
    def __init__(self, 
                 content: str, 
                 knowledge_type: str, 
                 source: str,
                 status: KnowledgeStatus,
                 evidence: List['Knowledge'] = None,
                 confidence_score: float = None):
        self.content = content
        self.type = knowledge_type
        self.source = source  
        self.status = status
        self.evidence = evidence or []
        
        # Confidence score based on status
        if status == KnowledgeStatus.VERIFIED:
            self.confidence = 1.0
        elif status == KnowledgeStatus.DEDUCED:
            self.confidence = 0.95
        elif status == KnowledgeStatus.UNKNOWN:
            self.confidence = 0.0
        elif status == KnowledgeStatus.CONFLICTED:
            self.confidence = 0.0
```

#### Rigorous Inference Rules

```python
class RigorousInferenceEngine:
    """Only builds knowledge that can be PROVEN from evidence"""
    
    def _try_infer(self, k1: Knowledge, k2: Knowledge) -> Optional[Knowledge]:
        """Attempt inference - return None if not rigorous enough"""
        
        # Rule 1: Function Usage Verification
        if (k1.type == 'function_definition' and k2.type == 'function_call'):
            if self._exact_function_match(k1, k2):
                return Knowledge(
                    content=f"Function '{k1.function_name}' is called at {k2.location}",
                    knowledge_type='verified_usage',
                    source='verified_inference',
                    status=KnowledgeStatus.VERIFIED,
                    evidence=[k1, k2]
                )
                
        # Rule 2: Inheritance Verification  
        if (k1.type == 'class_definition' and k2.type == 'inheritance_declaration'):
            if self._exact_inheritance_match(k1, k2):
                return Knowledge(
                    content=f"Class '{k1.class_name}' is a base class for '{k2.child_class}'",
                    knowledge_type='inheritance_relationship',
                    source='verified_inference', 
                    status=KnowledgeStatus.VERIFIED,
                    evidence=[k1, k2]
                )
                
        # Rule 3: Import-Usage Connection
        if (k1.type == 'import_statement' and k2.type == 'symbol_usage'):
            if self._exact_import_usage_match(k1, k2):
                return Knowledge(
                    content=f"Module '{k1.module_name}' is actively used via '{k2.symbol_name}'",
                    knowledge_type='dependency_usage',
                    source='verified_inference',
                    status=KnowledgeStatus.VERIFIED,
                    evidence=[k1, k2]
                )
                
        # If no rigorous rule applies, don't guess
        return None
        
    def _exact_function_match(self, definition: Knowledge, call: Knowledge) -> bool:
        """Verify exact function name match - no assumptions"""
        def_name = self._extract_function_name(definition.content)
        call_name = self._extract_function_name(call.content)
        return def_name == call_name and def_name is not None
        
    def _handle_insufficient_evidence(self, k1: Knowledge, k2: Knowledge) -> Optional[Knowledge]:
        """Create UNKNOWN knowledge when evidence is insufficient"""
        
        # Example: Similar function names but not exact match
        if (k1.type == 'function_definition' and k2.type == 'function_call'):
            def_name = self._extract_function_name(k1.content)
            call_name = self._extract_function_name(k2.content)
            
            if def_name and call_name and self._similar_but_not_exact(def_name, call_name):
                return Knowledge(
                    content=f"UNKNOWN: Relationship between function '{def_name}' and call '{call_name}' - similar names but cannot verify exact match",
                    knowledge_type='unknown_relationship',
                    source='insufficient_evidence',
                    status=KnowledgeStatus.UNKNOWN,
                    evidence=[k1, k2]
                )
                
        return None
        
    def _handle_conflicting_evidence(self, evidences: List[Knowledge]) -> Knowledge:
        """Create CONFLICTED knowledge when sources contradict"""
        
        conflicting_contents = [e.content for e in evidences]
        return Knowledge(
            content=f"CONFLICTED: Multiple contradictory statements found: {conflicting_contents}",
            knowledge_type='conflicted_information',
            source='conflicting_evidence',
            status=KnowledgeStatus.CONFLICTED,
            evidence=evidences
        )
```

#### Rigorous Evidence Standards

```python
class EvidenceStandards:
    """Defines what counts as 'rigorous enough' evidence"""
    
    @staticmethod
    def is_rigorous_enough(inference_type: str, evidence: List[Knowledge]) -> bool:
        """Determine if evidence meets rigor standards for inference type"""
        
        standards = {
            'function_usage': EvidenceStandards._verify_function_usage,
            'inheritance_relationship': EvidenceStandards._verify_inheritance,
            'dependency_usage': EvidenceStandards._verify_dependency,
            'architectural_pattern': EvidenceStandards._verify_pattern,
        }
        
        if inference_type not in standards:
            return False  # Unknown inference types are not rigorous
            
        return standards[inference_type](evidence)
        
    @staticmethod 
    def _verify_function_usage(evidence: List[Knowledge]) -> bool:
        """Function usage requires exact name match and valid locations"""
        if len(evidence) != 2:
            return False
            
        definition, call = evidence
        return (
            definition.type == 'function_definition' and
            call.type == 'function_call' and
            definition.function_name == call.function_name and
            definition.function_name is not None and
            call.location is not None
        )
        
    @staticmethod
    def _verify_inheritance(evidence: List[Knowledge]) -> bool:
        """Inheritance requires explicit 'class Child(Parent):' syntax"""
        if len(evidence) != 2:
            return False
            
        parent, inheritance = evidence
        return (
            parent.type == 'class_definition' and
            inheritance.type == 'inheritance_declaration' and
            parent.class_name == inheritance.parent_class and
            parent.class_name is not None
        )
        
    @staticmethod
    def _verify_dependency(evidence: List[Knowledge]) -> bool:
        """Dependency usage requires import + actual symbol usage"""
        if len(evidence) != 2:
            return False
            
        import_stmt, usage = evidence
        return (
            import_stmt.type == 'import_statement' and
            usage.type == 'symbol_usage' and
            import_stmt.provides_symbol(usage.symbol_name)
        )
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

```python
class KnowledgeChatBot:
    """Main chat bot that builds knowledge from multiple information sources"""
    
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.inference_engine = InferenceEngine()
        self.information_sources = []  # List of all information sources
        self.query_engine = QueryEngine()
        
    def add_information_source(self, source: InformationSource):
        """Add any type of information source"""
        self.information_sources.append(source)
        self._build_knowledge_from_source(source)
        
    def add_github_repo(self, repo_path: str):
        """Convenience method for adding GitHub repo"""
        source = GitHubRepoSource(repo_path)
        self.add_information_source(source)
        
    def add_pdf(self, pdf_path: str):
        """Convenience method for adding PDF"""
        source = PDFSource(pdf_path)
        self.add_information_source(source)
        
    def _get_user_source(self) -> UserInteractionSource:
        """Get or create user interaction source"""
        for source in self.information_sources:
            if isinstance(source, UserInteractionSource):
                return source
        # Create user source if doesn't exist
        user_source = UserInteractionSource()
        self.information_sources.append(user_source)
        return user_source
        
    def _build_knowledge_from_source(self, source: InformationSource):
        """Build knowledge from any information source"""
        print(f"📚 Processing {source.get_source_type()} source...")
        
        # Extract raw information
        raw_info = source.extract_information()
        
        # Convert to initial knowledge
        print("🧠 Building initial knowledge...")
        initial_knowledge = self._raw_to_knowledge(raw_info, source.get_source_type())
        
        # Add to knowledge base
        for knowledge in initial_knowledge:
            self.knowledge_base.add_knowledge(knowledge)
            
        # Run inference to build new knowledge
        print("🔄 Inferring new knowledge...")
        self.inference_engine.expand_knowledge_base(self.knowledge_base)
        
        print("✅ Knowledge building complete!")
        
    def _raw_to_knowledge(self, raw_info: List[RawInformation], source_type: str) -> List[Knowledge]:
        """Convert raw information to Knowledge objects"""
        knowledge_list = []
        for info in raw_info:
            knowledge = Knowledge(
                content=info.content,
                knowledge_type=self._determine_knowledge_type(info),
                source=f"{source_type}:{info.source_metadata.get('location', 'unknown')}",
                status=KnowledgeStatus.VERIFIED,  # Raw information is always verified
                evidence=[]
            )
            knowledge_list.append(knowledge)
        return knowledge_list
        
    def chat(self, user_input: str) -> str:
        """Handle user interaction - questions and responses to bot questions"""
        
        user_source = self._get_user_source()
        
        # Check if we have pending questions for the user
        if user_source.has_pending_questions():
            # User is responding to our question
            question = user_source.get_next_question()
            user_source.add_user_response(question.question, user_input)
            
            # Process this new user information
            self._build_knowledge_from_source(user_source)
            
            return "Thank you! I've learned from your response. What else would you like to know?"
            
        # Normal question processing
        if not self.information_sources:
            return "Please add an information source first. I need something to learn from!"
            
        result = self.query_engine.answer(user_input, self.knowledge_base)
        
        # Check if we need to ask user for clarification
        if result.has_uncertain_knowledge():
            uncertain_item = result.get_most_relevant_uncertain()
            question = self._generate_clarification_question(uncertain_item)
            user_source.ask_user(question, uncertain_item)
            
            return f"{result.generate_response()}\n\n❓ {question}"
            
        return result.generate_response()
        
    def _generate_clarification_question(self, knowledge: Knowledge) -> str:
        """Generate question to clarify uncertain knowledge"""
        if knowledge.status == KnowledgeStatus.DEDUCED:
            return f"I deduced that {knowledge.content}. Is this correct?"
        elif knowledge.status == KnowledgeStatus.UNKNOWN:
            return f"I'm unsure about: {knowledge.content}. Can you clarify?"
        elif knowledge.status == KnowledgeStatus.CONFLICTED:
            conflicts = [e.content for e in knowledge.evidence]
            return f"I found conflicting information:\n" + \
                   "\n".join(f"- {c}" for c in conflicts) + \
                   "\nWhich is correct?"
        return f"Can you help me understand: {knowledge.content}?"
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

## Usage Example

```python
# User workflow with unified information sources
bot = KnowledgeChatBot()

# 1. Add multiple information sources
bot.add_github_repo("/path/to/repo")  # Static source
bot.add_pdf("/path/to/paper.pdf")     # Static source  
# User interactions will be added automatically as an information source

# 2. User asks questions - bot learns from both static sources AND user responses
response = bot.chat("What is the architecture of this project?")
# Bot may ask for clarification: "I deduced X. Is this correct?"

response = bot.chat("Yes, but specifically it uses microservices")
# Bot processes user response as new information and rebuilds knowledge

response = bot.chat("How does the authentication work?")
# Now bot has knowledge from: repo files + user clarification + inference
# Returns: "Based on github_repo and user_interaction sources: ..."

# 3. All information sources are treated equally in knowledge building
print(bot.knowledge_base.get_knowledge_by_source('github_repo'))      # Code analysis
print(bot.knowledge_base.get_knowledge_by_source('user_interaction'))  # User responses
print(bot.knowledge_base.get_knowledge_by_source('pdf_document'))      # Paper content
```

### Information Source Equality

The key insight is that **all information sources are equal**:

```python
# These are all just different ways to get RawInformation:
github_info = github_source.extract_information()  # From files
user_info = user_source.extract_information()      # From user responses  
pdf_info = pdf_source.extract_information()        # From documents
db_info = database_source.extract_information()    # From records

# All get processed the same way:
for info_list in [github_info, user_info, pdf_info, db_info]:
    knowledge = bot._raw_to_knowledge(info_list, source_type)
    bot.knowledge_base.add_knowledge(knowledge)
    bot.inference_engine.expand_knowledge_base(bot.knowledge_base)
```

## Knowledge Building Examples

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

1. **Iterative Knowledge Building**: The system doesn't just store information - it actively builds new knowledge through inference

2. **Source Constraint**: The bot only knows what you teach it - no external knowledge leaks

3. **Transparent Reasoning**: Users can see how knowledge was derived (fact → inference → deep insight)

4. **Flexible Inference**: Support both breadth-first (explore all options) and depth-first (follow reasoning chains) strategies

5. **Progressive Learning**: Each source adds to existing knowledge, enabling cross-source insights

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