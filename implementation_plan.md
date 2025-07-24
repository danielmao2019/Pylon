# Knowledge Chat Bot Implementation Plan

## Project Overview

A chat bot that builds deep, layered knowledge from provided sources. Users provide sources (repositories, papers, databases) → the bot builds knowledge through iterative inference → users can then ask anything about those sources.

## Document Structure

**System Design:** Core innovation, information sources, architecture, and knowledge building strategies  
**Implementation:** System components and query processing  
**Development:** UI design, technical stack, timeline, and usage examples

---

# Part I: System Design

## Key Innovation: Continuous Knowledge Building

### The Dynamic Knowledge Insight

**Traditional chat bots**: Knowledge is static after initialization, only retrieval during chat  
**Our approach**: Knowledge continuously grows during conversation through multiple mechanisms:

1. **User Interaction as Information Source**: Confirmations, clarifications treated as new information
2. **On-Demand Source Processing**: User points to new docs → immediate knowledge building  
3. **Recursive Knowledge Building**: Knowledge base itself becomes source for higher-level knowledge
4. **Conversational Learning**: Every chat interaction potentially adds new knowledge

### Continuous Knowledge Building Scenarios

```
Scenario 1: User Confirmation
  user: "What does Parser do?"
  bot: "I deduced Parser handles files. Is this correct?"  
  user: "Yes, specifically XML files"
  → User response becomes new VERIFIED knowledge
  → Triggers inference to build XML-related knowledge
  → Knowledge base grows during conversation

Scenario 2: User Points to New Document  
  user: "Read this PDF and tell me about the methodology"
  → System immediately processes PDF as new information source
  → Builds knowledge from PDF content
  → Runs inference to connect with existing knowledge
  → Knowledge base expands in real-time

Scenario 3: Knowledge-on-Knowledge Building
  → Existing knowledge base contains facts about functions
  → System analyzes knowledge patterns to infer architectural insights  
  → Knowledge base becomes source for meta-knowledge
  → Higher-level understanding emerges from existing knowledge
```

```
// All information sources follow same interface (including knowledge base itself):
sources = [
    GitHubRepo("/path/to/repo"),        // Static: files, structure, dependencies
    UserInteraction(),                  // Interactive: confirmations, clarifications  
    PDFDocument("/path/to/paper"),      // Static: text, references, figures
    Database("connection_string"),      // Dynamic: records, relationships
    KnowledgeBaseSource(existing_kb)   // Recursive: knowledge-on-knowledge building
]

// Same processing pipeline for all sources (including recursive):
for each source:
    raw_info = source.extract_information()     // Source-specific extraction
    knowledge = convert_to_knowledge(raw_info)  // Standardize format
    knowledge_base.add(knowledge)               // Store with source tracking
    inference_engine.infer_new_knowledge()     // Build derived knowledge
    
// Recursive Knowledge Building:
// The knowledge_base itself can be treated as an information source
// to build higher-level meta-knowledge and architectural insights
```

### Recursive Knowledge Building Design

```
KnowledgeBaseSource Implementation Plan:

Purpose: Treat existing knowledge base as information source for meta-analysis

Data Extraction Strategy:
  - Pattern Analysis: Identify recurring patterns in existing knowledge
  - Relationship Mining: Find implicit connections between knowledge items  
  - Architectural Inference: Derive system-level insights from component knowledge
  - Abstraction Building: Create higher-level concepts from detailed facts

Extraction Examples:
  Raw Knowledge: [10 function definitions, 5 class definitions, 8 imports]
  →
  Meta Knowledge: "This module follows object-oriented design patterns"
  
  Raw Knowledge: [error handling in 15 functions, try-catch patterns, logging calls]  
  →
  Meta Knowledge: "System has comprehensive error handling architecture"

Recursive Trigger Conditions:
  - Knowledge base reaches certain size threshold
  - User asks architectural/pattern questions
  - Periodic meta-analysis runs
  - New domain of knowledge added (trigger cross-domain analysis)
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

## Information Sources: The Foundation

All information sources implement the same interface to maintain consistency:

### Unified Source Architecture

```
Information Source Types:
  - GitHubRepo: Code repositories (AST parsing, documentation, dependencies)
  - UserInteraction: Confirmations and corrections (bidirectional, verified)
  - PDFDocument: Papers and documents (text, structure, references)
  - Database: Structured data (records, relationships, metadata)
  - KnowledgeBaseSource: Recursive meta-analysis of existing knowledge

KnowledgeBaseSource Details:
  Purpose: Treat existing knowledge base as information source for meta-analysis
  
  Data Extraction Strategy:
    - Pattern Analysis: Identify recurring patterns in existing knowledge
    - Relationship Mining: Find implicit connections between knowledge items  
    - Architectural Inference: Derive system-level insights from component knowledge
    - Abstraction Building: Create higher-level concepts from detailed facts
  
  Extraction Examples:
    Raw Knowledge: [10 function definitions, 5 class definitions, 8 imports]
    → Meta Knowledge: "This module follows object-oriented design patterns"
    
    Raw Knowledge: [error handling in 15 functions, try-catch patterns, logging calls]  
    → Meta Knowledge: "System has comprehensive error handling architecture"

Common Interface:
  - get_source_type() → Source identifier
  - extract_information() → RawInformation list  
  - is_available() → Availability check
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

Flexible Inference Algorithm:
  current_layer = all_facts_and_relationships
  while current_layer has items and depth < MAX_DEPTH:
    next_layer = []
    
    // Try different inference patterns:
    for each knowledge_item k in current_layer:
      // Unary inferences (single item analysis)
      unary_inferences = try_unary_inference(k)
      next_layer.add_all(unary_inferences)
      
      // Binary inferences (pairs)
      for each other_item k2 in current_layer where k2 != k:
        binary_inference = try_binary_inference(k, k2)
        if binary_inference: next_layer.add(binary_inference)
        
        // Triplet inferences (three items needed)
        for each third_item k3 in current_layer where k3 != k and k3 != k2:
          triplet_inference = try_triplet_inference(k, k2, k3)
          if triplet_inference: next_layer.add(triplet_inference)
          
    current_layer = next_layer
    depth++

Inference Pattern Examples:
  Unary:    "function_definition" → "this_is_a_utility_function" (pattern analysis)
  Binary:   "class_def" + "inheritance" → "inheritance_relationship"  
  Triplet:  "import" + "class_def" + "usage" → "external_dependency_pattern"
  N-ary:    Multiple related functions → "module_functionality_pattern"
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
    
    // Try all inference patterns for current knowledge
    best_next = find_strongest_inference_from(current, focus_area)
    // This could be unary, binary, triplet, etc. - whatever fits best
    
    if best_next and not already_known(best_next):
      knowledge_base.add(best_next)  // Add to knowledge base immediately
      stack.push(best_next)          // Continue this reasoning chain

Key Differences from BFS:
  - BFS: Explores broadly across all domains at each layer
  - DFS: Follows deep reasoning chains in specific domain
  - Both strategies add knowledge to knowledge_base immediately when found
  
Knowledge Base Update Pattern (Same for Both):
  - When new knowledge inferred → immediately add to knowledge_base
  - This enables incremental knowledge building during inference
  - No difference in update timing between BFS and DFS
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

```
Knowledge Confidence Levels:
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

Conflict Detection and Resolution:
  1. Conflict Detection:
     - Check if new knowledge contradicts existing knowledge
     - Compare content semantically (not just exact match)
     - Identify potential conflicts across all confidence levels
     
  2. Conflict Resolution Strategy:
     if no_conflict: Add to appropriate collection by status
     elif conflict_detected: Create CONFLICTED knowledge item with all sources
     elif supports_existing: Strengthen confidence of existing knowledge
     
  3. Conflict Examples:
     - "Parser handles JSON" vs "Parser handles XML" → CONFLICT
     - "Function foo() takes 2 args" vs "Function foo() takes 3 args" → CONFLICT  
     - "Class A inherits B" + "Class A inherits B" → SUPPORT (strengthen)
```

### Rigorous Inference Engine

```
Purpose: Build new knowledge using rigorous evidence-based inference

Strategy Implementation:
  - Breadth-First: Expand knowledge layer by layer across all domains
  - Depth-First: Follow reasoning chains deep in specific focus areas
  - Hybrid: BFS for broad coverage, then DFS for deep insights

Evidence Requirements:
  - Only rigorous inferences using exact matching
  - Every inference must cite source evidence
  - Proper categorization as VERIFIED/DEDUCED/UNKNOWN/CONFLICTED
  - No guessing or fuzzy matching allowed

Flexible Inference Patterns:
  - Unary: Single knowledge item analysis (e.g., "function_definition" → "utility_function")
  - Binary: Pairs of knowledge items (e.g., "class_def" + "inheritance" → "inheritance_relationship")
  - Triplet: Three items needed (e.g., "import" + "class_def" + "usage" → "dependency_pattern")
  - N-ary: Multiple related items → higher-level patterns

Rigorous Inference Rules (only build knowledge that can be PROVEN):
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

Evidence Standards by Inference Type:
  function_usage: Requires exactly 2 pieces of evidence (function_definition + function_call)
  inheritance_relationship: Must have class_definition + inheritance_declaration  
  dependency_usage: Must have import_statement + symbol_usage
  
Handling Uncertainty:
  - If evidence insufficient → create UNKNOWN knowledge
  - If sources contradict → create CONFLICTED knowledge  
  - If no rigorous rule applies → create nothing (don't guess)
```

### Information Processing Flow

**Knowledge Base with Conflict Detection:**

```
KnowledgeBase.add_knowledge(new_knowledge):

  1. Conflict Detection:
     - Check if new knowledge contradicts existing knowledge
     - Compare content semantically (not just exact match)
     - Identify potential conflicts across all confidence levels
     
  2. Conflict Resolution Strategy:
     if no_conflict:
       - Add to appropriate collection by status
       - Update source tracking
     elif conflict_detected:
       - Create CONFLICTED knowledge item
       - Include all conflicting sources as evidence
       - Mark original conflicting items as superseded
       - Add conflict to conflicts collection
     elif supports_existing:
       - Strengthen confidence of existing knowledge
       - Add as supporting evidence
       
  3. Knowledge Collections:
     - verified_knowledge: High confidence facts
     - deduced_knowledge: Logical inferences  
     - unknown_items: Cannot determine from evidence
     - conflicts: Contradictory information
     - source_tracker: Maps source_type → knowledge items

  4. Conflict Detection Examples:
     - "Parser handles JSON" vs "Parser handles XML" → CONFLICT
     - "Function foo() takes 2 args" vs "Function foo() takes 3 args" → CONFLICT  
     - "Class A inherits B" + "Class A inherits B" → SUPPORT (strengthen)
```

*Note: Response structure and query processing details are covered in the Query Processing section.*

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



### Core System Components

```
Chat Bot Processing Flow:

1. Input Analysis:
   - New information provided → Process as information source
   - New document referenced → Add source + build knowledge  
   - Question asked → Determine response strategy + build knowledge
   - Confirmation/correction → Update knowledge + trigger inference

2. Knowledge Building Triggers:
   - User confirmations/corrections → Build verified knowledge
   - New documents mentioned → Extract + infer knowledge
   - Knowledge base growth → Trigger meta-analysis
   - Pattern questions → Analyze existing knowledge patterns

3. Response Strategies:
   - Knowledge Base Mode: Query existing knowledge, mark gaps for clarification
   - RAG Mode: Direct source retrieval with potential knowledge building
   - Hybrid Mode: Combine knowledge base + RAG, discover new connections
```



## Query Processing

**Response Generation Strategy:**

```
Query Processing Pipeline:
  1. Question Analysis → Determine query type and response strategy
  2. Knowledge Retrieval → Semantic search + keyword matching from knowledge base
  3. Context Building → Rank by relevance and confidence level
  4. Response Generation → LLM prompt with source constraints and uncertainty indicators

Response Format with Confidence Levels:
  ✓ VERIFIED FACTS: Direct evidence from sources
  → LOGICAL DEDUCTIONS: Rigorous inferences with evidence
  ? UNCLEAR AREAS: Insufficient evidence for determination
  ⚠ CONFLICTING INFORMATION: Contradictory sources requiring clarification

Key Constraints:
  - Only use information from provided sources (no external knowledge)
  - Always cite sources and show confidence levels
  - Generate clarification questions for uncertain knowledge
```

---

# Part III: Development

## Web Interface Design

**UI Layout Strategy:**

```
Two-Column Web Interface:
  Left Column (2/3 width): Main Chat Interface
    - Chat history with message bubbles
    - Knowledge metadata expandable for each bot response
    - Text input for user questions
    
  Right Column (1/3 width): Knowledge Confirmation Panel
    - Pending confirmation questions with context
    - Correct/Wrong buttons for quick feedback
    - Comment boxes for detailed corrections
    - Learning progress statistics and charts
    
  Sidebar: Source Management
    - Add new sources (GitHub, PDF, Database)
    - Select knowledge building strategy (BFS/DFS/Hybrid)
    - View loaded sources with statistics
```

## Active Knowledge Confirmation System

**Curiosity Engine Design:**

```
Purpose: Generate confirmation questions during knowledge building and chat

Components:
  - Uncertainty Detector: Identifies knowledge needing confirmation
  - Question Generator: Creates natural language questions
  - Context Tracker: Links questions to knowledge items
  - Confirmation Processor: Updates knowledge from user feedback

Curiosity Triggers (when to ask questions):
  1. Low Confidence Deductions (confidence < 0.8)
  2. Conflicting Information Detected
  3. Pattern Recognition Uncertainty  
  4. Cross-Source Inconsistencies
  5. Missing Critical Information

Active Confirmation Process:
  1. During knowledge building/inference:
     - Check each new knowledge item for uncertainty
     - Generate confirmation question if needed
     - Add to pending confirmations queue
     
  2. During response generation:
     - Identify knowledge used in response
     - Mark uncertain knowledge for potential confirmation
     - Include confirmation requests in response
     
  3. User feedback processing:
     - Positive confirmation → upgrade confidence to VERIFIED
     - Negative confirmation → create CONFLICTED knowledge
     - User corrections → add new VERIFIED knowledge from user input

Question Types:
  - Deductions: "I deduced X from Y. Is this correct?"
  - Conflicts: "Found conflicting info: A vs B. Which is correct?"
  - Patterns: "I noticed pattern X. Does this make sense?"
  - Gaps: "Couldn't determine X from sources. Can you clarify?"
```

**Verbose Knowledge Display Design:**

```
Purpose: Show users exactly which knowledge was used in responses

Knowledge Tracking Strategy:
  - Record which knowledge items retrieved for each response
  - Track confidence levels and original sources
  - Maintain evidence chains showing how knowledge was derived

Display Format:
  - Group by confidence (Verified → Deduced → Uncertain)
  - Color-code by reliability (Green/Yellow/Red)
  - Show source attribution and evidence chains
  - Expandable details for full context

User Correction Flow:
  - User clicks "This understanding is wrong"
  - Correction dialog opens with comment box
  - User input processed as new verified knowledge
  - Knowledge base updated and inferences re-run
```

**Enhanced Chat Architecture:**

```
Active Confirmation Integration:

chat_with_curiosity() method:
  - Process user input for new information
  - Generate response using current knowledge
  - Simultaneously generate confirmation questions
  - Return response + knowledge metadata + confirmations

process_confirmation() method:
  - Locate original knowledge item
  - Update confidence based on user feedback
  - Process corrections as new verified knowledge
  - Trigger inference updates on corrected knowledge

Benefits:
  - Continuous learning during conversation
  - Transparent reasoning builds user trust
  - Active curiosity prevents knowledge gaps
  - User corrections improve future responses
```

---

## Technical Stack

### Core Dependencies

```
Knowledge & Retrieval:
  - ChromaDB (dev), Qdrant (production) - Vector databases
  - all-MiniLM-L6-v2 - Embedding model for semantic similarity
  - NetworkX - Knowledge graph representation

LLM Integration:
  - GPT-4 or local models - Text generation
  - LangChain - LLM orchestration and prompt management

Source Parsing:
  - GitPython - Git repository operations
  - AST (built-in) - Python code parsing
  - tree-sitter - Multi-language parsing support

Web Interface:
  - Streamlit - Web UI framework
  - streamlit-chat - Chat components

Future Extensions:
  - PyMuPDF - PDF parsing and text extraction
  - SQLAlchemy - Database connections and ORM
```

## Implementation Timeline

```
Phase 1 (Weeks 1-2): Core Knowledge System
  - Knowledge representation with confidence levels
  - Basic inference engine and source management

Phase 2 (Weeks 3-4): Information Processing  
  - GitHub repository parser with AST
  - Inference engine enhancement (BFS/DFS strategies)
  - Confidence scoring and validation

Phase 3 (Weeks 5-6): User Interface & Integration
  - Streamlit chat interface with confirmation panel
  - LLM integration and curiosity engine
  - Testing, optimization, and deployment
```

## Usage Example: Learning from Code Repository

**Multi-Layer Knowledge Building Process:**

```
1. System Setup:
   - Add GitHub repository as information source  
   - User interaction source created automatically
   - Initial facts extracted via AST parsing

2. Knowledge Building Layers:
   Layer 0: Direct Facts (class definitions, functions, imports, documentation)
   Layer 1: Basic Relationships (inheritance, dependencies, cross-references)  
   Layer 2: Architectural Insights (design patterns, framework identification)
   Layer 3: Meta-Knowledge (system architecture, design philosophy)

3. Interactive Learning Flow:
   user: "What is the architecture of this project?"
   bot: "Based on code analysis: [response with confidence levels]
        ❓ I deduced X uses microservices. Is this correct?"
   
   user: "Yes, specifically REST microservices with Docker"
   bot: [Processes correction as verified knowledge → triggers inference]
   
   user: "How does authentication work?"  
   bot: "Based on github_repo and user_interaction: [enhanced response]"
```

### Detailed Example: Understanding FastAPI Repository

**Initial Facts (Direct Extraction):**
```
- Class FastAPI defined in main.py:15
- Function get() defined in routing.py:45
- Function post() defined in routing.py:67
- FastAPI inherits from Starlette
- Import statement: from pydantic import BaseModel
```

**First-Level Inferences:**
```
- FastAPI is a web framework (from inheritance + HTTP method functions)
- Project uses Pydantic for data validation (from imports + usage patterns)
- Decorator pattern used for routing (from @app.get decorator analysis)
```

**Deep Inferences (Multi-Level):**
```
- FastAPI emphasizes type safety (Pydantic integration + type hints usage)
- Architecture follows dependency injection pattern (from decorator analysis + parameter inspection)
- Framework designed for API development with automatic documentation (OpenAPI integration detected)
```

## Core Design Principles

1. **Unified Information Sources**: All sources (files, databases, user input) use same interface
2. **Rigorous Evidence-Based Inference**: No guessing - only provable inferences with clear confidence levels  
3. **Continuous Knowledge Building**: Knowledge grows during conversation, not just initialization
4. **Transparent Reasoning**: Full provenance tracking from sources to conclusions
5. **Active Curiosity**: System asks clarifying questions to resolve uncertainty

This creates a learning system that builds layered understanding through iterative inference and user interaction.