---
```markdown
# System Design Document

## The Agent Loop Architecture
The core loop is implemented in `loop.py` as a state-machine `while` loop. It enforces a strict 8-step hard cap.

**Step-by-Step Flow:**
1. **Pre-flight Refusal Checks:** Before the LLM is even called, the user query is checked against Regex filters for non-cricket topics, coding requests, and investment advice. If triggered, the agent exits at Step 0, ensuring 0ms LLM latency and zero token cost.
2. **System Prompt Construction:** Temporal bounds and context are injected dynamically into the system prompt based on the question.
3. **Planning Step (Bonus A):** The LLM outputs a `Plan:` explaining its reasoning before making tool calls.
4. **Execution & Evaluation:** Tools are executed safely within `try/except` blocks. Errors are fed back into the message history so the LLM can self-correct (e.g., fixing bad SQL syntax).
5. **Termination:** The loop breaks when the LLM returns standard text without a `tool_calls` array, or if the 8-step hard cap is reached.

## Tool Schemas
1. **`query_data`**: 
   * **Input:** Natural language query.
   * **Output:** Formatted markdown table from SQLite.
   * **Design Note:** The tool translates natural language to SQL using a highly specific "cheat sheet" injected into its system prompt to ensure accurate aggregations (e.g., distinguishing between highest match score vs highest score off a single ball).
2. **`search_docs`**: 
   * **Input:** Specific search string (must include exact entity names).
   * **Output:** Top-3 vector chunks with source filenames.
3. **`web_search`**: 
   * **Input:** Short search phrase (<10 words).
   * **Output:** Clean snippets with URLs.

## Preventing Infinite Loops
Instead of just relying on the 8-step hard cap, I implemented an **Anti-Repetition Guardrail** at the code level. 
Inside the loop, the agent tracks `previous_tool_signatures` (a hash of `tool_name + arguments`). If the LLM attempts to execute the exact same tool with the exact same arguments twice in a row (a common panic-behavior in small LLMs), the Python loop intercepts it. It immediately injects a system message saying *"You already executed this exact tool... You MUST change your query."* This breaks the infinite decoding loop and forces the agent to try a different strategy.