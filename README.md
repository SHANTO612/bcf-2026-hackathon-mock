# Practice Problem: Contact Information Parser

Welcome to the practice problem! This exercise will help you get familiar with the hackathon workflow, tooling, and evaluation process before tackling the main challenge.

## Problem Statement

Build a REST API that:
1. Extracts structured contact information from natural language text using an LLM
2. Validates the extracted contact against a database to check if they exist

**Input:** A natural language sentence containing contact details
**Output:** Structured JSON with extracted fields + database validation

### Example

**Request:**
```json
{
  "text": "You can reach John Smith at john.smith@acme.com or call him at 555-123-4567",
  "llm": "gpt-4o-mini"
}
```

**Response:**
```json
{
  "name": "John Smith",
  "email": "john.smith@acme.com",
  "phone": "555-123-4567",
  "found_in_database": true,
  "company": "Acme Corporation"
}
```

If the contact is NOT found in the database:
```json
{
  "name": "Unknown Person",
  "email": "unknown@test.com",
  "phone": "555-000-0000",
  "found_in_database": false,
  "company": null
}
```

### Supported LLMs

Your API may support one or more of these LLM options (passed in the `llm` field):
- `gemini-2.5-flash`
- `gemini-2.5-flash-preview`
- `gpt-4o-mini`

**Recommendation:** Use a Gemini model (preferably `gemini-2.5-flash`) unless you have a specific reason to use GPT.

### LLM API keys (required)

You will need an API key for the provider backing the model you use:

- **Gemini**: Create/manage keys in [Google AI Studio](https://aistudio.google.com/app/apikey)  
  Documentation: [Using Gemini API keys](https://ai.google.dev/gemini-api/docs/api-key)
- **OpenAI (GPT)**: Create/manage keys in the [OpenAI API dashboard](https://platform.openai.com/api-keys)  
  Documentation: [Developer quickstart](https://platform.openai.com/docs/quickstart)

Store keys in environment variables and do not commit them to git.

---

## Checkpoints

Work through these checkpoints to build your solution step by step.

### Checkpoint 1: Environment Setup

You can run the database using **Docker** (recommended) or **locally**.

#### Option A: Docker (recommended)

1. Make sure you have Docker installed
2. Start the provided services:
   ```bash
   docker-compose up -d
   ```
3. Verify PostgreSQL is running:
   ```bash
   docker exec -it practice_db psql -U postgres -d practice_db -c "SELECT 'Hello, Database!';"
   ```
4. Open pgweb to explore the database: http://localhost:8082

#### Option B: Run PostgreSQL locally

1. Install PostgreSQL and make sure the `psql` CLI is available
2. Create the database (example):
   ```bash
   createdb -U postgres practice_db
   ```
3. Load the schema + seed data from `data/init.sql`:
   ```bash
   psql -U postgres -d practice_db -f data/init.sql
   ```
4. Verify it worked:
   ```bash
   psql -U postgres -d practice_db -c "SELECT 'Hello, Database!';"
   ```

You can use any PostgreSQL client (pgAdmin, DBeaver, etc.) to inspect the data.

**Why a database?** While this practice problem doesn't require database queries, the main hackathon will. This checkpoint ensures your database setup works correctly.

### Checkpoint 2: Explore the Sample Database

Connect to the database and explore the schema:

```bash
# Docker:
docker exec -it practice_db psql -U postgres -d practice_db

# Local:
psql -U postgres -d practice_db
```

Run these commands:
```sql
\dt                           -- List all tables
SELECT * FROM contacts;       -- View sample contacts
SELECT * FROM companies;      -- View sample companies
```

Note the contacts in the database - your API will need to check if extracted contacts exist here!

### Checkpoint 3: Create Your API Server

Create a REST API with the following endpoint:

**POST /parse**

Request body:
```json
{
  "text": "Contact Jane Doe at jane.doe@global.com, phone: 555-234-5678",
  "llm": "gpt-4o-mini"
}
```

Response:
```json
{
  "name": "Jane Doe",
  "email": "jane.doe@global.com",
  "phone": "555-234-5678",
  "found_in_database": true,
  "company": "Global Industries"
}
```

**Requirements:**
1. Use the specified LLM to extract `name`, `email`, and `phone` from the text
2. Query the database to check if a contact with that name exists
3. If found, set `found_in_database: true` and include their `company`
4. If not found, set `found_in_database: false` and `company: null`
5. If a field is not present in the text, return `null` for that field

**Health Check Endpoint:**

**GET /health**

Response:
```json
{
  "status": "ok",
  "database": "connected"
}
```

### Checkpoint 4: Get Structured Output from LLM

The key skill here is getting the LLM to return properly structured JSON. Tips:

1. **Use system prompts** to instruct the LLM on the exact output format
2. **Use JSON mode** if the LLM API supports it (e.g., OpenAI's `response_format`)
3. **Validate the response** before returning it to ensure it matches the expected schema

Example system prompt:
```
Extract contact information from the given text.
Return a JSON object with these fields:
- name: The person's full name (string or null)
- email: The email address (string or null)
- phone: The phone number (string or null)

Return ONLY the JSON object, no other text.
```

### Checkpoint 5: Test with the Checker Script

Once your server is running, test it against our test cases:

```bash
# Make sure your server is running on port 8000
python checker/checker.py --url http://localhost:8000
```

The checker will:
1. Send test requests to your `/parse` endpoint
2. Compare your responses against expected outputs
3. Report how many tests passed

**Goal:** Pass all test cases before moving to the main challenge!

---

## API Specification

### POST /parse

Parses contact information from natural language text and validates against database.

**Request:**
| Field | Type   | Required | Description |
|-------|--------|----------|-------------|
| text  | string | Yes      | Natural language text containing contact info |
| llm   | string | Yes      | LLM to use: `gemini-2.5-flash`, `gemini-2.5-flash-preview`, or `gpt-4o-mini` |

**API keys:** If you select a Gemini model, you’ll need a Gemini API key. If you select a GPT model, you’ll need an OpenAI API key.
- Gemini key docs: [Using Gemini API keys](https://ai.google.dev/gemini-api/docs/api-key)
- OpenAI key dashboard: [OpenAI API keys](https://platform.openai.com/api-keys)

**Response:**
| Field             | Type          | Description |
|-------------------|---------------|-------------|
| name              | string\|null  | Extracted full name |
| email             | string\|null  | Extracted email address |
| phone             | string\|null  | Extracted phone number |
| found_in_database | boolean       | Whether the contact was found in database |
| company           | string\|null  | Company name if found, otherwise null |

### GET /health

Health check endpoint.

**Response:**
| Field    | Type   | Description |
|----------|--------|-------------|
| status   | string | Should be "ok" |
| database | string | Should be "connected" |

---

## File Structure

```
mock-problem/
├── README.md              # This file
├── docker-compose.yml     # Docker services configuration
├── data/
│   └── init.sql          # Database initialization script
└── checker/
    ├── checker.py        # Test runner script
    └── test_cases.json   # Test cases
```

---

## Tips for Success

1. **Start simple** - Get a basic endpoint working before adding LLM integration
2. **Test incrementally** - Run the checker after each change
3. **Handle edge cases** - What if the text has no email? No phone?
4. **Check your JSON** - Make sure field names match exactly (`name`, not `Name`)

Good luck!
