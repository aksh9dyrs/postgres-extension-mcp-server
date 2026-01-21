# MCP PostgreSQL Extensions Server

A clean, minimal Model Context Protocol (MCP) server that exposes PostgreSQL database extensions as tools for Claude Desktop.

## 🚀 What This Does

This server connects Claude to your PostgreSQL databases, giving you natural language access to 10 powerful PostgreSQL extensions across 5 separate databases.

## 📁 Project Structure

```
MCP IDE/
└── backend/
    ├── mcp_postgres_server.py    # Main MCP server (54 tools)
    ├── config.py                 # Database configuration
    ├── .env                      # Environment variables
    ├── claude_desktop_config.json # Claude Desktop setup
    ├── requirements.txt          # Python dependencies
    └── README.md                # This file
```

## ⚡ Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your databases in `.env`**

3. **Add to Claude Desktop config**

4. **Run the server:**
   ```bash
   python mcp_postgres_server.py
   ```

## 🎯 Available Extensions

- **PostGIS** - Geographic data (4 tools)
- **pgcrypto** - Encryption/decryption (5 tools)  
- **pg_stat_statements** - Query performance (6 tools)
- **pg_prewarm** - Cache management (3 tools)
- **pg_partman** - Partition management (7 tools)
- **pg_cron** - Job scheduling (6 tools)
- **hypopg** - Hypothetical indexes (6 tools)
- **TimescaleDB** - Time-series data (8 tools)
- **Infrastructure** - System monitoring (5 tools)
- **Apache AGE** - Graph database (4 tools)

**Total: 54 tools across 10 extensions**

## 💬 Example Queries

Ask Claude things like:
- "What are the slowest running queries?"
- "Decrypt John Doe's email address securely"
- "Show me geographic data for nearby hospitals"
- "Schedule a weekly database cleanup job"

## 🔧 Configuration

Your `.env` file should contain:
```bash
DATABASE_URL=postgresql://username:password@localhost:5432/db1
DATABASE_URL2=postgresql://username:password@localhost:5432/db2
DATABASE_URL3=postgresql://username:password@localhost:5432/db3
DATABASE_URL4=postgresql://username:password@localhost:5432/db4
DATABASE_URL5=postgresql://username:password@localhost:5432/db5
SAMBANOVA_API_KEY=your_api_key_here
```

## ✅ Requirements

- Python 3.8+
- PostgreSQL with extensions installed
- SambaNova API key for natural language processing