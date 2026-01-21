"""
PostgreSQL Extensions MCP Server

A self-contained MCP server that exposes all PostgreSQL extensions as tools for Claude:
- PostGIS (geospatial operations)
- pgcrypto (encryption/decryption)  
- pg_stat_statements (query performance)
- pg_prewarm (cache warming)
- pg_partman (partition management)
- pg_cron (cron job scheduling)
- hypopg (hypothetical indexes)
- TimescaleDB (time-series data)
- Infrastructure monitoring
- Apache AGE (graph database)
"""

import asyncio
import os
import json
import re
import sys
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Debug configuration
LOCAL_MODE = os.getenv('LOCAL_MODE', 'false').lower() == 'true'
SAMBANOVA_API_KEY = os.getenv('SAMBANOVA_API_KEY')

print(f"DEBUG: LOCAL_MODE: {LOCAL_MODE}")
print(f"DEBUG: SAMBANOVA_API_KEY: {SAMBANOVA_API_KEY}")

# Database configurations from config.py
DATABASE_URL = os.getenv('DATABASE_URL')
DATABASE_URL2 = os.getenv('DATABASE_URL2')
DATABASE_URL3 = os.getenv('DATABASE_URL3')
DATABASE_URL4 = os.getenv('DATABASE_URL4')
DATABASE_URL5 = os.getenv('DATABASE_URL5')

print(f"DEBUG: DATABASE_URL: {DATABASE_URL}")

# Database connection functions
@contextmanager
def get_db_connection(db_url):
    """Get database connection with context manager."""
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def execute_query(db_url: str, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
    """Execute a database query and return results."""
    try:
        with get_db_connection(db_url) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                
                # Check if it's a SELECT query
                if query.strip().upper().startswith('SELECT'):
                    results = cur.fetchall()
                    return {'success': True, 'data': results}
                else:
                    conn.commit()
                    return {'success': True, 'message': 'Query executed successfully'}
                    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def call_llm_api(prompt: str, extension: str) -> Dict[str, Any]:
    """Call SambaNova API for natural language processing."""
    if LOCAL_MODE or not SAMBANOVA_API_KEY:
        return {
            'sql': f"SELECT 'Local mode - no LLM call' as message",
            'extension': extension,
            'explanation': 'Running in local mode, no API call made'
        }
    
    # Extension-specific schemas
    schemas = {
        'pgcrypto': """
Schema:
appointments(appointment_id, patient_id, doctor_id, appointment_date, reason)
doctors(doctor_id, name, specialty)
lab_results(result_id, patient_id, test_name, result_value, test_date)
patients(patient_id, full_name, email, phone, dob, gender)
Sensitive fields (email, phone) are encrypted with pgcrypto.
Database: postgres (port 5432)
""",
        'pg_stat_statements': """
Schema:
pg_stat_statements extension tracking all SQL queries
patients, appointments, doctors, lab_results, customers, stores, delivery_zones
Database: postgres (port 5432)
""",
        'pg_prewarm': """
Schema:
Can preload any table: patients, appointments, doctors, lab_results, customers, stores, delivery_zones
Database: postgres (port 5432)
""",
        'postgis': """
Schema:
customers(customer_id, name, location) - Customer locations with geographic data
delivery_zones(zone_id, name, boundary) - Delivery zone boundaries with geographic data
stores(store_id, name, location) - Store locations with geographic data
spatial_ref_sys, topology schema
Geometry functions: ST_Distance, ST_Within, ST_Contains, ST_Intersects, etc.
Database: postgres (port 5432)
""",
        'pg_partman': """
Schema:
transactions - partitioned table with date-based partitions
transactions_p20250923-transactions_p20250928, transactions_p2025_09_20-transactions_p2025_09_22 - partitioned subsets
Database: postgres (port 5433)
""",
        'pg_cron': """
Schema:
traffic_data(id, location, vehicle_count, timestamp)
traffic_summary(id, location, avg_vehicles, peak_vehicles, total_readings)
All schemas
Database: traffic (port 5434)
""",
        'timescaledb': """
Schema:
energy_usage(device_id, usage_kwh, timestamp) - hypertable for time-series energy data
hourly_power_usage - aggregated power usage by hour
Database: postgres (port 5111)
""",
        'hypopg': """
Schema:
products(product_id, name, price, ...) - product catalog
hypopg_hidden_indexes, hypopg_list_indexes - system tables for hypothetical indexes
Database: postgres (port 5445)
""",
        'infrastructure': """
Schema:
System monitoring tables for database health checks
patients, appointments, doctors, lab_results, customers, stores, delivery_zones tables
Database: postgres (port 5432)
""",
        'apache_age': """
Schema:
Transaction graph data with Cypher query support
Database: postgres (port 5433)
""",
    }
    
    schema = schemas.get(extension, "Various database tables")
    
    system_prompt = f"""You are an expert assistant for a database using PostgreSQL and these extensions: {extension}.
{schema}
You answer natural language questions by:
- Identifying the relevant extension ({extension})
- Generating SQL for the task
- Providing a short explanation of the result or optimization.
Return JSON: {{"sql": "...", "extension": "{extension}", "explanation": "...""}}.
If the question is not about these extensions, reply: {{"error": "Unknown extension or unsupported query."}}"""

    payload = {
        "model": "Llama-4-Maverick-17B-128E-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }
    
    headers = {
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            "https://api.sambanova.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            return json.loads(content)
        else:
            return {
                'sql': f"SELECT 'API Error: {response.status_code}' as error",
                'extension': extension,
                'explanation': f'API call failed with status {response.status_code}'
            }
            
    except Exception as e:
        return {
            'sql': f"SELECT 'Error: {str(e)}' as error",
            'extension': extension,
            'explanation': f'Error calling LLM API: {str(e)}'
        }

# Initialize FastMCP server
mcp = FastMCP("PostgreSQL Extensions Server")

# PostGIS Tools (DATABASE_URL - port 5432, customers/deliveryzones/stores/spatial_ref_sys/topology)
@mcp.tool()
def get_customers_outside_zones() -> Dict[str, Any]:
    """Find customers located outside of delivery zones using PostGIS."""
    query = """
    SELECT c.customer_id, c.name, ST_AsText(c.location) as customer_location
    FROM customers c
    WHERE NOT EXISTS (
        SELECT 1 FROM deliveryzones dz 
        WHERE ST_Within(c.location, dz.boundary)
    )
    """
    result = execute_query(DATABASE_URL, query)
    result['extension'] = 'postgis'
    return result

@mcp.tool()
def get_nearest_store_to_customer(customer_name: str) -> Dict[str, Any]:
    """Find the nearest store to a specific customer using PostGIS distance calculations."""
    query = """
    SELECT s.store_id, s.name, 
           ST_Distance(s.location, c.location) as distance_meters,
           ST_AsText(s.location) as store_location
    FROM stores s, customers c
    WHERE c.name = %s
    ORDER BY ST_Distance(s.location, c.location)
    LIMIT 1
    """
    result = execute_query(DATABASE_URL, query, (customer_name,))
    result['extension'] = 'postgis'
    return result

@mcp.tool()
def get_stores_within_km_of_city(city_name: str, radius_km: float) -> Dict[str, Any]:
    """Find all stores within a specified radius of a city using PostGIS."""
    query = """
    SELECT s.store_id, s.name,
           ST_Distance(s.location, ST_GeomFromText('POINT(-74.0060 40.7128)', 4326)) as distance_meters
    FROM stores s
    WHERE ST_DWithin(s.location, ST_GeomFromText('POINT(-74.0060 40.7128)', 4326), %s * 1000)
    ORDER BY distance_meters
    """
    result = execute_query(DATABASE_URL, query, (radius_km,))
    result['extension'] = 'postgis'
    return result

@mcp.tool()
def postgis_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about geographic data using PostGIS."""
    llm_result = call_llm_api(question, 'postgis')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'postgis'}
    
    result = execute_query(DATABASE_URL, llm_result['sql'])
    result.update(llm_result)
    return result

# pgcrypto Tools (DATABASE_URL - port 5432, patients/doctors/appointments/labresults)
@mcp.tool()
def decrypt_patient_email(patient_name: str) -> Dict[str, Any]:
    """Decrypt a patient's email address using pgcrypto."""
    query = "SELECT pgp_sym_decrypt(email, 'secret') as email FROM patients WHERE full_name = %s"
    result = execute_query(DATABASE_URL, query, (patient_name,))
    result['extension'] = 'pgcrypto'
    return result

@mcp.tool()
def find_patient_by_encrypted_email(encrypted_search: str) -> Dict[str, Any]:
    """Find patients by searching encrypted email data using pgcrypto."""
    query = """
    SELECT patient_id, full_name, 
           pgp_sym_decrypt(email, 'secret') as decrypted_email
    FROM patients 
    WHERE pgp_sym_decrypt(email, 'secret') ILIKE %s
    """
    result = execute_query(DATABASE_URL, query, (f'%{encrypted_search}%',))
    result['extension'] = 'pgcrypto'
    return result

@mcp.tool()
def get_all_patient_emails_decrypted() -> Dict[str, Any]:
    """Get all patient emails in decrypted format using pgcrypto."""
    query = """
    SELECT patient_id, full_name, 
           pgp_sym_decrypt(email, 'secret') as email,
           pgp_sym_decrypt(phone, 'secret') as phone
    FROM patients
    ORDER BY patient_id
    """
    result = execute_query(DATABASE_URL, query)
    result['extension'] = 'pgcrypto'
    return result

@mcp.tool()
def encrypt_new_patient_data(name: str, email: str, phone: str) -> Dict[str, Any]:
    """Encrypt and insert new patient data using pgcrypto."""
    query = """
    INSERT INTO patients (full_name, email, phone)
    VALUES (%s, pgp_sym_encrypt(%s, 'secret'), pgp_sym_encrypt(%s, 'secret'))
    RETURNING patient_id, full_name
    """
    result = execute_query(DATABASE_URL, query, (name, email, phone))
    result['extension'] = 'pgcrypto'
    return result

@mcp.tool()
def pgcrypto_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about encrypted hospital data using pgcrypto."""
    llm_result = call_llm_api(question, 'pgcrypto')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'pgcrypto'}
    
    result = execute_query(DATABASE_URL, llm_result['sql'])
    result.update(llm_result)
    return result

# pg_stat_statements Tools (DATABASE_URL - port 5432, hospital data)
@mcp.tool()
def get_slowest_queries(limit: int = 5) -> Dict[str, Any]:
    """Get the slowest running queries using pg_stat_statements."""
    query = """
    SELECT query, calls, total_exec_time, mean_exec_time, 
           (total_exec_time/sum(total_exec_time) OVER()) * 100 as percent_of_total
    FROM pg_stat_statements 
    ORDER BY mean_exec_time DESC 
    LIMIT %s
    """
    result = execute_query(DATABASE_URL, query, (limit,))
    result['extension'] = 'pg_stat_statements'
    return result

@mcp.tool()
def get_most_called_queries(limit: int = 5) -> Dict[str, Any]:
    """Get the most frequently called queries using pg_stat_statements."""
    query = """
    SELECT query, calls, total_exec_time, mean_exec_time
    FROM pg_stat_statements 
    ORDER BY calls DESC 
    LIMIT %s
    """
    result = execute_query(DATABASE_URL, query, (limit,))
    result['extension'] = 'pg_stat_statements'
    return result

@mcp.tool()
def get_query_performance_summary() -> Dict[str, Any]:
    """Get overall query performance summary using pg_stat_statements."""
    query = """
    SELECT 
        count(*) as total_queries,
        sum(calls) as total_calls,
        sum(total_exec_time) as total_time_ms,
        avg(mean_exec_time) as avg_query_time_ms,
        max(max_exec_time) as slowest_query_ms
    FROM pg_stat_statements
    """
    result = execute_query(DATABASE_URL, query)
    result['extension'] = 'pg_stat_statements'
    return result

@mcp.tool()
def find_queries_by_pattern(pattern: str) -> Dict[str, Any]:
    """Find queries matching a specific pattern using pg_stat_statements."""
    query = """
    SELECT query, calls, total_exec_time, mean_exec_time
    FROM pg_stat_statements 
    WHERE query ILIKE %s
    ORDER BY mean_exec_time DESC
    """
    result = execute_query(DATABASE_URL, query, (f'%{pattern}%',))
    result['extension'] = 'pg_stat_statements'
    return result

@mcp.tool()
def reset_pg_stat_statements() -> Dict[str, Any]:
    """Reset pg_stat_statements statistics."""
    query = "SELECT pg_stat_statements_reset()"
    result = execute_query(DATABASE_URL, query)
    result['extension'] = 'pg_stat_statements'
    return result

@mcp.tool()
def pg_stat_statements_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about query performance using pg_stat_statements."""
    llm_result = call_llm_api(question, 'pg_stat_statements')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'pg_stat_statements'}
    
    result = execute_query(DATABASE_URL, llm_result['sql'])
    result.update(llm_result)
    return result

# pg_prewarm Tools (DATABASE_URL - port 5432, patients/doctors/appointments/labresults)
@mcp.tool()
def prewarm_table(table_name: str) -> Dict[str, Any]:
    """Prewarm a specific table into buffer cache using pg_prewarm."""
    query = "SELECT pg_prewarm(%s)"
    result = execute_query(DATABASE_URL, query, (table_name,))
    result['extension'] = 'pg_prewarm'
    return result

@mcp.tool()
def prewarm_patients_table() -> Dict[str, Any]:
    """Prewarm the patients table for better performance."""
    query = "SELECT pg_prewarm('patients')"
    result = execute_query(DATABASE_URL, query)
    result['extension'] = 'pg_prewarm'
    return result

@mcp.tool()
def pg_prewarm_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about cache warming using pg_prewarm."""
    llm_result = call_llm_api(question, 'pg_prewarm')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'pg_prewarm'}
    
    result = execute_query(DATABASE_URL, llm_result['sql'])
    result.update(llm_result)
    return result

# pg_partman Tools (DATABASE_URL2 - port 5433, partman/public schemas)
@mcp.tool()
def get_partition_info() -> Dict[str, Any]:
    """Get information about partitioned tables using pg_partman."""
    query = """
    SELECT schemaname, tablename
    FROM pg_tables 
    WHERE schemaname IN ('partman', 'public')
    """
    result = execute_query(DATABASE_URL2, query)
    result['extension'] = 'pg_partman'
    return result

@mcp.tool()
def run_partition_maintenance() -> Dict[str, Any]:
    """Run partition maintenance tasks using pg_partman."""
    query = "SELECT partman.run_maintenance_proc()"
    result = execute_query(DATABASE_URL2, query)
    result['extension'] = 'pg_partman'
    return result

@mcp.tool()
def pg_partman_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about partition management using pg_partman."""
    llm_result = call_llm_api(question, 'pg_partman')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'pg_partman'}
    
    result = execute_query(DATABASE_URL2, llm_result['sql'])
    result.update(llm_result)
    return result

# pg_cron Tools (DATABASE_URL3 - port 5434, traffic_data/traffic_summary)
@mcp.tool()
def get_scheduled_jobs() -> Dict[str, Any]:
    """Get all scheduled cron jobs using pg_cron."""
    query = """
    SELECT jobid, schedule, command, nodename, nodeport, database, username, active
    FROM cron.job
    ORDER BY jobid
    """
    result = execute_query(DATABASE_URL3, query)
    result['extension'] = 'pg_cron'
    return result

@mcp.tool()
def schedule_database_job(schedule: str, command: str, job_name: str) -> Dict[str, Any]:
    """Schedule a new database job using pg_cron."""
    query = """
    SELECT cron.schedule(%s, %s, %s)
    """
    result = execute_query(DATABASE_URL3, query, (job_name, schedule, command))
    result['extension'] = 'pg_cron'
    return result

@mcp.tool()
def get_job_run_details(job_id: int) -> Dict[str, Any]:
    """Get execution details for a specific cron job."""
    query = """
    SELECT jobid, runid, job_pid, database, username, command, status, 
           return_message, start_time, end_time
    FROM cron.job_run_details 
    WHERE jobid = %s
    ORDER BY start_time DESC
    LIMIT 10
    """
    result = execute_query(DATABASE_URL3, query, (job_id,))
    result['extension'] = 'pg_cron'
    return result

@mcp.tool()
def unschedule_job(job_id: int) -> Dict[str, Any]:
    """Unschedule a cron job using pg_cron."""
    query = "SELECT cron.unschedule(%s)"
    result = execute_query(DATABASE_URL3, query, (job_id,))
    result['extension'] = 'pg_cron'
    return result

@mcp.tool()
def get_traffic_data() -> Dict[str, Any]:
    """Get traffic data from traffic_data table."""
    query = """
    SELECT * FROM traffic_data 
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    ORDER BY timestamp DESC
    LIMIT 100
    """
    result = execute_query(DATABASE_URL3, query)
    result['extension'] = 'pg_cron'
    return result

@mcp.tool()
def get_traffic_summary() -> Dict[str, Any]:
    """Get traffic summary statistics from traffic_summary table."""
    query = """
    SELECT * FROM traffic_summary 
    ORDER BY location
    """
    result = execute_query(DATABASE_URL3, query)
    result['extension'] = 'pg_cron'
    return result

@mcp.tool()
def pg_cron_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about scheduled jobs using pg_cron."""
    llm_result = call_llm_api(question, 'pg_cron')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'pg_cron'}
    
    result = execute_query(DATABASE_URL3, llm_result['sql'])
    result.update(llm_result)
    return result

# hypopg Tools (DATABASE_URL4 - port 5445, products/public schema)
@mcp.tool()
def create_hypothetical_index(table_name: str, column_name: str) -> Dict[str, Any]:
    """Create a hypothetical index using hypopg for testing."""
    query = "SELECT * FROM hypopg_create_index('CREATE INDEX ON %s (%s)')" % (table_name, column_name)
    result = execute_query(DATABASE_URL4, query)
    result['extension'] = 'hypopg'
    return result

@mcp.tool()
def list_hypothetical_indexes() -> Dict[str, Any]:
    """List all hypothetical indexes created with hypopg."""
    query = "SELECT * FROM hypopg()"
    result = execute_query(DATABASE_URL4, query)
    result['extension'] = 'hypopg'
    return result

@mcp.tool()
def explain_with_hypothetical_indexes(sql_query: str) -> Dict[str, Any]:
    """Explain a query plan with hypothetical indexes using hypopg."""
    query = f"EXPLAIN (FORMAT JSON) {sql_query}"
    result = execute_query(DATABASE_URL4, query)
    result['extension'] = 'hypopg'
    return result

@mcp.tool()
def drop_hypothetical_index(index_id: int) -> Dict[str, Any]:
    """Drop a specific hypothetical index using hypopg."""
    query = "SELECT hypopg_drop_index(%s)"
    result = execute_query(DATABASE_URL4, query, (index_id,))
    result['extension'] = 'hypopg'
    return result

@mcp.tool()
def reset_hypothetical_indexes() -> Dict[str, Any]:
    """Remove all hypothetical indexes using hypopg."""
    query = "SELECT hypopg_reset()"
    result = execute_query(DATABASE_URL4, query)
    result['extension'] = 'hypopg'
    return result

@mcp.tool()
def hypopg_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about hypothetical indexes using hypopg."""
    llm_result = call_llm_api(question, 'hypopg')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'hypopg'}
    
    result = execute_query(DATABASE_URL4, llm_result['sql'])
    result.update(llm_result)
    return result

# TimescaleDB Tools (DATABASE_URL5 - port 5111, energy_usage)
@mcp.tool()
def get_energy_usage_by_time_bucket(interval: str = '1 hour') -> Dict[str, Any]:
    """Get energy usage aggregated by time buckets using TimescaleDB."""
    query = f"""
    SELECT time_bucket('{interval}', timestamp) AS time_bucket,
           device_id,
           AVG(usage_kwh) as avg_usage,
           MAX(usage_kwh) as peak_usage,
           COUNT(*) as readings
    FROM energy_usage
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    GROUP BY time_bucket, device_id
    ORDER BY time_bucket DESC, device_id
    """
    result = execute_query(DATABASE_URL5, query)
    result['extension'] = 'timescaledb'
    return result

@mcp.tool()
def get_peak_energy_usage_times() -> Dict[str, Any]:
    """Get peak energy usage times using TimescaleDB time-series functions."""
    query = """
    SELECT time_bucket('15 minutes', timestamp) AS time_bucket,
           SUM(usage_kwh) as total_usage
    FROM energy_usage
    WHERE timestamp >= NOW() - INTERVAL '7 days'
    GROUP BY time_bucket
    ORDER BY total_usage DESC
    LIMIT 10
    """
    result = execute_query(DATABASE_URL5, query)
    result['extension'] = 'timescaledb'
    return result

@mcp.tool()
def get_device_energy_trends(device_id: str, hours: int = 24) -> Dict[str, Any]:
    """Get energy usage trends for a specific device using TimescaleDB."""
    query = """
    SELECT time_bucket('1 hour', timestamp) AS hour,
           device_id,
           AVG(usage_kwh) as avg_usage,
           MIN(usage_kwh) as min_usage,
           MAX(usage_kwh) as max_usage
    FROM energy_usage
    WHERE device_id = %s AND timestamp >= NOW() - INTERVAL '%s hours'
    GROUP BY hour, device_id
    ORDER BY hour
    """
    result = execute_query(DATABASE_URL5, query, (device_id, hours))
    result['extension'] = 'timescaledb'
    return result

@mcp.tool()
def create_timescale_continuous_aggregate(view_name: str, interval: str = '1 day') -> Dict[str, Any]:
    """Create a continuous aggregate for energy data using TimescaleDB."""
    query = f"""
    CREATE MATERIALIZED VIEW {view_name}
    WITH (timescaledb.continuous) AS
    SELECT time_bucket('{interval}', timestamp) AS bucket,
           device_id,
           AVG(usage_kwh) as avg_usage,
           SUM(usage_kwh) as total_usage
    FROM energy_usage
    GROUP BY bucket, device_id
    """
    result = execute_query(DATABASE_URL5, query)
    result['extension'] = 'timescaledb'
    return result

@mcp.tool()
def get_timescale_hypertables() -> Dict[str, Any]:
    """Get information about TimescaleDB hypertables."""
    query = """
    SELECT hypertable_schema, hypertable_name, num_dimensions, 
           num_chunks, compression_enabled, replication_factor
    FROM timescaledb_information.hypertables
    """
    result = execute_query(DATABASE_URL5, query)
    result['extension'] = 'timescaledb'
    return result

@mcp.tool()
def compress_timescale_chunks(table_name: str, older_than: str = '7 days') -> Dict[str, Any]:
    """Compress old chunks in a TimescaleDB hypertable."""
    query = f"""
    SELECT compress_chunk(i, if_not_compressed => true)
    FROM show_chunks('{table_name}', older_than => INTERVAL '{older_than}') i
    """
    result = execute_query(DATABASE_URL5, query)
    result['extension'] = 'timescaledb'
    return result

@mcp.tool()
def get_timescale_chunk_info(table_name: str) -> Dict[str, Any]:
    """Get chunk information for a TimescaleDB hypertable."""
    query = f"""
    SELECT chunk_schema, chunk_name, range_start, range_end, 
           is_compressed, compressed_chunk_schema, compressed_chunk_name
    FROM timescaledb_information.chunks
    WHERE hypertable_name = '{table_name}'
    ORDER BY range_start DESC
    """
    result = execute_query(DATABASE_URL5, query)
    result['extension'] = 'timescaledb'
    return result

@mcp.tool()
def timescaledb_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about time-series data using TimescaleDB."""
    llm_result = call_llm_api(question, 'timescaledb')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'timescaledb'}
    
    result = execute_query(DATABASE_URL5, llm_result['sql'])
    result.update(llm_result)
    return result

# Infrastructure Monitoring Tools (DATABASE_URL - port 5432)
@mcp.tool()
def get_database_connection_stats() -> Dict[str, Any]:
    """Get database connection statistics for infrastructure monitoring."""
    query = """
    SELECT 
        count(*) as total_connections,
        count(*) FILTER (WHERE state = 'active') as active_connections,
        count(*) FILTER (WHERE state = 'idle') as idle_connections,
        max(backend_start) as oldest_connection
    FROM pg_stat_activity
    WHERE pid <> pg_backend_pid()
    """
    result = execute_query(DATABASE_URL, query)
    result['extension'] = 'infrastructure'
    return result

@mcp.tool()
def get_database_size_info() -> Dict[str, Any]:
    """Get database size information for infrastructure monitoring."""
    query = """
    SELECT 
        datname as database_name,
        pg_size_pretty(pg_database_size(datname)) as size,
        pg_database_size(datname) as size_bytes
    FROM pg_database
    WHERE datistemplate = false
    ORDER BY pg_database_size(datname) DESC
    """
    result = execute_query(DATABASE_URL, query)
    result['extension'] = 'infrastructure'
    return result

@mcp.tool()
def get_table_sizes() -> Dict[str, Any]:
    """Get table size information for infrastructure monitoring."""
    query = """
    SELECT 
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
        pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) as table_size,
        pg_total_relation_size(schemaname||'.'||tablename) as total_size_bytes
    FROM pg_tables
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    LIMIT 10
    """
    result = execute_query(DATABASE_URL, query)
    result['extension'] = 'infrastructure'
    return result

@mcp.tool()
def get_index_usage_stats() -> Dict[str, Any]:
    """Get index usage statistics for infrastructure monitoring."""
    query = """
    SELECT 
        schemaname,
        tablename,
        indexname,
        idx_scan,
        idx_tup_read,
        idx_tup_fetch
    FROM pg_stat_user_indexes
    WHERE idx_scan > 0
    ORDER BY idx_scan DESC
    LIMIT 20
    """
    result = execute_query(DATABASE_URL, query)
    result['extension'] = 'infrastructure'
    return result

@mcp.tool()
def infrastructure_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about infrastructure monitoring."""
    llm_result = call_llm_api(question, 'infrastructure')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'infrastructure'}
    
    result = execute_query(DATABASE_URL, llm_result['sql'])
    result.update(llm_result)
    return result

# Apache AGE Tools (DATABASE_URL2 - port 5433, graph schemas)
@mcp.tool()
def create_age_graph(graph_name: str) -> Dict[str, Any]:
    """Create a new graph using Apache AGE."""
    query = f"SELECT * FROM ag_catalog.create_graph('{graph_name}')"
    result = execute_query(DATABASE_URL2, query)
    result['extension'] = 'apache_age'
    return result

@mcp.tool()
def get_age_graphs() -> Dict[str, Any]:
    """List all graphs in Apache AGE."""
    query = "SELECT * FROM ag_catalog.ag_graph"
    result = execute_query(DATABASE_URL2, query)
    result['extension'] = 'apache_age'
    return result

@mcp.tool()
def cypher_query(graph_name: str, cypher_query: str) -> Dict[str, Any]:
    """Execute a Cypher query on an Apache AGE graph."""
    query = f"SELECT * FROM cypher('{graph_name}', $${cypher_query}$$) as (result agtype)"
    result = execute_query(DATABASE_URL2, query)
    result['extension'] = 'apache_age'
    return result

@mcp.tool()
def apache_age_natural_language_query(question: str) -> Dict[str, Any]:
    """Process natural language questions about graph data using Apache AGE."""
    llm_result = call_llm_api(question, 'apache_age')
    if 'error' in llm_result:
        return {'success': False, 'error': llm_result['error'], 'extension': 'apache_age'}
    
    result = execute_query(DATABASE_URL2, llm_result['sql'])
    result.update(llm_result)
    return result

if __name__ == "__main__":
    try:
        import sys
        import logging
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        print("Server interrupted", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
