import psycopg2
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS
from typing import Optional, Tuple

def get_db_connection():
    """Create a database connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def check_db_connection() -> str:
    """Check database connection status."""
    conn = get_db_connection()
    status = "connected" if conn else "disconnected"
    if conn:
        conn.close()
    return status

def search_contact_by_name(name: str) -> Tuple[bool, Optional[str]]:
    """Search for a contact by name and return (found, company_name)."""
    if not name:
        return False, None
        
    conn = get_db_connection()
    if not conn:
        return False, None
        
    found_in_database = False
    company = None
    
    try:
        cur = conn.cursor()
        # Check for name match (case insensitive)
        query = """
        SELECT co.name 
        FROM contacts c 
        LEFT JOIN companies co ON c.company_id = co.company_id 
        WHERE LOWER(c.first_name || ' ' || c.last_name) = LOWER(%s)
        """
        cur.execute(query, (name,))
        result = cur.fetchone()
        
        if result:
            found_in_database = True
            company = result[0]
        
        cur.close()
    except Exception as e:
        print(f"Database query error: {e}")
    finally:
        conn.close()
        
    return found_in_database, company
