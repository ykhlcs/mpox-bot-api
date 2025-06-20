import sqlite3
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = "mpox_bot.db"
DB_VERSION = 3  # Incremented version

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create version table if not exists
    c.execute('''
        CREATE TABLE IF NOT EXISTS Version (
            version INTEGER PRIMARY KEY
        )
    ''')
    
    # Get current version
    c.execute('SELECT version FROM Version ORDER BY version DESC LIMIT 1')
    current_version = c.fetchone()
    current_version = current_version[0] if current_version else 0
    
 # ===== Migration 2 to 3 =====
    if current_version < 3:
        logger.info("Migrating database to version 3")
        
        # Check if Message table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Message'")
        if c.fetchone():
            # Add intent_id column if missing
            c.execute("PRAGMA table_info(Message)")
            columns = [col[1] for col in c.fetchall()]
            if 'intent_id' not in columns:
                c.execute('ALTER TABLE Message ADD COLUMN intent_id INTEGER REFERENCES Intent(intent_id)')
        
        # Update version
        c.execute('INSERT INTO Version (version) VALUES (3)')
        conn.commit()
    
    # ===== Create Tables (Version 2 Schema) =====
    # User table remains the same
    c.execute('''
        CREATE TABLE IF NOT EXISTS User (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            first_name TEXT,
            last_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Updated Intent table
    c.execute('''
        CREATE TABLE IF NOT EXISTS Intent (
            intent_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            category TEXT  -- NEW COLUMN
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS Message (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT REFERENCES User(user_id),
            content TEXT NOT NULL,
            intent_id INTEGER REFERENCES Intent(intent_id),
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS Response (
            response_id INTEGER PRIMARY KEY AUTOINCREMENT,
            intent_id INTEGER REFERENCES Intent(intent_id),
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS Misinformation (
            misinfo_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            verification_status TEXT DEFAULT 'pending',
            source_url TEXT
        )
    ''')
    
    # Create indexes
    c.execute('CREATE INDEX IF NOT EXISTS idx_message_content ON Message(content)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_misinfo_content ON Misinformation(content)')
    
    # Pre-populate intents
    intents = [
        ("misinfo_check", "Verification of potential misinformation", "classification"),
        ("transmission_risk", "Questions about transmission scenarios", "scenario"),
        ("symptom_query", "Inquiries about mpox symptoms", "information"),
        ("prevention_info", "Questions about prevention methods", "information"),
        ("news_request", "Requests for latest mpox news", "service"),
        ("general_question", "Other health-related questions", "information"),
        ("greeting", "Initial user greetings", "service"),
        ("off_topic", "Non-health related queries", "service")
    ]

    # Update existing intents
    for name, desc, category in intents:
        c.execute('''
            INSERT OR IGNORE INTO Intent (name, description, category)
            VALUES (?, ?, ?)
        ''', (name, desc, category))
    
    c.executemany('''
        INSERT OR IGNORE INTO Intent (name, description, category) VALUES (?, ?, ?)
    ''', intents)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized with simplified schema")

# ===== USER OPERATIONS =====
def log_user(user_id, username=None, first_name=None, last_name=None):
    """Log or update user information"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO User (user_id, username, first_name, last_name)
            VALUES (?, ?, ?, ?)
        ''', (user_id, username, first_name, last_name))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error logging user: {str(e)}")
        return False
    finally:
        conn.close()

# ===== MESSAGE OPERATIONS =====
def log_message(user_id, content, intent_name=None):
    """Log message with intent mapping"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Get intent ID if provided
        intent_id = None
        if intent_name:
            c.execute('SELECT intent_id FROM Intent WHERE name = ?', (intent_name,))
            intent_row = c.fetchone()
            intent_id = intent_row[0] if intent_row else None
        
        c.execute('''
            INSERT INTO Message (user_id, content, intent_id)
            VALUES (?, ?, ?)
        ''', (user_id, content, intent_id))
        
        message_id = c.lastrowid
        conn.commit()
        return message_id
    except Exception as e:
        logger.error(f"Error logging message: {str(e)}")
        return None
    finally:
        conn.close()

# ===== MISINFORMATION LOGGING =====
def log_misinformation(content, source_url=None):
    """Log detected misinformation claim"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT OR IGNORE INTO Misinformation (content, source_url)
            VALUES (?, ?)
        ''', (content, source_url))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error logging misinformation: {str(e)}")
        return False
    finally:
        conn.close()

# ===== RESPONSE LOGGING =====
def log_response(intent_name, response_content):
    """Log a bot response"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            INSERT INTO Response (intent_id, content)
            VALUES (
                (SELECT intent_id FROM Intent WHERE name = ?),
                ?
            )
        ''', (intent_name, response_content))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error logging response: {str(e)}")
        return False
    finally:
        conn.close()

# ===== ANALYSIS QUERIES =====
def get_misinformation_stats():
    """Get summary of misinformation claims"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            SELECT verification_status, COUNT(*) 
            FROM Misinformation 
            GROUP BY verification_status
        ''')
        return c.fetchall()
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return []
    finally:
        conn.close()

def export_to_csv(filename="mpox_bot_data.csv"):
    """Export data to CSV"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Export messages
        with open(f"messages_{filename}", "w") as f:
            c = conn.cursor()
            c.execute("SELECT * FROM Message")
            columns = [desc[0] for desc in c.description]
            f.write(",".join(columns) + "\n")
            for row in c.fetchall():
                f.write(",".join(map(str, row)) + "\n")
        
        # Export misinformation
        with open(f"misinformation_{filename}", "w") as f:
            c = conn.cursor()
            c.execute("SELECT * FROM Misinformation")
            columns = [desc[0] for desc in c.description]
            f.write(",".join(columns) + "\n")
            for row in c.fetchall():
                f.write(",".join(map(str, row)) + "\n")
        
        return True
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        return False
    finally:
        conn.close()

# Initialize database when imported
init_db()