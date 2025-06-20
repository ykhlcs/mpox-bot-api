import sqlite3

def migrate_database():
    conn = sqlite3.connect('mpox_bot.db')
    c = conn.cursor()
    
    try:
        # Add intent_id column to Message table
        c.execute("PRAGMA table_info(Message)")
        columns = [col[1] for col in c.fetchall()]
        if 'intent_id' not in columns:
            c.execute('''
                ALTER TABLE Message
                ADD COLUMN intent_id INTEGER REFERENCES Intent(intent_id)
            ''')
            print("Added intent_id column to Message table")
        
        # Ensure Intent table exists
        c.execute('''
            CREATE TABLE IF NOT EXISTS Intent (
                intent_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        ''')
        
        # Pre-populate intents
        intents = [
            ("misinfo_check", "Verification of potential misinformation"),
            ("transmission_risk", "Questions about transmission scenarios"),
            ("symptom_query", "Inquiries about mpox symptoms"),
            ("prevention_info", "Questions about prevention methods"),
            ("news_request", "Requests for latest mpox news"),
            ("general_question", "Other health-related questions"),
            ("joke_request", "Requests for jokes/humor"),
            ("off_topic", "Non-mpox related queries"),
            ("greeting", "Initial greetings"),
            ("casual_reply", "Thank you/casual responses")
        ]
        
        c.executemany('''
            INSERT OR IGNORE INTO Intent (name, description) VALUES (?, ?)
        ''', intents)
        
        conn.commit()
        print("Database migration successful")
    except Exception as e:
        print(f"Migration failed: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()