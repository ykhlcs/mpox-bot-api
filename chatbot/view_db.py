import sqlite3
import pandas as pd

def view_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    print(f"Database: {db_path}")
    print("Tables:", ", ".join(tables))
    
    while True:
        print("\nOptions:")
        print("1. View table data")
        print("2. Run custom query")
        print("3. Export table to CSV")
        print("4. Exit")
        
        choice = input("Select option: ")
        
        if choice == "1":
            table_name = input("Enter table name: ")
            if table_name in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                print(f"\nFirst 10 rows of {table_name}:")
                print(df.head(10))
            else:
                print("Invalid table name")
                
        elif choice == "2":
            query = input("Enter SQL query: ")
            try:
                df = pd.read_sql_query(query, conn)
                print(f"\nQuery results ({len(df)} rows):")
                print(df)
            except Exception as e:
                print(f"Error: {str(e)}")
                
        elif choice == "3":
            table_name = input("Enter table name to export: ")
            if table_name in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                csv_file = f"{table_name}.csv"
                df.to_csv(csv_file, index=False)
                print(f"Exported to {csv_file}")
            else:
                print("Invalid table name")
                
        elif choice == "4":
            break
            
    conn.close()

if __name__ == "__main__":
    view_database("mpox_bot.db")