import re
import shutil
from datetime import datetime
from pathlib import Path

def update_yaml_line_by_line(file_path, line_number, new_date):
    file_path = Path(file_path)
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    
    try:
        # Create backup
        shutil.copy2(file_path, backup_path)
        
        # Read all lines
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Update the specific line (convert to 0-based indexing)
        target_line_index = line_number - 1

        # Replace with new date, keeping the exact format
        lines[target_line_index] = f"  test_end_date: '{new_date}'\n"
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)
        
        print(f"Successfully updated line {line_number} to: test_end_date: '{new_date}'")
        
        # Remove backup on success
        backup_path.unlink()
    except Exception as e:
        print(f"Error updating YAML file: {e}")
        # Restore from backup if it exists
        if backup_path.exists():
            shutil.move(backup_path, file_path)
            print("Restored original file from backup")
        raise
        
    except Exception as e:
        print(f"Error updating YAML file: {e}")
        # Restore from backup if it exists
        if backup_path.exists():
            shutil.move(backup_path, file_path)
            print("Restored original file from backup")
        raise

# Example usage
if __name__ == "__main__":
    # Update to today's date
    today = datetime.now().strftime('%Y-%m-%d')
    yaml_file_path = 'config/parameters.yml'
    update_yaml_line_by_line(yaml_file_path, 11, today)
    
