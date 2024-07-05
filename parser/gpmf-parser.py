import sys
from parse import recursive, parse_value  # Import the necessary functions

def parse_gpmd_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            # Use the recursive function to parse the data
            for element, parents in recursive(data):
                try:
                    value = parse_value(element)
                except ValueError:
                    value = element.data
                print(" > ".join([x.decode('ascii') for x in parents]), element.key.decode('ascii'), value)
    except Exception as e:
        print(f"Error parsing GPMD file: {e}", file=sys.stderr)

if __name__ == "__main__":
    gpmd_file_path = r'C:\Users\Gebruiker\Q2-Research-ML-Design\[4] GUI\GPMF_Parser\GH010038.MP4.gpmf'
    parse_gpmd_file(gpmd_file_path)
