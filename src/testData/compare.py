import csv
import sys

def load_csv(filename):
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        data = []
        for row in reader:
            for val in row:
                try:
                    data.append(float(val.strip()))
                except ValueError:
                    print(f"Invalid float value in {filename}: {val}")
                    sys.exit(1)
        return data

def compare_csvs(file1, file2, precision=4):
    data1 = load_csv(file1)
    data2 = load_csv(file2)

    if len(data1) != len(data2):
        print(f"Length mismatch: {len(data1)} vs {len(data2)}")
        return False
    tolerance = 1e-4  # 4 decimal places


    mismatch_found = False
    for i, (v1, v2) in enumerate(zip(data1, data2)):
        if abs(v1 - v2) > tolerance:
            print(f"Mismatch at index {i}: {v1:.10f} vs {v2:.10f}")

    if not mismatch_found:
        print("✅ All values match up to 4 decimal places!")
    else:
        print("❌ Mismatches found.")

    return not mismatch_found

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare two CSVs up to 4 decimal places.")
    parser.add_argument("csv1", help="First CSV file")
    parser.add_argument("csv2", help="Second CSV file")
    args = parser.parse_args()

    compare_csvs(args.csv1, args.csv2)
