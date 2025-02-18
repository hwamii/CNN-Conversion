with open('weights.csv', encoding='utf-8-sig') as f:  # Remove BOM automatically
    for i, line in enumerate(f):
        values = line.strip().split(',')
        try:
            numbers = [float(v) for v in values]
        except ValueError:
            print(f'❌ Error in row {i}: {values}')