import pandas as pd

# Read the CSV file, treating Code column as text to preserve leading zeros
df = pd.read_csv('data/ISIC_Rev_4_english_structure.Txt', dtype={'Code': 'string'})

print("First few rows:")
print(df.head(10))
print(f"\nTotal number of records: {len(df)}")

# Count how many records have codes with 4 or more characters
codes_4_or_more = df[df['Code'].str.len() >= 4]
count_4_or_more = len(codes_4_or_more)

print(f"\nNumber of records with codes having 4 or more characters: {count_4_or_more}")

# Show some examples of codes with 4 or more characters
print(f"\nFirst 10 examples of codes with 4+ characters:")
print(codes_4_or_more['Code'].head(10).tolist())

# Let's also see the distribution of code lengths
code_lengths = df['Code'].str.len().value_counts().sort_index()
print(f"\nDistribution of code lengths:")
for length, count in code_lengths.items():
    print(f"Length {length}: {count} records")
