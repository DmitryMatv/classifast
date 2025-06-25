import os
import pandas as pd


# Define file paths for Excel and Parquet
excel_file_path = "data/2022-NAICS-Codes-listed-numerically-2-Digit-through-6-Digit.xlsx"  # UNGM_UNSPSC_26-May-2025..xlsx # unspsc-english-v260801.1.xlsx
parquet_file_path = os.path.splitext(excel_file_path)[0] + ".parquet"

# Try to load from Parquet; if not found, load from Excel and save to Parquet
if os.path.exists(parquet_file_path):
    print(f"Loading data from existing Parquet file: {parquet_file_path}")
    df = pd.read_parquet(parquet_file_path, engine="pyarrow")
else:
    print(f"Loading data from Excel file: {excel_file_path}")
    # This block replaces the original pd.read_excel call
    df = pd.read_excel(
        excel_file_path,
        # header=12,
        dtype=str,
        # nrows=10000,  # sampling
    )
    print(f"Saving data to Parquet file for future use: {parquet_file_path}")
    df.to_parquet(parquet_file_path, engine="pyarrow", compression="snappy")

"""
# Read CSV
# ETIM Classes
csv_path = "ETIMARTCLASS.csv"

with open(csv_path, 'rb') as rawdata:
    chars = chardet.detect(rawdata.read())
print(chars['encoding'])

df = pd.read_csv(csv_path,  
                 encoding=chars['encoding'], 
                 sep=';',
                 header=0,
                 dtype=str,
                 #nrows=10000 # sampling
                 ) 

# ETIM Groups
dfETIMgroup = pd.read_csv("ETIMARTGROUP.csv",
                       encoding=chars['encoding'],
                       sep=';',
                       dtype=str)

df = df.merge(dfETIMgroup,
              on='ARTGROUPID', 
              how='left')

df.drop(columns=['ARTGROUPID', 'ARTCLASSVERSION', 'ARTCLASSVERSIONDATE'], inplace=True)

# ETIM Synonyms
dfETIMsynonyms = pd.read_csv("ETIMARTCLASSSYNONYMMAP.csv",
                       encoding=chars['encoding'],
                       sep=';',
                       dtype=str)

synonyms_grouped = dfETIMsynonyms.groupby('ARTCLASSID')['CLASSSYNONYM'].agg(lambda x: ', '.join(x))

df = df.merge(synonyms_grouped,
              on='ARTCLASSID', 
              how='left')

#df['CLASSSYNONYM'].fillna('', inplace=True)


df = df.astype(str)
df = df.reset_index(drop=True)

df.to_csv('test.csv', sep=';', index=False)

"""


# Get basic information about the dataframe structure
df.info()

# Basic information about the dataframe
print("\nShape of the dataframe (rows, columns):", df.shape)
print("\nFirst and last rows of the dataframe:")
print(df.head(5))
print("\n...")  # Indicate there are skipped entries
print(df.tail())

print("\nColumn names:\n", list(df.columns))

# Additional information
print("\nSummary statistics:")
print(df.describe(include="all"))

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Number of unique values in each column
print("\nNumber of unique values in each column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")


# Identical rows
print("\nCompletely identical rows:")
print(df.duplicated().sum())


# Count number of duplicate keys
key_col = "Seq. No."

print("\nNumber keys count:")
print(df[key_col].value_counts()[df[key_col].value_counts() > 1])

print("\nAll rows with the same key (sorted by key):")
df_key_duplicates = df[df.duplicated(subset=[key_col], keep=False)]
print(df_key_duplicates)
df_key_duplicates.to_excel("Data/duplicate_keys.xlsx")  # Save to Excel


# Examine the hierarchical structure based on Parent key with limits
parent_col = "Segment"
title_col = "Segment Title"

if parent_col in df.columns:
    print("\nHierarchy structure:")
    root_items = df[df[parent_col].isnull()]
    print(f"Root level items: {len(root_items)}")

    # Count items at different levels
    levels = {}
    for parent in df[parent_col].dropna().unique():
        child_count = len(df[df[parent_col] == parent])
        levels[parent] = child_count

    # Sort by child count for better visualization
    sorted_levels = sorted(levels.items(), key=lambda x: x[1], reverse=True)

    # Show only top N and bottom N
    N = 5  # Change this to show more or fewer items

    # Aggregate children counts for each parent
    parent_child_counts = df.groupby(parent_col)[key_col].count().reset_index()

    # Sort parents by number of children in descending order
    sorted_parents = parent_child_counts.sort_values(key_col, ascending=False)

    # Print top N parents with their human-readable names and child count
    print(f"\nTop {N} parents by number of children:")
    for _, row in sorted_parents.head(N).iterrows():
        parent_key = row[parent_col]
        child_count = row[key_col]

        # Find the human-readable name for this parent
        parent_name_rows = df[df[key_col] == parent_key]
        parent_name = (
            parent_name_rows[title_col].iloc[0]
            if not parent_name_rows.empty
            else "Unknown"
        )

        print(f"Parent {parent_key} ({parent_name}): {child_count} children")

    if len(sorted_levels) > N * 2:
        print("\n...")  # Indicate there are skipped entries

    print(f"\nBottom {N} parents by number of children:")
    for parent, count in sorted_levels[-N:]:
        parent_name = (
            df[df[key_col] == parent][title_col].values[0]
            if parent in df[key_col].values
            else "Unknown"
        )
        print(f"Parent {parent} ({parent_name}): {count} children")
