import pandas as pd
import numpy as np

# Read the CSV file (ISIC_Rev_4_english_structure.Txt)
print("Reading ISIC_Rev_4_english_structure.Txt...")
csv_file = pd.read_csv("data/ISIC_Rev_4_english_structure.Txt")
print(f"CSV file shape: {csv_file.shape}")
print("CSV file columns:", csv_file.columns.tolist())
print("\nFirst few rows of CSV file:")
print(csv_file.head())

# Read the Excel file (ISICRev4_Titles.xlsx)
print("\n" + "=" * 60)
print("Reading ISICRev4_Titles.xlsx...")
excel_file = pd.read_excel("data/ISICRev4_Titles.xlsx")
print(f"Excel file shape: {excel_file.shape}")
print("Excel file columns:", excel_file.columns.tolist())
print("\nFirst few rows of Excel file:")
print(excel_file.head())

# Check if both files have the same number of rows
print("\n" + "=" * 60)
print("COMPARISON ANALYSIS")
print("=" * 60)
print(f"CSV file rows: {len(csv_file)}")
print(f"Excel file rows: {len(excel_file)}")
print(f"Same number of rows: {len(csv_file) == len(excel_file)}")

# Check if ExplanatoryNoteExclusion column exists in Excel file
if "ExplanatoryNoteExclusion" in excel_file.columns:
    print("\n✓ ExplanatoryNoteExclusion column found in Excel file")

    # Get the Description column from CSV and ExplanatoryNoteExclusion from Excel
    csv_descriptions = csv_file["Description"].astype(str).str.strip()
    excel_explanatory = excel_file["ExplanatoryNoteExclusion"].astype(str).str.strip()

    print(
        f"\nCSV Description column - Non-null values: {csv_descriptions.notna().sum()}"
    )
    print(
        f"Excel ExplanatoryNoteExclusion column - Non-null values: {excel_explanatory.notna().sum()}"
    )

    # Handle NaN values
    csv_descriptions_clean = csv_descriptions.fillna("")
    excel_explanatory_clean = excel_explanatory.fillna("")

    # Compare the values
    matches = csv_descriptions_clean == excel_explanatory_clean
    num_matches = matches.sum()
    total_rows = len(csv_descriptions_clean)

    print(f"\nMATCH RESULTS:")
    print(f"Total rows compared: {total_rows}")
    print(f"Exact matches: {num_matches}")
    print(f"Match percentage: {(num_matches/total_rows)*100:.2f}%")

    if num_matches == total_rows:
        print("✓ ALL VALUES MATCH!")
    else:
        print(f"✗ {total_rows - num_matches} values DO NOT match")

        # Show mismatches
        mismatches = ~matches
        if mismatches.any():
            print(f"\nFirst 10 mismatches:")
            mismatch_df = pd.DataFrame(
                {
                    "Index": csv_file.index[mismatches][:10],
                    "CSV_Description": csv_descriptions_clean[mismatches][:10],
                    "Excel_ExplanatoryNoteExclusion": excel_explanatory_clean[
                        mismatches
                    ][:10],
                }
            )
            print(mismatch_df.to_string(index=False))

            # Check if it's just ordering issue
            csv_set = set(csv_descriptions_clean)
            excel_set = set(excel_explanatory_clean)

            print(f"\nSET COMPARISON:")
            print(f"Unique values in CSV Description: {len(csv_set)}")
            print(f"Unique values in Excel ExplanatoryNoteExclusion: {len(excel_set)}")
            print(f"Values in CSV but not in Excel: {len(csv_set - excel_set)}")
            print(f"Values in Excel but not in CSV: {len(excel_set - csv_set)}")

            if csv_set == excel_set:
                print("✓ Same unique values (possibly different order)")
            else:
                print("✗ Different sets of unique values")

                # Show some examples of differences
                if csv_set - excel_set:
                    print(f"\nSample values in CSV but not in Excel:")
                    for val in list(csv_set - excel_set)[:5]:
                        print(f"  '{val}'")

                if excel_set - csv_set:
                    print(f"\nSample values in Excel but not in CSV:")
                    for val in list(excel_set - csv_set)[:5]:
                        print(f"  '{val}'")

else:
    print("\n✗ ExplanatoryNoteExclusion column not found in Excel file")
    print("Available columns:", excel_file.columns.tolist())
