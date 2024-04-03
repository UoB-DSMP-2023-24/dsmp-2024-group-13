import pandas as pd

def preprocess_data(df, relevant_columns=None):
    """
    Preprocess the given DataFrame by filtering relevant columns, checking for missing values, and cleaning the data.

    Parameters:
    df (DataFrame): The data frame to preprocess.
    relevant_columns (list): A list of column names to keep in the DataFrame.

    Returns:
    DataFrame: A cleaned DataFrame with relevant columns and no missing cdr3 sequences.
    dict: A report dictionary with information on missing values and unique values in the DataFrame.
    """
    if relevant_columns is None:
        relevant_columns = [
            'cdr3.alpha', 'v.alpha', 'j.alpha', 'cdr3.beta', 'v.beta', 'd.beta', 'j.beta',
            'species', 'mhc.a', 'mhc.b', 'antigen.gene', 'antigen.epitope', 'vdjdb.score', 'mhc.class'
        ]

    # Filtering the DataFrame to keep only relevant columns
    filtered_data = df[relevant_columns]

    # Checking for missing values in crucial columns
    missing_values = filtered_data.isnull().sum()

    # Examining the number of unique values in categorical columns for potential encoding strategies
    unique_values = filtered_data.nunique()

    # Removing rows with missing cdr3 sequences
    df_cleaned = filtered_data.dropna(subset=['cdr3.alpha', 'cdr3.beta'])
    df_cleaned.reset_index(drop=True, inplace=True)
    
    # Creating a report dictionary
    report = {
        "Missing Values": missing_values,
        "Unique Values": unique_values
    }

    return df_cleaned, report

# Usage example:
# Assuming 'df' is your DataFrame containing the raw data
df_cleaned, report = preprocess_data(df)
print(df_cleaned.head())  # Displays the first few rows of the cleaned DataFrame
print(report)  # Prints out the report of missing and unique values



def filter_by_length_range(df, column_name):
    """
    Asks the user for length bounds and filters the DataFrame to include rows where the length of
    the specified column's sequence falls within the provided bounds.

    Parameters:
    df (DataFrame): The data frame to filter.
    column_name (str): The name of the sequence column to check (e.g., 'cdr3.alpha').

    Returns:
    DataFrame: A DataFrame filtered by the specified length range.
    """
    # Ask the user for length bounds
    lower_bound = int(input(f"Enter lower bound for {column_name} length: "))
    upper_bound = int(input(f"Enter upper bound for {column_name} length: "))
    
    # Calculate the sequence lengths
    df[column_name + '.length'] = df[column_name].apply(len)
    
    # Filter based on the length range
    return df[(df[column_name + '.length'] >= lower_bound) & (df[column_name + '.length'] <= upper_bound)]

def filter_by_species(df):
    """
    Asks the user for species to filter by and filters the DataFrame to include rows where the
    species column matches any of the species provided.

    Parameters:
    df (DataFrame): The data frame to filter.

    Returns:
    DataFrame: A DataFrame filtered by the specified species.
    """
    # Ask the user for species to filter by
    input_species = input("Enter the species to filter by (separated by commas): ")
    species_to_filter = [species.strip() for species in input_species.split(',')]
    
    return df[df['species'].isin(species_to_filter)]


def filter_by_minimum_score(df, column='vdjdb.score'):
    """
    Filters the DataFrame based on a minimum score inputted by the user for a specified column.
    
    Parameters:
    df (DataFrame): The data frame to filter.
    column (str): The name of the column to apply the filter on. Defaults to 'vdjdb.score'.
    
    Returns:
    DataFrame: A DataFrame filtered based on the user-specified minimum score.
    """
    # Prompting user for minimum score
    min_score = input(f"Enter the minimum score (inclusive) for {column}: ")
    
    # Validating user input
    try:
        min_score = int(min_score)
        if min_score < 0 or min_score > 3:
            print("Score out of range. Please enter a value between 0 and 3.")
            return df
    except ValueError:
        print("Invalid input. Please enter an integer value.")
        return df
    
    # Filtering the DataFrame
    filtered_df = df[df[column] >= min_score]
    
    return filtered_df

def filter_by_mhc_class(df, column='mhc.class'):
    """
    Filters the DataFrame based on a user-specified MHC class ('MHCI' or 'MHCII').

    Parameters:
    df (DataFrame): The data frame to filter.
    column (str): The name of the column to apply the filter on. Defaults to 'mhc.class'.

    Returns:
    DataFrame: A DataFrame filtered based on the user-specified MHC class.
    """
    # Prompting user for MHC class
    mhc_class = input(f"Enter the MHC class to keep ('MHCI' or 'MHCII'): ").strip()

    # Validating user input
    if mhc_class not in ['MHCI', 'MHCII']:
        print("Invalid input. Please enter 'MHCI' or 'MHCII'.")
        return df
    
    # Filtering the DataFrame
    filtered_df = df[df[column] == mhc_class]
    
    return filtered_df

"""
example usage:

# Assuming df is  DataFrame
df = df_cleaned

# Apply length range filters
df_filtered_alpha = filter_by_length_range(df, 'cdr3.alpha')
df_filtered_beta = filter_by_length_range(df, 'cdr3.beta')
df_filtered_epitope = filter_by_length_range(df, 'antigen.epitope')
df_filtered_min_score = filter_by_minimum_score(df)

# Intersect the filtered DataFrames to get only rows that meet all criteria
df_length_filtered = df_filtered_alpha.merge(df_filtered_beta).merge(df_filtered_epitope)

# Further filter by species
df_final_filtered = filter_by_species(df_length_filtered)

"""