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